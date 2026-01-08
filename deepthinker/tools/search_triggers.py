"""
Search Trigger Logic for DeepThinker 2.0.

Provides objective-aware web search triggering rules, per-phase quotas,
and rationale logging for search decisions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Keywords that should trigger web search
SEARCH_TRIGGER_KEYWORDS = [
    "analyze", "analysis", "risk", "assessment", "forecast",
    "long-term", "future", "predict", "trend", "market",
    "comparison", "compare", "benchmark", "best practice",
    "state of the art", "sota", "latest", "current",
    "research", "study", "data", "statistics", "evidence",
]

# Phase-specific search quotas (minimum expected searches)
PHASE_SEARCH_QUOTAS = {
    "recon": 1,
    "reconnaissance": 1,
    "context": 1,
    "research": 2,
    "analysis": 1,
    "deep": 1,
    "deep_analysis": 2,
    "design": 0,
    "implementation": 0,
    "testing": 0,
    "synthesis": 0,
}


@dataclass
class SearchDecision:
    """
    Result of a search trigger decision.
    
    Attributes:
        should_search: Whether search should be triggered
        reason: Explanation for the decision
        quota: Remaining quota for this phase
        queries: Suggested search queries (if search triggered)
        triggered_by: What triggered the decision (keywords, uncertainty, quota, etc.)
    """
    should_search: bool
    reason: str
    quota: int = 0
    queries: List[str] = field(default_factory=list)
    triggered_by: str = ""


class SearchTriggerManager:
    """
    Manages web search triggering decisions for DeepThinker missions.
    
    Provides:
    - Objective-aware keyword detection
    - Per-phase search quotas
    - Uncertainty-based triggering
    - Rationale logging for all decisions
    """
    
    def __init__(
        self,
        enable_search: bool = True,
        global_quota: int = 10,
        trigger_keywords: Optional[List[str]] = None,
        phase_quotas: Optional[Dict[str, int]] = None,
        max_searches_per_mission: int = 5,
    ):
        """
        Initialize search trigger manager.
        
        Args:
            enable_search: Whether search is enabled at all
            global_quota: Maximum searches per mission (legacy, kept for compatibility)
            trigger_keywords: Custom keywords that trigger search
            phase_quotas: Custom per-phase quotas
            max_searches_per_mission: Maximum searches per mission (anti-hallucination budget)
        """
        self.enable_search = enable_search
        self.global_quota = global_quota
        self.max_searches_per_mission = max_searches_per_mission
        self.trigger_keywords = trigger_keywords or SEARCH_TRIGGER_KEYWORDS
        self.phase_quotas = phase_quotas or PHASE_SEARCH_QUOTAS.copy()
        
        # Tracking
        self._searches_performed: Dict[str, int] = {}  # phase -> count
        self._total_searches: int = 0
        self._searches_used: int = 0  # Budget tracking
        self._decision_log: List[Dict[str, Any]] = []
    
    def should_trigger_search(
        self,
        objective: str,
        phase_name: str,
        uncertainty: float = 0.5,
        data_needs: Optional[List[str]] = None,
        unresolved_questions: Optional[List[str]] = None,
        force: bool = False,
    ) -> SearchDecision:
        """
        Determine if a web search should be triggered.
        
        Decision logic (in priority order):
        1. If search disabled globally, don't search
        2. If global quota exhausted, don't search
        3. If phase quota exhausted and not forced, don't search
        4. If forced, search
        5. If objective contains trigger keywords, search
        6. If high uncertainty (> 0.6), search
        7. If data needs or unresolved questions exist, search
        8. If phase quota not yet met, search
        
        Args:
            objective: Mission objective text
            phase_name: Current phase name
            uncertainty: Uncertainty score (0-1)
            data_needs: List of identified data needs
            unresolved_questions: List of unresolved questions
            force: Force search regardless of criteria
            
        Returns:
            SearchDecision with decision and rationale
        """
        data_needs = data_needs or []
        unresolved_questions = unresolved_questions or []
        
        # Normalize phase name
        phase_key = self._normalize_phase_name(phase_name)
        
        # Get quotas
        phase_quota = self.phase_quotas.get(phase_key, 0)
        searches_in_phase = self._searches_performed.get(phase_key, 0)
        remaining_phase_quota = max(0, phase_quota - searches_in_phase)
        remaining_global_quota = max(0, self.global_quota - self._total_searches)
        
        # Check: search disabled
        if not self.enable_search:
            return self._log_decision(
                should_search=False,
                reason="Internet search disabled globally",
                phase=phase_key,
                triggered_by="disabled"
            )
        
        # Check: search budget exhausted (anti-hallucination budget guard)
        if self._searches_used >= self.max_searches_per_mission:
            return self._log_decision(
                should_search=False,
                reason=f"Search budget exhausted ({self._searches_used}/{self.max_searches_per_mission})",
                phase=phase_key,
                triggered_by="budget_exhausted"
            )
        
        # Check: global quota exhausted (legacy check)
        if remaining_global_quota <= 0:
            return self._log_decision(
                should_search=False,
                reason=f"Global search quota exhausted ({self._total_searches}/{self.global_quota})",
                phase=phase_key,
                triggered_by="quota_exhausted"
            )
        
        # Check: forced
        if force:
            queries = self._generate_queries(objective, data_needs, unresolved_questions)
            return self._log_decision(
                should_search=True,
                reason="Search forced by caller",
                phase=phase_key,
                quota=remaining_phase_quota,
                queries=queries,
                triggered_by="forced"
            )
        
        # Check: objective keywords
        keyword_match = self._check_keywords(objective)
        if keyword_match:
            queries = self._generate_queries(objective, data_needs, unresolved_questions)
            return self._log_decision(
                should_search=True,
                reason=f"Objective contains trigger keywords: {keyword_match[:3]}",
                phase=phase_key,
                quota=remaining_phase_quota,
                queries=queries,
                triggered_by="keywords"
            )
        
        # Check: high uncertainty
        if uncertainty > 0.6:
            queries = self._generate_queries(objective, data_needs, unresolved_questions)
            return self._log_decision(
                should_search=True,
                reason=f"High uncertainty ({uncertainty:.2f} > 0.6) - search for evidence",
                phase=phase_key,
                quota=remaining_phase_quota,
                queries=queries,
                triggered_by="uncertainty"
            )
        
        # Check: data needs or questions
        if data_needs or unresolved_questions:
            queries = self._generate_queries(objective, data_needs, unresolved_questions)
            return self._log_decision(
                should_search=True,
                reason=f"Data needs ({len(data_needs)}) or questions ({len(unresolved_questions)}) require search",
                phase=phase_key,
                quota=remaining_phase_quota,
                queries=queries,
                triggered_by="data_needs"
            )
        
        # Check: phase quota not met
        if remaining_phase_quota > 0 and phase_quota > 0:
            queries = self._generate_queries(objective, data_needs, unresolved_questions)
            return self._log_decision(
                should_search=True,
                reason=f"Phase quota not met ({searches_in_phase}/{phase_quota})",
                phase=phase_key,
                quota=remaining_phase_quota,
                queries=queries,
                triggered_by="phase_quota"
            )
        
        # Default: don't search
        return self._log_decision(
            should_search=False,
            reason=f"No search criteria met: uncertainty={uncertainty:.2f}, "
                   f"data_needs={len(data_needs)}, questions={len(unresolved_questions)}",
            phase=phase_key,
            triggered_by="no_criteria"
        )
    
    def record_search(self, phase_name: str, query_count: int = 1) -> None:
        """
        Record that a search was performed.
        
        Args:
            phase_name: Phase where search occurred
            query_count: Number of queries executed
        """
        phase_key = self._normalize_phase_name(phase_name)
        
        if phase_key not in self._searches_performed:
            self._searches_performed[phase_key] = 0
        
        self._searches_performed[phase_key] += query_count
        self._total_searches += query_count
        self._searches_used += query_count  # Update budget usage
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search statistics.
        
        Returns:
            Dictionary with search statistics
        """
        return {
            "enabled": self.enable_search,
            "total_searches": self._total_searches,
            "global_quota": self.global_quota,
            "global_remaining": max(0, self.global_quota - self._total_searches),
            "searches_per_phase": self._searches_performed.copy(),
            "decision_count": len(self._decision_log),
            "budget_used": self._searches_used,
            "budget_max": self.max_searches_per_mission,
            "budget_remaining": max(0, self.max_searches_per_mission - self._searches_used),
        }
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get all search decisions made."""
        return self._decision_log.copy()
    
    def _check_keywords(self, text: str) -> List[str]:
        """Check if text contains trigger keywords."""
        text_lower = text.lower()
        matches = []
        
        for keyword in self.trigger_keywords:
            if keyword.lower() in text_lower:
                matches.append(keyword)
        
        return matches
    
    def _normalize_phase_name(self, phase_name: str) -> str:
        """Normalize phase name for quota lookup."""
        lower = phase_name.lower().strip()
        
        # Map common variations
        mappings = {
            "reconnaissance": "recon",
            "context gathering": "context",
            "situation analysis": "analysis",
            "deep analysis": "deep",
            "final synthesis": "synthesis",
            "synthesis & report": "synthesis",
        }
        
        for full, short in mappings.items():
            if full in lower:
                return short
        
        # Check for keyword matches
        for quota_key in PHASE_SEARCH_QUOTAS:
            if quota_key in lower:
                return quota_key
        
        return lower.replace(" ", "_")
    
    def _generate_queries(
        self,
        objective: str,
        data_needs: List[str],
        unresolved_questions: List[str],
    ) -> List[str]:
        """
        Generate search queries based on needs.
        
        Args:
            objective: Mission objective
            data_needs: Identified data needs
            unresolved_questions: Unresolved questions
            
        Returns:
            List of search query strings
        """
        queries = []
        
        # Add objective-based query (first 100 chars)
        if objective:
            queries.append(objective[:100].strip())
        
        # Add data needs as queries
        for need in data_needs[:2]:
            if len(need) > 10:
                queries.append(need[:100].strip())
        
        # Add questions as queries
        for question in unresolved_questions[:2]:
            if len(question) > 10:
                # Remove question marks for better search
                q = question.rstrip("?").strip()[:100]
                queries.append(q)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)
        
        return unique[:3]  # Max 3 queries
    
    def _log_decision(
        self,
        should_search: bool,
        reason: str,
        phase: str,
        quota: int = 0,
        queries: Optional[List[str]] = None,
        triggered_by: str = "",
    ) -> SearchDecision:
        """Log and create a search decision."""
        decision = SearchDecision(
            should_search=should_search,
            reason=reason,
            quota=quota,
            queries=queries or [],
            triggered_by=triggered_by
        )
        
        self._decision_log.append({
            "phase": phase,
            "should_search": should_search,
            "reason": reason,
            "triggered_by": triggered_by,
            "queries": queries or [],
        })
        
        # Log for visibility
        if should_search:
            logger.info(f"Search triggered for {phase}: {reason}")
        else:
            logger.debug(f"Search skipped for {phase}: {reason}")
        
        return decision


def should_trigger_search(
    objective: str,
    phase: str,
    uncertainty: float = 0.5,
    enable_search: bool = True,
) -> bool:
    """
    Convenience function for simple search decision.
    
    Args:
        objective: Mission objective
        phase: Current phase name
        uncertainty: Uncertainty score
        enable_search: Whether search is enabled
        
    Returns:
        True if search should be triggered
    """
    if not enable_search:
        return False
    
    # Quick keyword check
    obj_lower = objective.lower()
    for keyword in SEARCH_TRIGGER_KEYWORDS:
        if keyword.lower() in obj_lower:
            return True
    
    # High uncertainty check
    if uncertainty > 0.6:
        return True
    
    # Phase quota check
    phase_lower = phase.lower()
    for quota_phase in ["recon", "research", "deep"]:
        if quota_phase in phase_lower:
            return True
    
    return False


def get_phase_search_quota(phase: str) -> int:
    """
    Get search quota for a phase.
    
    Args:
        phase: Phase name
        
    Returns:
        Quota for this phase type
    """
    phase_lower = phase.lower()
    
    for quota_phase, quota in PHASE_SEARCH_QUOTAS.items():
        if quota_phase in phase_lower:
            return quota
    
    return 0

