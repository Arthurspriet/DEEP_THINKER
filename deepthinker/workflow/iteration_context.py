"""
Iteration Context Manager for DeepThinker 2.0.

Manages stateful context evolution across iterations, ensuring councils
receive enriched, evolving context each loop rather than static inputs.

Key responsibilities:
- Update ResearchContext after each iteration
- Compress findings into prior_knowledge summaries
- Derive subgoals from focus areas
- Track iteration progress and context deltas
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ContextDelta:
    """
    Represents changes to context between iterations.
    
    Used for logging and tracking progress.
    """
    iteration: int
    focus_areas_added: List[str] = field(default_factory=list)
    focus_areas_removed: List[str] = field(default_factory=list)
    questions_added: List[str] = field(default_factory=list)
    questions_resolved: List[str] = field(default_factory=list)
    data_needs_added: List[str] = field(default_factory=list)
    data_needs_resolved: List[str] = field(default_factory=list)
    subgoals_added: List[str] = field(default_factory=list)
    web_searches_performed: int = 0
    prior_knowledge_lines_added: int = 0
    
    def summary(self) -> str:
        """Generate a summary of context changes."""
        changes = []
        if self.focus_areas_added:
            changes.append(f"+{len(self.focus_areas_added)} focus areas")
        if self.focus_areas_removed:
            changes.append(f"-{len(self.focus_areas_removed)} focus areas")
        if self.questions_added:
            changes.append(f"+{len(self.questions_added)} questions")
        if self.questions_resolved:
            changes.append(f"✓{len(self.questions_resolved)} questions resolved")
        if self.data_needs_added:
            changes.append(f"+{len(self.data_needs_added)} data needs")
        if self.data_needs_resolved:
            changes.append(f"✓{len(self.data_needs_resolved)} data needs resolved")
        if self.web_searches_performed:
            changes.append(f"{self.web_searches_performed} web searches")
        if self.prior_knowledge_lines_added:
            changes.append(f"+{self.prior_knowledge_lines_added} lines knowledge")
        
        return f"Iteration {self.iteration}: " + ", ".join(changes) if changes else f"Iteration {self.iteration}: No changes"


@dataclass
class IterationState:
    """
    Tracks the full state of iteration context.
    
    This is stored in MissionState and evolved each iteration.
    """
    current_iteration: int = 0
    focus_areas: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    data_needs: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    prior_knowledge: str = ""
    evidence_requests: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    requires_evidence: bool = False
    web_searches_total: int = 0
    # History for tracking
    delta_history: List[ContextDelta] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_iteration": self.current_iteration,
            "focus_areas": self.focus_areas,
            "unresolved_questions": self.unresolved_questions,
            "data_needs": self.data_needs,
            "subgoals": self.subgoals,
            "prior_knowledge_length": len(self.prior_knowledge),
            "evidence_requests": self.evidence_requests,
            "gaps": self.gaps,
            "requires_evidence": self.requires_evidence,
            "web_searches_total": self.web_searches_total,
            "delta_history_count": len(self.delta_history)
        }


class IterationContextManager:
    """
    Manages context evolution across research and synthesis iterations.
    
    This is the core component that ensures each iteration receives
    different, evolved context based on prior findings and feedback.
    """
    
    def __init__(self, max_prior_knowledge_chars: int = 5000):
        """
        Initialize the context manager.
        
        Args:
            max_prior_knowledge_chars: Maximum chars for prior_knowledge summary
        """
        self.max_prior_knowledge_chars = max_prior_knowledge_chars
        self._state: Optional[IterationState] = None
    
    def initialize(self) -> IterationState:
        """Initialize fresh iteration state."""
        self._state = IterationState()
        return self._state
    
    def get_state(self) -> Optional[IterationState]:
        """Get current iteration state."""
        return self._state
    
    def set_state(self, state: IterationState) -> None:
        """Set iteration state (e.g., from restored mission)."""
        self._state = state
    
    def update_from_research(
        self,
        findings: Any,  # ResearchFindings
        evaluation: Optional[Any] = None  # ResearchEvaluation
    ) -> ContextDelta:
        """
        Update context based on research findings and evaluation.
        
        This is the primary method for evolving context after a research iteration.
        
        Args:
            findings: ResearchFindings from ResearcherCouncil
            evaluation: Optional ResearchEvaluation from EvaluatorCouncil
            
        Returns:
            ContextDelta describing what changed
        """
        if self._state is None:
            self._state = IterationState()
        
        self._state.current_iteration += 1
        delta = ContextDelta(iteration=self._state.current_iteration)
        
        # Track previous state for delta calculation
        prev_focus = set(self._state.focus_areas)
        prev_questions = set(self._state.unresolved_questions)
        prev_data_needs = set(self._state.data_needs)
        
        # Extract findings
        if hasattr(findings, 'gaps'):
            new_gaps = [g for g in findings.gaps if g not in self._state.gaps]
            self._state.gaps.extend(new_gaps)
            delta.focus_areas_added.extend(new_gaps)
        
        if hasattr(findings, 'unresolved_questions'):
            new_questions = [q for q in findings.unresolved_questions 
                           if q not in self._state.unresolved_questions]
            self._state.unresolved_questions.extend(new_questions)
            delta.questions_added.extend(new_questions)
        
        if hasattr(findings, 'evidence_requests'):
            new_evidence = [e for e in findings.evidence_requests
                          if e not in self._state.evidence_requests]
            self._state.evidence_requests.extend(new_evidence)
            delta.data_needs_added.extend(new_evidence)
        
        if hasattr(findings, 'next_focus_areas'):
            new_focus = [f for f in findings.next_focus_areas
                        if f not in self._state.focus_areas]
            self._state.focus_areas.extend(new_focus)
            delta.focus_areas_added.extend(new_focus)
        
        if hasattr(findings, 'web_search_count'):
            delta.web_searches_performed = findings.web_search_count
            self._state.web_searches_total += findings.web_search_count
        
        # Compress and add to prior knowledge
        if hasattr(findings, 'key_points') and findings.key_points:
            new_knowledge = self._compress_findings(findings)
            if new_knowledge:
                old_len = len(self._state.prior_knowledge)
                self._state.prior_knowledge = self._merge_prior_knowledge(
                    self._state.prior_knowledge,
                    new_knowledge
                )
                delta.prior_knowledge_lines_added = len(self._state.prior_knowledge) - old_len
        
        # Apply evaluation feedback if provided
        if evaluation is not None:
            self._apply_evaluation(evaluation, delta)
        
        # Derive subgoals from focus areas
        self._state.subgoals = self._derive_subgoals(self._state.focus_areas)
        delta.subgoals_added = self._state.subgoals
        
        # Determine if evidence is required for next iteration
        self._state.requires_evidence = bool(
            self._state.data_needs or 
            self._state.evidence_requests or
            self._state.current_iteration > 1
        )
        
        # Calculate removed items (resolved in this iteration)
        current_focus = set(self._state.focus_areas)
        current_questions = set(self._state.unresolved_questions)
        current_data_needs = set(self._state.data_needs)
        
        delta.focus_areas_removed = list(prev_focus - current_focus)
        delta.questions_resolved = list(prev_questions - current_questions)
        delta.data_needs_resolved = list(prev_data_needs - current_data_needs)
        
        # Store delta in history
        self._state.delta_history.append(delta)
        
        logger.info(delta.summary())
        
        return delta
    
    def _apply_evaluation(self, evaluation: Any, delta: ContextDelta) -> None:
        """Apply evaluation feedback to state."""
        if hasattr(evaluation, 'gaps'):
            new_gaps = [g for g in evaluation.gaps if g not in self._state.gaps]
            self._state.gaps.extend(new_gaps)
            delta.focus_areas_added.extend(new_gaps)
        
        if hasattr(evaluation, 'unresolved_questions'):
            new_questions = [q for q in evaluation.unresolved_questions
                           if q not in self._state.unresolved_questions]
            self._state.unresolved_questions.extend(new_questions)
            delta.questions_added.extend(new_questions)
        
        if hasattr(evaluation, 'evidence_requests'):
            new_evidence = [e for e in evaluation.evidence_requests
                          if e not in self._state.evidence_requests]
            self._state.evidence_requests.extend(new_evidence)
            delta.data_needs_added.extend(new_evidence)
        
        if hasattr(evaluation, 'next_focus_areas'):
            new_focus = [f for f in evaluation.next_focus_areas
                        if f not in self._state.focus_areas]
            self._state.focus_areas.extend(new_focus)
            delta.focus_areas_added.extend(new_focus)
    
    def _compress_findings(self, findings: Any) -> str:
        """
        Compress findings into a knowledge summary.
        
        Args:
            findings: ResearchFindings object
            
        Returns:
            Compressed knowledge string
        """
        parts = []
        
        if hasattr(findings, 'iteration'):
            parts.append(f"## Iteration {findings.iteration} Findings")
        
        if hasattr(findings, 'key_points') and findings.key_points:
            parts.append("Key Points:")
            for point in findings.key_points[:5]:
                parts.append(f"  - {point}")
        
        if hasattr(findings, 'recommendations') and findings.recommendations:
            parts.append("Recommendations:")
            for rec in findings.recommendations[:3]:
                parts.append(f"  - {rec}")
        
        return "\n".join(parts)
    
    def _merge_prior_knowledge(self, existing: str, new: str) -> str:
        """
        Merge new knowledge with existing, respecting size limits.
        
        Args:
            existing: Existing prior knowledge
            new: New knowledge to add
            
        Returns:
            Merged knowledge string within size limit
        """
        combined = f"{existing}\n\n{new}" if existing else new
        
        # Trim if too long
        if len(combined) > self.max_prior_knowledge_chars:
            # Keep the most recent content
            combined = combined[-self.max_prior_knowledge_chars:]
            # Try to find a clean break point
            break_point = combined.find('\n\n')
            if break_point > 0 and break_point < 500:
                combined = combined[break_point + 2:]
        
        return combined.strip()
    
    def _derive_subgoals(self, focus_areas: List[str], limit: int = 5) -> List[str]:
        """
        Convert focus areas into actionable subgoals.
        
        Args:
            focus_areas: List of focus areas
            limit: Maximum number of subgoals
            
        Returns:
            List of subgoal strings
        """
        subgoals = []
        
        for area in focus_areas[:limit]:
            # Simple transformation - could be enhanced with LLM
            if area.lower().startswith("how "):
                subgoals.append(f"Investigate: {area}")
            elif area.lower().startswith("what "):
                subgoals.append(f"Determine: {area}")
            elif area.lower().startswith("why "):
                subgoals.append(f"Explain: {area}")
            else:
                subgoals.append(f"Research: {area}")
        
        return subgoals
    
    def get_research_context_updates(self) -> Dict[str, Any]:
        """
        Get updates to apply to ResearchContext for next iteration.
        
        Returns:
            Dictionary with context field updates
        """
        if self._state is None:
            return {}
        
        return {
            "focus_areas": self._state.focus_areas[:5],
            "unresolved_questions": self._state.unresolved_questions[:5],
            "data_needs": self._state.data_needs[:5] + self._state.evidence_requests[:5],
            "prior_knowledge": self._state.prior_knowledge,
            "subgoals": self._state.subgoals[:5],
            "requires_evidence": self._state.requires_evidence
        }
    
    def should_continue_research(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8
    ) -> tuple:
        """
        Determine if research should continue based on state.
        
        Args:
            max_iterations: Maximum allowed iterations
            confidence_threshold: Confidence level to stop at
            
        Returns:
            Tuple of (should_continue, reason)
        """
        if self._state is None:
            return True, "No state initialized"
        
        if self._state.current_iteration >= max_iterations:
            return False, f"Maximum iterations ({max_iterations}) reached"
        
        # Continue if there are unresolved items
        if self._state.gaps:
            return True, f"{len(self._state.gaps)} gaps remaining"
        
        if self._state.unresolved_questions:
            return True, f"{len(self._state.unresolved_questions)} questions remaining"
        
        if self._state.evidence_requests:
            return True, f"{len(self._state.evidence_requests)} evidence requests pending"
        
        # If no work items, can stop
        return False, "All issues resolved"
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all iterations for logging.
        
        Returns:
            Summary dictionary
        """
        if self._state is None:
            return {"status": "not initialized"}
        
        return {
            "total_iterations": self._state.current_iteration,
            "current_focus_areas": len(self._state.focus_areas),
            "current_questions": len(self._state.unresolved_questions),
            "current_data_needs": len(self._state.data_needs),
            "total_web_searches": self._state.web_searches_total,
            "prior_knowledge_size": len(self._state.prior_knowledge),
            "deltas": [d.summary() for d in self._state.delta_history]
        }


# Module-level convenience functions

def create_context_manager() -> IterationContextManager:
    """Create a new IterationContextManager instance."""
    return IterationContextManager()


def compress_findings_to_knowledge(findings: Any, max_chars: int = 2000) -> str:
    """
    Compress ResearchFindings into a knowledge summary.
    
    Args:
        findings: ResearchFindings object
        max_chars: Maximum characters in output
        
    Returns:
        Compressed knowledge string
    """
    parts = []
    
    if hasattr(findings, 'summary'):
        parts.append(findings.summary[:500])
    
    if hasattr(findings, 'key_points') and findings.key_points:
        parts.append("Key Findings:")
        for point in findings.key_points[:5]:
            parts.append(f"- {point[:100]}")
    
    if hasattr(findings, 'recommendations') and findings.recommendations:
        parts.append("Recommendations:")
        for rec in findings.recommendations[:3]:
            parts.append(f"- {rec[:100]}")
    
    result = "\n".join(parts)
    return result[:max_chars] if len(result) > max_chars else result

