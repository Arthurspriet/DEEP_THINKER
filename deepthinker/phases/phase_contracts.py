"""
Phase Contracts for DeepThinker Phase Purity Enforcement.

Defines what each phase can and cannot produce, and enforces these
contracts through semantic inspection of outputs.

Phase Contracts:
- Reconnaissance: Sources, facts, unknowns, questions. NO recommendations/conclusions.
- Analysis: Synthesis, causal links, validated facts. NO new sources/recommendations.
- Synthesis: Recommendations, scenarios, conclusions. NO raw research/new facts.

Violations trigger:
- Contamination score computation
- Evaluator score penalties
- Potential corrective iterations
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ContaminationType(str, Enum):
    """Types of phase contamination."""
    PREMATURE_RECOMMENDATION = "premature_recommendation"
    PREMATURE_CONCLUSION = "premature_conclusion"
    PREMATURE_SYNTHESIS = "premature_synthesis"
    LATE_SOURCING = "late_sourcing"
    LATE_RESEARCH = "late_research"
    RAW_RESEARCH_IN_SYNTHESIS = "raw_research_in_synthesis"
    NEW_FACTS_IN_SYNTHESIS = "new_facts_in_synthesis"


@dataclass
class PhaseContract:
    """
    Contract defining allowed and forbidden outputs for a phase.
    
    Attributes:
        phase_name: Name of the phase
        allowed_types: Output types allowed in this phase
        forbidden_types: Output types forbidden in this phase
        allowed_patterns: Regex patterns for allowed content
        forbidden_patterns: Regex patterns for forbidden content
        max_speculation_ratio: Max fraction of speculative content
        requires_sources: Whether outputs must cite sources
    """
    phase_name: str
    allowed_types: Set[str] = field(default_factory=set)
    forbidden_types: Set[str] = field(default_factory=set)
    allowed_patterns: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    max_speculation_ratio: float = 0.5
    requires_sources: bool = False
    
    def __post_init__(self):
        """Compile regex patterns."""
        self._compiled_allowed = [
            re.compile(p, re.IGNORECASE) for p in self.allowed_patterns
        ]
        self._compiled_forbidden = [
            re.compile(p, re.IGNORECASE) for p in self.forbidden_patterns
        ]


@dataclass
class PhaseViolation:
    """
    A single phase contract violation.
    
    Attributes:
        contamination_type: Type of contamination detected
        description: Human-readable description
        excerpt: Text excerpt that triggered violation
        severity: Severity level (0-1)
        line_number: Approximate line number
    """
    contamination_type: ContaminationType
    description: str
    excerpt: str = ""
    severity: float = 0.5
    line_number: int = 0


@dataclass
class PhaseContamination:
    """
    Result of phase contamination analysis.
    
    Attributes:
        phase_name: Phase that was analyzed
        violations: List of detected violations
        contamination_score: Overall contamination score (0-1)
        is_clean: Whether output passed contamination check
        recommended_penalty: Suggested evaluator score penalty
        should_retry: Whether corrective iteration is recommended
    """
    phase_name: str
    violations: List[PhaseViolation] = field(default_factory=list)
    contamination_score: float = 0.0
    is_clean: bool = True
    recommended_penalty: float = 0.0
    should_retry: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "phase_name": self.phase_name,
            "violation_count": len(self.violations),
            "contamination_score": self.contamination_score,
            "is_clean": self.is_clean,
            "recommended_penalty": self.recommended_penalty,
            "should_retry": self.should_retry,
            "violation_types": [v.contamination_type.value for v in self.violations],
        }


# =============================================================================
# Phase Contract Definitions
# =============================================================================

PHASE_CONTRACTS: Dict[str, PhaseContract] = {
    "reconnaissance": PhaseContract(
        phase_name="reconnaissance",
        allowed_types={"sources", "facts", "unknowns", "questions", "landscape", "hypotheses"},
        forbidden_types={"recommendations", "conclusions", "synthesis", "action_items", "final_report"},
        allowed_patterns=[
            r'\b(source|reference|according to|cited|found)\b',
            r'\b(fact|evidence|data|information)\b',
            r'\b(unknown|unclear|question|investigate)\b',
            r'\b(landscape|overview|context|background)\b',
        ],
        forbidden_patterns=[
            r'\b(recommend|should|must|advise|suggest)\b.{0,50}\b(implement|adopt|use|action)\b',
            r'\b(in conclusion|therefore we recommend|final assessment)\b',
            r'\b(action items?|next steps?|implementation plan)\b',
            r'\b(we conclude|our verdict|decision is)\b',
        ],
        max_speculation_ratio=0.3,
        requires_sources=True,
    ),
    
    "analysis": PhaseContract(
        phase_name="analysis",
        allowed_types={"synthesis", "causal_links", "validated_facts", "analysis", "findings", "trade_offs"},
        forbidden_types={"new_sources", "recommendations", "final_report", "raw_research"},
        allowed_patterns=[
            r'\b(analysis|finding|pattern|trend|correlation)\b',
            r'\b(because|therefore|thus|leads to|causes)\b',
            r'\b(trade-?off|balance|tension|versus)\b',
            r'\b(validates?|confirms?|supports?|contradicts?)\b',
        ],
        forbidden_patterns=[
            r'\baccording to (?:new |recent |additional )(?:source|research|study)\b',
            r'\b(recommend|should|must|advise)\b.{0,30}\b(implement|adopt|action)\b',
            r'\b(action items?|next steps?|implementation)\b',
        ],
        max_speculation_ratio=0.4,
        requires_sources=False,
    ),
    
    "deep_analysis": PhaseContract(
        phase_name="deep_analysis",
        allowed_types={"scenarios", "risk_analysis", "trade_offs", "validated_hypotheses", "stress_tests"},
        forbidden_types={"final_report", "implementation"},
        allowed_patterns=[
            r'\b(scenario|case|possibility|outcome)\b',
            r'\b(risk|vulnerability|failure mode|edge case)\b',
            r'\b(trade-?off|balance|tension|cost|benefit)\b',
            r'\b(stress test|boundary|limit|extreme)\b',
        ],
        forbidden_patterns=[
            r'\b(final report|executive summary)\b',
            r'\b(implement|code|build|deploy)\b.{0,20}\b(now|immediately)\b',
        ],
        max_speculation_ratio=0.5,
        requires_sources=False,
    ),
    
    "synthesis": PhaseContract(
        phase_name="synthesis",
        allowed_types={"recommendations", "scenarios", "conclusions", "action_items", "final_report"},
        forbidden_types={"raw_research", "new_facts", "new_sources"},
        allowed_patterns=[
            r'\b(recommend|suggest|advise|propose)\b',
            r'\b(conclusion|summary|synthesis|overall)\b',
            r'\b(action item|next step|implementation)\b',
            r'\b(scenario|strategy|approach|plan)\b',
        ],
        forbidden_patterns=[
            r'\bnew (?:research|source|study|data|evidence)\s+(?:shows|indicates|suggests)\b',
            r'\baccording to (?:new|recent|additional|fresh)\b',
            r'\bwe (?:found|discovered|identified) (?:new|additional)\b',
        ],
        max_speculation_ratio=0.3,
        requires_sources=False,
    ),
    
    "implementation": PhaseContract(
        phase_name="implementation",
        allowed_types={"code", "implementation", "tests", "documentation"},
        forbidden_types={"recommendations", "synthesis", "analysis"},
        allowed_patterns=[
            r'\b(code|function|class|method|implementation)\b',
            r'\b(test|assert|expect|verify)\b',
            r'\b(documentation|docstring|comment)\b',
        ],
        forbidden_patterns=[
            r'\b(recommend|should consider|might want to)\b',
            r'\b(in conclusion|overall|synthesis)\b',
            r'\b(analysis suggests|findings indicate)\b',
        ],
        max_speculation_ratio=0.1,
        requires_sources=False,
    ),
}


class PhaseGuard:
    """
    Guards phase purity by detecting contamination.
    
    Inspects outputs for content that doesn't belong in the current phase
    and computes contamination scores for penalizing evaluations.
    
    Usage:
        guard = PhaseGuard()
        result = guard.inspect_output(output, "reconnaissance")
        if not result.is_clean:
            # Apply penalty to evaluator score
            adjusted_score = score - result.recommended_penalty
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        contamination_threshold: float = 0.3
    ):
        """
        Initialize the phase guard.
        
        Args:
            strict_mode: If True, lower thresholds for violation detection
            contamination_threshold: Score threshold for triggering retry
        """
        self.strict_mode = strict_mode
        self.contamination_threshold = contamination_threshold
        self._validation_log: List[Dict] = []
    
    def get_contract(self, phase_name: str) -> PhaseContract:
        """Get contract for a phase, with fallback to default."""
        phase_lower = phase_name.lower()
        
        # Direct match
        if phase_lower in PHASE_CONTRACTS:
            return PHASE_CONTRACTS[phase_lower]
        
        # Fuzzy match
        for key, contract in PHASE_CONTRACTS.items():
            if key in phase_lower or phase_lower in key:
                return contract
        
        # Default permissive contract
        return PhaseContract(
            phase_name=phase_name,
            allowed_types=set(),
            forbidden_types=set(),
        )
    
    def inspect_output(
        self,
        output: str,
        phase_name: str
    ) -> PhaseContamination:
        """
        Inspect output for phase contamination.
        
        Args:
            output: Output text to inspect
            phase_name: Current phase name
            
        Returns:
            PhaseContamination result
        """
        contract = self.get_contract(phase_name)
        violations = []
        
        # Check forbidden patterns
        for i, pattern in enumerate(contract._compiled_forbidden):
            matches = pattern.finditer(output)
            for match in matches:
                # Determine contamination type
                contamination_type = self._classify_violation(
                    match.group(), phase_name
                )
                
                # Get context around match
                start = max(0, match.start() - 50)
                end = min(len(output), match.end() + 50)
                excerpt = output[start:end].strip()
                
                # Estimate line number
                line_num = output[:match.start()].count('\n') + 1
                
                violations.append(PhaseViolation(
                    contamination_type=contamination_type,
                    description=f"Forbidden pattern in {phase_name}: '{match.group()}'",
                    excerpt=excerpt,
                    severity=0.6 if self.strict_mode else 0.4,
                    line_number=line_num,
                ))
        
        # Check for forbidden artifact types mentioned
        output_lower = output.lower()
        for forbidden_type in contract.forbidden_types:
            if forbidden_type.lower() in output_lower:
                # Check it's not just mentioning but actually producing
                if self._is_producing_artifact(output, forbidden_type):
                    violations.append(PhaseViolation(
                        contamination_type=self._classify_violation(
                            forbidden_type, phase_name
                        ),
                        description=f"Producing forbidden artifact type: {forbidden_type}",
                        severity=0.5,
                    ))
        
        # Compute contamination score
        contamination_score = self._compute_contamination_score(
            violations, output, contract
        )
        
        # Determine if clean and penalties
        is_clean = contamination_score < self.contamination_threshold
        recommended_penalty = min(3.0, contamination_score * 5)
        should_retry = contamination_score > 0.5
        
        result = PhaseContamination(
            phase_name=phase_name,
            violations=violations,
            contamination_score=contamination_score,
            is_clean=is_clean,
            recommended_penalty=recommended_penalty,
            should_retry=should_retry,
        )
        
        # Log
        self._log_inspection(phase_name, result)
        
        return result
    
    def _classify_violation(
        self,
        matched_text: str,
        phase_name: str
    ) -> ContaminationType:
        """Classify violation type based on matched text and phase."""
        text_lower = matched_text.lower()
        phase_lower = phase_name.lower()
        
        # Premature recommendations
        if any(w in text_lower for w in ["recommend", "should", "advise", "action"]):
            if "recon" in phase_lower or "analysis" in phase_lower:
                return ContaminationType.PREMATURE_RECOMMENDATION
        
        # Premature conclusions
        if any(w in text_lower for w in ["conclusion", "verdict", "final"]):
            if "recon" in phase_lower or "analysis" in phase_lower:
                return ContaminationType.PREMATURE_CONCLUSION
        
        # Late sourcing
        if any(w in text_lower for w in ["new source", "new research", "additional study"]):
            if "synthesis" in phase_lower:
                return ContaminationType.LATE_SOURCING
        
        # Raw research in synthesis
        if "synthesis" in phase_lower:
            if any(w in text_lower for w in ["found", "discovered", "identified"]):
                return ContaminationType.RAW_RESEARCH_IN_SYNTHESIS
        
        return ContaminationType.PREMATURE_SYNTHESIS
    
    def _is_producing_artifact(self, output: str, artifact_type: str) -> bool:
        """Check if output is actually producing an artifact type."""
        # Look for section headers indicating production
        patterns = [
            rf'##?\s*{artifact_type}',
            rf'{artifact_type}\s*:\s*\n',
            rf'(?:my|our|the)\s+{artifact_type}',
        ]
        
        for pattern in patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        
        return False
    
    def _compute_contamination_score(
        self,
        violations: List[PhaseViolation],
        output: str,
        contract: PhaseContract
    ) -> float:
        """Compute overall contamination score."""
        if not violations:
            return 0.0
        
        # Base score from violations
        base_score = sum(v.severity for v in violations) / max(len(violations), 1)
        
        # Scale by violation density (violations per 1000 chars)
        density = len(violations) / (len(output) / 1000 + 1)
        density_factor = min(1.0, density / 5)  # Cap at 5 violations per 1000 chars
        
        # Combine
        score = 0.7 * base_score + 0.3 * density_factor
        
        return min(1.0, score)
    
    def compute_contamination_score(
        self,
        output: str,
        phase_name: str
    ) -> float:
        """
        Convenience method to get just the contamination score.
        
        Args:
            output: Output to check
            phase_name: Current phase
            
        Returns:
            Contamination score (0-1)
        """
        result = self.inspect_output(output, phase_name)
        return result.contamination_score
    
    def trigger_corrective_iteration(
        self,
        contamination: PhaseContamination,
        state: Any = None
    ) -> bool:
        """
        Determine if corrective iteration should be triggered.
        
        Args:
            contamination: Contamination analysis result
            state: Optional mission state for context
            
        Returns:
            True if corrective iteration recommended
        """
        if contamination.should_retry:
            logger.warning(
                f"Phase '{contamination.phase_name}' requires corrective iteration: "
                f"contamination={contamination.contamination_score:.2f}"
            )
            return True
        
        return False
    
    def get_correction_prompt(self, contamination: PhaseContamination) -> str:
        """
        Generate a correction prompt for contaminated output.
        
        Args:
            contamination: Contamination analysis result
            
        Returns:
            Prompt text for correction
        """
        violations_str = "\n".join(
            f"- {v.contamination_type.value}: {v.description}"
            for v in contamination.violations[:5]
        )
        
        contract = self.get_contract(contamination.phase_name)
        allowed_str = ", ".join(contract.allowed_types)
        forbidden_str = ", ".join(contract.forbidden_types)
        
        return f"""Your previous output violated phase contracts for '{contamination.phase_name}'.

## Violations Detected:
{violations_str}

## Phase Contract:
- ALLOWED: {allowed_str}
- FORBIDDEN: {forbidden_str}

## Correction Required:
Please revise your output to:
1. Remove any {forbidden_str}
2. Focus only on {allowed_str}
3. Keep analysis appropriate for the {contamination.phase_name} phase

Do NOT include recommendations, conclusions, or synthesis unless this is the synthesis phase.
"""
    
    def _log_inspection(self, phase_name: str, result: PhaseContamination) -> None:
        """Log inspection result."""
        entry = {
            "phase": phase_name,
            "contamination_score": result.contamination_score,
            "violation_count": len(result.violations),
            "is_clean": result.is_clean,
        }
        self._validation_log.append(entry)
        
        if not result.is_clean:
            logger.warning(
                f"Phase contamination in '{phase_name}': "
                f"score={result.contamination_score:.2f}, "
                f"violations={len(result.violations)}"
            )
    
    def get_validation_log(self) -> List[Dict]:
        """Get validation log."""
        return self._validation_log.copy()
    
    def clear_validation_log(self) -> None:
        """Clear validation log."""
        self._validation_log.clear()


# Global guard instance
_guard: Optional[PhaseGuard] = None


def get_phase_guard(strict_mode: bool = False) -> PhaseGuard:
    """Get the global phase guard instance."""
    global _guard
    if _guard is None:
        _guard = PhaseGuard(strict_mode=strict_mode)
    return _guard

