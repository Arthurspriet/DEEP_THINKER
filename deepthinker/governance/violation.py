"""
Violation Schema for Normative Control Layer.

Defines typed violations with severity scoring for governance decisions.
Violations are categorized by type and marked as hard (can trigger BLOCK)
or soft (accumulate toward warnings).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ViolationType(str, Enum):
    """
    Categories of governance violations.
    
    Organized by domain:
    - EPISTEMIC_*: Knowledge quality violations
    - PHASE_*: Phase purity violations
    - STRUCTURAL_*: Output structure violations
    - CONFIDENCE_*: Confidence reporting violations
    """
    
    # Epistemic violations - knowledge quality
    EPISTEMIC_UNGROUNDED_CLAIM = "epistemic_ungrounded_claim"
    EPISTEMIC_UNGROUNDED_FACT = "epistemic_ungrounded_fact"
    EPISTEMIC_LOW_SOURCE_COUNT = "epistemic_low_source_count"
    EPISTEMIC_HIGH_SPECULATION = "epistemic_high_speculation"
    EPISTEMIC_MISSING_CITATIONS = "epistemic_missing_citations"
    EPISTEMIC_BUDGET_NOT_MET = "epistemic_budget_not_met"
    
    # Phase purity violations
    PHASE_CONTAMINATION = "phase_contamination"
    PHASE_PREMATURE_RECOMMENDATION = "phase_premature_recommendation"
    PHASE_PREMATURE_CONCLUSION = "phase_premature_conclusion"
    PHASE_LATE_SOURCING = "phase_late_sourcing"
    PHASE_RAW_RESEARCH_IN_SYNTHESIS = "phase_raw_research_in_synthesis"
    
    # Structural violations - output format/completeness
    STRUCTURAL_MISSING_SCENARIOS = "structural_missing_scenarios"
    STRUCTURAL_MALFORMED_SCENARIO = "structural_malformed_scenario"
    STRUCTURAL_INSUFFICIENT_CONTENT = "structural_insufficient_content"
    STRUCTURAL_SCENARIO_NOT_DISTINCT = "structural_scenario_not_distinct"
    
    # Confidence violations
    CONFIDENCE_INFLATION = "confidence_inflation"
    CONFIDENCE_EXCEEDS_GROUNDING = "confidence_exceeds_grounding"


@dataclass
class Violation:
    """
    A single governance violation.
    
    Attributes:
        type: Category of violation
        severity: Severity score from 0.0 (trivial) to 1.0 (critical)
        description: Human-readable description of the violation
        phase_name: Phase where violation occurred
        is_hard: If True, this violation alone can trigger BLOCK status
        details: Optional additional context/evidence
        rule_id: Identifier of the rule that triggered this violation
    """
    
    type: ViolationType
    severity: float
    description: str
    phase_name: str
    is_hard: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    rule_id: str = ""
    
    def __post_init__(self):
        """Validate and normalize violation."""
        # Clamp severity to valid range
        self.severity = max(0.0, min(1.0, self.severity))
        
        # Generate rule_id if not provided
        if not self.rule_id:
            self.rule_id = f"{self.type.value}_{self.phase_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "severity": self.severity,
            "description": self.description,
            "phase_name": self.phase_name,
            "is_hard": self.is_hard,
            "details": self.details,
            "rule_id": self.rule_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Violation":
        """Create from dictionary."""
        return cls(
            type=ViolationType(data["type"]),
            severity=data.get("severity", 0.5),
            description=data.get("description", ""),
            phase_name=data.get("phase_name", ""),
            is_hard=data.get("is_hard", False),
            details=data.get("details", {}),
            rule_id=data.get("rule_id", ""),
        )
    
    def __str__(self) -> str:
        """Human-readable representation."""
        hard_marker = " [HARD]" if self.is_hard else ""
        return f"[{self.type.value}]{hard_marker} {self.description} (severity={self.severity:.2f})"


# Default severity mappings for violation types
DEFAULT_VIOLATION_SEVERITIES: Dict[ViolationType, float] = {
    # Epistemic - moderate to high severity
    ViolationType.EPISTEMIC_UNGROUNDED_CLAIM: 0.6,
    ViolationType.EPISTEMIC_LOW_SOURCE_COUNT: 0.5,
    ViolationType.EPISTEMIC_HIGH_SPECULATION: 0.5,
    ViolationType.EPISTEMIC_MISSING_CITATIONS: 0.4,
    
    # Phase purity - moderate severity (allow recovery)
    ViolationType.PHASE_CONTAMINATION: 0.4,
    ViolationType.PHASE_PREMATURE_RECOMMENDATION: 0.5,
    ViolationType.PHASE_PREMATURE_CONCLUSION: 0.5,
    ViolationType.PHASE_LATE_SOURCING: 0.4,
    ViolationType.PHASE_RAW_RESEARCH_IN_SYNTHESIS: 0.4,
    
    # Structural - high severity (hard constraints)
    ViolationType.STRUCTURAL_MISSING_SCENARIOS: 0.9,
    ViolationType.STRUCTURAL_MALFORMED_SCENARIO: 0.7,
    ViolationType.STRUCTURAL_INSUFFICIENT_CONTENT: 0.6,
    ViolationType.STRUCTURAL_SCENARIO_NOT_DISTINCT: 0.5,
    
    # Confidence - moderate severity
    ViolationType.CONFIDENCE_INFLATION: 0.4,
    ViolationType.CONFIDENCE_EXCEEDS_GROUNDING: 0.5,
}


# Violations that are hard by default (can trigger BLOCK alone)
DEFAULT_HARD_VIOLATIONS = {
    ViolationType.STRUCTURAL_MISSING_SCENARIOS,
    ViolationType.STRUCTURAL_MALFORMED_SCENARIO,
}


def create_violation(
    violation_type: ViolationType,
    description: str,
    phase_name: str,
    severity: Optional[float] = None,
    is_hard: Optional[bool] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Violation:
    """
    Factory function to create violations with sensible defaults.
    
    Args:
        violation_type: Type of violation
        description: Human-readable description
        phase_name: Phase where violation occurred
        severity: Override default severity
        is_hard: Override default hard status
        details: Additional context
        
    Returns:
        Configured Violation instance
    """
    return Violation(
        type=violation_type,
        severity=severity if severity is not None else DEFAULT_VIOLATION_SEVERITIES.get(violation_type, 0.5),
        description=description,
        phase_name=phase_name,
        is_hard=is_hard if is_hard is not None else violation_type in DEFAULT_HARD_VIOLATIONS,
        details=details or {},
    )

