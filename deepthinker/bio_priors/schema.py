"""
BioPattern Schema for Bio Priors.

Defines the BioPattern dataclass that represents a biological strategy pattern
with strict validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .signals import PressureSignals


# Valid fields that can appear in system_mapping
VALID_SYSTEM_MAPPING_KEYS = {
    "exploration_bias_delta",
    "depth_budget_delta",
    "redundancy_check",
    "force_falsification_step",
    "branch_pruning_suggested",
    "confidence_penalty_delta",
    "retrieval_diversify",
    "council_diversity_min",
}

# Valid maturity levels
VALID_MATURITY = {"draft", "stable"}


class BioPatternValidationError(Exception):
    """Raised when a BioPattern fails validation."""
    pass


@dataclass
class BioPattern:
    """
    A biological strategy pattern.
    
    Represents a pattern from biology that can be mapped to reasoning
    system modulations via PressureSignals.
    
    Attributes:
        id: Unique identifier (must start with "BIO_")
        name: Human-readable name
        problem_class: List of problem classes this pattern addresses
        conditions: Human-readable conditions that trigger this pattern
        mechanism: Description of the biological mechanism
        system_mapping: Maps to PressureSignals fields
        expected_tradeoffs: Benefits and risks of applying this pattern
        tests: Test conditions to validate the pattern
        citations: Optional literature citations
        maturity: "draft" or "stable"
        weight: Pattern weight (0.0 to 1.0)
    """
    id: str
    name: str
    problem_class: List[str]
    conditions: List[str]
    mechanism: str
    system_mapping: Dict[str, Any]
    expected_tradeoffs: Dict[str, List[str]] = field(default_factory=dict)
    tests: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    maturity: str = "draft"
    weight: float = 0.3
    
    def __post_init__(self) -> None:
        """Validate after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate this pattern against schema rules.
        
        Raises:
            BioPatternValidationError: If validation fails
        """
        errors = []
        
        # ID must start with BIO_
        if not self.id.startswith("BIO_"):
            errors.append(f"id must start with 'BIO_', got '{self.id}'")
        
        # Weight must be in [0, 1]
        if not 0.0 <= self.weight <= 1.0:
            errors.append(f"weight must be in [0, 1], got {self.weight}")
        
        # Maturity must be valid
        if self.maturity not in VALID_MATURITY:
            errors.append(
                f"maturity must be one of {VALID_MATURITY}, got '{self.maturity}'"
            )
        
        # Must have at least 1 problem_class
        if not self.problem_class:
            errors.append("must have at least 1 problem_class")
        
        # Must have at least 1 system_mapping key
        if not self.system_mapping:
            errors.append("must have at least 1 system_mapping key")
        
        # system_mapping keys must be valid PressureSignals fields
        invalid_keys = set(self.system_mapping.keys()) - VALID_SYSTEM_MAPPING_KEYS
        if invalid_keys:
            errors.append(
                f"system_mapping has invalid keys: {invalid_keys}. "
                f"Valid keys: {VALID_SYSTEM_MAPPING_KEYS}"
            )
        
        # expected_tradeoffs should have benefits and/or risks
        if self.expected_tradeoffs:
            valid_tradeoff_keys = {"benefits", "risks"}
            invalid_tradeoff_keys = set(self.expected_tradeoffs.keys()) - valid_tradeoff_keys
            if invalid_tradeoff_keys:
                errors.append(
                    f"expected_tradeoffs has invalid keys: {invalid_tradeoff_keys}. "
                    f"Valid keys: {valid_tradeoff_keys}"
                )
        
        if errors:
            raise BioPatternValidationError(
                f"BioPattern '{self.id}' validation failed:\n  - " +
                "\n  - ".join(errors)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BioPattern":
        """
        Create BioPattern from dictionary.
        
        Args:
            data: Dictionary with pattern fields
            
        Returns:
            BioPattern instance
            
        Raises:
            BioPatternValidationError: If validation fails
        """
        return cls(
            id=data.get("id", "BIO_UNKNOWN"),
            name=data.get("name", "Unknown Pattern"),
            problem_class=data.get("problem_class", []),
            conditions=data.get("conditions", []),
            mechanism=data.get("mechanism", ""),
            system_mapping=data.get("system_mapping", {}),
            expected_tradeoffs=data.get("expected_tradeoffs", {}),
            tests=data.get("tests", []),
            citations=data.get("citations", []),
            maturity=data.get("maturity", "draft"),
            weight=data.get("weight", 0.3),
        )
    
    def to_pressure_signals(self) -> PressureSignals:
        """
        Convert system_mapping to PressureSignals.
        
        Returns:
            PressureSignals with values from system_mapping
        """
        return PressureSignals.from_dict(self.system_mapping)
    
    def matches_context(
        self,
        phase: str,
        has_stagnation: bool = False,
        has_high_contradiction: bool = False,
        has_high_drift: bool = False,
        has_high_branching: bool = False,
        is_early_phase: bool = False,
        is_late_phase: bool = False,
    ) -> float:
        """
        Compute match score for a given context.
        
        Returns a score from 0.0 to 1.0 indicating how well this
        pattern matches the current context.
        
        Args:
            phase: Current phase name
            has_stagnation: Evidence production stagnated
            has_high_contradiction: High contradiction rate
            has_high_drift: High drift from goal
            has_high_branching: High plan branching
            is_early_phase: In early phase
            is_late_phase: In late phase
            
        Returns:
            Match score (0.0 to 1.0)
        """
        score = 0.0
        matches = 0
        
        # Check problem_class for context signals
        problem_classes_lower = [p.lower() for p in self.problem_class]
        
        if has_stagnation and any(
            kw in pc for pc in problem_classes_lower
            for kw in ["stagnation", "exploration", "foraging", "search"]
        ):
            score += 1.0
            matches += 1
        
        if has_high_contradiction and any(
            kw in pc for pc in problem_classes_lower
            for kw in ["contradiction", "conflict", "inconsistency", "error", "redundancy"]
        ):
            score += 1.0
            matches += 1
        
        if has_high_drift and any(
            kw in pc for pc in problem_classes_lower
            for kw in ["drift", "alignment", "homeostasis", "balance", "immune"]
        ):
            score += 1.0
            matches += 1
        
        if has_high_branching and any(
            kw in pc for pc in problem_classes_lower
            for kw in ["complexity", "budget", "metabolic", "pruning", "resource"]
        ):
            score += 1.0
            matches += 1
        
        if is_early_phase and any(
            kw in pc for pc in problem_classes_lower
            for kw in ["exploration", "foraging", "discovery", "developmental"]
        ):
            score += 0.5
            matches += 1
        
        if is_late_phase and any(
            kw in pc for pc in problem_classes_lower
            for kw in ["verification", "validation", "redundancy", "falsification"]
        ):
            score += 0.5
            matches += 1
        
        # Normalize score
        if matches > 0:
            return min(1.0, score / max(1, matches))
        
        # Base score for any pattern
        return 0.1



