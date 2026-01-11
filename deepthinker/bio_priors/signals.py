"""
Pressure Signals for Bio Priors.

Defines the PressureSignals dataclass that represents bounded modulation
signals emitted by the bio prior engine.

These signals are:
- BOUNDED: All values are clamped to safe ranges
- NON-AUTHORITATIVE: They suggest, not mandate
- TRACEABLE: Include version and intent metadata
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


# Bounds for each signal field
BOUNDS = {
    "exploration_bias_delta": (-0.2, 0.2),
    "depth_budget_delta": (-2, 2),
    "confidence_penalty_delta": (0.0, 0.2),
    "council_diversity_min": (1, 4),
}


@dataclass
class PressureSignals:
    """
    Bounded pressure signals for reasoning modulation.
    
    These signals are emitted by the bio prior engine and may influence
    reasoning dynamics. They are advisory and bounded.
    
    Modulation fields (bounded):
        exploration_bias_delta: Bias toward exploration [-0.2, +0.2]
        depth_budget_delta: Adjustment to depth budget [-2, +2]
        redundancy_check: Suggest redundant verification
        force_falsification_step: Suggest falsification attempt
        branch_pruning_suggested: Suggest pruning low-value branches
        confidence_penalty_delta: Confidence reduction [0.0, 0.2]
        retrieval_diversify: Suggest diversifying retrieval
        council_diversity_min: Minimum council diversity [1, 4]
    
    Traceability fields (non-authoritative):
        bounds_version: Version of bounds specification
        intent: Statement of intent (always "modulation_only")
    """
    # Modulation fields (bounded)
    exploration_bias_delta: float = 0.0
    depth_budget_delta: int = 0
    redundancy_check: bool = False
    force_falsification_step: bool = False
    branch_pruning_suggested: bool = False
    confidence_penalty_delta: float = 0.0
    retrieval_diversify: bool = False
    council_diversity_min: int = 1
    
    # Traceability fields (non-authoritative)
    bounds_version: str = "v1"
    intent: str = "modulation_only"
    
    def clamp(self) -> "PressureSignals":
        """
        Clamp all values to their defined bounds.
        
        Returns:
            New PressureSignals with clamped values
        """
        return PressureSignals(
            exploration_bias_delta=_clamp_float(
                self.exploration_bias_delta,
                *BOUNDS["exploration_bias_delta"]
            ),
            depth_budget_delta=_clamp_int(
                self.depth_budget_delta,
                *BOUNDS["depth_budget_delta"]
            ),
            redundancy_check=self.redundancy_check,
            force_falsification_step=self.force_falsification_step,
            branch_pruning_suggested=self.branch_pruning_suggested,
            confidence_penalty_delta=_clamp_float(
                self.confidence_penalty_delta,
                *BOUNDS["confidence_penalty_delta"]
            ),
            retrieval_diversify=self.retrieval_diversify,
            council_diversity_min=_clamp_int(
                self.council_diversity_min,
                *BOUNDS["council_diversity_min"]
            ),
            bounds_version=self.bounds_version,
            intent=self.intent,
        )
    
    def merge(
        self,
        other: "PressureSignals",
        weight_self: float = 1.0,
        weight_other: float = 1.0,
    ) -> "PressureSignals":
        """
        Merge with another PressureSignals using weighted combination.
        
        - Floats: Weighted sum
        - Ints: Weighted round
        - Bools: OR
        - Then clamp
        
        Args:
            other: Other PressureSignals to merge with
            weight_self: Weight for this instance's values
            weight_other: Weight for other instance's values
            
        Returns:
            New merged and clamped PressureSignals
        """
        total_weight = weight_self + weight_other
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero
        
        # Weighted float combination
        exploration = (
            self.exploration_bias_delta * weight_self +
            other.exploration_bias_delta * weight_other
        ) / total_weight
        
        confidence = (
            self.confidence_penalty_delta * weight_self +
            other.confidence_penalty_delta * weight_other
        ) / total_weight
        
        # Weighted int combination (round)
        depth = round(
            (self.depth_budget_delta * weight_self +
             other.depth_budget_delta * weight_other) / total_weight
        )
        
        diversity = round(
            (self.council_diversity_min * weight_self +
             other.council_diversity_min * weight_other) / total_weight
        )
        
        # Bool OR
        redundancy = self.redundancy_check or other.redundancy_check
        falsification = self.force_falsification_step or other.force_falsification_step
        pruning = self.branch_pruning_suggested or other.branch_pruning_suggested
        retrieval = self.retrieval_diversify or other.retrieval_diversify
        
        result = PressureSignals(
            exploration_bias_delta=exploration,
            depth_budget_delta=depth,
            redundancy_check=redundancy,
            force_falsification_step=falsification,
            branch_pruning_suggested=pruning,
            confidence_penalty_delta=confidence,
            retrieval_diversify=retrieval,
            council_diversity_min=diversity,
            bounds_version=self.bounds_version,
            intent=self.intent,
        )
        
        return result.clamp()
    
    def scale(self, alpha: float) -> "PressureSignals":
        """
        Scale numeric fields by alpha factor (NOT bools).
        
        Args:
            alpha: Scaling factor
            
        Returns:
            New scaled and clamped PressureSignals
        """
        result = PressureSignals(
            exploration_bias_delta=self.exploration_bias_delta * alpha,
            depth_budget_delta=round(self.depth_budget_delta * alpha),
            redundancy_check=self.redundancy_check,  # Bools unchanged
            force_falsification_step=self.force_falsification_step,
            branch_pruning_suggested=self.branch_pruning_suggested,
            confidence_penalty_delta=self.confidence_penalty_delta * alpha,
            retrieval_diversify=self.retrieval_diversify,
            council_diversity_min=max(1, round(self.council_diversity_min * alpha)),
            bounds_version=self.bounds_version,
            intent=self.intent,
        )
        
        return result.clamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PressureSignals":
        """Create from dictionary."""
        return cls(
            exploration_bias_delta=data.get("exploration_bias_delta", 0.0),
            depth_budget_delta=data.get("depth_budget_delta", 0),
            redundancy_check=data.get("redundancy_check", False),
            force_falsification_step=data.get("force_falsification_step", False),
            branch_pruning_suggested=data.get("branch_pruning_suggested", False),
            confidence_penalty_delta=data.get("confidence_penalty_delta", 0.0),
            retrieval_diversify=data.get("retrieval_diversify", False),
            council_diversity_min=data.get("council_diversity_min", 1),
            bounds_version=data.get("bounds_version", "v1"),
            intent=data.get("intent", "modulation_only"),
        ).clamp()
    
    @classmethod
    def zero(cls) -> "PressureSignals":
        """Create a zero/neutral PressureSignals."""
        return cls()
    
    def has_any_signal(self) -> bool:
        """Check if any signal is non-default/non-zero."""
        return (
            self.exploration_bias_delta != 0.0 or
            self.depth_budget_delta != 0 or
            self.redundancy_check or
            self.force_falsification_step or
            self.branch_pruning_suggested or
            self.confidence_penalty_delta != 0.0 or
            self.retrieval_diversify or
            self.council_diversity_min != 1
        )


def _clamp_float(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float to bounds."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(min_val, min(max_val, value))


def _clamp_int(value: int, min_val: int, max_val: int) -> int:
    """Clamp an int to bounds."""
    return max(min_val, min(max_val, int(value)))



