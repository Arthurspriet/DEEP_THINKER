"""
Policy Features for Learning Module.

Provides feature extraction for stop/escalate predictions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PolicyState:
    """
    State representation for policy prediction.
    
    Contains all features used to predict P(stop), P(escalate), P(switch_mode).
    
    Attributes:
        # Identification
        mission_id: Mission identifier
        phase_id: Phase identifier
        phase_type: Type of phase
        phase_number: Which phase number (0-indexed)
        
        # Score features
        current_score: Current overall score (0-1)
        score_history: Recent score history
        score_trend: Computed score trend
        
        # Quality features
        consistency_score: Current consistency score
        grounding_score: Current grounding score
        
        # Disagreement features
        disagreement_rate: Judge disagreement rate
        contradiction_count: Number of contradictions
        
        # Time features
        time_remaining_minutes: Remaining mission time
        time_budget_used_pct: Fraction of time budget used
        
        # Alignment features
        alignment_drift_risk: Current drift risk (0-1)
        alignment_corrections: Number of corrections
        
        # Tool features
        tool_success_rate: Recent tool success rate
        tool_calls_count: Number of tool calls
        
        # Resource features
        tokens_used: Tokens consumed
        cost_usd: USD cost estimate
    """
    
    # Identification
    mission_id: str = ""
    phase_id: str = ""
    phase_type: str = ""
    phase_number: int = 0
    
    # Score features
    current_score: float = 0.5
    score_history: List[float] = field(default_factory=list)
    score_trend: float = 0.0
    
    # Quality features
    consistency_score: float = 0.5
    grounding_score: float = 0.5
    
    # Disagreement features
    disagreement_rate: float = 0.0
    contradiction_count: int = 0
    
    # Time features
    time_remaining_minutes: float = 60.0
    time_budget_used_pct: float = 0.0
    
    # Alignment features
    alignment_drift_risk: float = 0.0
    alignment_corrections: int = 0
    
    # Tool features
    tool_success_rate: float = 1.0
    tool_calls_count: int = 0
    
    # Resource features
    tokens_used: int = 0
    cost_usd: float = 0.0
    
    def __post_init__(self):
        """Compute derived features."""
        if self.score_history and not self.score_trend:
            self.score_trend = self._compute_trend()
    
    def _compute_trend(self) -> float:
        """Compute score trend from history."""
        if len(self.score_history) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(self.score_history)
        x_mean = (n - 1) / 2
        y_mean = sum(self.score_history) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.score_history))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def to_features(self) -> Dict[str, float]:
        """
        Convert to feature dictionary for prediction.
        
        Returns:
            Dictionary of feature name -> float value
        """
        return {
            "current_score": self.current_score,
            "score_trend": self.score_trend,
            "consistency_score": self.consistency_score,
            "grounding_score": self.grounding_score,
            "disagreement_rate": self.disagreement_rate,
            "contradiction_count": float(min(self.contradiction_count, 10)) / 10,
            "time_remaining_minutes": min(self.time_remaining_minutes, 120) / 120,
            "time_budget_used_pct": self.time_budget_used_pct,
            "alignment_drift_risk": self.alignment_drift_risk,
            "alignment_corrections": float(min(self.alignment_corrections, 5)) / 5,
            "tool_success_rate": self.tool_success_rate,
            "tool_calls_count": float(min(self.tool_calls_count, 20)) / 20,
            "tokens_used_k": min(self.tokens_used, 100000) / 100000,
            "cost_usd": min(self.cost_usd, 1.0),
            "phase_number": float(min(self.phase_number, 10)) / 10,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "phase_type": self.phase_type,
            "phase_number": self.phase_number,
            "current_score": self.current_score,
            "score_history": self.score_history,
            "score_trend": self.score_trend,
            "consistency_score": self.consistency_score,
            "grounding_score": self.grounding_score,
            "disagreement_rate": self.disagreement_rate,
            "contradiction_count": self.contradiction_count,
            "time_remaining_minutes": self.time_remaining_minutes,
            "time_budget_used_pct": self.time_budget_used_pct,
            "alignment_drift_risk": self.alignment_drift_risk,
            "alignment_corrections": self.alignment_corrections,
            "tool_success_rate": self.tool_success_rate,
            "tool_calls_count": self.tool_calls_count,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyState":
        """Create from dictionary."""
        return cls(
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            phase_type=data.get("phase_type", ""),
            phase_number=data.get("phase_number", 0),
            current_score=data.get("current_score", 0.5),
            score_history=data.get("score_history", []),
            score_trend=data.get("score_trend", 0.0),
            consistency_score=data.get("consistency_score", 0.5),
            grounding_score=data.get("grounding_score", 0.5),
            disagreement_rate=data.get("disagreement_rate", 0.0),
            contradiction_count=data.get("contradiction_count", 0),
            time_remaining_minutes=data.get("time_remaining_minutes", 60.0),
            time_budget_used_pct=data.get("time_budget_used_pct", 0.0),
            alignment_drift_risk=data.get("alignment_drift_risk", 0.0),
            alignment_corrections=data.get("alignment_corrections", 0),
            tool_success_rate=data.get("tool_success_rate", 1.0),
            tool_calls_count=data.get("tool_calls_count", 0),
            tokens_used=data.get("tokens_used", 0),
            cost_usd=data.get("cost_usd", 0.0),
        )


def get_policy_feature_names() -> List[str]:
    """Get ordered list of policy feature names."""
    return [
        "current_score",
        "score_trend",
        "consistency_score",
        "grounding_score",
        "disagreement_rate",
        "contradiction_count",
        "time_remaining_minutes",
        "time_budget_used_pct",
        "alignment_drift_risk",
        "alignment_corrections",
        "tool_success_rate",
        "tool_calls_count",
        "tokens_used_k",
        "cost_usd",
        "phase_number",
    ]


