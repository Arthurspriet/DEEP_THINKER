"""
Contextual Features for Bandits.

Provides feature extraction for contextual bandit decisions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BanditContext:
    """
    Context for bandit decision making.
    
    Provides features that can be used for contextual bandits.
    Currently used for logging; contextual selection is future work.
    
    Attributes:
        mission_id: Mission identifier
        phase_id: Phase identifier
        phase_type: Type of phase (reconnaissance, synthesis, etc.)
        
        # Time features
        time_remaining_minutes: Remaining mission time
        time_budget_used_pct: Fraction of time budget used
        
        # Quality features
        current_score: Current overall score
        score_trend: Recent score trend (positive = improving)
        
        # Complexity features
        difficulty_estimate: Estimated difficulty (0-1)
        retry_count: Number of retries so far
        
        # Alignment features
        alignment_drift_risk: Current alignment drift risk (0-1)
        alignment_corrections: Number of corrections applied
        
        # Resource features
        tokens_used: Tokens used so far
        gpu_pressure: Current GPU pressure level
    """
    
    mission_id: str = ""
    phase_id: str = ""
    phase_type: str = ""
    
    # Time features
    time_remaining_minutes: float = 60.0
    time_budget_used_pct: float = 0.0
    
    # Quality features
    current_score: float = 0.5
    score_trend: float = 0.0
    
    # Complexity features
    difficulty_estimate: float = 0.5
    retry_count: int = 0
    
    # Alignment features
    alignment_drift_risk: float = 0.0
    alignment_corrections: int = 0
    
    # Resource features
    tokens_used: int = 0
    gpu_pressure: str = "low"
    
    def to_features(self) -> Dict[str, float]:
        """
        Convert to feature dictionary for bandit.
        
        Returns:
            Dictionary of feature name -> float value
        """
        # Encode categorical features
        gpu_pressure_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        phase_type_features = self._encode_phase_type()
        
        features = {
            "time_remaining_minutes": self.time_remaining_minutes,
            "time_budget_used_pct": self.time_budget_used_pct,
            "current_score": self.current_score,
            "score_trend": self.score_trend,
            "difficulty_estimate": self.difficulty_estimate,
            "retry_count": float(self.retry_count),
            "alignment_drift_risk": self.alignment_drift_risk,
            "alignment_corrections": float(self.alignment_corrections),
            "tokens_used_k": self.tokens_used / 1000.0,
            "gpu_pressure": gpu_pressure_map.get(self.gpu_pressure, 0.5),
        }
        
        # Add phase type features
        features.update(phase_type_features)
        
        return features
    
    def _encode_phase_type(self) -> Dict[str, float]:
        """One-hot encode phase type."""
        phase_types = [
            "reconnaissance", "research", "planning",
            "synthesis", "evaluation", "execution"
        ]
        
        features = {}
        for pt in phase_types:
            key = f"phase_type_{pt}"
            features[key] = 1.0 if self.phase_type.lower() == pt else 0.0
        
        return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "phase_type": self.phase_type,
            "time_remaining_minutes": self.time_remaining_minutes,
            "time_budget_used_pct": self.time_budget_used_pct,
            "current_score": self.current_score,
            "score_trend": self.score_trend,
            "difficulty_estimate": self.difficulty_estimate,
            "retry_count": self.retry_count,
            "alignment_drift_risk": self.alignment_drift_risk,
            "alignment_corrections": self.alignment_corrections,
            "tokens_used": self.tokens_used,
            "gpu_pressure": self.gpu_pressure,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditContext":
        """Create from dictionary."""
        return cls(
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            phase_type=data.get("phase_type", ""),
            time_remaining_minutes=data.get("time_remaining_minutes", 60.0),
            time_budget_used_pct=data.get("time_budget_used_pct", 0.0),
            current_score=data.get("current_score", 0.5),
            score_trend=data.get("score_trend", 0.0),
            difficulty_estimate=data.get("difficulty_estimate", 0.5),
            retry_count=data.get("retry_count", 0),
            alignment_drift_risk=data.get("alignment_drift_risk", 0.0),
            alignment_corrections=data.get("alignment_corrections", 0),
            tokens_used=data.get("tokens_used", 0),
            gpu_pressure=data.get("gpu_pressure", "low"),
        )


def get_feature_names() -> List[str]:
    """Get ordered list of feature names."""
    return [
        "time_remaining_minutes",
        "time_budget_used_pct",
        "current_score",
        "score_trend",
        "difficulty_estimate",
        "retry_count",
        "alignment_drift_risk",
        "alignment_corrections",
        "tokens_used_k",
        "gpu_pressure",
        "phase_type_reconnaissance",
        "phase_type_research",
        "phase_type_planning",
        "phase_type_synthesis",
        "phase_type_evaluation",
        "phase_type_execution",
    ]

