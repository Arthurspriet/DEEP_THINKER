"""
Reward Weights for DeepThinker.

Configurable component weights for reward computation.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    """
    Weights for combining reward signal components.
    
    The final reward is computed as:
        reward = (
            score_delta * score_delta_weight +
            consistency * consistency_weight +
            grounding * grounding_weight -
            cost * cost_weight -
            alignment * alignment_weight -
            time * time_weight
        )
    
    Weights should be positive. The signs are applied in RewardSignal.compute_reward().
    
    Attributes:
        score_delta: Weight for score improvement
        consistency: Weight for consistency improvement
        grounding: Weight for grounding improvement
        cost: Weight for cost penalty
        alignment: Weight for alignment penalty
        time: Weight for time penalty
    """
    
    score_delta: float = 0.40
    consistency: float = 0.15
    grounding: float = 0.15
    cost: float = 0.15
    alignment: float = 0.10
    time: float = 0.05
    
    def __post_init__(self):
        """Validate weights are non-negative."""
        for attr in ["score_delta", "consistency", "grounding", "cost", "alignment", "time"]:
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"Weight {attr} must be non-negative, got {value}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "score_delta": self.score_delta,
            "consistency": self.consistency,
            "grounding": self.grounding,
            "cost": self.cost,
            "alignment": self.alignment,
            "time": self.time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardWeights":
        """Create from dictionary."""
        return cls(
            score_delta=float(data.get("score_delta", 0.40)),
            consistency=float(data.get("consistency", 0.15)),
            grounding=float(data.get("grounding", 0.15)),
            cost=float(data.get("cost", 0.15)),
            alignment=float(data.get("alignment", 0.10)),
            time=float(data.get("time", 0.05)),
        )
    
    @classmethod
    def load(cls, path: str) -> "RewardWeights":
        """
        Load weights from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            RewardWeights instance
        """
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to load weights from {path}: {e}")
        
        return cls()  # Return defaults
    
    def save(self, path: str) -> bool:
        """
        Save weights to JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to save weights to {path}: {e}")
            return False
    
    def normalize(self) -> "RewardWeights":
        """
        Return normalized weights that sum to 1.0.
        
        Returns:
            New RewardWeights instance with normalized values
        """
        total = (
            self.score_delta + self.consistency + self.grounding +
            self.cost + self.alignment + self.time
        )
        
        if total == 0:
            return RewardWeights()
        
        return RewardWeights(
            score_delta=self.score_delta / total,
            consistency=self.consistency / total,
            grounding=self.grounding / total,
            cost=self.cost / total,
            alignment=self.alignment / total,
            time=self.time / total,
        )


# Default weights instance
DEFAULT_REWARD_WEIGHTS = RewardWeights()


