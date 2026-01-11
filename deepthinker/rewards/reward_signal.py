"""
Reward Signal for DeepThinker.

Versioned, deterministic reward computation with full audit trail.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .config import RewardConfig, get_reward_config
from .reward_weights import RewardWeights, DEFAULT_REWARD_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class RewardSignal:
    """
    Unified reward signal for learning components.
    
    Key properties:
    - Versioned: schema version for reproducibility
    - Deterministic: same inputs always produce same outputs
    - Auditable: raw components and normalization metadata preserved
    - Clamped: hard limits on penalty contributions for safety
    
    Usage:
        signal = RewardSignal.from_phase_outcome(
            score_delta=0.1,
            consistency_delta=0.05,
            cost_tokens=5000,
            cost_latency_ms=2000,
            alignment_events=1,
            time_budget_used_pct=0.4,
        )
        reward = signal.compute_reward()  # in [-1, +1]
    
    Attributes:
        reward_version: Schema version string
        components_raw: Pre-normalization component values
        normalization_meta: Normalization parameters used
        
        # Normalized components (post-normalization, pre-clamp)
        score_delta: Normalized score improvement
        consistency_delta: Normalized consistency improvement
        grounding_delta: Normalized grounding improvement
        cost_penalty: Normalized cost penalty
        alignment_penalty: Normalized alignment penalty
        time_penalty: Normalized time penalty
        
        # Clamped values (post-clamp)
        cost_penalty_clamped: Cost penalty after hard clamp
        alignment_penalty_clamped: Alignment penalty after hard clamp
        time_penalty_clamped: Time penalty after hard clamp
        
        # Final reward
        reward: Final computed reward in [-1, +1]
        
        # Metadata
        decision_type: Type of decision this reward is for
        mission_id: Mission this reward belongs to
        phase_id: Phase this reward belongs to
        timestamp: When reward was computed
    """
    
    # Versioning
    reward_version: str = "1.0.0"
    
    # Raw components (pre-normalization)
    components_raw: Dict[str, float] = field(default_factory=dict)
    
    # Normalization metadata
    normalization_meta: Dict[str, Any] = field(default_factory=dict)
    
    # Normalized components (post-normalization, pre-clamp)
    score_delta: float = 0.0
    consistency_delta: float = 0.0
    grounding_delta: float = 0.0
    cost_penalty: float = 0.0
    alignment_penalty: float = 0.0
    time_penalty: float = 0.0
    
    # Clamped values (from config)
    cost_penalty_clamped: float = 0.0
    alignment_penalty_clamped: float = 0.0
    time_penalty_clamped: float = 0.0
    
    # Final reward
    reward: float = 0.0
    
    # Metadata
    decision_type: str = ""
    mission_id: str = ""
    phase_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_phase_outcome(
        cls,
        score_delta: float = 0.0,
        consistency_delta: float = 0.0,
        grounding_delta: float = 0.0,
        cost_tokens: int = 0,
        cost_latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        alignment_drift_events: int = 0,
        alignment_corrections: int = 0,
        time_budget_used_pct: float = 0.0,
        mission_id: str = "",
        phase_id: str = "",
        decision_type: str = "",
        config: Optional[RewardConfig] = None,
    ) -> "RewardSignal":
        """
        Create a RewardSignal from phase outcome metrics.
        
        Args:
            score_delta: Overall score change (can be negative)
            consistency_delta: Consistency score change
            grounding_delta: Grounding score change
            cost_tokens: Tokens consumed
            cost_latency_ms: Latency in milliseconds
            cost_usd: USD cost estimate
            alignment_drift_events: Number of drift events
            alignment_corrections: Number of corrections applied
            time_budget_used_pct: Fraction of time budget used (0-1)
            mission_id: Mission identifier
            phase_id: Phase identifier
            decision_type: Type of decision
            config: RewardConfig (uses global if None)
            
        Returns:
            RewardSignal with normalized components
        """
        config = config or get_reward_config()
        
        # Store raw components
        components_raw = {
            "score_delta": score_delta,
            "consistency_delta": consistency_delta,
            "grounding_delta": grounding_delta,
            "cost_tokens": cost_tokens,
            "cost_latency_ms": cost_latency_ms,
            "cost_usd": cost_usd,
            "alignment_drift_events": alignment_drift_events,
            "alignment_corrections": alignment_corrections,
            "time_budget_used_pct": time_budget_used_pct,
        }
        
        # Store normalization metadata
        normalization_meta = {
            "score_normalizer": "passthrough",
            "cost_normalizer": "log_scale",
            "baseline_cost_tokens": config.baseline_cost_tokens,
            "baseline_latency_ms": config.baseline_latency_ms,
            "baseline_cost_usd": config.baseline_cost_usd,
            "config_version": config.version,
        }
        
        # Normalize cost penalty (log scale to handle wide range)
        # Cost = weighted combination of tokens, latency, and USD
        token_ratio = cost_tokens / max(1, config.baseline_cost_tokens)
        latency_ratio = cost_latency_ms / max(1, config.baseline_latency_ms)
        usd_ratio = cost_usd / max(0.001, config.baseline_cost_usd)
        
        # Log scale to compress large values
        cost_penalty = (
            0.5 * math.log1p(token_ratio) +
            0.3 * math.log1p(latency_ratio) +
            0.2 * math.log1p(usd_ratio)
        ) / math.log1p(2)  # Normalize so baseline = ~1.0
        
        # Normalize alignment penalty
        # Each drift event or correction adds to penalty
        alignment_penalty = 0.0
        if alignment_drift_events > 0:
            alignment_penalty += 0.1 * alignment_drift_events
        if alignment_corrections > 0:
            alignment_penalty += 0.05 * alignment_corrections
        
        # Time penalty (linear, 0 at start, 1 at budget exhaustion)
        time_penalty = min(1.0, max(0.0, time_budget_used_pct))
        
        signal = cls(
            reward_version=config.version,
            components_raw=components_raw,
            normalization_meta=normalization_meta,
            score_delta=score_delta,
            consistency_delta=consistency_delta,
            grounding_delta=grounding_delta,
            cost_penalty=cost_penalty,
            alignment_penalty=alignment_penalty,
            time_penalty=time_penalty,
            mission_id=mission_id,
            phase_id=phase_id,
            decision_type=decision_type,
        )
        
        return signal
    
    def compute_reward(
        self,
        weights: Optional[RewardWeights] = None,
        config: Optional[RewardConfig] = None,
    ) -> float:
        """
        Compute final reward with hard clamps.
        
        This is a deterministic computation:
        1. Apply hard clamps to penalties
        2. Compute weighted sum
        3. Clamp to [-1, +1]
        
        Args:
            weights: RewardWeights (uses defaults if None)
            config: RewardConfig (uses global if None)
            
        Returns:
            Final reward in [-1, +1]
        """
        weights = weights or DEFAULT_REWARD_WEIGHTS
        config = config or get_reward_config()
        
        # Apply hard clamps from config
        self.cost_penalty_clamped = min(self.cost_penalty, config.cost_penalty_clamp)
        self.alignment_penalty_clamped = min(self.alignment_penalty, config.alignment_penalty_clamp)
        self.time_penalty_clamped = min(self.time_penalty, config.time_penalty_clamp)
        
        # Compute weighted sum
        # Positive components (improvements)
        positive = (
            weights.score_delta * self.score_delta +
            weights.consistency * self.consistency_delta +
            weights.grounding * self.grounding_delta
        )
        
        # Negative components (penalties)
        negative = (
            weights.cost * self.cost_penalty_clamped +
            weights.alignment * self.alignment_penalty_clamped +
            weights.time * self.time_penalty_clamped
        )
        
        raw_reward = positive - negative
        
        # Clamp to [-1, +1]
        self.reward = max(-1.0, min(1.0, raw_reward))
        
        return self.reward
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Fully serializable for replay and audit.
        """
        return {
            "reward_version": self.reward_version,
            "components_raw": self.components_raw,
            "normalization_meta": self.normalization_meta,
            "score_delta": self.score_delta,
            "consistency_delta": self.consistency_delta,
            "grounding_delta": self.grounding_delta,
            "cost_penalty": self.cost_penalty,
            "alignment_penalty": self.alignment_penalty,
            "time_penalty": self.time_penalty,
            "cost_penalty_clamped": self.cost_penalty_clamped,
            "alignment_penalty_clamped": self.alignment_penalty_clamped,
            "time_penalty_clamped": self.time_penalty_clamped,
            "reward": self.reward,
            "decision_type": self.decision_type,
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardSignal":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            reward_version=data.get("reward_version", "1.0.0"),
            components_raw=data.get("components_raw", {}),
            normalization_meta=data.get("normalization_meta", {}),
            score_delta=data.get("score_delta", 0.0),
            consistency_delta=data.get("consistency_delta", 0.0),
            grounding_delta=data.get("grounding_delta", 0.0),
            cost_penalty=data.get("cost_penalty", 0.0),
            alignment_penalty=data.get("alignment_penalty", 0.0),
            time_penalty=data.get("time_penalty", 0.0),
            cost_penalty_clamped=data.get("cost_penalty_clamped", 0.0),
            alignment_penalty_clamped=data.get("alignment_penalty_clamped", 0.0),
            time_penalty_clamped=data.get("time_penalty_clamped", 0.0),
            reward=data.get("reward", 0.0),
            decision_type=data.get("decision_type", ""),
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            timestamp=timestamp,
        )
    
    def __str__(self) -> str:
        return (
            f"RewardSignal(reward={self.reward:.3f}, "
            f"score_delta={self.score_delta:.3f}, "
            f"cost_penalty={self.cost_penalty_clamped:.3f})"
        )




