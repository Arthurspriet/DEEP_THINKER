"""
Alignment Learning for DeepThinker.

Provides:
- AlignmentActionBandit: Constrained bandit for alignment actions
- Per-mission caps for disruptive actions
- CUSUM remains hard safety override

Safety constraints:
- Existing CUSUM thresholds remain hard safety override
- Per-mission caps on disruptive actions
- Cannot override user-triggered events
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..bandits.generalized_bandit import GeneralizedBandit
from ..bandits.config import BanditConfig, get_bandit_config
from .learning_config import AlignmentLearningConfig, get_alignment_learning_config

logger = logging.getLogger(__name__)


class AlignmentAction(str, Enum):
    """
    Alignment actions that can be learned.
    
    From least to most disruptive:
    """
    REANCHOR_INTERNAL = "reanchor_internal"
    """Soft re-anchor to north star (internal only)."""
    
    INCREASE_SKEPTIC_WEIGHT = "increase_skeptic_weight"
    """Increase skeptic weight in councils."""
    
    SWITCH_DEEPEN_MODE_TO_EVIDENCE = "switch_deepen_mode_to_evidence"
    """Switch deepening mode to evidence gathering."""
    
    PRUNE_OR_PARK_FOCUS_AREAS = "prune_or_park_focus_areas"
    """Prune or park drifting focus areas."""
    
    TRIGGER_USER_EVENT_DRIFT_CONFIRMATION = "trigger_user_event_drift_confirmation"
    """Trigger user confirmation for drift (most disruptive)."""


# Disruptiveness ranking (higher = more disruptive)
ACTION_DISRUPTIVENESS = {
    AlignmentAction.REANCHOR_INTERNAL: 1,
    AlignmentAction.INCREASE_SKEPTIC_WEIGHT: 2,
    AlignmentAction.SWITCH_DEEPEN_MODE_TO_EVIDENCE: 3,
    AlignmentAction.PRUNE_OR_PARK_FOCUS_AREAS: 4,
    AlignmentAction.TRIGGER_USER_EVENT_DRIFT_CONFIRMATION: 5,
}


@dataclass
class AlignmentContext:
    """
    Context for alignment action selection.
    
    Attributes:
        drift_score: Current drift score (0-1, higher = more drift)
        cusum_value: Current CUSUM detector value
        phases_since_anchor: Phases since last re-anchor
        recent_drift_events: Number of drift events in recent history
        current_relevance: Current relevance score
        skeptic_weight: Current skeptic weight
        evidence_mode_active: Whether evidence mode is active
    """
    drift_score: float = 0.0
    cusum_value: float = 0.0
    phases_since_anchor: int = 0
    recent_drift_events: int = 0
    current_relevance: float = 0.5
    skeptic_weight: float = 0.3
    evidence_mode_active: bool = False
    
    def to_features(self) -> Dict[str, float]:
        """Convert to feature dictionary for bandit."""
        return {
            "drift_score": self.drift_score,
            "cusum_value": min(1.0, self.cusum_value / 5.0),  # Normalize
            "phases_since_anchor": min(1.0, self.phases_since_anchor / 10.0),
            "recent_drift_events": min(1.0, self.recent_drift_events / 5.0),
            "current_relevance": self.current_relevance,
            "skeptic_weight": self.skeptic_weight,
            "evidence_mode_active": 1.0 if self.evidence_mode_active else 0.0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_score": self.drift_score,
            "cusum_value": self.cusum_value,
            "phases_since_anchor": self.phases_since_anchor,
            "recent_drift_events": self.recent_drift_events,
            "current_relevance": self.current_relevance,
            "skeptic_weight": self.skeptic_weight,
            "evidence_mode_active": self.evidence_mode_active,
        }


@dataclass
class AlignmentOutcome:
    """
    Outcome of an alignment action.
    
    Used to compute reward for learning.
    """
    action: AlignmentAction
    drift_before: float
    drift_after: float
    relevance_before: float
    relevance_after: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def drift_reduction(self) -> float:
        """How much drift was reduced (positive = good)."""
        return self.drift_before - self.drift_after
    
    @property
    def relevance_delta(self) -> float:
        """Change in relevance (negative = bad)."""
        return self.relevance_after - self.relevance_before
    
    def compute_reward(self, config: AlignmentLearningConfig) -> float:
        """
        Compute reward for this action.
        
        Reward = drift_reduction - penalty * relevance_loss
        """
        reward = config.drift_reduction_weight * self.drift_reduction
        
        if self.relevance_delta < 0:
            reward += config.relevance_penalty_weight * self.relevance_delta
        
        return reward
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "drift_before": self.drift_before,
            "drift_after": self.drift_after,
            "relevance_before": self.relevance_before,
            "relevance_after": self.relevance_after,
            "drift_reduction": self.drift_reduction,
            "relevance_delta": self.relevance_delta,
            "timestamp": self.timestamp.isoformat(),
        }


class AlignmentActionBandit:
    """
    Constrained bandit for alignment actions.
    
    Safety constraints:
    - Existing CUSUM thresholds remain hard safety override
    - Per-mission caps on disruptive actions
    - Cannot override user-triggered events
    
    Usage:
        bandit = AlignmentActionBandit()
        
        # Select action (only if CUSUM triggered)
        action = bandit.select_action(context, cusum_triggered=True)
        
        if action:
            # Apply action
            ...
            
            # Record outcome
            bandit.record_outcome(AlignmentOutcome(
                action=action,
                drift_before=0.6,
                drift_after=0.3,
                relevance_before=0.8,
                relevance_after=0.75,
            ))
    """
    
    def __init__(
        self,
        config: Optional[AlignmentLearningConfig] = None,
        bandit_config: Optional[BanditConfig] = None,
    ):
        """
        Initialize the alignment action bandit.
        
        Args:
            config: Alignment learning configuration
            bandit_config: Bandit configuration
        """
        self.config = config or get_alignment_learning_config()
        self.bandit_config = bandit_config or get_bandit_config()
        
        # Initialize underlying bandit
        self.bandit = GeneralizedBandit(
            decision_class="alignment_action",
            arms=[a.value for a in AlignmentAction],
            config=self.bandit_config,
        )
        
        # Per-mission action counts (reset per mission)
        self._action_counts: Dict[AlignmentAction, int] = defaultdict(int)
        self._current_mission_id: str = ""
    
    def reset_mission(self, mission_id: str) -> None:
        """
        Reset per-mission caps for new mission.
        
        Args:
            mission_id: New mission identifier
        """
        self._action_counts = defaultdict(int)
        self._current_mission_id = mission_id
    
    def select_action(
        self,
        context: AlignmentContext,
        cusum_triggered: bool = False,
        mission_id: str = "",
    ) -> Optional[AlignmentAction]:
        """
        Select alignment action.
        
        Returns None if:
        - CUSUM not triggered (safety floor)
        - Per-mission cap exceeded
        - Bandit frozen
        - Feature disabled
        
        Args:
            context: Current alignment context
            cusum_triggered: Whether CUSUM threshold was exceeded
            mission_id: Mission identifier (for per-mission caps)
            
        Returns:
            AlignmentAction or None
        """
        if not self.config.enabled:
            return None
        
        # Safety floor: CUSUM must trigger first
        if self.config.cusum_override_enabled and not cusum_triggered:
            return None
        
        # Reset caps if new mission
        if mission_id and mission_id != self._current_mission_id:
            self.reset_mission(mission_id)
        
        # Get bandit recommendation
        features = context.to_features()
        recommended_arm = self.bandit.select(features)
        
        try:
            action = AlignmentAction(recommended_arm)
        except ValueError:
            action = AlignmentAction.REANCHOR_INTERNAL
        
        # Check per-mission caps
        action = self._check_caps(action)
        
        # Record selection
        self._action_counts[action] += 1
        
        logger.debug(
            f"[ALIGNMENT_BANDIT] Selected: {action.value} "
            f"(count={self._action_counts[action]}, "
            f"drift={context.drift_score:.2f})"
        )
        
        return action
    
    def _check_caps(self, action: AlignmentAction) -> AlignmentAction:
        """
        Check per-mission caps and fallback if needed.
        
        Args:
            action: Originally recommended action
            
        Returns:
            Action to use (may be fallback)
        """
        # Define caps
        caps = {
            AlignmentAction.PRUNE_OR_PARK_FOCUS_AREAS: self.config.max_prune_per_mission,
            AlignmentAction.TRIGGER_USER_EVENT_DRIFT_CONFIRMATION: self.config.max_user_events_per_mission,
            AlignmentAction.SWITCH_DEEPEN_MODE_TO_EVIDENCE: self.config.max_evidence_mode_switches,
            AlignmentAction.INCREASE_SKEPTIC_WEIGHT: self.config.max_skeptic_increases,
        }
        
        # Check if cap exceeded
        if action in caps:
            if self._action_counts[action] >= caps[action]:
                # Fallback to less disruptive action
                logger.debug(
                    f"[ALIGNMENT_BANDIT] Cap exceeded for {action.value}, "
                    f"falling back to reanchor"
                )
                return AlignmentAction.REANCHOR_INTERNAL
        
        return action
    
    def record_outcome(self, outcome: AlignmentOutcome) -> bool:
        """
        Record outcome for learning.
        
        Args:
            outcome: AlignmentOutcome to record
            
        Returns:
            True if recorded successfully
        """
        if not self.config.enabled:
            return False
        
        # Compute reward
        reward = outcome.compute_reward(self.config)
        
        # Update bandit
        success = self.bandit.update(outcome.action.value, reward)
        
        logger.debug(
            f"[ALIGNMENT_BANDIT] Recorded: {outcome.action.value} -> "
            f"reward={reward:.3f} (drift_reduction={outcome.drift_reduction:.2f}, "
            f"relevance_delta={outcome.relevance_delta:.2f})"
        )
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            "enabled": self.config.enabled,
            "mission_id": self._current_mission_id,
            "action_counts": dict(self._action_counts),
            "caps": {
                "prune": self.config.max_prune_per_mission,
                "user_events": self.config.max_user_events_per_mission,
                "evidence_switches": self.config.max_evidence_mode_switches,
            },
            "bandit_stats": self.bandit.get_stats(),
        }
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history for current mission."""
        return [
            {"action": action.value, "count": count}
            for action, count in self._action_counts.items()
            if count > 0
        ]


# Global bandit instance
_bandit: Optional[AlignmentActionBandit] = None


def get_alignment_action_bandit(
    config: Optional[AlignmentLearningConfig] = None,
) -> AlignmentActionBandit:
    """Get global alignment action bandit instance."""
    global _bandit
    if _bandit is None:
        _bandit = AlignmentActionBandit(config=config)
    return _bandit


def reset_alignment_action_bandit() -> None:
    """Reset global bandit (mainly for testing)."""
    global _bandit
    _bandit = None


