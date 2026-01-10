"""
Alignment Learning Configuration for DeepThinker.

Module-local configuration for the alignment auto-tuning subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlignmentLearningConfig:
    """
    Configuration for the alignment learning subsystem.
    
    Attributes:
        enabled: Whether alignment learning is enabled
        
        # Per-mission caps for disruptive actions
        max_prune_per_mission: Maximum focus area prunes per mission
        max_user_events_per_mission: Maximum user confirmation events
        max_evidence_mode_switches: Maximum evidence mode switches
        max_skeptic_increases: Maximum skeptic weight increases
        
        # Safety overrides
        cusum_override_enabled: Keep CUSUM as hard safety floor
        
        # Bandit settings (uses bandits/config.py for general settings)
        bandit_decision_class: Decision class name for alignment bandit
        
        # Learning settings
        drift_reduction_weight: Weight for drift reduction in reward
        relevance_penalty_weight: Weight for relevance loss in reward
    """
    
    enabled: bool = False
    
    # Per-mission caps for disruptive actions
    max_prune_per_mission: int = 2
    max_user_events_per_mission: int = 1
    max_evidence_mode_switches: int = 3
    max_skeptic_increases: int = 5
    
    # Safety overrides
    cusum_override_enabled: bool = True
    
    # Bandit settings
    bandit_decision_class: str = "alignment_action"
    
    # Learning settings
    drift_reduction_weight: float = 1.0
    relevance_penalty_weight: float = 0.5
    
    @classmethod
    def from_env(cls) -> "AlignmentLearningConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            ALIGNMENT_LEARNING_ENABLED: "true" to enable
            ALIGNMENT_MAX_PRUNE_PER_MISSION: int
            ALIGNMENT_MAX_USER_EVENTS_PER_MISSION: int
            ALIGNMENT_MAX_EVIDENCE_MODE_SWITCHES: int
            ALIGNMENT_MAX_SKEPTIC_INCREASES: int
            ALIGNMENT_CUSUM_OVERRIDE_ENABLED: "true" to keep CUSUM safety
            ALIGNMENT_DRIFT_REDUCTION_WEIGHT: float
            ALIGNMENT_RELEVANCE_PENALTY_WEIGHT: float
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default
        
        def get_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        return cls(
            enabled=get_bool("ALIGNMENT_LEARNING_ENABLED", False),
            max_prune_per_mission=get_int("ALIGNMENT_MAX_PRUNE_PER_MISSION", 2),
            max_user_events_per_mission=get_int("ALIGNMENT_MAX_USER_EVENTS_PER_MISSION", 1),
            max_evidence_mode_switches=get_int("ALIGNMENT_MAX_EVIDENCE_MODE_SWITCHES", 3),
            max_skeptic_increases=get_int("ALIGNMENT_MAX_SKEPTIC_INCREASES", 5),
            cusum_override_enabled=get_bool("ALIGNMENT_CUSUM_OVERRIDE_ENABLED", True),
            drift_reduction_weight=get_float("ALIGNMENT_DRIFT_REDUCTION_WEIGHT", 1.0),
            relevance_penalty_weight=get_float("ALIGNMENT_RELEVANCE_PENALTY_WEIGHT", 0.5),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "max_prune_per_mission": self.max_prune_per_mission,
            "max_user_events_per_mission": self.max_user_events_per_mission,
            "max_evidence_mode_switches": self.max_evidence_mode_switches,
            "max_skeptic_increases": self.max_skeptic_increases,
            "cusum_override_enabled": self.cusum_override_enabled,
            "bandit_decision_class": self.bandit_decision_class,
            "drift_reduction_weight": self.drift_reduction_weight,
            "relevance_penalty_weight": self.relevance_penalty_weight,
        }


# Global config instance (lazy-loaded)
_config: Optional[AlignmentLearningConfig] = None


def get_alignment_learning_config(force_reload: bool = False) -> AlignmentLearningConfig:
    """
    Get the global alignment learning configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        AlignmentLearningConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = AlignmentLearningConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[ALIGNMENT_LEARNING] Enabled with caps=[prune={_config.max_prune_per_mission}, "
                f"user_events={_config.max_user_events_per_mission}, "
                f"evidence_switches={_config.max_evidence_mode_switches}], "
                f"cusum_override={_config.cusum_override_enabled}"
            )
    
    return _config


def reset_alignment_learning_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None


