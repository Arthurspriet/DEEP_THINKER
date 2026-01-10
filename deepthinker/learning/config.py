"""
Learning Configuration for DeepThinker.

Module-local configuration for the learning subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LearnedPolicyMode(str, Enum):
    """
    Operating mode for learned policy.
    
    OFF: Disabled entirely - no predictions, no logging
    SHADOW: Log predictions but don't change behavior (for analysis)
    ADVISORY: Log + surface to orchestrator (can ignore)
    ACTIVE: Predictions drive decisions (rules still safety floor)
    """
    OFF = "off"
    SHADOW = "shadow"
    ADVISORY = "advisory"
    ACTIVE = "active"


@dataclass
class LearningConfig:
    """
    Configuration for the learning subsystem.
    
    Attributes:
        enabled: Whether learning features are enabled
        policy_mode: Operating mode for learned policy
        model_path: Path to learned model weights
        shadow_log_path: Path for shadow mode logging
        
        # Prediction thresholds
        stop_threshold: P(stop) threshold for recommending stop
        escalate_threshold: P(escalate) threshold for recommending escalate
        
        # Feature extraction
        score_trend_window: Number of recent scores for trend computation
    """
    
    enabled: bool = False
    policy_mode: LearnedPolicyMode = LearnedPolicyMode.OFF
    model_path: str = "kb/models/stop_escalate/weights.json"
    shadow_log_path: str = "kb/learning/shadow_logs.jsonl"
    
    # Prediction thresholds
    stop_threshold: float = 0.7
    escalate_threshold: float = 0.6
    
    # Feature extraction
    score_trend_window: int = 5
    
    @classmethod
    def from_env(cls) -> "LearningConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            LEARNING_ENABLED: "true" to enable
            LEARNED_POLICY_MODE: "off", "shadow", "advisory", "active"
            LEARNING_MODEL_PATH: path string
            LEARNING_SHADOW_LOG_PATH: path string
            LEARNING_STOP_THRESHOLD: float
            LEARNING_ESCALATE_THRESHOLD: float
            LEARNING_SCORE_TREND_WINDOW: int
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
        
        # Parse policy mode
        mode_str = os.environ.get("LEARNED_POLICY_MODE", "off").lower()
        try:
            policy_mode = LearnedPolicyMode(mode_str)
        except ValueError:
            policy_mode = LearnedPolicyMode.OFF
        
        return cls(
            enabled=get_bool("LEARNING_ENABLED", False),
            policy_mode=policy_mode,
            model_path=os.environ.get("LEARNING_MODEL_PATH", "kb/models/stop_escalate/weights.json"),
            shadow_log_path=os.environ.get("LEARNING_SHADOW_LOG_PATH", "kb/learning/shadow_logs.jsonl"),
            stop_threshold=get_float("LEARNING_STOP_THRESHOLD", 0.7),
            escalate_threshold=get_float("LEARNING_ESCALATE_THRESHOLD", 0.6),
            score_trend_window=get_int("LEARNING_SCORE_TREND_WINDOW", 5),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "policy_mode": self.policy_mode.value,
            "model_path": self.model_path,
            "shadow_log_path": self.shadow_log_path,
            "stop_threshold": self.stop_threshold,
            "escalate_threshold": self.escalate_threshold,
            "score_trend_window": self.score_trend_window,
        }


# Global config instance (lazy-loaded)
_config: Optional[LearningConfig] = None


def get_learning_config(force_reload: bool = False) -> LearningConfig:
    """
    Get the global learning configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        LearningConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = LearningConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[LEARNING] Enabled with policy_mode={_config.policy_mode.value}, "
                f"stop_threshold={_config.stop_threshold}, "
                f"escalate_threshold={_config.escalate_threshold}"
            )
    
    return _config


def reset_learning_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None


