"""
Reward Configuration for DeepThinker.

Module-local configuration for the rewards subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """
    Configuration for the rewards subsystem.
    
    Attributes:
        enabled: Whether reward computation is enabled
        version: Schema version for reward signals
        
        # Hard clamps (safety bounds)
        cost_penalty_clamp: Maximum cost penalty contribution
        alignment_penalty_clamp: Maximum alignment penalty contribution
        time_penalty_clamp: Maximum time penalty contribution
        
        # Normalization baselines
        baseline_cost_tokens: Baseline token count for normalization
        baseline_latency_ms: Baseline latency for normalization
        baseline_cost_usd: Baseline USD cost for normalization
        
        # Persistence
        store_path: Path to reward history storage
    """
    
    enabled: bool = False
    version: str = "1.0.0"
    
    # Hard clamps (safety bounds - penalties cannot exceed these)
    cost_penalty_clamp: float = 0.3
    alignment_penalty_clamp: float = 0.2
    time_penalty_clamp: float = 0.15
    
    # Normalization baselines
    baseline_cost_tokens: int = 10000
    baseline_latency_ms: int = 5000
    baseline_cost_usd: float = 0.05
    
    # Persistence
    store_path: str = "kb/rewards/reward_history.jsonl"
    
    @classmethod
    def from_env(cls) -> "RewardConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            REWARD_ENABLED: "true" to enable
            REWARD_VERSION: Schema version string
            REWARD_COST_PENALTY_CLAMP: float
            REWARD_ALIGNMENT_PENALTY_CLAMP: float
            REWARD_TIME_PENALTY_CLAMP: float
            REWARD_BASELINE_COST_TOKENS: int
            REWARD_BASELINE_LATENCY_MS: int
            REWARD_BASELINE_COST_USD: float
            REWARD_STORE_PATH: path string
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
            enabled=get_bool("REWARD_ENABLED", False),
            version=os.environ.get("REWARD_VERSION", "1.0.0"),
            cost_penalty_clamp=get_float("REWARD_COST_PENALTY_CLAMP", 0.3),
            alignment_penalty_clamp=get_float("REWARD_ALIGNMENT_PENALTY_CLAMP", 0.2),
            time_penalty_clamp=get_float("REWARD_TIME_PENALTY_CLAMP", 0.15),
            baseline_cost_tokens=get_int("REWARD_BASELINE_COST_TOKENS", 10000),
            baseline_latency_ms=get_int("REWARD_BASELINE_LATENCY_MS", 5000),
            baseline_cost_usd=get_float("REWARD_BASELINE_COST_USD", 0.05),
            store_path=os.environ.get("REWARD_STORE_PATH", "kb/rewards/reward_history.jsonl"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "version": self.version,
            "cost_penalty_clamp": self.cost_penalty_clamp,
            "alignment_penalty_clamp": self.alignment_penalty_clamp,
            "time_penalty_clamp": self.time_penalty_clamp,
            "baseline_cost_tokens": self.baseline_cost_tokens,
            "baseline_latency_ms": self.baseline_latency_ms,
            "baseline_cost_usd": self.baseline_cost_usd,
            "store_path": self.store_path,
        }


# Global config instance (lazy-loaded)
_config: Optional[RewardConfig] = None


def get_reward_config(force_reload: bool = False) -> RewardConfig:
    """
    Get the global reward configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        RewardConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = RewardConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[REWARDS] Enabled with version={_config.version}, "
                f"clamps=[cost={_config.cost_penalty_clamp}, "
                f"alignment={_config.alignment_penalty_clamp}, "
                f"time={_config.time_penalty_clamp}]"
            )
    
    return _config


def reset_reward_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None




