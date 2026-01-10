"""
Bandit Configuration for DeepThinker.

Module-local configuration for the bandits subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BanditConfig:
    """
    Configuration for the bandits subsystem.
    
    Attributes:
        enabled: Whether bandits are enabled
        schema_version: Schema version for bandit state validation
        freeze_mode: If True, bandits are read-only (no updates)
        min_trials_before_exploit: Minimum trials per arm before exploitation
        exploration_bonus: UCB exploration coefficient (c parameter)
        lambda_cost: Cost penalty weight for reward computation
        algorithm: Selection algorithm ("ucb" or "thompson")
        
        # Persistence
        store_dir: Directory for bandit state files
    """
    
    enabled: bool = False
    schema_version: str = "1.0.0"
    freeze_mode: bool = False
    min_trials_before_exploit: int = 10
    exploration_bonus: float = 1.0
    lambda_cost: float = 0.1
    algorithm: str = "ucb"  # "ucb" or "thompson"
    
    # Persistence
    store_dir: str = "kb/bandits/"
    
    @classmethod
    def from_env(cls) -> "BanditConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            BANDIT_ENABLED: "true" to enable
            BANDIT_SCHEMA_VERSION: Schema version string
            BANDIT_FREEZE_MODE: "true" for read-only mode
            BANDIT_MIN_TRIALS_BEFORE_EXPLOIT: int
            BANDIT_EXPLORATION_BONUS: float
            BANDIT_LAMBDA_COST: float
            BANDIT_ALGORITHM: "ucb" or "thompson"
            BANDIT_STORE_DIR: path string
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
        
        algorithm = os.environ.get("BANDIT_ALGORITHM", "ucb").lower()
        if algorithm not in ("ucb", "thompson"):
            algorithm = "ucb"
        
        return cls(
            enabled=get_bool("BANDIT_ENABLED", False),
            schema_version=os.environ.get("BANDIT_SCHEMA_VERSION", "1.0.0"),
            freeze_mode=get_bool("BANDIT_FREEZE_MODE", False),
            min_trials_before_exploit=get_int("BANDIT_MIN_TRIALS_BEFORE_EXPLOIT", 10),
            exploration_bonus=get_float("BANDIT_EXPLORATION_BONUS", 1.0),
            lambda_cost=get_float("BANDIT_LAMBDA_COST", 0.1),
            algorithm=algorithm,
            store_dir=os.environ.get("BANDIT_STORE_DIR", "kb/bandits/"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "schema_version": self.schema_version,
            "freeze_mode": self.freeze_mode,
            "min_trials_before_exploit": self.min_trials_before_exploit,
            "exploration_bonus": self.exploration_bonus,
            "lambda_cost": self.lambda_cost,
            "algorithm": self.algorithm,
            "store_dir": self.store_dir,
        }


# Global config instance (lazy-loaded)
_config: Optional[BanditConfig] = None


def get_bandit_config(force_reload: bool = False) -> BanditConfig:
    """
    Get the global bandit configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        BanditConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = BanditConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[BANDITS] Enabled with schema_version={_config.schema_version}, "
                f"freeze_mode={_config.freeze_mode}, "
                f"min_trials={_config.min_trials_before_exploit}, "
                f"algorithm={_config.algorithm}"
            )
    
    return _config


def reset_bandit_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None


