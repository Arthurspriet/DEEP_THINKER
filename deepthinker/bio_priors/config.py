"""
Bio Priors Configuration.

Manages configuration for the bio priors subsystem.
Supports loading from environment variables.

Configuration is opt-in: DEEPTHINKER_BIO_PRIORS_ENABLED=False by default.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Valid modes for bio priors
VALID_MODES = ("off", "advisory", "shadow", "soft")


@dataclass
class BioPriorConfig:
    """
    Configuration for the Bio Priors system.
    
    All settings are configurable via environment variables.
    
    Attributes:
        enabled: Master switch for bio priors (default: False)
        mode: Operating mode - "off" | "advisory" | "shadow" | "soft" (default: "off")
        topk: Number of top patterns to select (default: 3)
        max_pressure: Maximum pressure scaling factor (default: 1.0)
    
    Mode definitions:
        - off: No evaluation, no output
        - advisory: Run engine, log output, no behavioral changes
        - shadow: Run engine, log output + "would_apply" diff, no behavioral changes
        - soft: Run engine, apply bounded modifications (v1: depth_budget_delta only)
    """
    enabled: bool = False
    mode: str = "off"
    topk: int = 3
    max_pressure: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.mode not in VALID_MODES:
            logger.warning(
                f"[BIO_PRIORS] Invalid mode '{self.mode}', defaulting to 'off'. "
                f"Valid modes: {VALID_MODES}"
            )
            self.mode = "off"
        
        if self.topk < 1:
            logger.warning(f"[BIO_PRIORS] topk must be >= 1, got {self.topk}, defaulting to 3")
            self.topk = 3
        
        if self.max_pressure < 0.0 or self.max_pressure > 2.0:
            logger.warning(
                f"[BIO_PRIORS] max_pressure should be in [0.0, 2.0], "
                f"got {self.max_pressure}, clamping"
            )
            self.max_pressure = max(0.0, min(2.0, self.max_pressure))
        
        # If enabled but mode is off, treat as disabled
        if self.enabled and self.mode == "off":
            logger.debug("[BIO_PRIORS] Enabled but mode='off', effectively disabled")
    
    @property
    def is_active(self) -> bool:
        """Check if bio priors are active (enabled AND mode != off)."""
        return self.enabled and self.mode != "off"
    
    @property
    def should_apply(self) -> bool:
        """Check if bio priors should apply changes (soft mode only)."""
        return self.enabled and self.mode == "soft"
    
    @property
    def should_compute_diff(self) -> bool:
        """Check if bio priors should compute 'would_apply' diff (shadow or soft)."""
        return self.enabled and self.mode in ("shadow", "soft")
    
    @classmethod
    def from_env(cls) -> "BioPriorConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            DEEPTHINKER_BIO_PRIORS_ENABLED: "true" to enable (default: false)
            DEEPTHINKER_BIO_PRIORS_MODE: "off"|"advisory"|"shadow"|"soft" (default: "off")
            DEEPTHINKER_BIO_PRIORS_TOPK: Number of top patterns (default: 3)
            DEEPTHINKER_BIO_PRIORS_MAX_PRESSURE: Max pressure scale (default: 1.0)
        
        Returns:
            BioPriorConfig instance
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
        
        def get_str(key: str, default: str) -> str:
            return os.environ.get(key, default).lower()
        
        config = cls(
            enabled=get_bool("DEEPTHINKER_BIO_PRIORS_ENABLED", False),
            mode=get_str("DEEPTHINKER_BIO_PRIORS_MODE", "off"),
            topk=get_int("DEEPTHINKER_BIO_PRIORS_TOPK", 3),
            max_pressure=get_float("DEEPTHINKER_BIO_PRIORS_MAX_PRESSURE", 1.0),
        )
        
        if config.is_active:
            logger.info(
                f"[BIO_PRIORS] Enabled with config: "
                f"mode={config.mode}, topk={config.topk}, "
                f"max_pressure={config.max_pressure}"
            )
        
        return config
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "topk": self.topk,
            "max_pressure": self.max_pressure,
            "is_active": self.is_active,
            "should_apply": self.should_apply,
        }


# Global config instance (lazy-loaded)
_config: Optional[BioPriorConfig] = None


def get_bio_prior_config(force_reload: bool = False) -> BioPriorConfig:
    """
    Get the global bio prior configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        BioPriorConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = BioPriorConfig.from_env()
    
    return _config


def reset_bio_prior_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None



