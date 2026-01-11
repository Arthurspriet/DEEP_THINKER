"""
Replay Configuration for DeepThinker.

Module-local configuration for the replay subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    """
    Replay fidelity mode.
    
    DECISIONS_ONLY: No model calls, replay routing/bandit only (default)
    WITH_JUDGES: Re-run judge scoring (expensive)
    FULL: Full re-execution (very expensive, requires model access)
    """
    DECISIONS_ONLY = "decisions_only"
    WITH_JUDGES = "with_judges"
    FULL = "full"


@dataclass
class ReplayConfig:
    """
    Configuration for the replay subsystem.
    
    Attributes:
        enabled: Whether replay is enabled
        mode: Replay fidelity mode
        output_path: Directory for replay results
        
        # Comparison settings
        include_regret_analysis: Whether to compute regret
        max_decisions_per_replay: Limit on decisions to replay
    """
    
    enabled: bool = False
    mode: ReplayMode = ReplayMode.DECISIONS_ONLY
    output_path: str = "kb/replay/"
    
    # Comparison settings
    include_regret_analysis: bool = True
    max_decisions_per_replay: int = 1000
    
    @classmethod
    def from_env(cls) -> "ReplayConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            REPLAY_ENABLED: "true" to enable
            REPLAY_MODE: "decisions_only", "with_judges", "full"
            REPLAY_OUTPUT_PATH: path string
            REPLAY_INCLUDE_REGRET: "true" to compute regret
            REPLAY_MAX_DECISIONS: int
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        # Parse replay mode
        mode_str = os.environ.get("REPLAY_MODE", "decisions_only").lower()
        try:
            mode = ReplayMode(mode_str)
        except ValueError:
            mode = ReplayMode.DECISIONS_ONLY
        
        return cls(
            enabled=get_bool("REPLAY_ENABLED", False),
            mode=mode,
            output_path=os.environ.get("REPLAY_OUTPUT_PATH", "kb/replay/"),
            include_regret_analysis=get_bool("REPLAY_INCLUDE_REGRET", True),
            max_decisions_per_replay=get_int("REPLAY_MAX_DECISIONS", 1000),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "output_path": self.output_path,
            "include_regret_analysis": self.include_regret_analysis,
            "max_decisions_per_replay": self.max_decisions_per_replay,
        }


# Global config instance (lazy-loaded)
_config: Optional[ReplayConfig] = None


def get_replay_config(force_reload: bool = False) -> ReplayConfig:
    """
    Get the global replay configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        ReplayConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = ReplayConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[REPLAY] Enabled with mode={_config.mode.value}, "
                f"output_path={_config.output_path}"
            )
    
    return _config


def reset_replay_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None




