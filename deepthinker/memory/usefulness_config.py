"""
Memory Usefulness Configuration for DeepThinker.

Module-local configuration for the memory usefulness subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryUsefulnessConfig:
    """
    Configuration for the memory usefulness subsystem.
    
    Attributes:
        enabled: Whether memory usefulness prediction is enabled
        weights_path: Path to learned model weights
        counterfactual_log_path: Path for counterfactual logging
        
        # Budget caps per phase type (in tokens)
        budget_tokens_recon: Token budget for reconnaissance phases
        budget_tokens_synthesis: Token budget for synthesis phases
        budget_tokens_default: Default token budget
        
        # Filtering thresholds
        min_helpfulness_threshold: Minimum P(helpful) to inject
        max_memories_per_phase: Maximum memories to inject per phase
        
        # Decay settings
        age_decay_factor: Decay factor per phase of age
    """
    
    enabled: bool = False
    weights_path: str = "kb/models/memory_usefulness/weights.json"
    counterfactual_log_path: str = "kb/memory/counterfactual_logs.jsonl"
    
    # Budget caps per phase type
    budget_tokens_recon: int = 500
    budget_tokens_synthesis: int = 1500
    budget_tokens_default: int = 1000
    
    # Filtering thresholds
    min_helpfulness_threshold: float = 0.3
    max_memories_per_phase: int = 10
    
    # Decay settings
    age_decay_factor: float = 0.95
    
    @classmethod
    def from_env(cls) -> "MemoryUsefulnessConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            MEMORY_USEFULNESS_ENABLED: "true" to enable
            MEMORY_USEFULNESS_WEIGHTS_PATH: path string
            MEMORY_USEFULNESS_COUNTERFACTUAL_LOG_PATH: path string
            MEMORY_USEFULNESS_BUDGET_RECON: int
            MEMORY_USEFULNESS_BUDGET_SYNTHESIS: int
            MEMORY_USEFULNESS_BUDGET_DEFAULT: int
            MEMORY_USEFULNESS_MIN_HELPFULNESS: float
            MEMORY_USEFULNESS_MAX_PER_PHASE: int
            MEMORY_USEFULNESS_AGE_DECAY: float
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
            enabled=get_bool("MEMORY_USEFULNESS_ENABLED", False),
            weights_path=os.environ.get(
                "MEMORY_USEFULNESS_WEIGHTS_PATH",
                "kb/models/memory_usefulness/weights.json"
            ),
            counterfactual_log_path=os.environ.get(
                "MEMORY_USEFULNESS_COUNTERFACTUAL_LOG_PATH",
                "kb/memory/counterfactual_logs.jsonl"
            ),
            budget_tokens_recon=get_int("MEMORY_USEFULNESS_BUDGET_RECON", 500),
            budget_tokens_synthesis=get_int("MEMORY_USEFULNESS_BUDGET_SYNTHESIS", 1500),
            budget_tokens_default=get_int("MEMORY_USEFULNESS_BUDGET_DEFAULT", 1000),
            min_helpfulness_threshold=get_float("MEMORY_USEFULNESS_MIN_HELPFULNESS", 0.3),
            max_memories_per_phase=get_int("MEMORY_USEFULNESS_MAX_PER_PHASE", 10),
            age_decay_factor=get_float("MEMORY_USEFULNESS_AGE_DECAY", 0.95),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "weights_path": self.weights_path,
            "counterfactual_log_path": self.counterfactual_log_path,
            "budget_tokens_recon": self.budget_tokens_recon,
            "budget_tokens_synthesis": self.budget_tokens_synthesis,
            "budget_tokens_default": self.budget_tokens_default,
            "min_helpfulness_threshold": self.min_helpfulness_threshold,
            "max_memories_per_phase": self.max_memories_per_phase,
            "age_decay_factor": self.age_decay_factor,
        }
    
    def get_budget_for_phase(self, phase_name: str) -> int:
        """Get token budget for a specific phase type."""
        phase_lower = phase_name.lower()
        
        if any(kw in phase_lower for kw in ["recon", "gather", "initial", "explore"]):
            return self.budget_tokens_recon
        elif any(kw in phase_lower for kw in ["synth", "final", "report", "conclude"]):
            return self.budget_tokens_synthesis
        else:
            return self.budget_tokens_default


# Global config instance (lazy-loaded)
_config: Optional[MemoryUsefulnessConfig] = None


def get_memory_usefulness_config(force_reload: bool = False) -> MemoryUsefulnessConfig:
    """
    Get the global memory usefulness configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        MemoryUsefulnessConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = MemoryUsefulnessConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[MEMORY_USEFULNESS] Enabled with min_helpfulness={_config.min_helpfulness_threshold}, "
                f"budgets=[recon={_config.budget_tokens_recon}, "
                f"synthesis={_config.budget_tokens_synthesis}]"
            )
    
    return _config


def reset_memory_usefulness_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None




