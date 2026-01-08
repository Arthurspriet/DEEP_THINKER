"""
Configuration for Cognitive Constitution.

Centralized configuration for constitution enforcement:
- Mode: off | shadow | enforce
- Thresholds for invariant checks
- Feature flags for individual invariants

All settings are configurable via environment variables.
All feature flags default to OFF for backward compatibility.
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConstitutionMode(str, Enum):
    """Operating mode for constitution enforcement."""
    OFF = "off"
    """Constitution disabled - no checks, no logging."""
    
    SHADOW = "shadow"
    """Log violations but don't enforce - advisory mode."""
    
    ENFORCE = "enforce"
    """Full enforcement - block learning, stop deepening on violations."""


@dataclass
class ConstitutionConfig:
    """
    Configuration for the Cognitive Constitution.
    
    All settings are configurable via environment variables.
    All feature flags default to OFF for backward compatibility.
    
    Attributes:
        mode: Operating mode (off, shadow, enforce)
        
        # Evidence Conservation
        evidence_threshold: Minimum score delta requiring new evidence
        
        # No-Free-Lunch Depth
        depth_penalty: Penalty for unproductive depth rounds
        max_unproductive_rounds: Max rounds without measurable gain
        
        # Goodhart Shield
        divergence_threshold: Target/shadow divergence sensitivity
        shadow_window: Number of phases to track for shadow metrics
        
        # Blinding
        blinding_enabled: Whether to sanitize judge inputs
        
        # Ledger
        ledger_enabled: Whether to write to constitution ledger
        ledger_base_dir: Base directory for ledger files
        
        # Individual invariant toggles
        evidence_conservation_enabled: Check evidence conservation
        monotonic_uncertainty_enabled: Check compression uncertainty
        no_free_lunch_enabled: Check depth productivity
        goodhart_shield_enabled: Check target/shadow divergence
    """
    
    # Master mode
    mode: ConstitutionMode = ConstitutionMode.OFF
    
    # Evidence Conservation thresholds
    evidence_threshold: float = 0.01  # Min score delta requiring evidence
    min_evidence_for_score_increase: int = 1  # Min new evidence objects
    
    # Monotonic Uncertainty thresholds
    compression_uncertainty_margin: float = 0.05  # Allowed uncertainty reduction
    
    # No-Free-Lunch Depth thresholds
    depth_penalty: float = 0.1  # Penalty for unproductive depth
    max_unproductive_rounds: int = 2  # Max rounds without gain
    min_gain_per_round: float = 0.02  # Minimum measurable gain per round
    
    # Goodhart Shield thresholds
    divergence_threshold: float = 0.05  # Target/shadow divergence sensitivity
    target_improvement_threshold: float = 0.03  # Target metric improvement
    shadow_improvement_threshold: float = 0.01  # Shadow metric improvement
    shadow_window: int = 5  # Phases to track for shadow metrics
    
    # Blinding
    blinding_enabled: bool = True  # Sanitize judge inputs
    
    # Ledger
    ledger_enabled: bool = True  # Write to constitution ledger
    ledger_base_dir: str = "kb/constitution"  # Base directory
    
    # Individual invariant toggles (all default ON when mode != OFF)
    evidence_conservation_enabled: bool = True
    monotonic_uncertainty_enabled: bool = True
    no_free_lunch_enabled: bool = True
    goodhart_shield_enabled: bool = True
    
    # Privacy
    hash_sensitive_data: bool = True  # Hash sensitive params in ledger
    max_excerpt_length: int = 200  # Max chars for text excerpts
    
    @property
    def is_enabled(self) -> bool:
        """Check if constitution is enabled (shadow or enforce)."""
        return self.mode != ConstitutionMode.OFF
    
    @property
    def is_enforcing(self) -> bool:
        """Check if constitution is in enforce mode."""
        return self.mode == ConstitutionMode.ENFORCE
    
    @classmethod
    def from_env(cls) -> "ConstitutionConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            CONSTITUTION_MODE: off | shadow | enforce (default: off)
            CONSTITUTION_EVIDENCE_THRESHOLD: float (default: 0.01)
            CONSTITUTION_DEPTH_PENALTY: float (default: 0.1)
            CONSTITUTION_DIVERGENCE_THRESHOLD: float (default: 0.05)
            CONSTITUTION_BLINDING_ENABLED: true | false (default: true)
            CONSTITUTION_LEDGER_ENABLED: true | false (default: true)
            CONSTITUTION_LEDGER_DIR: path (default: kb/constitution)
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
        
        # Parse mode
        mode_str = os.environ.get("CONSTITUTION_MODE", "off").lower()
        try:
            mode = ConstitutionMode(mode_str)
        except ValueError:
            logger.warning(f"Invalid CONSTITUTION_MODE '{mode_str}', defaulting to 'off'")
            mode = ConstitutionMode.OFF
        
        return cls(
            mode=mode,
            evidence_threshold=get_float("CONSTITUTION_EVIDENCE_THRESHOLD", 0.01),
            min_evidence_for_score_increase=get_int("CONSTITUTION_MIN_EVIDENCE", 1),
            compression_uncertainty_margin=get_float("CONSTITUTION_UNCERTAINTY_MARGIN", 0.05),
            depth_penalty=get_float("CONSTITUTION_DEPTH_PENALTY", 0.1),
            max_unproductive_rounds=get_int("CONSTITUTION_MAX_UNPRODUCTIVE_ROUNDS", 2),
            min_gain_per_round=get_float("CONSTITUTION_MIN_GAIN_PER_ROUND", 0.02),
            divergence_threshold=get_float("CONSTITUTION_DIVERGENCE_THRESHOLD", 0.05),
            target_improvement_threshold=get_float("CONSTITUTION_TARGET_IMPROVEMENT", 0.03),
            shadow_improvement_threshold=get_float("CONSTITUTION_SHADOW_IMPROVEMENT", 0.01),
            shadow_window=get_int("CONSTITUTION_SHADOW_WINDOW", 5),
            blinding_enabled=get_bool("CONSTITUTION_BLINDING_ENABLED", True),
            ledger_enabled=get_bool("CONSTITUTION_LEDGER_ENABLED", True),
            ledger_base_dir=os.environ.get("CONSTITUTION_LEDGER_DIR", "kb/constitution"),
            evidence_conservation_enabled=get_bool("CONSTITUTION_EVIDENCE_CONSERVATION", True),
            monotonic_uncertainty_enabled=get_bool("CONSTITUTION_MONOTONIC_UNCERTAINTY", True),
            no_free_lunch_enabled=get_bool("CONSTITUTION_NO_FREE_LUNCH", True),
            goodhart_shield_enabled=get_bool("CONSTITUTION_GOODHART_SHIELD", True),
            hash_sensitive_data=get_bool("CONSTITUTION_HASH_SENSITIVE", True),
            max_excerpt_length=get_int("CONSTITUTION_MAX_EXCERPT", 200),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "evidence_threshold": self.evidence_threshold,
            "min_evidence_for_score_increase": self.min_evidence_for_score_increase,
            "compression_uncertainty_margin": self.compression_uncertainty_margin,
            "depth_penalty": self.depth_penalty,
            "max_unproductive_rounds": self.max_unproductive_rounds,
            "min_gain_per_round": self.min_gain_per_round,
            "divergence_threshold": self.divergence_threshold,
            "target_improvement_threshold": self.target_improvement_threshold,
            "shadow_improvement_threshold": self.shadow_improvement_threshold,
            "shadow_window": self.shadow_window,
            "blinding_enabled": self.blinding_enabled,
            "ledger_enabled": self.ledger_enabled,
            "ledger_base_dir": self.ledger_base_dir,
            "evidence_conservation_enabled": self.evidence_conservation_enabled,
            "monotonic_uncertainty_enabled": self.monotonic_uncertainty_enabled,
            "no_free_lunch_enabled": self.no_free_lunch_enabled,
            "goodhart_shield_enabled": self.goodhart_shield_enabled,
        }


# Global config instance
_config: Optional[ConstitutionConfig] = None


def get_constitution_config() -> ConstitutionConfig:
    """Get global constitution config (lazy-loaded from env)."""
    global _config
    if _config is None:
        _config = ConstitutionConfig.from_env()
        if _config.is_enabled:
            logger.info(f"[CONSTITUTION] Loaded config: mode={_config.mode.value}")
    return _config


def reset_constitution_config() -> None:
    """Reset global config (for testing)."""
    global _config
    _config = None

