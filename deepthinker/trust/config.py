"""
Trust Configuration for DeepThinker.

Module-local configuration for the trust subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrustConfig:
    """
    Configuration for the trust subsystem.
    
    Attributes:
        enabled: Whether trust metrics are enabled
        
        # Component weights for overall trust
        confidence_calibration_weight: Weight for judge agreement
        epistemic_uncertainty_weight: Weight for ClaimGraph signals
        memory_reliance_weight: Weight for memory usage
        tool_reliance_weight: Weight for tool usage
        evidence_recency_weight: Weight for EvidenceObject recency
        evidence_diversity_weight: Weight for EvidenceObject diversity
        
        # Thresholds
        low_trust_threshold: Below this, trust is considered low
        high_trust_threshold: Above this, trust is considered high
        
        # EvidenceObject settings
        evidence_recency_days: Days before evidence is considered stale
    """
    
    enabled: bool = False
    
    # Component weights (should sum to ~1.0)
    confidence_calibration_weight: float = 0.25
    epistemic_uncertainty_weight: float = 0.20
    memory_reliance_weight: float = 0.10
    tool_reliance_weight: float = 0.10
    evidence_recency_weight: float = 0.15
    evidence_diversity_weight: float = 0.20
    
    # Thresholds
    low_trust_threshold: float = 0.4
    high_trust_threshold: float = 0.75
    
    # EvidenceObject settings
    evidence_recency_days: int = 365
    
    @classmethod
    def from_env(cls) -> "TrustConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            TRUST_ENABLED: "true" to enable
            TRUST_CONFIDENCE_CALIBRATION_WEIGHT: float
            TRUST_EPISTEMIC_UNCERTAINTY_WEIGHT: float
            TRUST_MEMORY_RELIANCE_WEIGHT: float
            TRUST_TOOL_RELIANCE_WEIGHT: float
            TRUST_EVIDENCE_RECENCY_WEIGHT: float
            TRUST_EVIDENCE_DIVERSITY_WEIGHT: float
            TRUST_LOW_THRESHOLD: float
            TRUST_HIGH_THRESHOLD: float
            TRUST_EVIDENCE_RECENCY_DAYS: int
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
            enabled=get_bool("TRUST_ENABLED", False),
            confidence_calibration_weight=get_float("TRUST_CONFIDENCE_CALIBRATION_WEIGHT", 0.25),
            epistemic_uncertainty_weight=get_float("TRUST_EPISTEMIC_UNCERTAINTY_WEIGHT", 0.20),
            memory_reliance_weight=get_float("TRUST_MEMORY_RELIANCE_WEIGHT", 0.10),
            tool_reliance_weight=get_float("TRUST_TOOL_RELIANCE_WEIGHT", 0.10),
            evidence_recency_weight=get_float("TRUST_EVIDENCE_RECENCY_WEIGHT", 0.15),
            evidence_diversity_weight=get_float("TRUST_EVIDENCE_DIVERSITY_WEIGHT", 0.20),
            low_trust_threshold=get_float("TRUST_LOW_THRESHOLD", 0.4),
            high_trust_threshold=get_float("TRUST_HIGH_THRESHOLD", 0.75),
            evidence_recency_days=get_int("TRUST_EVIDENCE_RECENCY_DAYS", 365),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "confidence_calibration_weight": self.confidence_calibration_weight,
            "epistemic_uncertainty_weight": self.epistemic_uncertainty_weight,
            "memory_reliance_weight": self.memory_reliance_weight,
            "tool_reliance_weight": self.tool_reliance_weight,
            "evidence_recency_weight": self.evidence_recency_weight,
            "evidence_diversity_weight": self.evidence_diversity_weight,
            "low_trust_threshold": self.low_trust_threshold,
            "high_trust_threshold": self.high_trust_threshold,
            "evidence_recency_days": self.evidence_recency_days,
        }


# Global config instance (lazy-loaded)
_config: Optional[TrustConfig] = None


def get_trust_config(force_reload: bool = False) -> TrustConfig:
    """
    Get the global trust configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        TrustConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = TrustConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[TRUST] Enabled with thresholds=[low={_config.low_trust_threshold}, "
                f"high={_config.high_trust_threshold}], "
                f"evidence_recency_days={_config.evidence_recency_days}"
            )
    
    return _config


def reset_trust_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None

