"""
Trust Module for DeepThinker.

Provides trust and confidence metrics:
- TrustScore: Aggregate trust score with explanation
- TrustCalculator: Computes trust from multiple signals
- TrustExplanation: Human-readable trust explanation

Incorporates signals from:
- JudgeEnsemble (disagreement)
- ClaimGraph (contradictions)
- EvidenceObject (recency, diversity)
- Memory injection logs
- Tool usage

All features are gated behind TrustConfig flags.
"""

from .config import TrustConfig, get_trust_config, reset_trust_config
from .trust_metrics import (
    TrustScore,
    TrustExplanation,
    TrustCalculator,
    get_trust_calculator,
    reset_trust_calculator,
)

__all__ = [
    # Config
    "TrustConfig",
    "get_trust_config",
    "reset_trust_config",
    # Trust Metrics
    "TrustScore",
    "TrustExplanation",
    "TrustCalculator",
    "get_trust_calculator",
    "reset_trust_calculator",
]

