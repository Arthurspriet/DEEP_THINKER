"""
Bio Priors Package for DeepThinker.

Provides biological strategy patterns as soft background priors that emit
bounded pressure signals to modulate reasoning dynamics.

This system is:
- PURE: Engine has no side effects
- DETERMINISTIC: Same input always produces same output
- DEFAULT-OFF: Disabled by default, backward compatible
- NON-AUTHORITATIVE: Priors never assert facts, only modulate

Modes:
- off: No evaluation
- advisory: Log only, no behavioral changes
- shadow: Log + compute "would_apply" diff, no behavioral changes
- soft: Apply bounded modifications (v1: depth_budget_delta only)
"""

from .config import BioPriorConfig, get_bio_prior_config
from .signals import PressureSignals
from .schema import BioPattern
from .metrics import BioPriorContext, build_context, RECENT_WINDOW_STEPS
from .engine import BioPriorEngine, BioPriorOutput
from .loader import load_patterns, validate_pattern
from .integration import apply_bio_pressures_to_deepening_plan

__all__ = [
    # Config
    "BioPriorConfig",
    "get_bio_prior_config",
    # Core types
    "PressureSignals",
    "BioPattern",
    "BioPriorContext",
    "BioPriorOutput",
    # Functions
    "build_context",
    "load_patterns",
    "validate_pattern",
    "apply_bio_pressures_to_deepening_plan",
    # Constants
    "RECENT_WINDOW_STEPS",
    # Engine
    "BioPriorEngine",
]

