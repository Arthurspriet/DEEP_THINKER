"""
Cognitive Constitution Module for DeepThinker.

Implements a control-system layer enforcing four invariants:
1. Conservation of Evidence - No confidence increase without new evidence
2. Monotonic Uncertainty Under Compression - Compression cannot reduce uncertainty
3. No-Free-Lunch Depth - Deeper rounds must produce measurable gain
4. Anti-Gaming Divergence (Goodhart Shield) - Target vs shadow metric divergence detection

Plus:
5. Blinded Evaluation - Judges cannot see routing/model identifiers

All features are flag-gated (default OFF) and designed for cheap local models.

North Star Principle:
"No cognition without a ledger. No confidence without evidence. No learning without blinding."
"""

from .config import (
    ConstitutionConfig,
    ConstitutionMode,
    get_constitution_config,
    reset_constitution_config,
)
from .types import (
    EvidenceEvent,
    ScoreEvent,
    ContradictionEvent,
    DepthEvent,
    MemoryEvent,
    CompressionEvent,
    LearningUpdateEvent,
    ConstitutionViolationEvent,
    ConstitutionEventType,
)
from .constitution_spec import (
    ConstitutionSpec,
    InvariantType,
    InvariantSpec,
)
from .enforcement import (
    ConstitutionFlags,
    EnforcementAction,
)

__all__ = [
    # Config
    "ConstitutionConfig",
    "ConstitutionMode",
    "get_constitution_config",
    "reset_constitution_config",
    # Event types
    "EvidenceEvent",
    "ScoreEvent",
    "ContradictionEvent",
    "DepthEvent",
    "MemoryEvent",
    "CompressionEvent",
    "LearningUpdateEvent",
    "ConstitutionViolationEvent",
    "ConstitutionEventType",
    # Spec
    "ConstitutionSpec",
    "InvariantType",
    "InvariantSpec",
    # Enforcement
    "ConstitutionFlags",
    "EnforcementAction",
]

