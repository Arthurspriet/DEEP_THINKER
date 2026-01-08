"""
Cognitive Spine Core Module for DeepThinker 2.0.

The Cognitive Spine is the central unifying layer that enforces:
- Schema coherence across all councils and phases
- Predictable output structures via contracts
- Resource allocation discipline (tokens, depth, iterations)
- Phase boundary validation
- Consensus engine availability
- Memory compression between phases

DeepThinker 2.0 Additions:
- PhaseValidator: Enforces hard phase contracts
"""

from .cognitive_spine import (
    CognitiveSpine,
    ResourceBudget,
    ValidationResult,
    MemorySlot,
    SchemaVersion,
)

from .phase_validator import (
    PhaseValidator,
    ValidationResult as PhaseValidationResult,
    get_phase_validator,
)

__all__ = [
    "CognitiveSpine",
    "ResourceBudget",
    "ValidationResult",
    "MemorySlot",
    "SchemaVersion",
    # Phase validation (new in 2.0)
    "PhaseValidator",
    "PhaseValidationResult",
    "get_phase_validator",
]

