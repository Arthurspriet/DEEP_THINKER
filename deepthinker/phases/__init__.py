"""
Phases Module for DeepThinker.

Provides phase contract enforcement and purity validation.
Ensures outputs conform to phase expectations and prevents
phase contamination (e.g., recommendations during reconnaissance).

Components:
- PhaseContracts: Defines allowed/forbidden outputs per phase
- PhaseGuard: Validates outputs against phase contracts
- ContaminationDetector: Detects out-of-phase content
"""

from .phase_contracts import (
    ContaminationType,
    PhaseContract,
    PhaseContamination,
    PhaseGuard,
    PhaseViolation,
    PHASE_CONTRACTS,
    get_phase_guard,
)
from .recon_phases import (
    ReconPhaseType,
    ReconExplorationOutput,
    ReconGroundingOutput,
    SplitReconConfig,
    create_exploration_phase_from_recon,
    create_grounding_phase_from_recon,
    should_split_recon_phase,
)

__all__ = [
    "ContaminationType",
    "PhaseContract",
    "PhaseContamination",
    "PhaseGuard",
    "PhaseViolation",
    "PHASE_CONTRACTS",
    "get_phase_guard",
    # Split Recon phases (Epistemic Hardening)
    "ReconPhaseType",
    "ReconExplorationOutput",
    "ReconGroundingOutput",
    "SplitReconConfig",
    "create_exploration_phase_from_recon",
    "create_grounding_phase_from_recon",
    "should_split_recon_phase",
]

