"""
Mission Engine for DeepThinker 2.0.

Provides support for long-running, time-bounded autonomous missions
using the council architecture.

Enhanced with effort-based execution:
- EffortLevel determines depth of analysis based on time budget
- Multi-round phase and council execution
- Quality-based early stopping
"""

from .mission_types import (
    MissionConstraints,
    MissionPhase,
    MissionState,
    EffortLevel,
    EFFORT_PRESETS,
    infer_effort_level,
    build_constraints_from_time_budget,
)
from .mission_store import MissionStore
from .mission_orchestrator import MissionOrchestrator
from .mission_time_manager import (
    MissionTimeManager,
    PhaseTimeRecord,
    create_time_manager_from_constraints,
)
from .evidence_recovery import (
    EvidenceRecoveryState,
    RecoveryEntryReason,
    check_evidence_recovery_entry,
)
from .artifact_firewall import (
    ArtifactFirewall,
    FilteredArtifact,
    PromotionResult,
    create_sanitized_context,
)

__all__ = [
    "MissionConstraints",
    "MissionPhase", 
    "MissionState",
    "MissionStore",
    "MissionOrchestrator",
    "EffortLevel",
    "EFFORT_PRESETS",
    "infer_effort_level",
    "build_constraints_from_time_budget",
    "MissionTimeManager",
    "PhaseTimeRecord",
    "create_time_manager_from_constraints",
    # Epistemic Hardening
    "EvidenceRecoveryState",
    "RecoveryEntryReason",
    "check_evidence_recovery_entry",
    # Artifact Firewall (Phase 6)
    "ArtifactFirewall",
    "FilteredArtifact",
    "PromotionResult",
    "create_sanitized_context",
]

