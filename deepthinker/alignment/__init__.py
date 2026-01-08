"""
Alignment Control Layer for DeepThinker.

Provides hybrid alignment control to detect and correct goal drift
in long-horizon missions. Uses embedding-based time series analysis
combined with LLM evaluation and soft corrective actions.

Main entry point: run_alignment_check() from integration module.

Components:
- models: Data structures (NorthStarGoal, AlignmentPoint, etc.)
- config: Configuration management
- drift: Embedding-based drift detection
- evaluator: LLM-based alignment evaluation
- controller: Soft action controller
- integration: Orchestrator hook and action application
- persist: JSON logging and persistence
"""

from .models import (
    AlignmentAction,
    AlignmentAssessment,
    AlignmentPoint,
    AlignmentTrajectory,
    ControllerState,
    NorthStarGoal,
    UserDriftEvent,
)

from .config import (
    AlignmentConfig,
    get_alignment_config,
)

from .integration import (
    run_alignment_check,
    finalize_alignment,
    cleanup_alignment,
    get_alignment_manager,
    AlignmentManager,
)

__all__ = [
    # Models
    "AlignmentAction",
    "AlignmentAssessment",
    "AlignmentPoint",
    "AlignmentTrajectory",
    "ControllerState",
    "NorthStarGoal",
    "UserDriftEvent",
    # Config
    "AlignmentConfig",
    "get_alignment_config",
    # Integration
    "run_alignment_check",
    "finalize_alignment",
    "cleanup_alignment",
    "get_alignment_manager",
    "AlignmentManager",
]

