"""
Metrics Module for DeepThinker.

Provides mission-level quality scoring, tool effectiveness tracking,
and judge ensemble evaluation for data-driven orchestration decisions.

All features are gated behind config flags (defaults OFF).

Components:
- MetricsConfig: Centralized configuration with env overrides
- Scorecard: Structured quality scores for phases/missions
- JudgeEnsemble: Multi-model quality evaluation with disagreement tracking
- ToolTracker: Per-step tool usage recording with heuristic attribution
- EvidenceObject: Standardized schema for tool outputs
"""

from .config import (
    MetricsConfig,
    get_metrics_config,
    reset_metrics_config,
    should_sample,
)
from .scorecard import (
    Scorecard,
    ScorecardCost,
    ScorecardRuntime,
    ScorecardMetadata,
)
from .judge_ensemble import (
    JudgeEnsemble,
    JudgeResult,
    JudgeScores,
)
from .tool_tracker import (
    ToolTracker,
    ToolUsageRecord,
    tracked_tool,
    get_tool_tracker,
)
from .evidence_object import (
    EvidenceObject,
    EvidenceType,
)
from .orchestrator_hooks import (
    MetricsOrchestrationHook,
    PhaseMetricsContext,
    get_metrics_hook,
)

__all__ = [
    # Config
    "MetricsConfig",
    "get_metrics_config",
    "reset_metrics_config",
    "should_sample",
    # Scorecard
    "Scorecard",
    "ScorecardCost",
    "ScorecardRuntime",
    "ScorecardMetadata",
    # Judge
    "JudgeEnsemble",
    "JudgeResult",
    "JudgeScores",
    # Tool Tracking
    "ToolTracker",
    "ToolUsageRecord",
    "tracked_tool",
    "get_tool_tracker",
    # Evidence
    "EvidenceObject",
    "EvidenceType",
    # Orchestrator Integration
    "MetricsOrchestrationHook",
    "PhaseMetricsContext",
    "get_metrics_hook",
]

