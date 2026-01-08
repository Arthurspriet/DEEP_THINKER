"""
Replay Module for DeepThinker.

Provides counterfactual replay capabilities:
- CounterfactualReplayEngine: Offline replay with different policies
- ReplayResult: Structured replay outcome
- ReplayMode: Replay fidelity modes

Modes:
- DECISIONS_ONLY: No model calls, replay routing/bandit only (default)
- WITH_JUDGES: Re-run judge scoring (expensive)
- FULL: Full re-execution (very expensive, not implemented)

All features are gated behind ReplayConfig flags.
"""

from .config import ReplayConfig, ReplayMode, get_replay_config, reset_replay_config
from .counterfactual_engine import (
    CounterfactualReplayEngine,
    ReplayResult,
    DecisionReplay,
    get_counterfactual_engine,
    reset_counterfactual_engine,
)

__all__ = [
    # Config
    "ReplayConfig",
    "ReplayMode",
    "get_replay_config",
    "reset_replay_config",
    # Engine
    "CounterfactualReplayEngine",
    "ReplayResult",
    "DecisionReplay",
    "get_counterfactual_engine",
    "reset_counterfactual_engine",
]

