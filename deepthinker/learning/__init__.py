"""
Learning Module for DeepThinker.

Provides learned policy components:
- StopEscalatePredictor: Hybrid learned + rules policy
- PolicyState: State representation for prediction
- PolicyPrediction: Prediction output

Supports modes:
- OFF: Disabled entirely
- SHADOW: Log predictions, no behavior change
- ADVISORY: Log + surface to orchestrator
- ACTIVE: Predictions drive decisions (rules still safety floor)

All features are gated behind LearningConfig flags.
"""

from .config import (
    LearningConfig,
    LearnedPolicyMode,
    get_learning_config,
    reset_learning_config,
)
from .policy_features import PolicyState, get_policy_feature_names
from .stop_escalate_predictor import (
    StopEscalatePredictor,
    PolicyAction,
    PolicyPrediction,
    ShadowLogRecord,
    get_stop_escalate_predictor,
    reset_stop_escalate_predictor,
)

__all__ = [
    # Config
    "LearningConfig",
    "LearnedPolicyMode",
    "get_learning_config",
    "reset_learning_config",
    # Policy Features
    "PolicyState",
    "get_policy_feature_names",
    # Stop/Escalate Predictor
    "StopEscalatePredictor",
    "PolicyAction",
    "PolicyPrediction",
    "ShadowLogRecord",
    "get_stop_escalate_predictor",
    "reset_stop_escalate_predictor",
]

