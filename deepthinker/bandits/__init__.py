"""
Bandits Module for DeepThinker.

Provides generalized multi-armed bandit implementations:
- GeneralizedBandit: Versioned, contextual bandit with freeze mode
- BanditRegistry: Central registry for all decision classes
- BanditSchema: Schema versioning for state validation

Supports decision classes:
- model_tier: SMALL, MEDIUM, LARGE
- council_set: standard, deep, fast, research
- tool_selection: web_search, code_exec, rag_query, ...
- escalation_depth: 1_round, 2_rounds, 3_rounds

All features are gated behind BanditConfig flags.
"""

from .config import BanditConfig, get_bandit_config, reset_bandit_config
from .generalized_bandit import (
    GeneralizedBandit,
    BanditSchema,
    BanditArm,
    BanditState,
)
from .contextual_features import BanditContext, get_feature_names
from .bandit_registry import (
    BanditRegistry,
    get_bandit_registry,
    reset_bandit_registry,
    get_model_tier_bandit,
    get_council_set_bandit,
    get_tool_selection_bandit,
    get_escalation_depth_bandit,
    DECISION_CLASSES,
)

__all__ = [
    # Config
    "BanditConfig",
    "get_bandit_config",
    "reset_bandit_config",
    # Generalized Bandit
    "GeneralizedBandit",
    "BanditSchema",
    "BanditArm",
    "BanditState",
    # Contextual Features
    "BanditContext",
    "get_feature_names",
    # Registry
    "BanditRegistry",
    "get_bandit_registry",
    "reset_bandit_registry",
    "get_model_tier_bandit",
    "get_council_set_bandit",
    "get_tool_selection_bandit",
    "get_escalation_depth_bandit",
    "DECISION_CLASSES",
]

