"""
Routing Module for DeepThinker.

Provides ML-based routing decisions and bandit optimization
for orchestration:
- MLRouterAdvisor: Supervised router for council/model/rounds selection
- ModelTierBandit: UCB/Thompson bandit for model tier choice
- Feature extraction for routing decisions

All features are gated behind config flags (defaults OFF).
"""

from .ml_router import (
    MLRouterAdvisor,
    RoutingDecision,
    RoutingFeatures,
    get_ml_router,
)
from .bandit import (
    ModelTierBandit,
    BanditArm,
    BanditState,
    get_model_tier_bandit,
)
from .features import (
    extract_routing_features,
    RoutingContext,
)

__all__ = [
    # Router
    "MLRouterAdvisor",
    "RoutingDecision",
    "RoutingFeatures",
    "get_ml_router",
    # Bandit
    "ModelTierBandit",
    "BanditArm",
    "BanditState",
    "get_model_tier_bandit",
    # Features
    "extract_routing_features",
    "RoutingContext",
]

