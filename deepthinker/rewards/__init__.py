"""
Rewards Module for DeepThinker.

Provides unified reward signals for learning components:
- RewardSignal: Versioned, deterministic reward computation
- RewardWeights: Configurable component weights
- RewardStore: Persistence for reward history

All features are gated behind RewardConfig flags.
"""

from .config import RewardConfig, get_reward_config, reset_reward_config
from .reward_weights import RewardWeights
from .reward_signal import RewardSignal
from .reward_store import RewardStore, get_reward_store

__all__ = [
    "RewardConfig",
    "get_reward_config",
    "reset_reward_config",
    "RewardWeights",
    "RewardSignal",
    "RewardStore",
    "get_reward_store",
]


