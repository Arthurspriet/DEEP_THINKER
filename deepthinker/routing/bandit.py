"""
Model Tier Bandit for DeepThinker.

UCB/Thompson Sampling bandit for model tier selection.

MVP: ONE decision only - model tier choice for council/scout escalation.
Tool-choice bandit deferred to future.

Reward = score_delta - lambda * cost_delta

Bandit state persisted per mission (or global) under kb/bandits/.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..metrics.config import MetricsConfig, get_metrics_config

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    """
    State for a single bandit arm.
    
    Attributes:
        name: Arm name (e.g., "SMALL", "MEDIUM", "LARGE")
        pulls: Number of times arm was pulled
        total_reward: Sum of rewards received
        total_reward_sq: Sum of squared rewards (for variance)
        last_pull: Timestamp of last pull
    """
    name: str
    pulls: int = 0
    total_reward: float = 0.0
    total_reward_sq: float = 0.0
    last_pull: Optional[datetime] = None
    
    @property
    def mean_reward(self) -> float:
        """Average reward for this arm."""
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls
    
    @property
    def variance(self) -> float:
        """Variance of rewards (for Thompson sampling)."""
        if self.pulls < 2:
            return 1.0  # High uncertainty
        mean = self.mean_reward
        return (self.total_reward_sq / self.pulls) - (mean * mean)
    
    def ucb_score(self, total_pulls: int, exploration_bonus: float = 1.0) -> float:
        """
        Compute UCB1 score for this arm.
        
        UCB = mean + c * sqrt(log(n) / n_i)
        
        Args:
            total_pulls: Total pulls across all arms
            exploration_bonus: Exploration coefficient (c)
            
        Returns:
            UCB score
        """
        if self.pulls == 0:
            return float('inf')  # Unexplored arms get priority
        
        exploitation = self.mean_reward
        exploration = exploration_bonus * math.sqrt(
            math.log(total_pulls + 1) / self.pulls
        )
        return exploitation + exploration
    
    def thompson_sample(self) -> float:
        """
        Sample from posterior (Thompson Sampling).
        
        Uses Normal approximation for simplicity.
        
        Returns:
            Sampled value from posterior
        """
        if self.pulls == 0:
            # Prior: N(0, 1)
            return random.gauss(0, 1)
        
        # Posterior: N(mean, variance/n)
        mean = self.mean_reward
        std = math.sqrt(self.variance / max(self.pulls, 1))
        return random.gauss(mean, std)
    
    def update(self, reward: float) -> None:
        """
        Update arm with new reward.
        
        Args:
            reward: Observed reward
        """
        self.pulls += 1
        self.total_reward += reward
        self.total_reward_sq += reward * reward
        self.last_pull = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pulls": self.pulls,
            "total_reward": self.total_reward,
            "total_reward_sq": self.total_reward_sq,
            "last_pull": self.last_pull.isoformat() if self.last_pull else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditArm":
        last_pull = data.get("last_pull")
        if isinstance(last_pull, str):
            last_pull = datetime.fromisoformat(last_pull)
        
        return cls(
            name=data.get("name", ""),
            pulls=data.get("pulls", 0),
            total_reward=data.get("total_reward", 0.0),
            total_reward_sq=data.get("total_reward_sq", 0.0),
            last_pull=last_pull,
        )


@dataclass
class BanditState:
    """
    State for the entire bandit.
    
    Attributes:
        arms: Dictionary of arm name -> BanditArm
        total_pulls: Total pulls across all arms
        created_at: When bandit was created
        updated_at: Last update time
    """
    arms: Dict[str, BanditArm] = field(default_factory=dict)
    total_pulls: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
            "total_pulls": self.total_pulls,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditState":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.utcnow()
        
        arms_data = data.get("arms", {})
        arms = {name: BanditArm.from_dict(arm) for name, arm in arms_data.items()}
        
        return cls(
            arms=arms,
            total_pulls=data.get("total_pulls", 0),
            created_at=created_at,
            updated_at=updated_at,
        )


class ModelTierBandit:
    """
    UCB/Thompson Sampling bandit for model tier selection.
    
    Chooses between model tiers (SMALL, MEDIUM, LARGE) to
    maximize reward = score_delta - lambda * cost_delta.
    
    Usage:
        bandit = ModelTierBandit()
        tier = bandit.select()  # Get recommended tier
        
        # After phase execution:
        bandit.update(tier, score_delta=0.1, cost_delta=0.05)
    """
    
    # Default tiers
    DEFAULT_TIERS = ["SMALL", "MEDIUM", "LARGE"]
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        state_path: Optional[str] = None,
        algorithm: str = "ucb",
    ):
        """
        Initialize the bandit.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
            state_path: Optional path to state JSON file.
            algorithm: "ucb" or "thompson"
        """
        self.config = config or get_metrics_config()
        self.state_path = state_path or "kb/bandits/model_tier_bandit.json"
        self.algorithm = algorithm
        self._state: Optional[BanditState] = None
        self._load_state()
    
    def _load_state(self) -> None:
        """Load state from file or initialize."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                self._state = BanditState.from_dict(data)
                logger.info(
                    f"[BANDIT] Loaded state: {self._state.total_pulls} total pulls"
                )
            else:
                self._initialize_state()
        except Exception as e:
            logger.warning(f"[BANDIT] Failed to load state: {e}")
            self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize fresh bandit state."""
        self._state = BanditState(
            arms={
                tier: BanditArm(name=tier)
                for tier in self.DEFAULT_TIERS
            }
        )
        logger.info("[BANDIT] Initialized fresh bandit state")
    
    def _save_state(self) -> None:
        """Save state to file."""
        if self._state is None:
            return
        
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"[BANDIT] Failed to save state: {e}")
    
    def select(self, context_features: Optional[Dict[str, float]] = None) -> str:
        """
        Select a model tier using the bandit algorithm.
        
        Args:
            context_features: Optional context features (for contextual bandit)
            
        Returns:
            Selected tier name
        """
        if not self.config.bandit_enabled:
            return "MEDIUM"  # Default when disabled
        
        if self._state is None:
            self._initialize_state()
        
        state = self._state
        
        # Exploration phase: try all arms first
        if state.total_pulls < self.config.bandit_min_observations * len(state.arms):
            # Round-robin through arms
            for arm in state.arms.values():
                if arm.pulls < self.config.bandit_min_observations:
                    logger.debug(f"[BANDIT] Exploring arm: {arm.name}")
                    return arm.name
        
        # Exploitation phase
        if self.algorithm == "thompson":
            return self._thompson_select(state)
        else:
            return self._ucb_select(state)
    
    def _ucb_select(self, state: BanditState) -> str:
        """Select using UCB1 algorithm."""
        best_arm = None
        best_score = float('-inf')
        
        for arm in state.arms.values():
            score = arm.ucb_score(
                state.total_pulls,
                self.config.bandit_exploration_bonus,
            )
            if score > best_score:
                best_score = score
                best_arm = arm
        
        selected = best_arm.name if best_arm else "MEDIUM"
        logger.debug(f"[BANDIT] UCB selected: {selected} (score={best_score:.3f})")
        return selected
    
    def _thompson_select(self, state: BanditState) -> str:
        """Select using Thompson Sampling."""
        best_arm = None
        best_sample = float('-inf')
        
        for arm in state.arms.values():
            sample = arm.thompson_sample()
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        
        selected = best_arm.name if best_arm else "MEDIUM"
        logger.debug(f"[BANDIT] Thompson selected: {selected}")
        return selected
    
    def update(
        self,
        tier: str,
        score_delta: float,
        cost_delta: float,
    ) -> float:
        """
        Update bandit with observed outcome.
        
        Args:
            tier: The tier that was used
            score_delta: Score improvement achieved
            cost_delta: Cost incurred
            
        Returns:
            Computed reward
        """
        if not self.config.bandit_enabled:
            return 0.0
        
        if self._state is None:
            self._initialize_state()
        
        state = self._state
        
        # Compute reward
        reward = score_delta - self.config.bandit_lambda * cost_delta
        
        # Update arm
        if tier in state.arms:
            state.arms[tier].update(reward)
        else:
            # Unknown tier - add it
            arm = BanditArm(name=tier)
            arm.update(reward)
            state.arms[tier] = arm
        
        state.total_pulls += 1
        state.updated_at = datetime.utcnow()
        
        logger.debug(
            f"[BANDIT] Updated {tier}: reward={reward:.3f}, "
            f"pulls={state.arms[tier].pulls}, "
            f"mean={state.arms[tier].mean_reward:.3f}"
        )
        
        # Persist state
        self._save_state()
        
        return reward
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        if self._state is None:
            return {"initialized": False}
        
        state = self._state
        return {
            "initialized": True,
            "total_pulls": state.total_pulls,
            "arms": {
                name: {
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "ucb_score": arm.ucb_score(
                        state.total_pulls,
                        self.config.bandit_exploration_bonus,
                    ),
                }
                for name, arm in state.arms.items()
            },
            "best_arm": max(
                state.arms.values(),
                key=lambda a: a.mean_reward,
            ).name if state.arms else None,
        }
    
    def reset(self) -> None:
        """Reset bandit state."""
        self._initialize_state()
        self._save_state()
        logger.info("[BANDIT] Reset bandit state")


# Global bandit instance
_bandit: Optional[ModelTierBandit] = None


def get_model_tier_bandit(config: Optional[MetricsConfig] = None) -> ModelTierBandit:
    """Get global model tier bandit instance."""
    global _bandit
    if _bandit is None:
        _bandit = ModelTierBandit(config=config)
    return _bandit

