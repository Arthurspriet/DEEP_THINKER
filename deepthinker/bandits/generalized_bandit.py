"""
Generalized Bandit for DeepThinker.

Contextual multi-armed bandit with:
- Schema versioning for state validation
- Arms signature hash for schema migration detection
- Freeze mode for read-only operation
- min_trials_before_exploit for exploration

Supports UCB and Thompson Sampling algorithms.
"""

import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BanditConfig, get_bandit_config

logger = logging.getLogger(__name__)


@dataclass
class BanditSchema:
    """
    Versioned schema for bandit state validation.
    
    Used to detect schema changes and prevent loading incompatible state.
    
    Attributes:
        schema_version: Version string (e.g., "1.0.0")
        arms_signature_hash: Hash of sorted arm names
        decision_class: Name of the decision class
        algorithm: Selection algorithm ("ucb" or "thompson")
        created_at: When schema was created
    """
    schema_version: str = "1.0.0"
    arms_signature_hash: str = ""
    decision_class: str = ""
    algorithm: str = "ucb"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arms_signature_hash": self.arms_signature_hash,
            "decision_class": self.decision_class,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditSchema":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            arms_signature_hash=data.get("arms_signature_hash", ""),
            decision_class=data.get("decision_class", ""),
            algorithm=data.get("algorithm", "ucb"),
            created_at=created_at,
        )


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
        return max(0.01, (self.total_reward_sq / self.pulls) - (mean * mean))
    
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
    Complete state for a bandit.
    
    Attributes:
        schema: Schema information for validation
        arms: Dictionary of arm name -> BanditArm
        total_pulls: Total pulls across all arms
        updated_at: Last update time
    """
    schema: BanditSchema = field(default_factory=BanditSchema)
    arms: Dict[str, BanditArm] = field(default_factory=dict)
    total_pulls: int = 0
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
            "total_pulls": self.total_pulls,
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditState":
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.utcnow()
        
        schema = BanditSchema.from_dict(data.get("schema", {}))
        arms_data = data.get("arms", {})
        arms = {name: BanditArm.from_dict(arm) for name, arm in arms_data.items()}
        
        return cls(
            schema=schema,
            arms=arms,
            total_pulls=data.get("total_pulls", 0),
            updated_at=updated_at,
        )


class GeneralizedBandit:
    """
    Contextual bandit with versioning and freeze mode.
    
    Key features:
    - Schema versioning for state validation
    - Arms signature hash for schema migration detection
    - Freeze mode for read-only operation
    - min_trials_before_exploit for exploration phase
    
    Usage:
        bandit = GeneralizedBandit(
            decision_class="model_tier",
            arms=["SMALL", "MEDIUM", "LARGE"],
        )
        
        arm = bandit.select()  # Get recommended arm
        
        # After execution:
        updated = bandit.update(arm, reward=0.5)  # Returns False if frozen
    """
    
    def __init__(
        self,
        decision_class: str,
        arms: List[str],
        config: Optional[BanditConfig] = None,
        state_path: Optional[str] = None,
    ):
        """
        Initialize the bandit.
        
        Args:
            decision_class: Name of the decision class (e.g., "model_tier")
            arms: List of arm names
            config: Optional BanditConfig. Uses global if None.
            state_path: Optional path to state file. Auto-generated if None.
        """
        self.config = config or get_bandit_config()
        self.decision_class = decision_class
        self.arm_names = sorted(arms)  # Sort for consistent hashing
        
        # Compute state path
        if state_path:
            self.state_path = Path(state_path)
        else:
            self.state_path = Path(self.config.store_dir) / f"{decision_class}.json"
        
        # Initialize schema
        self.schema = BanditSchema(
            schema_version=self.config.schema_version,
            arms_signature_hash=self._compute_arms_hash(self.arm_names),
            decision_class=decision_class,
            algorithm=self.config.algorithm,
        )
        
        # Load or initialize state
        self._state: Optional[BanditState] = None
        self._load_state()
    
    @staticmethod
    def _compute_arms_hash(arms: List[str]) -> str:
        """Compute stable hash of arm names."""
        arms_str = ",".join(sorted(arms))
        return hashlib.sha256(arms_str.encode()).hexdigest()[:16]
    
    def _load_state(self) -> None:
        """Load state from file or initialize."""
        try:
            if self.state_path.exists():
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                
                loaded_state = BanditState.from_dict(data)
                
                # Validate schema
                if self.validate_schema(loaded_state):
                    self._state = loaded_state
                    logger.info(
                        f"[BANDIT:{self.decision_class}] Loaded state: "
                        f"{self._state.total_pulls} total pulls"
                    )
                    return
                else:
                    logger.warning(
                        f"[BANDIT:{self.decision_class}] Schema mismatch, "
                        f"reinitializing"
                    )
            
            self._initialize_state()
            
        except Exception as e:
            logger.warning(
                f"[BANDIT:{self.decision_class}] Failed to load state: {e}"
            )
            self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize fresh bandit state."""
        self._state = BanditState(
            schema=self.schema,
            arms={name: BanditArm(name=name) for name in self.arm_names},
        )
        logger.info(
            f"[BANDIT:{self.decision_class}] Initialized with arms: {self.arm_names}"
        )
    
    def _save_state(self) -> bool:
        """Save state to file."""
        if self._state is None:
            return False
        
        if self.config.freeze_mode:
            # Don't save in freeze mode
            return True
        
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.warning(
                f"[BANDIT:{self.decision_class}] Failed to save state: {e}"
            )
            return False
    
    def validate_schema(self, loaded_state: BanditState) -> bool:
        """
        Validate loaded state matches current schema.
        
        Args:
            loaded_state: State loaded from file
            
        Returns:
            True if schemas match
        """
        loaded_schema = loaded_state.schema
        
        if loaded_schema.schema_version != self.schema.schema_version:
            logger.warning(
                f"[BANDIT:{self.decision_class}] Schema version mismatch: "
                f"{loaded_schema.schema_version} != {self.schema.schema_version}"
            )
            return False
        
        if loaded_schema.arms_signature_hash != self.schema.arms_signature_hash:
            logger.warning(
                f"[BANDIT:{self.decision_class}] Arms signature mismatch: "
                f"{loaded_schema.arms_signature_hash} != {self.schema.arms_signature_hash}"
            )
            return False
        
        return True
    
    def select(self, context: Optional[Dict[str, float]] = None) -> str:
        """
        Select an arm using the configured algorithm.
        
        Args:
            context: Optional context features (for future contextual bandit)
            
        Returns:
            Selected arm name
        """
        if not self.config.enabled:
            # Return first arm when disabled
            return self.arm_names[0] if self.arm_names else ""
        
        if self._state is None:
            self._initialize_state()
        
        state = self._state
        
        # Forced exploration phase: min_trials_before_exploit per arm
        min_trials = self.config.min_trials_before_exploit
        if state.total_pulls < min_trials * len(state.arms):
            return self._explore_round_robin(state)
        
        # Exploitation phase
        if self.config.algorithm == "thompson":
            return self._thompson_select(state)
        else:
            return self._ucb_select(state)
    
    def _explore_round_robin(self, state: BanditState) -> str:
        """Round-robin exploration for under-explored arms."""
        for arm in state.arms.values():
            if arm.pulls < self.config.min_trials_before_exploit:
                logger.debug(
                    f"[BANDIT:{self.decision_class}] Exploring: {arm.name} "
                    f"(pulls={arm.pulls}/{self.config.min_trials_before_exploit})"
                )
                return arm.name
        
        # All arms have minimum trials, fall back to exploitation
        return self._ucb_select(state)
    
    def _ucb_select(self, state: BanditState) -> str:
        """Select using UCB1 algorithm."""
        best_arm = None
        best_score = float('-inf')
        
        for arm in state.arms.values():
            score = arm.ucb_score(
                state.total_pulls,
                self.config.exploration_bonus,
            )
            if score > best_score:
                best_score = score
                best_arm = arm
        
        selected = best_arm.name if best_arm else self.arm_names[0]
        logger.debug(
            f"[BANDIT:{self.decision_class}] UCB selected: {selected} "
            f"(score={best_score:.3f})"
        )
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
        
        selected = best_arm.name if best_arm else self.arm_names[0]
        logger.debug(f"[BANDIT:{self.decision_class}] Thompson selected: {selected}")
        return selected
    
    def update(self, arm: str, reward: float) -> bool:
        """
        Update bandit with observed reward.
        
        Args:
            arm: The arm that was pulled
            reward: Observed reward
            
        Returns:
            True if updated, False if frozen or error
        """
        if not self.config.enabled:
            return False
        
        if self.config.freeze_mode:
            logger.debug(
                f"[BANDIT:{self.decision_class}] Freeze mode: update rejected"
            )
            return False
        
        if self._state is None:
            self._initialize_state()
        
        state = self._state
        
        if arm not in state.arms:
            # Unknown arm - add it
            state.arms[arm] = BanditArm(name=arm)
        
        state.arms[arm].update(reward)
        state.total_pulls += 1
        state.updated_at = datetime.utcnow()
        
        logger.debug(
            f"[BANDIT:{self.decision_class}] Updated {arm}: "
            f"reward={reward:.3f}, "
            f"pulls={state.arms[arm].pulls}, "
            f"mean={state.arms[arm].mean_reward:.3f}"
        )
        
        # Persist state
        self._save_state()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        if self._state is None:
            return {
                "initialized": False,
                "decision_class": self.decision_class,
            }
        
        state = self._state
        return {
            "initialized": True,
            "decision_class": self.decision_class,
            "schema_version": state.schema.schema_version,
            "arms_signature_hash": state.schema.arms_signature_hash,
            "freeze_mode": self.config.freeze_mode,
            "total_pulls": state.total_pulls,
            "min_trials_before_exploit": self.config.min_trials_before_exploit,
            "exploration_complete": (
                state.total_pulls >= 
                self.config.min_trials_before_exploit * len(state.arms)
            ),
            "arms": {
                name: {
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "ucb_score": arm.ucb_score(
                        state.total_pulls,
                        self.config.exploration_bonus,
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
        logger.info(f"[BANDIT:{self.decision_class}] Reset bandit state")
    
    @property
    def is_frozen(self) -> bool:
        """Check if bandit is in freeze mode."""
        return self.config.freeze_mode
    
    @property
    def exploration_complete(self) -> bool:
        """Check if minimum exploration is complete."""
        if self._state is None:
            return False
        return (
            self._state.total_pulls >= 
            self.config.min_trials_before_exploit * len(self._state.arms)
        )

