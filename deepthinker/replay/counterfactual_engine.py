"""
Counterfactual Replay Engine for DeepThinker.

Enables offline replay of missions with different policies:
- Re-run routing/bandit decisions with new policies
- Compare old vs new scorecards
- Estimate regret

Modes:
- DECISIONS_ONLY: No model calls, replay routing/bandit only (default)
- WITH_JUDGES: Re-run judge scoring (expensive)
- FULL: Full re-execution (very expensive, not implemented)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .config import ReplayConfig, ReplayMode, get_replay_config

logger = logging.getLogger(__name__)


class Policy(Protocol):
    """Protocol for policies that can be replayed."""
    
    def select(self, context: Dict[str, Any]) -> str:
        """Select an action given context."""
        ...


@dataclass
class DecisionReplay:
    """
    Result of replaying a single decision.
    
    Attributes:
        decision_id: Original decision ID
        decision_type: Type of decision
        original_action: What was originally chosen
        counterfactual_action: What would be chosen now
        original_reward: Original reward (if logged)
        counterfactual_reward: Estimated counterfactual reward
        action_changed: Whether action differs
        context: Decision context
    """
    decision_id: str
    decision_type: str
    original_action: str
    counterfactual_action: str
    original_reward: float
    counterfactual_reward: float
    action_changed: bool
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "original_action": self.original_action,
            "counterfactual_action": self.counterfactual_action,
            "original_reward": self.original_reward,
            "counterfactual_reward": self.counterfactual_reward,
            "action_changed": self.action_changed,
            "context": self.context,
        }


@dataclass
class ReplayResult:
    """
    Result of replaying a mission.
    
    Attributes:
        mission_id: Mission identifier
        replay_mode: Replay fidelity mode
        replayed_at: When replay was run
        
        # Decision comparison
        total_decisions: Total decisions replayed
        decisions_changed: Decisions where action differs
        decision_replays: Individual decision replays
        
        # Reward comparison
        original_total_reward: Sum of original rewards
        counterfactual_total_reward: Sum of counterfactual rewards
        
        # Regret estimation
        estimated_regret: Positive regret (improvement potential)
        regret_by_type: Regret broken down by decision type
        
        # Policy info
        policy_name: Name of counterfactual policy
        policy_config: Policy configuration
    """
    mission_id: str
    replay_mode: ReplayMode
    replayed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Decision comparison
    total_decisions: int = 0
    decisions_changed: int = 0
    decision_replays: List[DecisionReplay] = field(default_factory=list)
    
    # Reward comparison
    original_total_reward: float = 0.0
    counterfactual_total_reward: float = 0.0
    
    # Regret estimation
    estimated_regret: float = 0.0
    regret_by_type: Dict[str, float] = field(default_factory=dict)
    
    # Policy info
    policy_name: str = ""
    policy_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "replay_mode": self.replay_mode.value,
            "replayed_at": self.replayed_at.isoformat(),
            "total_decisions": self.total_decisions,
            "decisions_changed": self.decisions_changed,
            "decision_replays": [d.to_dict() for d in self.decision_replays],
            "original_total_reward": self.original_total_reward,
            "counterfactual_total_reward": self.counterfactual_total_reward,
            "estimated_regret": self.estimated_regret,
            "regret_by_type": self.regret_by_type,
            "policy_name": self.policy_name,
            "policy_config": self.policy_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayResult":
        replayed_at = data.get("replayed_at")
        if isinstance(replayed_at, str):
            replayed_at = datetime.fromisoformat(replayed_at)
        elif replayed_at is None:
            replayed_at = datetime.utcnow()
        
        return cls(
            mission_id=data.get("mission_id", ""),
            replay_mode=ReplayMode(data.get("replay_mode", "decisions_only")),
            replayed_at=replayed_at,
            total_decisions=data.get("total_decisions", 0),
            decisions_changed=data.get("decisions_changed", 0),
            decision_replays=[],  # Don't deserialize full replays
            original_total_reward=data.get("original_total_reward", 0.0),
            counterfactual_total_reward=data.get("counterfactual_total_reward", 0.0),
            estimated_regret=data.get("estimated_regret", 0.0),
            regret_by_type=data.get("regret_by_type", {}),
            policy_name=data.get("policy_name", ""),
            policy_config=data.get("policy_config", {}),
        )


class CounterfactualReplayEngine:
    """
    Offline replay of missions with different policies.
    
    REPLAY_MODE determines what is re-computed:
    - decisions_only: Re-run routing/bandit with new policy, use logged rewards
    - with_judges: Re-score outputs with current judges
    - full: Full re-execution (requires model access, not implemented)
    
    Usage:
        engine = CounterfactualReplayEngine()
        
        # Replay with new bandit policy
        result = engine.replay_mission(
            mission_id="abc-123",
            new_policy=new_bandit,
        )
        
        print(f"Estimated regret: {result.estimated_regret}")
    """
    
    def __init__(self, config: Optional[ReplayConfig] = None):
        """
        Initialize the engine.
        
        Args:
            config: Optional ReplayConfig
        """
        self.config = config or get_replay_config()
        self.output_path = Path(self.config.output_path)
    
    def replay_mission(
        self,
        mission_id: str,
        new_policy: Policy,
        policy_name: str = "new_policy",
        policy_config: Optional[Dict[str, Any]] = None,
        decision_store_path: Optional[str] = None,
        reward_store_path: Optional[str] = None,
    ) -> ReplayResult:
        """
        Replay a mission with a new policy.
        
        Args:
            mission_id: Mission to replay
            new_policy: Policy to use for counterfactual
            policy_name: Name for the policy
            policy_config: Policy configuration
            decision_store_path: Path to decision store
            reward_store_path: Path to reward store
            
        Returns:
            ReplayResult with comparison
        """
        if not self.config.enabled:
            return ReplayResult(
                mission_id=mission_id,
                replay_mode=self.config.mode,
            )
        
        # Load historical decisions
        decisions = self._load_decisions(mission_id, decision_store_path)
        
        if not decisions:
            logger.warning(f"[REPLAY] No decisions found for {mission_id}")
            return ReplayResult(
                mission_id=mission_id,
                replay_mode=self.config.mode,
            )
        
        # Limit decisions if configured
        if len(decisions) > self.config.max_decisions_per_replay:
            decisions = decisions[:self.config.max_decisions_per_replay]
        
        # Replay based on mode
        if self.config.mode == ReplayMode.DECISIONS_ONLY:
            result = self._replay_decisions_only(
                mission_id, decisions, new_policy, policy_name, policy_config
            )
        elif self.config.mode == ReplayMode.WITH_JUDGES:
            result = self._replay_with_judges(
                mission_id, decisions, new_policy, policy_name, policy_config
            )
        else:
            raise NotImplementedError(
                f"FULL replay mode requires live execution, not implemented"
            )
        
        # Compute regret if enabled
        if self.config.include_regret_analysis:
            result.estimated_regret = self._estimate_regret(
                result.original_total_reward,
                result.counterfactual_total_reward,
            )
            result.regret_by_type = self._regret_by_type(result.decision_replays)
        
        # Store result
        self._store_result(result)
        
        return result
    
    def _load_decisions(
        self,
        mission_id: str,
        decision_store_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load decisions from store."""
        try:
            # Try to load from decision store
            if decision_store_path:
                store_path = Path(decision_store_path)
            else:
                store_path = Path(f"kb/missions/{mission_id}/decisions.jsonl")
            
            if not store_path.exists():
                return []
            
            decisions = []
            with open(store_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            decisions.append(json.loads(line))
                        except Exception:
                            continue
            
            return decisions
            
        except Exception as e:
            logger.warning(f"[REPLAY] Failed to load decisions: {e}")
            return []
    
    def _replay_decisions_only(
        self,
        mission_id: str,
        decisions: List[Dict[str, Any]],
        new_policy: Policy,
        policy_name: str,
        policy_config: Optional[Dict[str, Any]],
    ) -> ReplayResult:
        """
        Replay with decisions_only mode (no model calls).
        
        Uses logged rewards, only re-runs selection.
        """
        replays = []
        original_total = 0.0
        counterfactual_total = 0.0
        decisions_changed = 0
        
        for decision in decisions:
            # Extract context for policy
            context = self._extract_context(decision)
            
            # Get original action
            original_action = decision.get("selected_option", "")
            
            # Get original reward (from logged data or estimate)
            original_reward = self._get_decision_reward(decision)
            original_total += original_reward
            
            # Apply new policy
            try:
                counterfactual_action = new_policy.select(context)
            except Exception:
                counterfactual_action = original_action
            
            # Estimate counterfactual reward
            # In decisions_only mode, we can only estimate based on arm statistics
            counterfactual_reward = self._estimate_counterfactual_reward(
                decision, counterfactual_action
            )
            counterfactual_total += counterfactual_reward
            
            action_changed = original_action != counterfactual_action
            if action_changed:
                decisions_changed += 1
            
            replay = DecisionReplay(
                decision_id=decision.get("decision_id", ""),
                decision_type=decision.get("decision_type", ""),
                original_action=original_action,
                counterfactual_action=counterfactual_action,
                original_reward=original_reward,
                counterfactual_reward=counterfactual_reward,
                action_changed=action_changed,
                context=context,
            )
            replays.append(replay)
        
        return ReplayResult(
            mission_id=mission_id,
            replay_mode=ReplayMode.DECISIONS_ONLY,
            total_decisions=len(decisions),
            decisions_changed=decisions_changed,
            decision_replays=replays,
            original_total_reward=original_total,
            counterfactual_total_reward=counterfactual_total,
            policy_name=policy_name,
            policy_config=policy_config or {},
        )
    
    def _replay_with_judges(
        self,
        mission_id: str,
        decisions: List[Dict[str, Any]],
        new_policy: Policy,
        policy_name: str,
        policy_config: Optional[Dict[str, Any]],
    ) -> ReplayResult:
        """
        Replay with judge re-scoring (expensive).
        
        Note: This is a stub. Full implementation would call judges.
        """
        # For now, fall back to decisions_only
        logger.warning(
            "[REPLAY] with_judges mode not fully implemented, "
            "falling back to decisions_only"
        )
        
        result = self._replay_decisions_only(
            mission_id, decisions, new_policy, policy_name, policy_config
        )
        result.replay_mode = ReplayMode.WITH_JUDGES
        
        return result
    
    def _extract_context(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Extract policy context from decision."""
        constraints = decision.get("constraints_snapshot", {})
        
        return {
            "decision_type": decision.get("decision_type", ""),
            "phase_type": decision.get("phase_type", ""),
            "time_remaining": constraints.get("time_remaining_minutes", 60.0),
            "importance": constraints.get("importance", 0.5),
            "retry_count": constraints.get("retry_count", 0),
            "quality_score": constraints.get("quality_score"),
            "options": decision.get("options_considered", []),
        }
    
    def _get_decision_reward(self, decision: Dict[str, Any]) -> float:
        """Get reward for a decision (from logged data or estimate)."""
        # Check for explicit reward
        if "reward" in decision:
            return decision["reward"]
        
        # Estimate from confidence and outcome
        confidence = decision.get("confidence", 0.5)
        constraints = decision.get("constraints_snapshot", {})
        quality = constraints.get("quality_score", 0.5)
        
        if quality is not None:
            return confidence * quality
        
        return confidence * 0.5
    
    def _estimate_counterfactual_reward(
        self,
        decision: Dict[str, Any],
        counterfactual_action: str,
    ) -> float:
        """
        Estimate counterfactual reward for different action.
        
        In decisions_only mode, we use heuristics:
        - Same action -> same reward
        - Different action -> interpolate based on available data
        """
        original_action = decision.get("selected_option", "")
        original_reward = self._get_decision_reward(decision)
        
        if counterfactual_action == original_action:
            return original_reward
        
        # For different actions, apply uncertainty discount
        # Real implementation would use bandit arm statistics
        return original_reward * 0.9
    
    def _estimate_regret(
        self,
        original_total: float,
        counterfactual_total: float,
    ) -> float:
        """Estimate regret (positive improvement potential)."""
        return max(0.0, counterfactual_total - original_total)
    
    def _regret_by_type(
        self,
        replays: List[DecisionReplay],
    ) -> Dict[str, float]:
        """Break down regret by decision type."""
        regret_by_type: Dict[str, float] = {}
        
        for replay in replays:
            dt = replay.decision_type or "unknown"
            regret = max(0.0, replay.counterfactual_reward - replay.original_reward)
            regret_by_type[dt] = regret_by_type.get(dt, 0.0) + regret
        
        return regret_by_type
    
    def _store_result(self, result: ReplayResult) -> bool:
        """Store replay result."""
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            result_file = self.output_path / f"{result.mission_id}_replay.json"
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            
            return True
            
        except Exception as e:
            logger.warning(f"[REPLAY] Failed to store result: {e}")
            return False
    
    def get_replay_history(self, mission_id: str) -> List[ReplayResult]:
        """Get all replay results for a mission."""
        try:
            results = []
            
            pattern = f"{mission_id}_replay*.json"
            for file_path in self.output_path.glob(pattern):
                with open(file_path, "r") as f:
                    data = json.load(f)
                results.append(ReplayResult.from_dict(data))
            
            return sorted(results, key=lambda r: r.replayed_at)
            
        except Exception as e:
            logger.warning(f"[REPLAY] Failed to get history: {e}")
            return []


# Global engine instance
_engine: Optional[CounterfactualReplayEngine] = None


def get_counterfactual_engine(
    config: Optional[ReplayConfig] = None,
) -> CounterfactualReplayEngine:
    """Get global counterfactual engine instance."""
    global _engine
    if _engine is None:
        _engine = CounterfactualReplayEngine(config=config)
    return _engine


def reset_counterfactual_engine() -> None:
    """Reset global engine (mainly for testing)."""
    global _engine
    _engine = None

