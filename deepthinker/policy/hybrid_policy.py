"""
Hybrid Policy for DeepThinker.

Combines rules-based ScorecardPolicy with learned StopEscalatePredictor.

Rules remain the safety floor:
- If rules say STOP, ABORT, or FORCE_EVIDENCE -> always follow
- If rules say CONTINUE, learned policy can suggest alternatives
- Learned policy runs in SHADOW mode initially (log only)

Usage:
    hybrid = HybridPolicy()
    decision = hybrid.decide(scorecard, constraints...)
    
    # Decision includes both rules and learned components
    print(decision.rules_decision)    # From ScorecardPolicy
    print(decision.learned_prediction)  # From StopEscalatePredictor
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from ..learning.config import LearningConfig, LearnedPolicyMode, get_learning_config
from ..learning.policy_features import PolicyState
from ..learning.stop_escalate_predictor import (
    PolicyAction as LearnedAction,
    PolicyPrediction,
    StopEscalatePredictor,
    get_stop_escalate_predictor,
)
from ..metrics.config import MetricsConfig, get_metrics_config
from ..metrics.scorecard import Scorecard
from .scorecard_policy import (
    PolicyAction,
    PolicyDecision,
    ScorecardPolicy,
    get_scorecard_policy,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridPolicyDecision:
    """
    Decision from hybrid policy.
    
    Contains both rules and learned components for transparency.
    
    Attributes:
        action: Final recommended action
        rationale: Combined rationale
        confidence: Confidence in decision
        
        rules_decision: Decision from ScorecardPolicy (safety floor)
        learned_prediction: Prediction from StopEscalatePredictor (if enabled)
        learned_influenced: Whether learned policy influenced final action
        
        scorecard: Scorecard used for decision
        policy_state: PolicyState used for learned prediction
        timestamp: When decision was made
    """
    action: PolicyAction
    rationale: str
    confidence: float = 0.8
    
    # Components
    rules_decision: Optional[PolicyDecision] = None
    learned_prediction: Optional[PolicyPrediction] = None
    learned_influenced: bool = False
    
    scorecard: Optional[Scorecard] = None
    policy_state: Optional[PolicyState] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "rules_decision": self.rules_decision.to_dict() if self.rules_decision else None,
            "learned_prediction": self.learned_prediction.to_dict() if self.learned_prediction else None,
            "learned_influenced": self.learned_influenced,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "policy_state": self.policy_state.to_dict() if self.policy_state else None,
            "timestamp": self.timestamp.isoformat(),
        }


class HybridPolicy:
    """
    Hybrid policy combining rules and learned components.
    
    Rules are the safety floor:
    - STOP, ABORT, FORCE_EVIDENCE from rules always take precedence
    - When rules say CONTINUE, learned policy can influence
    
    Learned policy modes:
    - OFF: Only rules
    - SHADOW: Log predictions, but use rules only
    - ADVISORY: Log + include in decision (caller can ignore)
    - ACTIVE: Predictions can override CONTINUE (rules still safety floor)
    """
    
    def __init__(
        self,
        metrics_config: Optional[MetricsConfig] = None,
        learning_config: Optional[LearningConfig] = None,
    ):
        """
        Initialize hybrid policy.
        
        Args:
            metrics_config: Config for ScorecardPolicy
            learning_config: Config for StopEscalatePredictor
        """
        self.metrics_config = metrics_config or get_metrics_config()
        self.learning_config = learning_config or get_learning_config()
        
        self._rules_policy = ScorecardPolicy(config=self.metrics_config)
        self._learned_predictor = StopEscalatePredictor(config=self.learning_config)
    
    def decide(
        self,
        scorecard: Scorecard,
        time_remaining_minutes: float = 60.0,
        alignment_drift_risk: float = 0.0,
        retry_count: int = 0,
        max_retries: int = 3,
        mission_id: str = "",
        phase_id: str = "",
        phase_type: str = "",
        phase_number: int = 0,
        score_history: Optional[list] = None,
        tool_success_rate: float = 1.0,
        tool_calls_count: int = 0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> HybridPolicyDecision:
        """
        Make a hybrid policy decision.
        
        Args:
            scorecard: Current scorecard
            time_remaining_minutes: Remaining mission time
            alignment_drift_risk: Risk of alignment drift (0-1)
            retry_count: Current retry count
            max_retries: Maximum retries allowed
            mission_id: Mission identifier
            phase_id: Phase identifier
            phase_type: Type of phase
            phase_number: Which phase number
            score_history: Recent score history
            tool_success_rate: Recent tool success rate
            tool_calls_count: Number of tool calls
            tokens_used: Tokens consumed
            cost_usd: USD cost estimate
            
        Returns:
            HybridPolicyDecision with combined result
        """
        # 1. Get rules decision (safety floor)
        rules_decision = self._rules_policy.decide(
            scorecard=scorecard,
            time_remaining_minutes=time_remaining_minutes,
            alignment_drift_risk=alignment_drift_risk,
            retry_count=retry_count,
            max_retries=max_retries,
        )
        
        # 2. Build policy state for learned predictor
        time_total_minutes = time_remaining_minutes + (60 * 0.4)  # Rough estimate
        policy_state = PolicyState(
            mission_id=mission_id,
            phase_id=phase_id,
            phase_type=phase_type,
            phase_number=phase_number,
            current_score=scorecard.overall,
            score_history=score_history or [],
            consistency_score=scorecard.consistency,
            grounding_score=scorecard.evidence_grounding,
            disagreement_rate=0.0,  # Would come from judge ensemble
            contradiction_count=0,   # Would come from claim graph
            time_remaining_minutes=time_remaining_minutes,
            time_budget_used_pct=(1.0 - time_remaining_minutes / time_total_minutes) if time_total_minutes > 0 else 0.5,
            alignment_drift_risk=alignment_drift_risk,
            alignment_corrections=0,  # Would come from alignment controller
            tool_success_rate=tool_success_rate,
            tool_calls_count=tool_calls_count,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )
        
        # 3. Get learned prediction (if enabled)
        learned_prediction = None
        if self.learning_config.enabled and self.learning_config.policy_mode != LearnedPolicyMode.OFF:
            learned_prediction = self._learned_predictor.predict(policy_state)
        
        # 4. Combine decisions
        return self._combine_decisions(
            rules_decision=rules_decision,
            learned_prediction=learned_prediction,
            scorecard=scorecard,
            policy_state=policy_state,
        )
    
    def _combine_decisions(
        self,
        rules_decision: PolicyDecision,
        learned_prediction: Optional[PolicyPrediction],
        scorecard: Scorecard,
        policy_state: PolicyState,
    ) -> HybridPolicyDecision:
        """
        Combine rules and learned decisions.
        
        Safety floor logic:
        - If rules say STOP, ABORT, FORCE_EVIDENCE, INCREASE_SKEPTIC -> follow
        - If rules say CONTINUE and learned is SHADOW -> use rules
        - If rules say CONTINUE and learned is ADVISORY/ACTIVE -> consider learned
        """
        mode = self.learning_config.policy_mode
        
        # Safety actions from rules always take precedence
        safety_actions = {
            PolicyAction.STOP,
            PolicyAction.ABORT,
            PolicyAction.FORCE_EVIDENCE,
            PolicyAction.INCREASE_SKEPTIC,
        }
        
        if rules_decision.action in safety_actions:
            # Rules triggered safety action - use it
            return HybridPolicyDecision(
                action=rules_decision.action,
                rationale=f"[RULES] {rules_decision.rationale}",
                confidence=rules_decision.confidence,
                rules_decision=rules_decision,
                learned_prediction=learned_prediction,
                learned_influenced=False,
                scorecard=scorecard,
                policy_state=policy_state,
            )
        
        # Rules said CONTINUE - check learned policy
        if learned_prediction is None or mode == LearnedPolicyMode.SHADOW:
            # No learned prediction or shadow mode - use rules
            return HybridPolicyDecision(
                action=rules_decision.action,
                rationale=f"[RULES] {rules_decision.rationale}",
                confidence=rules_decision.confidence,
                rules_decision=rules_decision,
                learned_prediction=learned_prediction,
                learned_influenced=False,
                scorecard=scorecard,
                policy_state=policy_state,
            )
        
        # We have a learned prediction and mode is ADVISORY or ACTIVE
        if learned_prediction.recommended_action is None:
            # No recommendation from learned policy
            return HybridPolicyDecision(
                action=rules_decision.action,
                rationale=f"[RULES] {rules_decision.rationale}",
                confidence=rules_decision.confidence,
                rules_decision=rules_decision,
                learned_prediction=learned_prediction,
                learned_influenced=False,
                scorecard=scorecard,
                policy_state=policy_state,
            )
        
        # Map learned action to policy action
        learned_action = self._map_learned_action(learned_prediction.recommended_action)
        
        if mode == LearnedPolicyMode.ADVISORY:
            # Advisory mode - include learned suggestion in rationale
            if learned_action != PolicyAction.CONTINUE:
                rationale = (
                    f"[RULES] {rules_decision.rationale} | "
                    f"[LEARNED suggests {learned_action.value}] "
                    f"p_stop={learned_prediction.p_stop:.2f}, "
                    f"p_escalate={learned_prediction.p_escalate:.2f}"
                )
                return HybridPolicyDecision(
                    action=rules_decision.action,  # Still use rules in advisory
                    rationale=rationale,
                    confidence=rules_decision.confidence,
                    rules_decision=rules_decision,
                    learned_prediction=learned_prediction,
                    learned_influenced=False,  # Advisory doesn't change action
                    scorecard=scorecard,
                    policy_state=policy_state,
                )
            else:
                return HybridPolicyDecision(
                    action=rules_decision.action,
                    rationale=f"[RULES] {rules_decision.rationale}",
                    confidence=rules_decision.confidence,
                    rules_decision=rules_decision,
                    learned_prediction=learned_prediction,
                    learned_influenced=False,
                    scorecard=scorecard,
                    policy_state=policy_state,
                )
        
        elif mode == LearnedPolicyMode.ACTIVE:
            # Active mode - learned can override CONTINUE
            if learned_action != PolicyAction.CONTINUE and learned_prediction.confidence > 0.7:
                rationale = (
                    f"[LEARNED] {learned_action.value} "
                    f"(confidence={learned_prediction.confidence:.2f}) - "
                    f"overriding rules CONTINUE"
                )
                return HybridPolicyDecision(
                    action=learned_action,
                    rationale=rationale,
                    confidence=learned_prediction.confidence * 0.9,  # Slight penalty
                    rules_decision=rules_decision,
                    learned_prediction=learned_prediction,
                    learned_influenced=True,
                    scorecard=scorecard,
                    policy_state=policy_state,
                )
        
        # Default to rules
        return HybridPolicyDecision(
            action=rules_decision.action,
            rationale=f"[RULES] {rules_decision.rationale}",
            confidence=rules_decision.confidence,
            rules_decision=rules_decision,
            learned_prediction=learned_prediction,
            learned_influenced=False,
            scorecard=scorecard,
            policy_state=policy_state,
        )
    
    def _map_learned_action(self, learned_action: LearnedAction) -> PolicyAction:
        """Map learned action to policy action."""
        mapping = {
            LearnedAction.CONTINUE: PolicyAction.CONTINUE,
            LearnedAction.STOP: PolicyAction.STOP,
            LearnedAction.ESCALATE: PolicyAction.ESCALATE,
            LearnedAction.SWITCH_MODE: PolicyAction.FORCE_EVIDENCE,
        }
        return mapping.get(learned_action, PolicyAction.CONTINUE)


# Global hybrid policy instance
_hybrid_policy: Optional[HybridPolicy] = None


def get_hybrid_policy(
    metrics_config: Optional[MetricsConfig] = None,
    learning_config: Optional[LearningConfig] = None,
) -> HybridPolicy:
    """Get global hybrid policy instance."""
    global _hybrid_policy
    if _hybrid_policy is None:
        _hybrid_policy = HybridPolicy(
            metrics_config=metrics_config,
            learning_config=learning_config,
        )
    return _hybrid_policy


def reset_hybrid_policy() -> None:
    """Reset global hybrid policy (mainly for testing)."""
    global _hybrid_policy
    _hybrid_policy = None




