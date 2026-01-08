"""
Scorecard Policy for DeepThinker.

Rules-based policy that decides per phase iteration:
- stop if overall_score >= threshold AND consistency >= threshold
- escalate if goal_coverage low OR grounding low
- if time_budget low -> stop unless score critically low
- if drift/alignment warning -> force evidence mode / skeptic weight

Gated with config flag: SCORECARD_POLICY_ENABLED
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..metrics.config import MetricsConfig, get_metrics_config
from ..metrics.scorecard import Scorecard

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Actions that policy can recommend."""
    CONTINUE = "continue"
    """Continue with current approach."""
    
    STOP = "stop"
    """Stop phase - quality threshold met."""
    
    ESCALATE = "escalate"
    """Escalate to stronger models/more resources."""
    
    FORCE_EVIDENCE = "force_evidence"
    """Force evidence gathering mode."""
    
    INCREASE_SKEPTIC = "increase_skeptic"
    """Increase skeptic weight in councils."""
    
    ABORT = "abort"
    """Abort phase due to critical failure."""


@dataclass
class PolicyDecision:
    """
    Structured output from policy evaluation.
    
    Attributes:
        action: Recommended action
        rationale: Brief explanation of decision
        confidence: Confidence in decision (0-1)
        scorecard: Scorecard that triggered decision
        constraints_snapshot: State at decision time
        timestamp: When decision was made
        triggered_rules: List of rules that triggered
    """
    action: PolicyAction
    rationale: str
    confidence: float = 0.8
    scorecard: Optional[Scorecard] = None
    constraints_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    triggered_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "constraints_snapshot": self.constraints_snapshot,
            "timestamp": self.timestamp.isoformat(),
            "triggered_rules": self.triggered_rules,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyDecision":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        scorecard_data = data.get("scorecard")
        scorecard = Scorecard.from_dict(scorecard_data) if scorecard_data else None
        
        return cls(
            action=PolicyAction(data.get("action", "continue")),
            rationale=data.get("rationale", ""),
            confidence=data.get("confidence", 0.8),
            scorecard=scorecard,
            constraints_snapshot=data.get("constraints_snapshot", {}),
            timestamp=timestamp,
            triggered_rules=data.get("triggered_rules", []),
        )


class ScorecardPolicy:
    """
    Rules-based policy for orchestration decisions.
    
    Evaluates scorecards against thresholds and constraints
    to produce actionable decisions.
    
    Usage:
        policy = ScorecardPolicy()
        decision = policy.decide(scorecard, constraints)
        
        if decision.action == PolicyAction.STOP:
            # Phase can stop
        elif decision.action == PolicyAction.ESCALATE:
            # Need stronger models
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize the policy.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
        """
        self.config = config or get_metrics_config()
    
    def decide(
        self,
        scorecard: Scorecard,
        time_remaining_minutes: float = 60.0,
        alignment_drift_risk: float = 0.0,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> PolicyDecision:
        """
        Make a policy decision based on scorecard and constraints.
        
        Args:
            scorecard: Current scorecard
            time_remaining_minutes: Remaining mission time
            alignment_drift_risk: Risk of alignment drift (0-1)
            retry_count: Current retry count
            max_retries: Maximum retries allowed
            
        Returns:
            PolicyDecision with recommended action
        """
        if not self.config.scorecard_policy_enabled:
            return PolicyDecision(
                action=PolicyAction.CONTINUE,
                rationale="Policy disabled",
                confidence=1.0,
                scorecard=scorecard,
            )
        
        triggered_rules: List[str] = []
        constraints = {
            "time_remaining_minutes": time_remaining_minutes,
            "alignment_drift_risk": alignment_drift_risk,
            "retry_count": retry_count,
            "max_retries": max_retries,
        }
        
        # Rule 1: Check for stop condition (high quality)
        if self._check_stop_condition(scorecard):
            triggered_rules.append("stop_threshold_met")
            return PolicyDecision(
                action=PolicyAction.STOP,
                rationale=(
                    f"Quality threshold met: overall={scorecard.overall:.2f} >= "
                    f"{self.config.stop_overall_threshold}, "
                    f"consistency={scorecard.consistency:.2f} >= "
                    f"{self.config.stop_consistency_threshold}"
                ),
                confidence=0.9,
                scorecard=scorecard,
                constraints_snapshot=constraints,
                triggered_rules=triggered_rules,
            )
        
        # Rule 2: Check for alignment drift risk
        if alignment_drift_risk > 0.7:
            triggered_rules.append("high_alignment_drift")
            return PolicyDecision(
                action=PolicyAction.FORCE_EVIDENCE,
                rationale=f"High alignment drift risk: {alignment_drift_risk:.2f}",
                confidence=0.85,
                scorecard=scorecard,
                constraints_snapshot=constraints,
                triggered_rules=triggered_rules,
            )
        
        # Rule 3: Check for moderate alignment drift
        if alignment_drift_risk > 0.5:
            triggered_rules.append("moderate_alignment_drift")
            return PolicyDecision(
                action=PolicyAction.INCREASE_SKEPTIC,
                rationale=f"Moderate alignment drift risk: {alignment_drift_risk:.2f}",
                confidence=0.75,
                scorecard=scorecard,
                constraints_snapshot=constraints,
                triggered_rules=triggered_rules,
            )
        
        # Rule 4: Check for escalation need (low goal coverage or grounding)
        if self._check_escalation_condition(scorecard):
            triggered_rules.append("low_quality_escalation")
            
            # But if retries exhausted, abort
            if retry_count >= max_retries:
                triggered_rules.append("retries_exhausted")
                return PolicyDecision(
                    action=PolicyAction.ABORT,
                    rationale=(
                        f"Retries exhausted ({retry_count}/{max_retries}) with low quality: "
                        f"goal_coverage={scorecard.goal_coverage:.2f}, "
                        f"evidence_grounding={scorecard.evidence_grounding:.2f}"
                    ),
                    confidence=0.9,
                    scorecard=scorecard,
                    constraints_snapshot=constraints,
                    triggered_rules=triggered_rules,
                )
            
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                rationale=(
                    f"Low quality scores: goal_coverage={scorecard.goal_coverage:.2f} "
                    f"< {self.config.escalate_goal_coverage_threshold} or "
                    f"evidence_grounding={scorecard.evidence_grounding:.2f} "
                    f"< {self.config.escalate_grounding_threshold}"
                ),
                confidence=0.8,
                scorecard=scorecard,
                constraints_snapshot=constraints,
                triggered_rules=triggered_rules,
            )
        
        # Rule 5: Check time budget
        if time_remaining_minutes < self.config.time_critical_threshold_minutes:
            triggered_rules.append("time_critical")
            
            # If score is critically low, escalate anyway
            if scorecard.overall < 0.3:
                triggered_rules.append("critically_low_score")
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    rationale=(
                        f"Time critical ({time_remaining_minutes:.1f}min) but "
                        f"score critically low ({scorecard.overall:.2f})"
                    ),
                    confidence=0.7,
                    scorecard=scorecard,
                    constraints_snapshot=constraints,
                    triggered_rules=triggered_rules,
                )
            
            # Otherwise, stop to save time
            return PolicyDecision(
                action=PolicyAction.STOP,
                rationale=(
                    f"Time critical ({time_remaining_minutes:.1f}min), "
                    f"accepting current score ({scorecard.overall:.2f})"
                ),
                confidence=0.6,
                scorecard=scorecard,
                constraints_snapshot=constraints,
                triggered_rules=triggered_rules,
            )
        
        # Default: continue
        return PolicyDecision(
            action=PolicyAction.CONTINUE,
            rationale="No policy thresholds triggered",
            confidence=0.5,
            scorecard=scorecard,
            constraints_snapshot=constraints,
            triggered_rules=triggered_rules,
        )
    
    def _check_stop_condition(self, scorecard: Scorecard) -> bool:
        """Check if scorecard meets stop thresholds."""
        return (
            scorecard.overall >= self.config.stop_overall_threshold and
            scorecard.consistency >= self.config.stop_consistency_threshold
        )
    
    def _check_escalation_condition(self, scorecard: Scorecard) -> bool:
        """Check if scorecard indicates need for escalation."""
        return (
            scorecard.goal_coverage < self.config.escalate_goal_coverage_threshold or
            scorecard.evidence_grounding < self.config.escalate_grounding_threshold
        )
    
    def should_stop(self, scorecard: Scorecard) -> bool:
        """
        Quick check if phase should stop based on scorecard.
        
        Args:
            scorecard: Current scorecard
            
        Returns:
            True if phase should stop
        """
        if not self.config.scorecard_policy_enabled:
            return False
        return self._check_stop_condition(scorecard)
    
    def should_escalate(self, scorecard: Scorecard) -> bool:
        """
        Quick check if phase should escalate based on scorecard.
        
        Args:
            scorecard: Current scorecard
            
        Returns:
            True if phase should escalate
        """
        if not self.config.scorecard_policy_enabled:
            return False
        return self._check_escalation_condition(scorecard)
    
    def evaluate_phase_transition(
        self,
        score_before: Optional[Scorecard],
        score_after: Scorecard,
        time_remaining_minutes: float = 60.0,
    ) -> PolicyDecision:
        """
        Evaluate transition between phases based on before/after scores.
        
        Args:
            score_before: Scorecard at phase start (may be None)
            score_after: Scorecard at phase end
            time_remaining_minutes: Remaining mission time
            
        Returns:
            PolicyDecision for the transition
        """
        # Compute delta if we have both
        if score_before is not None:
            delta = score_after.overall - score_before.overall
            
            # If we made significant progress, might be worth continuing
            if delta > 0.1 and score_after.overall < self.config.stop_overall_threshold:
                return PolicyDecision(
                    action=PolicyAction.CONTINUE,
                    rationale=f"Making progress: delta={delta:.2f}",
                    confidence=0.7,
                    scorecard=score_after,
                    constraints_snapshot={"delta": delta},
                    triggered_rules=["positive_delta"],
                )
            
            # If we're regressing, might need to escalate
            if delta < -0.1:
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    rationale=f"Regression detected: delta={delta:.2f}",
                    confidence=0.8,
                    scorecard=score_after,
                    constraints_snapshot={"delta": delta},
                    triggered_rules=["negative_delta"],
                )
        
        # Fall back to standard decision
        return self.decide(
            scorecard=score_after,
            time_remaining_minutes=time_remaining_minutes,
        )


# Global policy instance
_policy: Optional[ScorecardPolicy] = None


def get_scorecard_policy(config: Optional[MetricsConfig] = None) -> ScorecardPolicy:
    """Get global scorecard policy instance."""
    global _policy
    if _policy is None:
        _policy = ScorecardPolicy(config=config)
    return _policy

