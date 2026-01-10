"""
Stop/Escalate Predictor for DeepThinker.

Hybrid learned + rules policy for stopping, escalating, or switching modes.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LearningConfig, LearnedPolicyMode, get_learning_config
from .policy_features import PolicyState, get_policy_feature_names

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Actions the policy can recommend."""
    CONTINUE = "continue"       # Continue current phase
    STOP = "stop"               # Stop mission (good enough)
    ESCALATE = "escalate"       # Add more rounds/depth
    SWITCH_MODE = "switch_mode" # Switch to evidence mode


@dataclass
class PolicyPrediction:
    """
    Prediction from the stop/escalate policy.
    
    Attributes:
        p_stop: Probability of stopping
        p_escalate: Probability of escalating
        p_switch_mode: Probability of switching mode
        recommended_action: The recommended action
        confidence: Confidence in recommendation
        shadow_logged: Whether this was logged in shadow mode
    """
    p_stop: float = 0.0
    p_escalate: float = 0.0
    p_switch_mode: float = 0.0
    recommended_action: Optional[PolicyAction] = None
    confidence: float = 0.0
    shadow_logged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_stop": self.p_stop,
            "p_escalate": self.p_escalate,
            "p_switch_mode": self.p_switch_mode,
            "recommended_action": self.recommended_action.value if self.recommended_action else None,
            "confidence": self.confidence,
            "shadow_logged": self.shadow_logged,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyPrediction":
        action = data.get("recommended_action")
        if action:
            action = PolicyAction(action)
        
        return cls(
            p_stop=data.get("p_stop", 0.0),
            p_escalate=data.get("p_escalate", 0.0),
            p_switch_mode=data.get("p_switch_mode", 0.0),
            recommended_action=action,
            confidence=data.get("confidence", 0.0),
            shadow_logged=data.get("shadow_logged", False),
        )


@dataclass
class ShadowLogRecord:
    """Record for shadow mode logging."""
    timestamp: datetime
    state: PolicyState
    prediction: PolicyPrediction
    actual_action: Optional[PolicyAction] = None
    reward_observed: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.to_dict(),
            "prediction": self.prediction.to_dict(),
            "actual_action": self.actual_action.value if self.actual_action else None,
            "reward_observed": self.reward_observed,
        }


class StopEscalatePredictor:
    """
    Hybrid learned + rules policy for stop/escalate/switch decisions.
    
    Modes:
    - OFF: Disabled entirely
    - SHADOW: Log predictions but don't change behavior
    - ADVISORY: Log + return prediction (caller decides)
    - ACTIVE: Predictions drive decisions (rules still safety floor)
    
    The predictor uses a simple logistic-style linear model with
    weights loaded from JSON.
    
    Usage:
        predictor = StopEscalatePredictor()
        
        prediction = predictor.predict(state)
        
        if prediction.recommended_action == PolicyAction.STOP:
            # Consider stopping
            pass
    """
    
    # Default weights for linear model
    DEFAULT_WEIGHTS = {
        "stop": {
            "intercept": -2.0,
            "current_score": 3.0,
            "score_trend": 1.0,
            "time_budget_used_pct": 1.5,
            "alignment_drift_risk": -0.5,
        },
        "escalate": {
            "intercept": -1.5,
            "current_score": -2.0,
            "score_trend": -1.5,
            "disagreement_rate": 2.0,
            "time_remaining_minutes": 0.5,
        },
        "switch_mode": {
            "intercept": -2.5,
            "grounding_score": -3.0,
            "contradiction_count": 1.5,
            "alignment_drift_risk": 2.0,
        },
    }
    
    def __init__(self, config: Optional[LearningConfig] = None):
        """
        Initialize the predictor.
        
        Args:
            config: Optional config. Uses global if None.
        """
        self.config = config or get_learning_config()
        self._weights = self._load_weights()
    
    def _load_weights(self) -> Dict[str, Dict[str, float]]:
        """Load weights from file or use defaults."""
        try:
            if os.path.exists(self.config.model_path):
                with open(self.config.model_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(
                f"[LEARNING] Failed to load weights: {e}"
            )
        
        return {k: v.copy() for k, v in self.DEFAULT_WEIGHTS.items()}
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for probability conversion."""
        import math
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    
    def _compute_logit(
        self,
        features: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Compute logit (linear combination) for a model."""
        logit = weights.get("intercept", 0.0)
        
        for name, value in features.items():
            if name in weights:
                logit += weights[name] * value
        
        return logit
    
    def predict(self, state: PolicyState) -> PolicyPrediction:
        """
        Predict P(stop), P(escalate), P(switch_mode).
        
        Args:
            state: Current policy state
            
        Returns:
            PolicyPrediction with probabilities and recommendation
        """
        if not self.config.enabled or self.config.policy_mode == LearnedPolicyMode.OFF:
            return PolicyPrediction(
                recommended_action=PolicyAction.CONTINUE,
            )
        
        features = state.to_features()
        
        # Compute probabilities
        p_stop = self._sigmoid(self._compute_logit(features, self._weights.get("stop", {})))
        p_escalate = self._sigmoid(self._compute_logit(features, self._weights.get("escalate", {})))
        p_switch_mode = self._sigmoid(self._compute_logit(features, self._weights.get("switch_mode", {})))
        
        # Determine recommended action
        action = PolicyAction.CONTINUE
        confidence = 0.0
        
        # Check thresholds
        if p_stop >= self.config.stop_threshold:
            action = PolicyAction.STOP
            confidence = p_stop
        elif p_escalate >= self.config.escalate_threshold:
            action = PolicyAction.ESCALATE
            confidence = p_escalate
        elif p_switch_mode >= 0.6:  # Switch mode has fixed threshold
            action = PolicyAction.SWITCH_MODE
            confidence = p_switch_mode
        else:
            # Default to continue with confidence based on how far from thresholds
            action = PolicyAction.CONTINUE
            confidence = 1.0 - max(p_stop, p_escalate, p_switch_mode)
        
        prediction = PolicyPrediction(
            p_stop=p_stop,
            p_escalate=p_escalate,
            p_switch_mode=p_switch_mode,
            recommended_action=action,
            confidence=confidence,
        )
        
        # Handle shadow mode
        if self.config.policy_mode == LearnedPolicyMode.SHADOW:
            self._log_shadow(state, prediction)
            prediction.shadow_logged = True
            # In shadow mode, don't return an action
            prediction.recommended_action = None
        
        return prediction
    
    def _log_shadow(self, state: PolicyState, prediction: PolicyPrediction) -> None:
        """Log prediction in shadow mode."""
        try:
            log_path = Path(self.config.shadow_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            record = ShadowLogRecord(
                timestamp=datetime.utcnow(),
                state=state,
                prediction=prediction,
            )
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
            
            logger.debug(
                f"[LEARNING] Shadow logged: p_stop={prediction.p_stop:.3f}, "
                f"p_escalate={prediction.p_escalate:.3f}"
            )
            
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to log shadow: {e}")
    
    def update_shadow_outcome(
        self,
        mission_id: str,
        phase_id: str,
        actual_action: PolicyAction,
        reward: float,
    ) -> bool:
        """
        Update the most recent shadow log with actual outcome.
        
        Args:
            mission_id: Mission identifier
            phase_id: Phase identifier
            actual_action: What action was actually taken
            reward: Observed reward
            
        Returns:
            True if updated successfully
        """
        if not self.config.enabled:
            return False
        
        try:
            log_path = Path(self.config.shadow_log_path)
            
            if not log_path.exists():
                return False
            
            # Append outcome record
            outcome = {
                "type": "outcome_update",
                "mission_id": mission_id,
                "phase_id": phase_id,
                "actual_action": actual_action.value,
                "reward": reward,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome) + "\n")
            
            return True
            
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to update outcome: {e}")
            return False
    
    def save_weights(self, weights: Dict[str, Dict[str, float]]) -> bool:
        """Save weights to file."""
        try:
            weights_path = Path(self.config.model_path)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(weights_path, "w") as f:
                json.dump(weights, f, indent=2)
            
            self._weights = weights
            return True
            
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to save weights: {e}")
            return False
    
    def get_shadow_statistics(self) -> Dict[str, Any]:
        """Get statistics from shadow logs."""
        log_path = Path(self.config.shadow_log_path)
        
        if not log_path.exists():
            return {"records": 0}
        
        try:
            predictions = []
            outcomes = []
            
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    if data.get("type") == "outcome_update":
                        outcomes.append(data)
                    else:
                        predictions.append(data)
            
            return {
                "predictions": len(predictions),
                "outcomes": len(outcomes),
                "avg_p_stop": (
                    sum(p.get("prediction", {}).get("p_stop", 0) for p in predictions) /
                    max(1, len(predictions))
                ),
                "avg_p_escalate": (
                    sum(p.get("prediction", {}).get("p_escalate", 0) for p in predictions) /
                    max(1, len(predictions))
                ),
            }
            
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to compute stats: {e}")
            return {"error": str(e)}


# Global predictor instance
_predictor: Optional[StopEscalatePredictor] = None


def get_stop_escalate_predictor(
    config: Optional[LearningConfig] = None,
) -> StopEscalatePredictor:
    """Get global stop/escalate predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = StopEscalatePredictor(config=config)
    return _predictor


def reset_stop_escalate_predictor() -> None:
    """Reset global predictor (mainly for testing)."""
    global _predictor
    _predictor = None


