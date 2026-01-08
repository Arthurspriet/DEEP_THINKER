"""
ML Router Advisor for DeepThinker.

Supervised router that advises on:
- Council set selection
- Model tier selection
- Number of rounds per phase (1-3)

Acts as an ADVISOR: outputs decision + confidence/rationale.
Orchestrator applies only if no constraint violation.

Supports:
- sklearn-based model (LogisticRegression/LightGBM)
- sklearn-free fallback with JSON-stored weights
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..metrics.config import MetricsConfig, get_metrics_config
from .features import RoutingContext, extract_routing_features, get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class RoutingFeatures:
    """
    Feature vector for routing decision.
    
    Attributes:
        features: Dictionary of feature name -> value
        context: Original context
        timestamp: When features were extracted
    """
    features: Dict[str, float] = field(default_factory=dict)
    context: Optional[RoutingContext] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_vector(self, feature_names: Optional[List[str]] = None) -> List[float]:
        """Convert to ordered list for model input."""
        if feature_names is None:
            feature_names = get_feature_names()
        return [self.features.get(name, 0.0) for name in feature_names]


@dataclass
class RoutingDecision:
    """
    Output from ML router advisor.
    
    Attributes:
        council_set: Recommended council set (e.g., "standard", "deep", "fast")
        model_tier: Recommended model tier ("SMALL", "MEDIUM", "LARGE")
        num_rounds: Recommended number of rounds (1-3)
        confidence: Confidence in recommendation (0-1)
        rationale: Brief explanation
        features_used: Features used for decision
        timestamp: When decision was made
    """
    council_set: str = "standard"
    model_tier: str = "MEDIUM"
    num_rounds: int = 1
    confidence: float = 0.5
    rationale: str = ""
    features_used: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "council_set": self.council_set,
            "model_tier": self.model_tier,
            "num_rounds": self.num_rounds,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "features_used": self.features_used,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            council_set=data.get("council_set", "standard"),
            model_tier=data.get("model_tier", "MEDIUM"),
            num_rounds=data.get("num_rounds", 1),
            confidence=data.get("confidence", 0.5),
            rationale=data.get("rationale", ""),
            features_used=data.get("features_used", {}),
            timestamp=timestamp,
        )


class MLRouterAdvisor:
    """
    ML-based routing advisor.
    
    Provides advisory decisions on council/model/rounds selection.
    The orchestrator is free to apply or ignore based on constraints.
    
    Usage:
        router = MLRouterAdvisor()
        decision = router.advise(context)
        
        if decision.confidence > 0.6:
            apply_routing(decision)
    """
    
    # Default weights for heuristic fallback
    DEFAULT_WEIGHTS = {
        "tier": {
            "difficulty_estimate": 0.5,
            "time_pressure": -0.3,
            "retry_count": 0.2,
            "input_length": 0.1,
        },
        "rounds": {
            "difficulty_estimate": 0.4,
            "time_pressure": -0.5,
            "alignment_drift_risk": 0.3,
        },
    }
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        weights_path: Optional[str] = None,
    ):
        """
        Initialize the router.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
            weights_path: Optional path to weights JSON file.
        """
        self.config = config or get_metrics_config()
        self.weights_path = weights_path or "kb/models/ml_router/weights.json"
        self._sklearn_model = None
        self._weights: Optional[Dict[str, Any]] = None
        self._load_weights()
    
    def _load_weights(self) -> None:
        """Load weights from file or use defaults."""
        try:
            if os.path.exists(self.weights_path):
                with open(self.weights_path, "r") as f:
                    self._weights = json.load(f)
                logger.info(f"[ML_ROUTER] Loaded weights from {self.weights_path}")
            else:
                self._weights = self.DEFAULT_WEIGHTS.copy()
                logger.info("[ML_ROUTER] Using default heuristic weights")
        except Exception as e:
            logger.warning(f"[ML_ROUTER] Failed to load weights: {e}")
            self._weights = self.DEFAULT_WEIGHTS.copy()
    
    def _try_load_sklearn_model(self) -> bool:
        """Try to load sklearn model if available."""
        try:
            import joblib
            model_path = Path(self.weights_path).parent / "router_model.joblib"
            if model_path.exists():
                self._sklearn_model = joblib.load(model_path)
                logger.info(f"[ML_ROUTER] Loaded sklearn model from {model_path}")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[ML_ROUTER] Failed to load sklearn model: {e}")
        return False
    
    def advise(self, context: RoutingContext) -> RoutingDecision:
        """
        Get routing advice for given context.
        
        Args:
            context: RoutingContext with current state
            
        Returns:
            RoutingDecision with recommendations
        """
        if not self.config.learning_router_enabled:
            return RoutingDecision(
                rationale="Router disabled",
                confidence=0.0,
            )
        
        # Extract features
        features = extract_routing_features(context)
        routing_features = RoutingFeatures(features=features, context=context)
        
        # Try sklearn model first
        if self._sklearn_model is not None:
            return self._sklearn_predict(routing_features)
        
        # Fallback to heuristic
        return self._heuristic_advise(routing_features)
    
    def _sklearn_predict(self, features: RoutingFeatures) -> RoutingDecision:
        """Use sklearn model for prediction."""
        try:
            X = [features.to_vector()]
            prediction = self._sklearn_model.predict(X)[0]
            probabilities = self._sklearn_model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            # Parse prediction (format: "tier_rounds", e.g., "MEDIUM_2")
            parts = str(prediction).split("_")
            tier = parts[0] if len(parts) > 0 else "MEDIUM"
            rounds = int(parts[1]) if len(parts) > 1 else 1
            
            return RoutingDecision(
                council_set="standard",
                model_tier=tier,
                num_rounds=min(max(rounds, 1), 3),
                confidence=confidence,
                rationale="sklearn model prediction",
                features_used=features.features,
            )
        except Exception as e:
            logger.warning(f"[ML_ROUTER] sklearn prediction failed: {e}")
            return self._heuristic_advise(features)
    
    def _heuristic_advise(self, features: RoutingFeatures) -> RoutingDecision:
        """Heuristic-based advice using weighted features."""
        f = features.features
        weights = self._weights or self.DEFAULT_WEIGHTS
        
        # Compute tier score (higher -> larger model)
        tier_weights = weights.get("tier", {})
        tier_score = sum(
            tier_weights.get(name, 0.0) * f.get(name, 0.0)
            for name in tier_weights
        )
        
        # Map tier score to tier
        if tier_score > 0.4:
            model_tier = "LARGE"
        elif tier_score > 0.1:
            model_tier = "MEDIUM"
        else:
            model_tier = "SMALL"
        
        # Compute rounds score (higher -> more rounds)
        rounds_weights = weights.get("rounds", {})
        rounds_score = sum(
            rounds_weights.get(name, 0.0) * f.get(name, 0.0)
            for name in rounds_weights
        )
        
        # Map rounds score to num_rounds
        if rounds_score > 0.3:
            num_rounds = 3
        elif rounds_score > 0.0:
            num_rounds = 2
        else:
            num_rounds = 1
        
        # Confidence based on feature certainty
        confidence = 0.5 + 0.3 * abs(tier_score)
        
        # Select council set based on task type
        if f.get("task_type_research", 0) > 0.5:
            council_set = "research"
        elif f.get("task_type_code", 0) > 0.5:
            council_set = "coder"
        else:
            council_set = "standard"
        
        rationale = self._build_rationale(f, model_tier, num_rounds)
        
        return RoutingDecision(
            council_set=council_set,
            model_tier=model_tier,
            num_rounds=num_rounds,
            confidence=min(confidence, 0.9),
            rationale=rationale,
            features_used=features.features,
        )
    
    def _build_rationale(
        self,
        features: Dict[str, float],
        tier: str,
        rounds: int,
    ) -> str:
        """Build human-readable rationale."""
        reasons = []
        
        difficulty = features.get("difficulty_estimate", 0.5)
        if difficulty > 0.6:
            reasons.append(f"high difficulty ({difficulty:.2f})")
        elif difficulty < 0.3:
            reasons.append(f"low difficulty ({difficulty:.2f})")
        
        time_pressure = features.get("time_pressure", 0.0)
        if time_pressure > 0.5:
            reasons.append(f"time pressure ({time_pressure:.2f})")
        
        drift_risk = features.get("alignment_drift_risk", 0.0)
        if drift_risk > 0.3:
            reasons.append(f"drift risk ({drift_risk:.2f})")
        
        if not reasons:
            reasons.append("standard heuristic")
        
        return f"Recommended {tier}/{rounds}r: {', '.join(reasons)}"
    
    def save_weights(self, weights: Dict[str, Any]) -> None:
        """
        Save weights to file.
        
        Args:
            weights: Weight dictionary to save
        """
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        with open(self.weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        self._weights = weights
        logger.info(f"[ML_ROUTER] Saved weights to {self.weights_path}")
    
    def record_outcome(
        self,
        decision: RoutingDecision,
        score_delta: float,
        cost_delta: float,
    ) -> None:
        """
        Record outcome for future training.
        
        This is used to collect training data for offline model updates.
        
        Args:
            decision: The routing decision that was made
            score_delta: Score improvement achieved
            cost_delta: Cost incurred
        """
        # Store outcome for offline training
        outcome_path = Path(self.weights_path).parent / "outcomes.jsonl"
        os.makedirs(os.path.dirname(outcome_path), exist_ok=True)
        
        record = {
            "decision": decision.to_dict(),
            "score_delta": score_delta,
            "cost_delta": cost_delta,
            "reward": score_delta - self.config.bandit_lambda * cost_delta,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            with open(outcome_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"[ML_ROUTER] Failed to record outcome: {e}")


# Global router instance
_router: Optional[MLRouterAdvisor] = None


def get_ml_router(config: Optional[MetricsConfig] = None) -> MLRouterAdvisor:
    """Get global ML router instance."""
    global _router
    if _router is None:
        _router = MLRouterAdvisor(config=config)
    return _router

