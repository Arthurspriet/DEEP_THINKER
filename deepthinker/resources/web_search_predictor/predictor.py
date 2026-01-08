"""
Web Search Predictor Runtime.

Core prediction logic with ML models and rule-based fallback.
Predicts whether a phase requires web search to avoid hallucinations.
"""

import logging
from typing import Any, Optional

import numpy as np

from .schemas import WebSearchContext, WebSearchExecutionPlan, WebSearchSystemState, WebSearchPrediction
from .config import (
    PREDICTOR_CONFIG,
    DEFAULT_HALLUCINATION_RISK,
    DEFAULT_EXPECTED_QUERIES,
)
from .feature_encoder import WebSearchFeatureEncoder
from .model_registry import WebSearchModelRegistry, ModelMetadata

# ML Influence Tracking (optional, non-invasive)
try:
    from ...observability.ml_influence import get_influence_tracker
    ML_INFLUENCE_AVAILABLE = True
except ImportError:
    ML_INFLUENCE_AVAILABLE = False
    get_influence_tracker = None

logger = logging.getLogger(__name__)


class WebSearchPredictor:
    """
    Web Search Predictor for phase execution.
    
    Predicts search_required, expected_queries, and hallucination_risk_without_search
    before phase execution using trained ML models when available,
    with rule-based fallback for cold-start and low-confidence scenarios.
    
    Safety guarantees:
    - Never raises exceptions from predict()
    - Always returns valid predictions (falls back to rules on any error)
    - Confidence gating enforced before using ML predictions
    - Shadow mode only: no behavior modification
    """
    
    def __init__(
        self,
        registry: Optional[WebSearchModelRegistry] = None,
        encoder: Optional[WebSearchFeatureEncoder] = None,
        auto_load: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            registry: Model registry (creates default if None)
            encoder: Feature encoder (creates default if None)
            auto_load: Whether to automatically load the latest model
        """
        self.encoder = encoder or WebSearchFeatureEncoder()
        self.registry = registry or WebSearchModelRegistry()
        
        self._model: Optional[Any] = None
        self._metadata: Optional[ModelMetadata] = None
        
        if auto_load:
            self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the latest compatible model.
        
        Returns:
            True if model was loaded successfully
        """
        try:
            result = self.registry.load_latest()
            if result is not None:
                self._model, self._metadata = result
                logger.info(
                    f"Loaded WebSearch model v{self._metadata.version} "
                    f"({self._metadata.model_type}, trained on {self._metadata.dataset_size} samples)"
                )
                return True
            else:
                logger.info("No trained WebSearch model available, using fallback rules")
                return False
        except Exception as e:
            logger.warning(f"Failed to load WebSearch model: {e}")
            return False
    
    @property
    def model_loaded(self) -> bool:
        """Check if a trained model is loaded."""
        return self._model is not None
    
    @property
    def model_version(self) -> str:
        """Get the current model version string."""
        if self._metadata is not None:
            return f"v{self._metadata.version}"
        return "v0-fallback"
    
    def predict(
        self,
        ctx: WebSearchContext,
        plan: WebSearchExecutionPlan,
        state: WebSearchSystemState,
        mission_id: Optional[str] = None,
    ) -> WebSearchPrediction:
        """
        Predict web search necessity for a phase.
        
        Args:
            ctx: Web search context information
            plan: Execution plan configuration
            state: Current system state
            mission_id: Optional mission ID for influence tracking
            
        Returns:
            WebSearchPrediction with predicted search necessity and confidence
            
        Note:
            This method never raises exceptions. On any error,
            it falls back to rule-based predictions.
        """
        prediction: WebSearchPrediction
        try:
            # Try ML prediction if model is available
            if self._model is not None:
                prediction = self._ml_predict(ctx, plan, state)
            else:
                # Fall back to rules
                prediction = self._fallback_predict(ctx, plan, state)
            
        except Exception as e:
            logger.warning(f"Web search prediction failed, using fallback: {e}")
            prediction = self._fallback_predict(ctx, plan, state)
        
        # Emit influence event (non-blocking, never fails)
        self._emit_influence_event(ctx, prediction, mission_id)
        
        return prediction
    
    def _ml_predict(
        self,
        ctx: WebSearchContext,
        plan: WebSearchExecutionPlan,
        state: WebSearchSystemState
    ) -> WebSearchPrediction:
        """
        Make prediction using trained ML model.
        
        Args:
            ctx: Web search context
            plan: Execution plan
            state: System state
            
        Returns:
            WebSearchPrediction from ML model
        """
        # Encode features
        features = self.encoder.encode(ctx, plan, state)
        features_2d = features.reshape(1, -1)
        
        # Get predictions from model
        # Model outputs: [search_probability, expected_queries, hallucination_risk]
        preds = self._model.predict(features_2d)[0]
        
        # Extract predictions
        search_probability = float(np.clip(preds[0], 0, 1))
        expected_queries = max(0, int(round(preds[1])))
        hallucination_risk = float(np.clip(preds[2], 0, 1))
        
        # Estimate confidence based on model's certainty
        confidence = self._estimate_confidence(search_probability)
        
        # Apply confidence threshold
        threshold = PREDICTOR_CONFIG.get("confidence_threshold", 0.65)
        
        if confidence >= threshold:
            return WebSearchPrediction(
                search_required=search_probability >= 0.5,
                expected_queries=expected_queries,
                hallucination_risk_without_search=hallucination_risk,
                confidence=confidence,
                model_version=self.model_version,
                used_fallback=False,
            )
        else:
            # Low confidence - fall back to rules
            logger.debug(
                f"ML confidence {confidence:.2f} below threshold {threshold}, "
                f"using fallback"
            )
            return self._fallback_predict(ctx, plan, state)
    
    def _estimate_confidence(self, probability: float) -> float:
        """
        Estimate prediction confidence from probability.
        
        Confidence is higher when probability is closer to 0 or 1.
        
        Args:
            probability: Predicted probability (0-1)
            
        Returns:
            Confidence score (0-1)
        """
        # Distance from 0.5 (uncertainty point)
        distance = abs(probability - 0.5)
        # Scale to 0.5-1.0 range
        confidence = 0.5 + distance
        return confidence
    
    def _fallback_predict(
        self,
        ctx: WebSearchContext,
        plan: WebSearchExecutionPlan,
        state: WebSearchSystemState
    ) -> WebSearchPrediction:
        """
        Make rule-based fallback prediction.
        
        Args:
            ctx: Web search context
            plan: Execution plan
            state: System state
            
        Returns:
            WebSearchPrediction using heuristic rules
        """
        # Determine if search is likely needed based on content flags
        search_required = (
            ctx.contains_dates or
            ctx.contains_named_entities or
            ctx.contains_factual_claims
        )
        
        # Also consider if planner already enabled search
        if plan.search_enabled_by_planner:
            search_required = True
        
        # Consider phase type
        high_risk_phases = {"research", "deep_analysis"}
        if ctx.phase_type.lower() in high_risk_phases:
            # Research and deep analysis phases have higher baseline need
            if ctx.contains_dates or ctx.contains_named_entities:
                search_required = True
        
        # Get default values based on phase type
        phase_type_lower = ctx.phase_type.lower()
        default_risk = DEFAULT_HALLUCINATION_RISK.get(phase_type_lower, 0.3)
        default_queries = DEFAULT_EXPECTED_QUERIES.get(phase_type_lower, 2)
        
        # Compute predictions
        if search_required:
            expected_queries = default_queries
            # Risk is lower if we predict search will be used
            hallucination_risk = 0.4 if any([
                ctx.contains_dates,
                ctx.contains_named_entities,
                ctx.contains_factual_claims
            ]) else 0.25
        else:
            expected_queries = 0
            hallucination_risk = 0.15
        
        # Adjust risk based on content flags
        if ctx.contains_dates and ctx.contains_factual_claims:
            hallucination_risk = max(hallucination_risk, 0.5)
        
        return WebSearchPrediction(
            search_required=search_required,
            expected_queries=expected_queries,
            hallucination_risk_without_search=hallucination_risk,
            confidence=0.35,  # Low confidence for fallback
            model_version="v0-fallback",
            used_fallback=True,
        )
    
    def _emit_influence_event(
        self,
        ctx: WebSearchContext,
        prediction: WebSearchPrediction,
        mission_id: Optional[str] = None,
    ) -> None:
        """
        Emit an influence event for ML governance tracking.
        
        This is non-invasive and never affects prediction behavior.
        All errors are silently caught to ensure no impact on prediction.
        
        Args:
            ctx: Web search context
            prediction: The prediction made
            mission_id: Optional mission ID
        """
        if not ML_INFLUENCE_AVAILABLE:
            return
        
        try:
            tracker = get_influence_tracker()
            tracker.record(
                mission_id=mission_id or "unknown",
                phase_name=ctx.phase_name,
                phase_type=ctx.phase_type,
                predictor_name="web_search",
                predictor_version=prediction.model_version,
                prediction_summary={
                    "search_required": prediction.search_required,
                    "expected_queries": prediction.expected_queries,
                    "hallucination_risk_without_search": prediction.hallucination_risk_without_search,
                },
                confidence=prediction.confidence,
                used_fallback=prediction.used_fallback,
                predictor_mode="shadow",  # Currently always shadow mode
            )
        except Exception:
            # Silently ignore any tracking errors
            pass
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self._metadata is None:
            return {
                "model_loaded": False,
                "version": "v0-fallback",
                "type": "rule-based",
            }
        
        return {
            "model_loaded": True,
            "version": f"v{self._metadata.version}",
            "type": self._metadata.model_type,
            "trained_at": self._metadata.trained_at,
            "dataset_size": self._metadata.dataset_size,
            "feature_version": self._metadata.feature_version,
            "validation_metrics": self._metadata.validation_metrics,
        }

