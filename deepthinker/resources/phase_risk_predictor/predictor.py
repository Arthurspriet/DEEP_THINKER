"""
Phase Risk Predictor Runtime.

Core prediction logic with ML models and rule-based fallback.
"""

import logging
from typing import Any, Optional

import numpy as np

from .schemas import (
    PhaseRiskContext,
    PhaseRiskExecutionPlan,
    PhaseRiskSystemState,
    PhaseRiskPrediction,
)
from .config import (
    PREDICTOR_CONFIG,
    DEFAULT_RETRY_PROBABILITIES,
    DEFAULT_FAILURE_MODE,
    KNOWN_FAILURE_MODES,
)
from .feature_encoder import PhaseRiskFeatureEncoder
from .model_registry import PhaseRiskModelRegistry, ModelMetadata

# ML Influence Tracking (optional, non-invasive)
try:
    from ...observability.ml_influence import get_influence_tracker
    ML_INFLUENCE_AVAILABLE = True
except ImportError:
    ML_INFLUENCE_AVAILABLE = False
    get_influence_tracker = None

logger = logging.getLogger(__name__)


class PhaseRiskPredictor:
    """
    Phase Risk Predictor for execution risk assessment.
    
    Predicts retry_probability, expected_retries, and dominant_failure_mode
    before phase execution using trained ML models when available,
    with rule-based fallback for cold-start and low-confidence scenarios.
    
    Safety guarantees:
    - Never raises exceptions from predict()
    - Always returns valid predictions (falls back to rules on any error)
    - Confidence gating enforced before using ML predictions
    """
    
    def __init__(
        self,
        registry: Optional[PhaseRiskModelRegistry] = None,
        encoder: Optional[PhaseRiskFeatureEncoder] = None,
        auto_load: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            registry: Model registry (creates default if None)
            encoder: Feature encoder (creates default if None)
            auto_load: Whether to automatically load the latest model
        """
        self.encoder = encoder or PhaseRiskFeatureEncoder()
        self.registry = registry or PhaseRiskModelRegistry()
        
        self._model: Optional[Any] = None
        self._metadata: Optional[ModelMetadata] = None
        
        if auto_load:
            self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the latest compatible model from registry.
        
        Returns:
            True if a model was loaded successfully
        """
        try:
            result = self.registry.load_latest()
            if result is not None:
                self._model, self._metadata = result
                logger.info(
                    f"Loaded risk prediction model v{self._metadata.version} "
                    f"(feature_version={self._metadata.feature_version})"
                )
                return True
            else:
                logger.info("No trained risk model available, using fallback predictions")
                return False
        except Exception as e:
            logger.warning(f"Failed to load risk prediction model: {e}")
            return False
    
    @property
    def has_model(self) -> bool:
        """Check if a trained model is loaded."""
        return self._model is not None
    
    @property
    def model_version(self) -> str:
        """Get current model version string."""
        if self._metadata is not None:
            return f"v{self._metadata.version}"
        return "v0-fallback"
    
    def predict(
        self,
        ctx: PhaseRiskContext,
        plan: PhaseRiskExecutionPlan,
        state: PhaseRiskSystemState,
        mission_id: Optional[str] = None,
    ) -> PhaseRiskPrediction:
        """
        Predict execution risk for a phase.
        
        Args:
            ctx: Phase risk context information
            plan: Execution plan configuration
            state: Current system state
            mission_id: Optional mission ID for influence tracking
            
        Returns:
            PhaseRiskPrediction with predicted risk and confidence
            
        Note:
            This method never raises exceptions. On any error,
            it falls back to rule-based predictions.
        """
        prediction: PhaseRiskPrediction
        try:
            # Try ML prediction if model is available
            if self._model is not None:
                prediction = self._ml_predict(ctx, plan, state)
            else:
                # Fall back to rules
                prediction = self._fallback_predict(ctx, plan, state)
            
        except Exception as e:
            logger.warning(f"Risk prediction failed, using fallback: {e}")
            prediction = self._fallback_predict(ctx, plan, state)
        
        # Emit influence event (non-blocking, never fails)
        self._emit_influence_event(ctx, prediction, mission_id)
        
        return prediction
    
    def _ml_predict(
        self,
        ctx: PhaseRiskContext,
        plan: PhaseRiskExecutionPlan,
        state: PhaseRiskSystemState
    ) -> PhaseRiskPrediction:
        """
        Make prediction using ML model.
        
        Args:
            ctx: Phase risk context
            plan: Execution plan
            state: System state
            
        Returns:
            PhaseRiskPrediction from ML model or fallback
        """
        # Encode features
        features = self.encoder.encode(ctx, plan, state)
        
        # Estimate confidence before prediction
        confidence = self._estimate_confidence(features, ctx, plan)
        
        # Check confidence threshold
        threshold = PREDICTOR_CONFIG.get("confidence_threshold", 0.6)
        if confidence < threshold:
            logger.debug(
                f"Low confidence ({confidence:.2f} < {threshold}), using fallback"
            )
            return self._fallback_predict(ctx, plan, state)
        
        # Make prediction
        features_2d = features.reshape(1, -1)
        raw_preds = self._model.predict(features_2d)
        
        # Extract predictions
        # Model outputs: [retry_probability, expected_retries, failure_mode_index]
        if len(raw_preds.shape) == 1:
            # Single sample prediction
            retry_probability = float(raw_preds[0])
            expected_retries = float(raw_preds[1]) if len(raw_preds) > 1 else retry_probability * plan.max_iterations
            failure_mode_idx = int(raw_preds[2]) if len(raw_preds) > 2 else -1
        else:
            # Multi-output format [samples, targets]
            retry_probability = float(raw_preds[0, 0])
            expected_retries = float(raw_preds[0, 1]) if raw_preds.shape[1] > 1 else retry_probability * plan.max_iterations
            failure_mode_idx = int(raw_preds[0, 2]) if raw_preds.shape[1] > 2 else -1
        
        # Clamp values to valid ranges
        retry_probability = max(0.0, min(1.0, retry_probability))
        expected_retries = max(0.0, expected_retries)
        
        # Map failure mode index to string
        if 0 <= failure_mode_idx < len(KNOWN_FAILURE_MODES):
            dominant_failure_mode = KNOWN_FAILURE_MODES[failure_mode_idx]
        else:
            dominant_failure_mode = DEFAULT_FAILURE_MODE
        
        return PhaseRiskPrediction(
            retry_probability=retry_probability,
            expected_retries=expected_retries,
            dominant_failure_mode=dominant_failure_mode,
            confidence=confidence,
            model_version=self.model_version,
            used_fallback=False
        )
    
    def _fallback_predict(
        self,
        ctx: PhaseRiskContext,
        plan: PhaseRiskExecutionPlan,
        state: PhaseRiskSystemState
    ) -> PhaseRiskPrediction:
        """
        Make prediction using rule-based fallback.
        
        Uses heuristics based on phase type, iteration index, and configuration.
        
        Args:
            ctx: Phase risk context
            plan: Execution plan
            state: System state
            
        Returns:
            PhaseRiskPrediction with rule-based estimates
        """
        # Get base retry probability for phase type
        phase_type = ctx.phase_type.lower()
        base_probability = DEFAULT_RETRY_PROBABILITIES.get(phase_type, 0.20)
        
        # Adjust for iteration index (later iterations have higher retry probability)
        iteration_factor = 0.1 * ctx.iteration_index
        
        # Adjust for existing retries (more retries indicate harder phase)
        retry_factor = 0.1 * ctx.retry_count_so_far
        
        # Compute final retry probability
        retry_probability = min(0.6, base_probability + iteration_factor + retry_factor)
        
        # Expected retries
        expected_retries = retry_probability * plan.max_iterations
        
        # Adjust for time pressure (less time = higher risk)
        if ctx.mission_time_remaining_seconds < 300:  # Less than 5 minutes
            retry_probability = min(0.8, retry_probability + 0.15)
            expected_retries *= 1.2
        
        # Adjust for resource constraints
        if state.gpu_load_ratio > 0.9:
            retry_probability = min(0.8, retry_probability + 0.1)
        if state.memory_pressure_ratio > 0.8:
            retry_probability = min(0.8, retry_probability + 0.1)
        
        return PhaseRiskPrediction(
            retry_probability=retry_probability,
            expected_retries=expected_retries,
            dominant_failure_mode=DEFAULT_FAILURE_MODE,
            confidence=0.3,  # Low confidence for fallback
            model_version="v0-fallback",
            used_fallback=True
        )
    
    def _estimate_confidence(
        self,
        features: np.ndarray,
        ctx: PhaseRiskContext,
        plan: PhaseRiskExecutionPlan
    ) -> float:
        """
        Estimate prediction confidence.
        
        Confidence is based on:
        - Training data coverage (encoded in metadata)
        - Feature novelty (how different from training distribution)
        - Phase type familiarity
        
        Args:
            features: Encoded feature vector
            ctx: Phase risk context
            plan: Execution plan
            
        Returns:
            Confidence score between 0 and 1
        """
        if self._metadata is None:
            return 0.0
        
        # Base confidence from dataset size
        dataset_size = self._metadata.dataset_size
        min_samples = PREDICTOR_CONFIG.get("min_training_samples", 30)
        
        # Sigmoid-like curve: confidence increases with more data
        size_confidence = min(1.0, dataset_size / (min_samples * 5))
        
        # Adjust based on validation error
        metrics = self._metadata.validation_metrics
        mae = metrics.get("retry_mae", 1.0)
        # Lower error = higher confidence
        error_confidence = max(0.0, 1.0 - (mae / 3.0))
        
        # Check AUC if available (for classification quality)
        auc = metrics.get("retry_auc", 0.5)
        auc_confidence = max(0.0, (auc - 0.5) * 2)  # Map 0.5-1.0 to 0-1
        
        # Combine confidences
        confidence = (
            size_confidence * 0.3 + 
            error_confidence * 0.4 + 
            auc_confidence * 0.3
        )
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    def _emit_influence_event(
        self,
        ctx: PhaseRiskContext,
        prediction: PhaseRiskPrediction,
        mission_id: Optional[str] = None,
    ) -> None:
        """
        Emit an influence event for ML governance tracking.
        
        This is non-invasive and never affects prediction behavior.
        All errors are silently caught to ensure no impact on prediction.
        
        Args:
            ctx: Phase risk context
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
                predictor_name="phase_risk",
                predictor_version=prediction.model_version,
                prediction_summary={
                    "retry_probability": prediction.retry_probability,
                    "expected_retries": prediction.expected_retries,
                    "dominant_failure_mode": prediction.dominant_failure_mode,
                },
                confidence=prediction.confidence,
                used_fallback=prediction.used_fallback,
                predictor_mode="shadow",  # Currently always shadow mode
            )
        except Exception:
            # Silently ignore any tracking errors
            pass
    
    def reload_model(self) -> bool:
        """
        Reload the latest model from registry.
        
        Useful after training a new model.
        
        Returns:
            True if a model was loaded
        """
        self._model = None
        self._metadata = None
        return self._load_model()
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "has_model": self.has_model,
            "model_version": self.model_version,
            "metadata": self._metadata.to_dict() if self._metadata else None,
            "feature_encoder_version": self.encoder.VERSION,
            "feature_dim": self.encoder.feature_dim,
        }

