"""
Cost & Time Predictor Runtime.

Core prediction logic with ML models and rule-based fallback.
"""

import logging
from typing import Any, Optional

import numpy as np

from .schemas import PhaseContext, ExecutionPlan, SystemState, CostTimePrediction
from .config import (
    PREDICTOR_CONFIG,
    DEFAULT_PHASE_TIMES,
    DEFAULT_GPU_RATIOS,
    DEFAULT_VRAM_BY_TIER,
)
from .feature_encoder import FeatureEncoder
from .model_registry import ModelRegistry, ModelMetadata

# ML Influence Tracking (optional, non-invasive)
try:
    from ...observability.ml_influence import get_influence_tracker
    ML_INFLUENCE_AVAILABLE = True
except ImportError:
    ML_INFLUENCE_AVAILABLE = False
    get_influence_tracker = None

logger = logging.getLogger(__name__)


class CostTimePredictor:
    """
    Cost and Time Predictor for phase execution.
    
    Predicts wall_time, gpu_seconds, and vram_peak before phase execution
    using trained ML models when available, with rule-based fallback
    for cold-start and low-confidence scenarios.
    
    Safety guarantees:
    - Never predicts VRAM exceeding available - safety margin
    - Never predicts time exceeding remaining - synthesis reserve
    - Always returns valid predictions (falls back to rules on any error)
    """
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        encoder: Optional[FeatureEncoder] = None,
        auto_load: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            registry: Model registry (creates default if None)
            encoder: Feature encoder (creates default if None)
            auto_load: Whether to automatically load the latest model
        """
        self.encoder = encoder or FeatureEncoder()
        self.registry = registry or ModelRegistry()
        
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
                    f"Loaded prediction model v{self._metadata.version} "
                    f"(feature_version={self._metadata.feature_version})"
                )
                return True
            else:
                logger.info("No trained model available, using fallback predictions")
                return False
        except Exception as e:
            logger.warning(f"Failed to load prediction model: {e}")
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
        ctx: PhaseContext,
        plan: ExecutionPlan,
        state: SystemState,
        mission_id: Optional[str] = None,
    ) -> CostTimePrediction:
        """
        Predict execution cost for a phase.
        
        Args:
            ctx: Phase context information
            plan: Execution plan configuration
            state: Current system state
            mission_id: Optional mission ID for influence tracking
            
        Returns:
            CostTimePrediction with predicted costs and confidence
            
        Note:
            This method never raises exceptions. On any error,
            it falls back to rule-based predictions.
        """
        prediction: CostTimePrediction
        try:
            # Try ML prediction if model is available
            if self._model is not None:
                prediction = self._ml_predict(ctx, plan, state)
            else:
                # Fall back to rules
                prediction = self._fallback_predict(ctx, plan, state)
            
        except Exception as e:
            logger.warning(f"Prediction failed, using fallback: {e}")
            prediction = self._fallback_predict(ctx, plan, state)
        
        # Emit influence event (non-blocking, never fails)
        self._emit_influence_event(ctx, prediction, mission_id)
        
        return prediction
    
    def _ml_predict(
        self,
        ctx: PhaseContext,
        plan: ExecutionPlan,
        state: SystemState
    ) -> CostTimePrediction:
        """
        Make prediction using ML model.
        
        Args:
            ctx: Phase context
            plan: Execution plan
            state: System state
            
        Returns:
            CostTimePrediction from ML model or fallback
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
        
        # Extract predictions (model outputs [wall_time, gpu_seconds, vram_peak])
        if len(raw_preds.shape) == 1:
            # Single sample prediction
            wall_time = float(raw_preds[0])
            gpu_seconds = float(raw_preds[1]) if len(raw_preds) > 1 else wall_time * 0.7
            vram_peak = float(raw_preds[2]) if len(raw_preds) > 2 else 15000
        else:
            # Multi-output format [samples, targets]
            wall_time = float(raw_preds[0, 0])
            gpu_seconds = float(raw_preds[0, 1]) if raw_preds.shape[1] > 1 else wall_time * 0.7
            vram_peak = float(raw_preds[0, 2]) if raw_preds.shape[1] > 2 else 15000
        
        # Apply safety constraints
        wall_time = self._clamp_time(wall_time, ctx)
        gpu_seconds = max(0.0, min(gpu_seconds, wall_time))
        vram_peak = self._clamp_vram(vram_peak, state)
        
        return CostTimePrediction(
            wall_time_seconds=wall_time,
            gpu_seconds=gpu_seconds,
            vram_peak_mb=vram_peak,
            confidence=confidence,
            model_version=self.model_version,
            used_fallback=False
        )
    
    def _fallback_predict(
        self,
        ctx: PhaseContext,
        plan: ExecutionPlan,
        state: SystemState
    ) -> CostTimePrediction:
        """
        Make prediction using rule-based fallback.
        
        Uses default phase times from PhaseTimeAllocator and
        VRAM estimates from MODEL_COSTS.
        
        Args:
            ctx: Phase context
            plan: Execution plan
            state: System state
            
        Returns:
            CostTimePrediction with rule-based estimates
        """
        # Get base time estimate for phase type
        phase_type = ctx.phase_type.lower()
        base_time = DEFAULT_PHASE_TIMES.get(phase_type, 90.0)
        
        # Adjust for effort level
        effort_multiplier = {
            "minimal": 0.6,
            "standard": 1.0,
            "thorough": 1.5,
        }.get(ctx.effort_level.lower(), 1.0)
        
        wall_time = base_time * effort_multiplier
        
        # Adjust for model tier (larger models take longer)
        tier_multiplier = {
            "small": 0.7,
            "medium": 1.0,
            "large": 1.3,
            "xlarge": 2.0,
        }.get(plan.model_tier.lower(), 1.0)
        
        wall_time *= tier_multiplier
        
        # Adjust for number of models
        model_count = len(plan.model_names)
        if model_count > 1:
            wall_time *= (1.0 + 0.2 * (model_count - 1))  # 20% overhead per additional model
        
        # Estimate GPU seconds
        gpu_ratio = DEFAULT_GPU_RATIOS.get(phase_type, 0.7)
        gpu_seconds = wall_time * gpu_ratio
        
        # Estimate VRAM
        vram_peak = DEFAULT_VRAM_BY_TIER.get(plan.model_tier.lower(), 15000)
        
        # If we have multiple models, scale VRAM (assume not all loaded simultaneously)
        if model_count > 1:
            vram_peak = int(vram_peak * 1.3)
        
        # Apply safety constraints
        wall_time = self._clamp_time(wall_time, ctx)
        vram_peak = self._clamp_vram(vram_peak, state)
        
        return CostTimePrediction(
            wall_time_seconds=wall_time,
            gpu_seconds=gpu_seconds,
            vram_peak_mb=vram_peak,
            confidence=0.3,  # Low confidence for fallback
            model_version="v0-fallback",
            used_fallback=True
        )
    
    def _estimate_confidence(
        self,
        features: np.ndarray,
        ctx: PhaseContext,
        plan: ExecutionPlan
    ) -> float:
        """
        Estimate prediction confidence.
        
        Confidence is based on:
        - Training data coverage (encoded in metadata)
        - Feature novelty (how different from training distribution)
        - Phase type familiarity
        
        Args:
            features: Encoded feature vector
            ctx: Phase context
            plan: Execution plan
            
        Returns:
            Confidence score between 0 and 1
        """
        if self._metadata is None:
            return 0.0
        
        # Base confidence from dataset size
        dataset_size = self._metadata.dataset_size
        min_samples = PREDICTOR_CONFIG.get("min_training_samples", 20)
        
        # Sigmoid-like curve: confidence increases with more data
        size_confidence = min(1.0, dataset_size / (min_samples * 5))
        
        # Adjust based on validation error
        mae = self._metadata.validation_mae.get("wall_time", 100.0)
        # Lower error = higher confidence
        error_confidence = max(0.0, 1.0 - (mae / 200.0))
        
        # Check if we've seen this phase type in training
        # (This is a heuristic - ideally we'd track phase type distribution)
        phase_confidence = 0.8  # Assume reasonable coverage
        
        # Combine confidences
        confidence = (size_confidence * 0.3 + error_confidence * 0.4 + phase_confidence * 0.3)
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    def _clamp_time(self, predicted_time: float, ctx: PhaseContext) -> float:
        """
        Clamp predicted time to safe bounds.
        
        Never exceeds remaining time minus synthesis reserve.
        
        Args:
            predicted_time: Raw predicted wall time
            ctx: Phase context with time budget info
            
        Returns:
            Clamped wall time
        """
        synthesis_reserve = PREDICTOR_CONFIG.get("synthesis_reserve_seconds", 120.0)
        max_time = max(0.0, ctx.time_remaining_seconds - synthesis_reserve)
        
        # Ensure positive and within bounds
        return max(1.0, min(predicted_time, max_time))
    
    def _clamp_vram(self, predicted_vram: float, state: SystemState) -> int:
        """
        Clamp predicted VRAM to safe bounds.
        
        Never exceeds available VRAM minus safety margin.
        
        Args:
            predicted_vram: Raw predicted VRAM
            state: System state with available VRAM
            
        Returns:
            Clamped VRAM in MB (integer)
        """
        safety_margin = PREDICTOR_CONFIG.get("vram_safety_margin_mb", 2000)
        max_vram = max(0, state.available_vram_mb - safety_margin)
        
        # Ensure positive and within bounds, minimum 1GB
        return max(1000, min(int(predicted_vram), max_vram))
    
    def _emit_influence_event(
        self,
        ctx: PhaseContext,
        prediction: CostTimePrediction,
        mission_id: Optional[str] = None,
    ) -> None:
        """
        Emit an influence event for ML governance tracking.
        
        This is non-invasive and never affects prediction behavior.
        All errors are silently caught to ensure no impact on prediction.
        
        Args:
            ctx: Phase context
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
                predictor_name="cost_time",
                predictor_version=prediction.model_version,
                prediction_summary={
                    "wall_time_seconds": prediction.wall_time_seconds,
                    "gpu_seconds": prediction.gpu_seconds,
                    "vram_peak_mb": prediction.vram_peak_mb,
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

