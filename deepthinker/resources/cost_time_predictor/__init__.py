"""
Cost & Time Predictor for DeepThinker 2.0.

ML-based prediction of phase execution costs (wall time, GPU seconds, VRAM peak)
with rule-based fallback for safety and cold-start scenarios.
"""

from .schemas import (
    PhaseContext,
    ExecutionPlan,
    SystemState,
    CostTimePrediction,
    PredictionEvaluation,
)
from .config import PREDICTOR_CONFIG
from .predictor import CostTimePredictor
from .eval_logger import EvaluationLogger
from .feature_encoder import FeatureEncoder
from .model_registry import ModelRegistry, ModelMetadata

__all__ = [
    # Schemas
    "PhaseContext",
    "ExecutionPlan",
    "SystemState",
    "CostTimePrediction",
    "PredictionEvaluation",
    # Core classes
    "CostTimePredictor",
    "EvaluationLogger",
    "FeatureEncoder",
    "ModelRegistry",
    "ModelMetadata",
    # Config
    "PREDICTOR_CONFIG",
]

