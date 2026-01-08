"""
Phase Risk Predictor for DeepThinker 2.0.

ML-based prediction of phase execution risk (retry probability, expected retries,
failure mode) with rule-based fallback for safety and cold-start scenarios.
"""

from .schemas import (
    PhaseRiskContext,
    PhaseRiskExecutionPlan,
    PhaseRiskSystemState,
    PhaseRiskPrediction,
    PhaseRiskEvaluation,
)
from .config import PREDICTOR_CONFIG
from .predictor import PhaseRiskPredictor
from .eval_logger import PhaseRiskEvaluationLogger
from .feature_encoder import PhaseRiskFeatureEncoder
from .model_registry import PhaseRiskModelRegistry, ModelMetadata

__all__ = [
    # Schemas
    "PhaseRiskContext",
    "PhaseRiskExecutionPlan",
    "PhaseRiskSystemState",
    "PhaseRiskPrediction",
    "PhaseRiskEvaluation",
    # Core classes
    "PhaseRiskPredictor",
    "PhaseRiskEvaluationLogger",
    "PhaseRiskFeatureEncoder",
    "PhaseRiskModelRegistry",
    "ModelMetadata",
    # Config
    "PREDICTOR_CONFIG",
]

