"""
Web Search Predictor for DeepThinker 2.0.

ML-based prediction of web search necessity to prevent hallucinations,
with rule-based fallback for safety and cold-start scenarios.

Shadow mode only - predictions are logged but do not modify system behavior.
"""

from .schemas import (
    WebSearchContext,
    WebSearchExecutionPlan,
    WebSearchSystemState,
    WebSearchPrediction,
    WebSearchEvaluation,
)
from .config import PREDICTOR_CONFIG
from .predictor import WebSearchPredictor
from .eval_logger import WebSearchEvaluationLogger
from .feature_encoder import (
    WebSearchFeatureEncoder,
    contains_dates,
    contains_named_entities,
    contains_factual_claims,
    analyze_content,
)
from .model_registry import WebSearchModelRegistry, ModelMetadata

__all__ = [
    # Schemas
    "WebSearchContext",
    "WebSearchExecutionPlan",
    "WebSearchSystemState",
    "WebSearchPrediction",
    "WebSearchEvaluation",
    # Core classes
    "WebSearchPredictor",
    "WebSearchEvaluationLogger",
    "WebSearchFeatureEncoder",
    "WebSearchModelRegistry",
    "ModelMetadata",
    # Content analysis helpers
    "contains_dates",
    "contains_named_entities",
    "contains_factual_claims",
    "analyze_content",
    # Config
    "PREDICTOR_CONFIG",
]

