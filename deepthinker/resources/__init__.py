"""
Resources package for DeepThinker 2.0.

Provides GPU resource management, capacity tracking, cost/time prediction,
phase risk prediction, and web search prediction.
"""

from .gpu_manager import GPUResourceManager, GPUResourceStats

# Cost/Time Predictor (optional - requires numpy, sklearn)
COST_PREDICTOR_AVAILABLE = False
try:
    from .cost_time_predictor import (
        CostTimePredictor,
        PhaseContext,
        ExecutionPlan,
        SystemState,
        CostTimePrediction,
        PREDICTOR_CONFIG,
    )
    COST_PREDICTOR_AVAILABLE = True
except ImportError:
    # Dependencies not available
    CostTimePredictor = None
    PhaseContext = None
    ExecutionPlan = None
    SystemState = None
    CostTimePrediction = None
    PREDICTOR_CONFIG = {"enabled": False, "mode": "shadow"}

# Phase Risk Predictor (optional - requires numpy, sklearn)
RISK_PREDICTOR_AVAILABLE = False
try:
    from .phase_risk_predictor import (
        PhaseRiskPredictor,
        PhaseRiskContext,
        PhaseRiskExecutionPlan,
        PhaseRiskSystemState,
        PhaseRiskPrediction,
        PhaseRiskEvaluationLogger,
        PREDICTOR_CONFIG as RISK_PREDICTOR_CONFIG,
    )
    RISK_PREDICTOR_AVAILABLE = True
except ImportError:
    # Dependencies not available
    PhaseRiskPredictor = None
    PhaseRiskContext = None
    PhaseRiskExecutionPlan = None
    PhaseRiskSystemState = None
    PhaseRiskPrediction = None
    PhaseRiskEvaluationLogger = None
    RISK_PREDICTOR_CONFIG = {"enabled": False, "mode": "shadow"}

# Web Search Predictor (optional - requires numpy, sklearn)
WEB_SEARCH_PREDICTOR_AVAILABLE = False
try:
    from .web_search_predictor import (
        WebSearchPredictor,
        WebSearchContext,
        WebSearchExecutionPlan,
        WebSearchSystemState,
        WebSearchPrediction,
        WebSearchEvaluationLogger,
        PREDICTOR_CONFIG as WEB_SEARCH_PREDICTOR_CONFIG,
        contains_dates,
        contains_named_entities,
        contains_factual_claims,
        analyze_content,
    )
    WEB_SEARCH_PREDICTOR_AVAILABLE = True
except ImportError:
    # Dependencies not available
    WebSearchPredictor = None
    WebSearchContext = None
    WebSearchExecutionPlan = None
    WebSearchSystemState = None
    WebSearchPrediction = None
    WebSearchEvaluationLogger = None
    WEB_SEARCH_PREDICTOR_CONFIG = {"enabled": False, "mode": "shadow"}
    contains_dates = None
    contains_named_entities = None
    contains_factual_claims = None
    analyze_content = None

__all__ = [
    "GPUResourceManager",
    "GPUResourceStats",
    # Cost/Time Predictor
    "COST_PREDICTOR_AVAILABLE",
    "CostTimePredictor",
    "PhaseContext",
    "ExecutionPlan",
    "SystemState",
    "CostTimePrediction",
    "PREDICTOR_CONFIG",
    # Phase Risk Predictor
    "RISK_PREDICTOR_AVAILABLE",
    "PhaseRiskPredictor",
    "PhaseRiskContext",
    "PhaseRiskExecutionPlan",
    "PhaseRiskSystemState",
    "PhaseRiskPrediction",
    "PhaseRiskEvaluationLogger",
    "RISK_PREDICTOR_CONFIG",
    # Web Search Predictor
    "WEB_SEARCH_PREDICTOR_AVAILABLE",
    "WebSearchPredictor",
    "WebSearchContext",
    "WebSearchExecutionPlan",
    "WebSearchSystemState",
    "WebSearchPrediction",
    "WebSearchEvaluationLogger",
    "WEB_SEARCH_PREDICTOR_CONFIG",
    "contains_dates",
    "contains_named_entities",
    "contains_factual_claims",
    "analyze_content",
]
