"""
Configuration constants for Web Search Predictor.

Follows the pattern established in cost_time_predictor/config.py
and phase_risk_predictor/config.py.
"""

from typing import Dict, Any, List

# =============================================================================
# Feature Encoding Constants
# =============================================================================

# Phase types (order matters for one-hot encoding)
PHASE_TYPES: List[str] = [
    "research",
    "design",
    "implementation",
    "testing",
    "synthesis",
    "deep_analysis",
]

# Model tiers (order matters for one-hot encoding)
MODEL_TIERS: List[str] = [
    "small",
    "medium",
    "large",
    "xlarge",
]

# Effort levels (order matters for one-hot encoding)
EFFORT_LEVELS: List[str] = [
    "minimal",
    "standard",
    "thorough",
]

# Known councils for multi-hot encoding
KNOWN_COUNCILS: List[str] = [
    "research_council",
    "planner_council",
    "coder_council",
    "evaluator_council",
    "simulation_council",
    "synthesis_council",
    "deep_analysis_council",
    "evidence_council",
    "explorer_council",
]

# =============================================================================
# Predictor Settings
# =============================================================================

PREDICTOR_CONFIG: Dict[str, Any] = {
    # Master enable switch
    "enabled": True,
    
    # Operating mode:
    # - "shadow": Log predictions vs actuals, no behavior change
    # - "advisory": Predictions available but rules enforce caps
    # - "active": Reserved for future (not implemented)
    "mode": "shadow",
    
    # Confidence threshold - below this, use fallback
    "confidence_threshold": 0.65,
    
    # Minimum training samples required
    "min_training_samples": 40,
    
    # Use XGBoost when dataset exceeds this size
    "use_xgboost_threshold": 150,
}

# =============================================================================
# Feature Vector Configuration
# =============================================================================

# Increment this when feature encoding changes
FEATURE_VECTOR_VERSION: int = 1

# Feature normalization constants
NORMALIZATION_CONSTANTS: Dict[str, float] = {
    "prompt_token_count_max": 8000.0,
    "iteration_index_max": 10.0,
    "max_iterations_max": 10.0,
    "available_time_max_seconds": 3600.0,  # 1 hour
}

# =============================================================================
# Default Fallback Values
# =============================================================================

# Default hallucination risk by phase type (when search not used)
DEFAULT_HALLUCINATION_RISK: Dict[str, float] = {
    "research": 0.50,      # High risk - external facts needed
    "design": 0.25,        # Lower risk - more conceptual
    "implementation": 0.20, # Lower risk - code-focused
    "testing": 0.15,       # Low risk - deterministic
    "synthesis": 0.35,     # Medium risk - aggregation may hallucinate
    "deep_analysis": 0.45, # High risk - factual claims
}

# Default expected queries by phase type
DEFAULT_EXPECTED_QUERIES: Dict[str, int] = {
    "research": 3,
    "design": 1,
    "implementation": 1,
    "testing": 0,
    "synthesis": 2,
    "deep_analysis": 4,
}

# =============================================================================
# Storage Paths
# =============================================================================

MODEL_STORAGE_DIR: str = "kb/models/web_search_predictor"
EVAL_LOG_PATH: str = "kb/orchestration/web_search_eval.jsonl"

