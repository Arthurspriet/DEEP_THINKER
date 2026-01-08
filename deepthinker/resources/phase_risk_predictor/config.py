"""
Configuration constants for Phase Risk Predictor.

Follows the pattern established in cost_time_predictor/config.py.
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

# Known failure modes
KNOWN_FAILURE_MODES: List[str] = [
    "timeout",
    "hallucination",
    "low_quality",
    "incoherent",
    "unknown",
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
    "confidence_threshold": 0.6,
    
    # Minimum training samples required
    "min_training_samples": 30,
    
    # Use XGBoost when dataset exceeds this size
    "use_xgboost_threshold": 120,
}

# =============================================================================
# Feature Vector Configuration
# =============================================================================

# Increment this when feature encoding changes
FEATURE_VECTOR_VERSION: int = 1

# Feature normalization constants
NORMALIZATION_CONSTANTS: Dict[str, float] = {
    "iteration_index_max": 10.0,
    "retry_count_max": 5.0,
    "max_iterations_max": 10.0,
    "model_count_max": 5.0,
    "time_remaining_max_seconds": 3600.0,  # 1 hour
    "vram_max_mb": 50000.0,
}

# =============================================================================
# Default Fallback Values
# =============================================================================

# Default retry probability by phase type
DEFAULT_RETRY_PROBABILITIES: Dict[str, float] = {
    "research": 0.15,
    "design": 0.20,
    "implementation": 0.25,
    "testing": 0.20,
    "synthesis": 0.30,
    "deep_analysis": 0.35,
}

# Default failure mode when unknown
DEFAULT_FAILURE_MODE: str = "unknown"

# =============================================================================
# Storage Paths
# =============================================================================

MODEL_STORAGE_DIR: str = "kb/models/phase_risk_predictor"
EVAL_LOG_PATH: str = "kb/orchestration/phase_risk_eval.jsonl"

