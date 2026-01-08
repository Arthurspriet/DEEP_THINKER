"""
Configuration constants for Cost & Time Predictor.

Follows the pattern established in model_costs.py.
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
    "confidence_threshold": 0.6,
    
    # Minimum training samples required
    "min_training_samples": 20,
    
    # Use XGBoost when dataset exceeds this size
    "use_xgboost_threshold": 100,
    
    # Safety margins
    "vram_safety_margin_mb": 2000,
    "synthesis_reserve_seconds": 120.0,
}

# =============================================================================
# Feature Vector Configuration
# =============================================================================

# Increment this when feature encoding changes
FEATURE_VECTOR_VERSION: int = 1

# Feature normalization constants
NORMALIZATION_CONSTANTS: Dict[str, float] = {
    "time_budget_max_seconds": 3600.0,      # 1 hour
    "iteration_index_max": 10.0,
    "model_count_max": 5.0,
    "max_iterations_max": 10.0,
    "timeout_max_seconds": 300.0,
    "vram_max_mb": 50000.0,
}

# =============================================================================
# Default Fallback Values
# =============================================================================

# Default phase time estimates (seconds) when no ML model available
# These align with PhaseTimeAllocator defaults
DEFAULT_PHASE_TIMES: Dict[str, float] = {
    "research": 90.0,
    "design": 90.0,
    "implementation": 120.0,
    "testing": 90.0,
    "synthesis": 120.0,
    "deep_analysis": 180.0,
}

# GPU utilization ratio by phase type (gpu_seconds / wall_time)
DEFAULT_GPU_RATIOS: Dict[str, float] = {
    "research": 0.7,
    "design": 0.6,
    "implementation": 0.8,
    "testing": 0.5,
    "synthesis": 0.8,
    "deep_analysis": 0.4,
}

# Default VRAM usage by model tier (MB)
DEFAULT_VRAM_BY_TIER: Dict[str, int] = {
    "small": 3000,
    "medium": 8000,
    "large": 15000,
    "xlarge": 40000,
}

# =============================================================================
# Storage Paths
# =============================================================================

MODEL_STORAGE_DIR: str = "kb/models/cost_time_predictor"
EVAL_LOG_PATH: str = "kb/orchestration/cost_time_eval.jsonl"

