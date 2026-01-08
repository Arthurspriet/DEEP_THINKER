"""
Feature Encoder for Cost & Time Predictor.

Provides deterministic, versioned encoding of prediction inputs
to feature vectors for ML models.
"""

import logging
from typing import List, Optional

import numpy as np

from .schemas import PhaseContext, ExecutionPlan, SystemState
from .config import (
    PHASE_TYPES,
    MODEL_TIERS,
    EFFORT_LEVELS,
    KNOWN_COUNCILS,
    FEATURE_VECTOR_VERSION,
    NORMALIZATION_CONSTANTS,
)

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Deterministic feature encoder for cost/time prediction.
    
    Encodes PhaseContext, ExecutionPlan, and SystemState into a
    fixed-size feature vector suitable for ML models.
    
    Feature layout (version 1):
    - One-hot: phase_type (6 features)
    - One-hot: model_tier (4 features)
    - One-hot: effort_level (3 features)
    - Multi-hot: councils_invoked (9 features)
    - Numeric: normalized scalar features (12 features)
    
    Total: 34 features
    """
    
    VERSION = FEATURE_VECTOR_VERSION
    
    # Number of numeric features (must match encode() implementation)
    NUM_NUMERIC_FEATURES = 12
    
    def __init__(self):
        """Initialize the feature encoder."""
        self._feature_names: Optional[List[str]] = None
    
    @property
    def feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        return (
            len(PHASE_TYPES) +           # one-hot phase_type
            len(MODEL_TIERS) +           # one-hot model_tier
            len(EFFORT_LEVELS) +         # one-hot effort_level
            len(KNOWN_COUNCILS) +        # multi-hot councils
            self.NUM_NUMERIC_FEATURES    # numeric features
        )
    
    def encode(
        self,
        ctx: PhaseContext,
        plan: ExecutionPlan,
        state: SystemState
    ) -> np.ndarray:
        """
        Encode inputs into a feature vector.
        
        Args:
            ctx: Phase context information
            plan: Execution plan configuration
            state: Current system state
            
        Returns:
            np.ndarray of shape (feature_dim,) with dtype float32
        """
        features: List[float] = []
        
        # === One-hot encodings ===
        
        # Phase type (6 features)
        features.extend(self._one_hot(ctx.phase_type, PHASE_TYPES))
        
        # Model tier (4 features)
        features.extend(self._one_hot(plan.model_tier, MODEL_TIERS))
        
        # Effort level (3 features)
        features.extend(self._one_hot(ctx.effort_level, EFFORT_LEVELS))
        
        # === Multi-hot encoding ===
        
        # Councils invoked (9 features)
        features.extend(self._multi_hot(plan.councils_invoked, KNOWN_COUNCILS))
        
        # === Numeric features (normalized) ===
        
        # Time budget (normalized to 0-1 range, max 1 hour)
        features.append(
            ctx.mission_time_budget_seconds / 
            NORMALIZATION_CONSTANTS["time_budget_max_seconds"]
        )
        
        # Time remaining (normalized)
        features.append(
            ctx.time_remaining_seconds / 
            NORMALIZATION_CONSTANTS["time_budget_max_seconds"]
        )
        
        # Time remaining ratio (0-1)
        if ctx.mission_time_budget_seconds > 0:
            features.append(
                ctx.time_remaining_seconds / ctx.mission_time_budget_seconds
            )
        else:
            features.append(1.0)
        
        # Iteration index (normalized)
        features.append(
            ctx.iteration_index / 
            NORMALIZATION_CONSTANTS["iteration_index_max"]
        )
        
        # Number of models (normalized)
        features.append(
            len(plan.model_names) / 
            NORMALIZATION_CONSTANTS["model_count_max"]
        )
        
        # Consensus enabled (binary)
        features.append(float(plan.consensus_enabled))
        
        # Max iterations (normalized)
        features.append(
            plan.max_iterations / 
            NORMALIZATION_CONSTANTS["max_iterations_max"]
        )
        
        # Per-call timeout (normalized)
        features.append(
            plan.per_call_timeout_seconds / 
            NORMALIZATION_CONSTANTS["timeout_max_seconds"]
        )
        
        # Search enabled (binary)
        features.append(float(plan.search_enabled))
        
        # Available VRAM (normalized)
        features.append(
            state.available_vram_mb / 
            NORMALIZATION_CONSTANTS["vram_max_mb"]
        )
        
        # GPU load ratio (already 0-1)
        features.append(state.gpu_load_ratio)
        
        # Memory pressure ratio (already 0-1)
        features.append(state.memory_pressure_ratio)
        
        # Convert to numpy array
        feature_array = np.array(features, dtype=np.float32)
        
        # Validate dimension
        if len(feature_array) != self.feature_dim:
            logger.warning(
                f"Feature dimension mismatch: got {len(feature_array)}, "
                f"expected {self.feature_dim}"
            )
        
        return feature_array
    
    def _one_hot(self, value: str, categories: List[str]) -> List[float]:
        """
        Create one-hot encoding for a categorical value.
        
        Args:
            value: The value to encode
            categories: List of possible categories
            
        Returns:
            List of floats (0.0 or 1.0) of length len(categories)
        """
        encoding = [0.0] * len(categories)
        
        # Normalize value for matching
        value_lower = value.lower().strip()
        
        for i, cat in enumerate(categories):
            if cat.lower() == value_lower:
                encoding[i] = 1.0
                break
        else:
            # Value not found - log warning but don't fail
            logger.debug(f"Unknown category value: {value}, expected one of {categories}")
        
        return encoding
    
    def _multi_hot(self, values: List[str], categories: List[str]) -> List[float]:
        """
        Create multi-hot encoding for a list of categorical values.
        
        Args:
            values: List of values to encode
            categories: List of possible categories
            
        Returns:
            List of floats (0.0 or 1.0) of length len(categories)
        """
        encoding = [0.0] * len(categories)
        
        # Create lowercase lookup
        cat_lower_map = {cat.lower(): i for i, cat in enumerate(categories)}
        
        for value in values:
            value_lower = value.lower().strip()
            if value_lower in cat_lower_map:
                encoding[cat_lower_map[value_lower]] = 1.0
            else:
                logger.debug(f"Unknown multi-hot value: {value}")
        
        return encoding
    
    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.
        
        Useful for model interpretability and debugging.
        
        Returns:
            List of feature names in order
        """
        if self._feature_names is not None:
            return self._feature_names
        
        names: List[str] = []
        
        # One-hot: phase_type
        for pt in PHASE_TYPES:
            names.append(f"phase_type_{pt}")
        
        # One-hot: model_tier
        for mt in MODEL_TIERS:
            names.append(f"model_tier_{mt}")
        
        # One-hot: effort_level
        for el in EFFORT_LEVELS:
            names.append(f"effort_level_{el}")
        
        # Multi-hot: councils
        for council in KNOWN_COUNCILS:
            names.append(f"council_{council}")
        
        # Numeric features
        names.extend([
            "time_budget_normalized",
            "time_remaining_normalized",
            "time_remaining_ratio",
            "iteration_index_normalized",
            "model_count_normalized",
            "consensus_enabled",
            "max_iterations_normalized",
            "per_call_timeout_normalized",
            "search_enabled",
            "available_vram_normalized",
            "gpu_load_ratio",
            "memory_pressure_ratio",
        ])
        
        self._feature_names = names
        return names
    
    def decode_feature_importance(
        self,
        importances: np.ndarray
    ) -> List[tuple]:
        """
        Map feature importances to human-readable names.
        
        Args:
            importances: Array of feature importances from model
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        names = self.get_feature_names()
        
        if len(importances) != len(names):
            logger.warning(
                f"Importance array length {len(importances)} doesn't match "
                f"feature count {len(names)}"
            )
            return []
        
        pairs = list(zip(names, importances))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return pairs

