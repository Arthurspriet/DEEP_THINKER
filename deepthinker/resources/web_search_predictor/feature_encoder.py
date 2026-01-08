"""
Feature Encoder for Web Search Predictor.

Provides deterministic, versioned encoding of prediction inputs
to feature vectors for ML models. Includes content analysis helpers
for detecting dates, named entities, and factual claims.
"""

import logging
import re
from typing import List, Optional

import numpy as np

from .schemas import WebSearchContext, WebSearchExecutionPlan, WebSearchSystemState
from .config import (
    PHASE_TYPES,
    MODEL_TIERS,
    EFFORT_LEVELS,
    KNOWN_COUNCILS,
    FEATURE_VECTOR_VERSION,
    NORMALIZATION_CONSTANTS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Content Analysis Patterns (for detecting search-relevant content)
# =============================================================================

# Date patterns: years, month names, date formats
DATE_PATTERNS = [
    r'\b(19|20)\d{2}\b',                          # Years: 1900-2099
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',        # Date formats: 12/31/2024
    r'\b(today|yesterday|tomorrow|last\s+week|next\s+month|this\s+year)\b',
]

# Named entity patterns: proper nouns, organization patterns
NAMED_ENTITY_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',       # Multi-word proper nouns
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-z]+\b', # Titles with names
    r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',           # Acronyms: NASA, FBI
    r'\b(?:Inc|Corp|LLC|Ltd|Co)\.\b',            # Company suffixes
    r'\b(?:University|Institute|Foundation|Company|Corporation)\s+of\b',
]

# Factual claim patterns: assertions requiring verification
FACTUAL_CLAIM_PATTERNS = [
    r'\b(?:is|are|was|were)\s+(?:the\s+)?(?:largest|smallest|first|oldest|newest)\b',
    r'\b(?:founded|established|created|invented)\s+(?:in|by)\b',
    r'\b(?:according\s+to|based\s+on|research\s+shows)\b',
    r'\b\d+(?:\.\d+)?\s*(?:percent|%|million|billion|trillion)\b',
    r'\b(?:currently|recently|officially|approximately)\b',
    r'\b(?:statistics|data|studies|research)\s+(?:show|indicate|suggest)\b',
]

# Compile patterns for efficiency
_DATE_REGEX = re.compile('|'.join(DATE_PATTERNS), re.IGNORECASE)
_ENTITY_REGEX = re.compile('|'.join(NAMED_ENTITY_PATTERNS))
_FACTUAL_REGEX = re.compile('|'.join(FACTUAL_CLAIM_PATTERNS), re.IGNORECASE)


def contains_dates(text: str) -> bool:
    """
    Check if text contains date references.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if date patterns are found
    """
    if not text:
        return False
    return bool(_DATE_REGEX.search(text))


def contains_named_entities(text: str) -> bool:
    """
    Check if text contains named entities (people, places, organizations).
    
    Args:
        text: Text to analyze
        
    Returns:
        True if named entity patterns are found
    """
    if not text:
        return False
    return bool(_ENTITY_REGEX.search(text))


def contains_factual_claims(text: str) -> bool:
    """
    Check if text contains factual claims requiring verification.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if factual claim patterns are found
    """
    if not text:
        return False
    return bool(_FACTUAL_REGEX.search(text))


def analyze_content(text: str) -> dict:
    """
    Analyze text for all search-relevant patterns.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with boolean flags for each pattern type
    """
    return {
        "contains_dates": contains_dates(text),
        "contains_named_entities": contains_named_entities(text),
        "contains_factual_claims": contains_factual_claims(text),
    }


class WebSearchFeatureEncoder:
    """
    Deterministic feature encoder for web search prediction.
    
    Encodes WebSearchContext, WebSearchExecutionPlan, and WebSearchSystemState
    into a fixed-size feature vector suitable for ML models.
    
    Feature layout (version 1):
    - One-hot: phase_type (6 features)
    - One-hot: effort_level (3 features)
    - One-hot: model_tier (4 features)
    - Multi-hot: councils_invoked (9 features)
    - Boolean: content flags (5 features)
    - Numeric: normalized scalar features (4 features)
    
    Total: 31 features
    """
    
    VERSION = FEATURE_VECTOR_VERSION
    
    # Number of boolean features
    NUM_BOOLEAN_FEATURES = 5
    
    # Number of numeric features (must match encode() implementation)
    NUM_NUMERIC_FEATURES = 4
    
    def __init__(self):
        """Initialize the feature encoder."""
        self._feature_names: Optional[List[str]] = None
    
    @property
    def feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        return (
            len(PHASE_TYPES) +           # one-hot phase_type
            len(EFFORT_LEVELS) +         # one-hot effort_level
            len(MODEL_TIERS) +           # one-hot model_tier
            len(KNOWN_COUNCILS) +        # multi-hot councils
            self.NUM_BOOLEAN_FEATURES +  # boolean flags
            self.NUM_NUMERIC_FEATURES    # numeric features
        )
    
    def encode(
        self,
        ctx: WebSearchContext,
        plan: WebSearchExecutionPlan,
        state: WebSearchSystemState
    ) -> np.ndarray:
        """
        Encode inputs into a feature vector.
        
        Args:
            ctx: Web search context information
            plan: Execution plan configuration
            state: Current system state
            
        Returns:
            np.ndarray of shape (feature_dim,) with dtype float32
        """
        features: List[float] = []
        
        # === One-hot encodings ===
        
        # Phase type (6 features)
        features.extend(self._one_hot(ctx.phase_type, PHASE_TYPES))
        
        # Effort level (3 features)
        features.extend(self._one_hot(ctx.effort_level, EFFORT_LEVELS))
        
        # Model tier (4 features)
        features.extend(self._one_hot(plan.model_tier, MODEL_TIERS))
        
        # === Multi-hot encoding ===
        
        # Councils invoked (9 features)
        features.extend(self._multi_hot(plan.councils_invoked, KNOWN_COUNCILS))
        
        # === Boolean features ===
        
        # Content analysis flags
        features.append(float(ctx.contains_dates))
        features.append(float(ctx.contains_named_entities))
        features.append(float(ctx.contains_factual_claims))
        
        # Plan flags
        features.append(float(plan.search_enabled_by_planner))
        features.append(float(plan.consensus_enabled))
        
        # === Numeric features (normalized) ===
        
        # Prompt token count (normalized)
        features.append(
            ctx.prompt_token_count / 
            NORMALIZATION_CONSTANTS["prompt_token_count_max"]
        )
        
        # Iteration index (normalized)
        features.append(
            ctx.iteration_index / 
            NORMALIZATION_CONSTANTS["iteration_index_max"]
        )
        
        # Max iterations (normalized)
        features.append(
            plan.max_iterations / 
            NORMALIZATION_CONSTANTS["max_iterations_max"]
        )
        
        # Available time (normalized)
        features.append(
            state.available_time_seconds / 
            NORMALIZATION_CONSTANTS["available_time_max_seconds"]
        )
        
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
        
        # One-hot: effort_level
        for el in EFFORT_LEVELS:
            names.append(f"effort_level_{el}")
        
        # One-hot: model_tier
        for mt in MODEL_TIERS:
            names.append(f"model_tier_{mt}")
        
        # Multi-hot: councils
        for council in KNOWN_COUNCILS:
            names.append(f"council_{council}")
        
        # Boolean features
        names.extend([
            "contains_dates",
            "contains_named_entities",
            "contains_factual_claims",
            "search_enabled_by_planner",
            "consensus_enabled",
        ])
        
        # Numeric features
        names.extend([
            "prompt_token_count_normalized",
            "iteration_index_normalized",
            "max_iterations_normalized",
            "available_time_normalized",
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

