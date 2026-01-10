"""
Configuration for HuggingFace Instruments Layer.

Provides feature flags and model configuration for:
- HF Embeddings (optional, requires meta.json compatibility)
- HF Reranker (cross-encoder reranking)
- Claim Extraction (multi-mode with fallback)

All settings are configurable via environment variables.
All feature flags default to OFF for backward compatibility.
Master switch (HF_INSTRUMENTS_ENABLED) must be True for any HF feature to activate.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Check if HuggingFace dependencies are available
try:
    import torch
    from transformers import AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.debug("HuggingFace dependencies not available (torch/transformers)")


@dataclass
class HFInstrumentsConfig:
    """
    Configuration for HuggingFace Instruments.
    
    All settings are configurable via environment variables with prefix DEEPTHINKER_HF_.
    All feature flags default to OFF for backward compatibility.
    
    Attributes:
        enabled: Master switch for all HF instruments
        embeddings_enabled: Use HF for query embeddings (requires meta.json match)
        reranker_enabled: Rerank after retrieval (if master enabled)
        claim_extractor_enabled: Extract claims (if master enabled)
        
        device: Device for HF models (auto|cpu|cuda)
        cache_max_models: Maximum models to keep in memory
        
        embed_model_id: HuggingFace model ID for embeddings
        rerank_model_id: HuggingFace model ID for cross-encoder reranking
        
        rerank_topn: Number of candidates to rerank
        rerank_max_length: Max tokens per query+passage for reranking
        rerank_batch_size: Batch size for reranking
        rerank_timeout_seconds: Timeout per batch for reranking
        
        claim_extractor_mode: Extraction mode (regex|hf|llm-json)
    """
    
    # Master switches
    enabled: bool = False
    embeddings_enabled: bool = False
    reranker_enabled: bool = True  # Enabled by default IF master is enabled
    claim_extractor_enabled: bool = True  # Enabled by default IF master is enabled
    
    # Device & cache
    device: str = "auto"  # auto|cpu|cuda
    cache_max_models: int = 2
    
    # Model IDs
    embed_model_id: str = "BAAI/bge-small-en-v1.5"
    rerank_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Reranker latency safety (Constraint 4)
    rerank_topn: int = 20
    rerank_max_length: int = 512
    rerank_batch_size: int = 8
    rerank_timeout_seconds: float = 5.0
    
    # Claim extraction (Constraint 3)
    claim_extractor_mode: str = "regex"  # regex | hf | llm-json
    
    def __post_init__(self):
        """Validate configuration."""
        if self.device not in ("auto", "cpu", "cuda"):
            logger.warning(f"Invalid device '{self.device}', defaulting to 'auto'")
            self.device = "auto"
        
        if self.claim_extractor_mode not in ("regex", "hf", "llm-json"):
            logger.warning(f"Invalid claim_extractor_mode '{self.claim_extractor_mode}', defaulting to 'regex'")
            self.claim_extractor_mode = "regex"
        
        if self.cache_max_models < 1:
            self.cache_max_models = 1
        
        if self.rerank_topn < 1:
            self.rerank_topn = 1
        
        if self.rerank_batch_size < 1:
            self.rerank_batch_size = 1
    
    def is_reranker_active(self) -> bool:
        """Check if reranker should be active."""
        return (
            HF_AVAILABLE and 
            self.enabled and 
            self.reranker_enabled
        )
    
    def is_embeddings_active(self) -> bool:
        """Check if HF embeddings should be active (still requires meta.json check)."""
        return (
            HF_AVAILABLE and 
            self.enabled and 
            self.embeddings_enabled
        )
    
    def is_claim_extractor_active(self) -> bool:
        """Check if claim extractor should be active."""
        return (
            self.enabled and 
            self.claim_extractor_enabled
        )
    
    def get_resolved_device(self) -> str:
        """Get the resolved device (handles 'auto')."""
        if self.device != "auto":
            return self.device
        
        if not HF_AVAILABLE:
            return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        
        return "cpu"


def _parse_bool(value: str) -> bool:
    """Parse a boolean from environment variable string."""
    return value.lower() in ("true", "yes", "1", "on")


def _parse_float(value: str, default: float) -> float:
    """Parse a float from environment variable string."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_int(value: str, default: int) -> int:
    """Parse an int from environment variable string."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def load_config_from_environment() -> HFInstrumentsConfig:
    """
    Load HF instruments configuration from environment variables.
    
    Environment variables (prefix DEEPTHINKER_HF_):
        DEEPTHINKER_HF_INSTRUMENTS_ENABLED
        DEEPTHINKER_HF_EMBEDDINGS_ENABLED
        DEEPTHINKER_HF_RERANKER_ENABLED
        DEEPTHINKER_HF_CLAIM_EXTRACTOR_ENABLED
        DEEPTHINKER_HF_DEVICE
        DEEPTHINKER_HF_CACHE_MAX_MODELS
        DEEPTHINKER_HF_EMBED_MODEL_ID
        DEEPTHINKER_HF_RERANK_MODEL_ID
        DEEPTHINKER_HF_RERANK_TOPN
        DEEPTHINKER_HF_RERANK_MAX_LENGTH
        DEEPTHINKER_HF_RERANK_BATCH_SIZE
        DEEPTHINKER_HF_RERANK_TIMEOUT_SECONDS
        DEEPTHINKER_HF_CLAIM_EXTRACTOR_MODE
    
    Returns:
        HFInstrumentsConfig with values from environment
    """
    config = HFInstrumentsConfig()
    
    # Master switches
    if os.environ.get("DEEPTHINKER_HF_INSTRUMENTS_ENABLED"):
        config.enabled = _parse_bool(os.environ["DEEPTHINKER_HF_INSTRUMENTS_ENABLED"])
    
    if os.environ.get("DEEPTHINKER_HF_EMBEDDINGS_ENABLED"):
        config.embeddings_enabled = _parse_bool(os.environ["DEEPTHINKER_HF_EMBEDDINGS_ENABLED"])
    
    if os.environ.get("DEEPTHINKER_HF_RERANKER_ENABLED"):
        config.reranker_enabled = _parse_bool(os.environ["DEEPTHINKER_HF_RERANKER_ENABLED"])
    
    if os.environ.get("DEEPTHINKER_HF_CLAIM_EXTRACTOR_ENABLED"):
        config.claim_extractor_enabled = _parse_bool(os.environ["DEEPTHINKER_HF_CLAIM_EXTRACTOR_ENABLED"])
    
    # Device & cache
    if os.environ.get("DEEPTHINKER_HF_DEVICE"):
        config.device = os.environ["DEEPTHINKER_HF_DEVICE"]
    
    if os.environ.get("DEEPTHINKER_HF_CACHE_MAX_MODELS"):
        config.cache_max_models = _parse_int(
            os.environ["DEEPTHINKER_HF_CACHE_MAX_MODELS"], 
            config.cache_max_models
        )
    
    # Model IDs
    if os.environ.get("DEEPTHINKER_HF_EMBED_MODEL_ID"):
        config.embed_model_id = os.environ["DEEPTHINKER_HF_EMBED_MODEL_ID"]
    
    if os.environ.get("DEEPTHINKER_HF_RERANK_MODEL_ID"):
        config.rerank_model_id = os.environ["DEEPTHINKER_HF_RERANK_MODEL_ID"]
    
    # Reranker settings
    if os.environ.get("DEEPTHINKER_HF_RERANK_TOPN"):
        config.rerank_topn = _parse_int(
            os.environ["DEEPTHINKER_HF_RERANK_TOPN"],
            config.rerank_topn
        )
    
    if os.environ.get("DEEPTHINKER_HF_RERANK_MAX_LENGTH"):
        config.rerank_max_length = _parse_int(
            os.environ["DEEPTHINKER_HF_RERANK_MAX_LENGTH"],
            config.rerank_max_length
        )
    
    if os.environ.get("DEEPTHINKER_HF_RERANK_BATCH_SIZE"):
        config.rerank_batch_size = _parse_int(
            os.environ["DEEPTHINKER_HF_RERANK_BATCH_SIZE"],
            config.rerank_batch_size
        )
    
    if os.environ.get("DEEPTHINKER_HF_RERANK_TIMEOUT_SECONDS"):
        config.rerank_timeout_seconds = _parse_float(
            os.environ["DEEPTHINKER_HF_RERANK_TIMEOUT_SECONDS"],
            config.rerank_timeout_seconds
        )
    
    # Claim extraction
    if os.environ.get("DEEPTHINKER_HF_CLAIM_EXTRACTOR_MODE"):
        config.claim_extractor_mode = os.environ["DEEPTHINKER_HF_CLAIM_EXTRACTOR_MODE"]
    
    # Run post-init validation
    config.__post_init__()
    
    return config


# Global configuration instance (loaded from environment)
_config: Optional[HFInstrumentsConfig] = None


def get_config() -> HFInstrumentsConfig:
    """Get the global HF instruments configuration."""
    global _config
    if _config is None:
        _config = load_config_from_environment()
    return _config


def reload_config() -> HFInstrumentsConfig:
    """Reload configuration from environment."""
    global _config
    _config = load_config_from_environment()
    return _config


def set_config(config: HFInstrumentsConfig) -> None:
    """Set the global configuration (for testing)."""
    global _config
    _config = config


__all__ = [
    "HF_AVAILABLE",
    "HFInstrumentsConfig",
    "get_config",
    "reload_config",
    "set_config",
    "load_config_from_environment",
]

