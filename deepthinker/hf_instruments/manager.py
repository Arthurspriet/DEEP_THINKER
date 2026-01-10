"""
HuggingFace Instrument Manager.

Provides process-level model caching and device management for HF instruments.
Implements LRU eviction when cache exceeds configured limits.

Thread-safe and designed for graceful degradation when HF is unavailable.
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .config import get_config, HF_AVAILABLE

if TYPE_CHECKING:
    from .reranker import CrossEncoderReranker
    from .embeddings import HFEmbedder

logger = logging.getLogger(__name__)


class HFInstrumentManager:
    """
    Process-level manager for HuggingFace models.
    
    Provides:
    - Lazy loading of models on first use
    - LRU cache with configurable max size
    - Thread-safe access
    - Device management (auto/cpu/cuda)
    - Graceful degradation when HF unavailable
    
    Usage:
        manager = get_instrument_manager()
        reranker = manager.get_reranker()
        if reranker:
            results = reranker.rerank(query, passages)
    """
    
    def __init__(self):
        """Initialize the instrument manager."""
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._device: Optional[str] = None
        
    @property
    def device(self) -> str:
        """Get the resolved device for HF models."""
        if self._device is None:
            config = get_config()
            self._device = config.get_resolved_device()
            logger.info(f"[HFInstrumentManager] Using device: {self._device}")
        return self._device
    
    def _cache_key(self, model_type: str, model_id: str) -> str:
        """Generate cache key for a model."""
        return f"{model_type}:{model_id}:{self.device}"
    
    def _evict_if_needed(self) -> None:
        """Evict oldest models if cache exceeds limit."""
        config = get_config()
        while len(self._cache) >= config.cache_max_models:
            if not self._cache:
                break
            key, model = self._cache.popitem(last=False)
            logger.info(f"[HFInstrumentManager] Evicted model from cache: {key}")
            # Attempt cleanup
            try:
                del model
            except Exception:
                pass
    
    def get_reranker(self) -> Optional["CrossEncoderReranker"]:
        """
        Get the cross-encoder reranker instance.
        
        Returns None if:
        - HF not available
        - Reranker not enabled in config
        - Model loading fails
        
        Returns:
            CrossEncoderReranker instance or None
        """
        config = get_config()
        
        if not config.is_reranker_active():
            return None
        
        cache_key = self._cache_key("reranker", config.rerank_model_id)
        
        with self._lock:
            # Check cache
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            
            # Load model
            try:
                from .reranker import CrossEncoderReranker
                
                self._evict_if_needed()
                
                logger.info(f"[HFInstrumentManager] Loading reranker: {config.rerank_model_id}")
                reranker = CrossEncoderReranker(
                    model_id=config.rerank_model_id,
                    device=self.device,
                )
                
                self._cache[cache_key] = reranker
                logger.info(f"[HFInstrumentManager] Reranker loaded successfully")
                return reranker
                
            except Exception as e:
                logger.warning(f"[HFInstrumentManager] Failed to load reranker: {e}")
                return None
    
    def get_embedder(self) -> Optional["HFEmbedder"]:
        """
        Get the HF embedder instance.
        
        Returns None if:
        - HF not available
        - Embeddings not enabled in config
        - Model loading fails
        
        Note: Even if this returns an embedder, it may still be incompatible
        with the index (dimension mismatch). Use check_embedding_compatibility()
        before actually using the embedder.
        
        Returns:
            HFEmbedder instance or None
        """
        config = get_config()
        
        if not config.is_embeddings_active():
            return None
        
        cache_key = self._cache_key("embedder", config.embed_model_id)
        
        with self._lock:
            # Check cache
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            
            # Load model
            try:
                from .embeddings import HFEmbedder
                
                self._evict_if_needed()
                
                logger.info(f"[HFInstrumentManager] Loading embedder: {config.embed_model_id}")
                embedder = HFEmbedder(
                    model_id=config.embed_model_id,
                    device=self.device,
                )
                
                self._cache[cache_key] = embedder
                logger.info(f"[HFInstrumentManager] Embedder loaded successfully")
                return embedder
                
            except Exception as e:
                logger.warning(f"[HFInstrumentManager] Failed to load embedder: {e}")
                return None
    
    def get_cached_models(self) -> Dict[str, str]:
        """Get list of currently cached models."""
        with self._lock:
            return {key: type(model).__name__ for key, model in self._cache.items()}
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            logger.info("[HFInstrumentManager] Cache cleared")
    
    def preload_reranker(self) -> bool:
        """
        Preload the reranker model.
        
        Returns:
            True if reranker was loaded successfully
        """
        return self.get_reranker() is not None
    
    def preload_embedder(self) -> bool:
        """
        Preload the embedder model.
        
        Returns:
            True if embedder was loaded successfully
        """
        return self.get_embedder() is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the manager.
        
        Returns:
            Dictionary with status information
        """
        config = get_config()
        
        return {
            "hf_available": HF_AVAILABLE,
            "enabled": config.enabled,
            "device": self.device,
            "cached_models": self.get_cached_models(),
            "cache_max_models": config.cache_max_models,
            "reranker_active": config.is_reranker_active(),
            "embeddings_active": config.is_embeddings_active(),
            "claim_extractor_active": config.is_claim_extractor_active(),
        }


# Module-level singleton
_manager_instance: Optional[HFInstrumentManager] = None
_manager_lock = threading.Lock()


def get_instrument_manager() -> HFInstrumentManager:
    """
    Get the global HF instrument manager instance.
    
    Returns:
        HFInstrumentManager singleton
    """
    global _manager_instance
    
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = HFInstrumentManager()
    
    return _manager_instance


def get_reranker() -> Optional["CrossEncoderReranker"]:
    """
    Convenience function to get the reranker.
    
    Returns:
        CrossEncoderReranker instance or None
    """
    return get_instrument_manager().get_reranker()


def get_embedder() -> Optional["HFEmbedder"]:
    """
    Convenience function to get the embedder.
    
    Returns:
        HFEmbedder instance or None
    """
    return get_instrument_manager().get_embedder()


__all__ = [
    "HFInstrumentManager",
    "get_instrument_manager",
    "get_reranker",
    "get_embedder",
]

