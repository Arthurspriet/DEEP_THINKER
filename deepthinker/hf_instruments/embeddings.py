"""
HuggingFace Embeddings for HF Instruments.

Provides HF-based embeddings with meta.json compatibility verification.

Constraint 1: HF embeddings must NEVER degrade retrieval.
- Always verify embedding_model_id + dimension + similarity_type + normalization
- HF_EMBEDDINGS_ENABLED is ignored unless meta confirms compatibility
- Log clear WARNING when skipping HF embeddings
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import get_config, HF_AVAILABLE
from .meta import IndexMeta, is_embedding_compatible, read_index_meta

logger = logging.getLogger(__name__)


class HFEmbedder:
    """
    HuggingFace-based embedder with compatibility verification.
    
    Always checks meta.json compatibility before use.
    Falls back gracefully if HF is unavailable or incompatible.
    
    Usage:
        embedder = HFEmbedder(model_id="BAAI/bge-small-en-v1.5")
        
        # Check compatibility before use
        if embedder.is_compatible_with_index(Path("kb/rag/global")):
            embedding = embedder.embed("query text")
    """
    
    def __init__(self, model_id: str, device: str = "cpu"):
        """
        Initialize the HF embedder.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to run on (cpu or cuda)
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._dimension: Optional[int] = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace not available, embedder will not work")
            return
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading embedding model: {self.model_id}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
            
            self.model.eval()
            
            # Get dimension from model config
            self._dimension = self.model.config.hidden_size
            
            self._loaded = True
            logger.info(f"Embedding model loaded on {self.device}, dim={self._dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._loaded = False
    
    @property
    def dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._dimension
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
    
    def is_compatible_with_index(self, index_dir: Path) -> bool:
        """
        Check if this embedder is compatible with an index.
        
        Constraint 1: HF embeddings must NEVER degrade retrieval.
        
        Args:
            index_dir: Directory containing the index (with meta.json)
            
        Returns:
            True if compatible, False otherwise (with WARNING logged)
        """
        if not self._loaded:
            logger.warning("Embedder not loaded - HF embeddings DISABLED")
            return False
        
        meta = read_index_meta(index_dir)
        return is_embedding_compatible(meta, self.model_id)
    
    def embed(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Embedding as numpy array, or None on failure
        """
        if not self._loaded:
            return None
        
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use CLS token or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output[0]
                else:
                    # Mean pooling over token embeddings
                    token_embeddings = outputs.last_hidden_state[0]
                    attention_mask = inputs['attention_mask'][0]
                    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 0)
                    sum_mask = torch.clamp(mask_expanded.sum(0), min=1e-9)
                    embedding = sum_embeddings / sum_mask
                
                embedding = embedding.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def embed_batch(
        self, 
        texts: List[str], 
        normalize: bool = True,
        batch_size: int = 32
    ) -> Optional[np.ndarray]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to L2-normalize the embeddings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_texts, dimension), or None on failure
        """
        if not self._loaded or not texts:
            return None
        
        try:
            import torch
            
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Use CLS token or mean pooling
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeddings = outputs.pooler_output
                    else:
                        # Mean pooling
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                    
                    embeddings = embeddings.cpu().numpy()
                
                # Normalize if requested
                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms > 0, norms, 1)
                    embeddings = embeddings / norms
                
                all_embeddings.append(embeddings)
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the embedder."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "loaded": self._loaded,
            "dimension": self._dimension,
        }


class CompatibleEmbedder:
    """
    Wrapper that provides HF embeddings only when compatible.
    
    Falls back to a provided fallback function when:
    - HF not available
    - Embeddings not enabled
    - Index metadata missing or incompatible
    
    Usage:
        def fallback_embed(text):
            return call_ollama_embeddings(text)
        
        embedder = CompatibleEmbedder(
            index_dir=Path("kb/rag/global"),
            fallback_fn=fallback_embed
        )
        
        embedding = embedder.embed("query text")  # Uses HF or fallback
    """
    
    def __init__(
        self,
        index_dir: Path,
        fallback_fn,
        hf_embedder: Optional[HFEmbedder] = None
    ):
        """
        Initialize the compatible embedder.
        
        Args:
            index_dir: Directory containing the index (with meta.json)
            fallback_fn: Fallback function for embeddings (text -> list[float])
            hf_embedder: Optional pre-loaded HF embedder
        """
        self.index_dir = index_dir
        self.fallback_fn = fallback_fn
        self._hf_embedder = hf_embedder
        self._use_hf: Optional[bool] = None
        self._checked = False
    
    def _check_compatibility(self) -> None:
        """Check HF compatibility with index."""
        if self._checked:
            return
        
        self._checked = True
        config = get_config()
        
        # Check if HF embeddings are enabled
        if not config.is_embeddings_active():
            logger.debug("HF embeddings not active, using fallback")
            self._use_hf = False
            return
        
        # Get or create HF embedder
        if self._hf_embedder is None:
            from .manager import get_embedder
            self._hf_embedder = get_embedder()
        
        if self._hf_embedder is None:
            logger.debug("HF embedder not available, using fallback")
            self._use_hf = False
            return
        
        # Check compatibility
        if self._hf_embedder.is_compatible_with_index(self.index_dir):
            logger.info(f"Using HF embeddings for {self.index_dir}")
            self._use_hf = True
        else:
            logger.debug("HF embeddings not compatible, using fallback")
            self._use_hf = False
    
    @property
    def using_hf(self) -> bool:
        """Check if using HF embeddings."""
        self._check_compatibility()
        return self._use_hf or False
    
    def embed(self, text: str) -> List[float]:
        """
        Embed text using HF (if compatible) or fallback.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as list of floats
        """
        self._check_compatibility()
        
        if self._use_hf and self._hf_embedder is not None:
            embedding = self._hf_embedder.embed(text)
            if embedding is not None:
                return embedding.tolist()
            # Fall through to fallback on HF failure
            logger.warning("HF embedding failed, using fallback")
        
        # Use fallback
        result = self.fallback_fn(text)
        return result if result else []


__all__ = [
    "HFEmbedder",
    "CompatibleEmbedder",
]

