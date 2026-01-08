"""
FAISS Index for storing and searching latent mission memories.

Stores mean vectors in FAISS for fast ANN search, and full memory tokens
in a pickle store for retrieval.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import faiss

from .config import INDEX_PATH, STORE_PATH

logger = logging.getLogger(__name__)


class LatentIndex:
    """
    FAISS index for latent mission memories.
    
    Stores:
    - FAISS index: mean of memory tokens per document (for search)
    - Pickle store: full memory tokens + metadata (for retrieval)
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the index.
        
        Args:
            dimension: Dimension of the embedding vectors (hidden_size)
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.store: Dict[str, Dict] = {}  # doc_id -> {memory_tokens, metadata}
        self._is_built = False
    
    def add(self, doc_id: str, mem_tokens: np.ndarray, metadata: Dict) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier (mission_id)
            mem_tokens: Memory tokens array of shape (MEMORY_TOKENS_PER_DOC, hidden_size)
            metadata: Metadata dict (mission_id, objective, created_at, status, etc.)
        """
        if self._is_built:
            raise RuntimeError("Cannot add documents after index is built. Create a new index.")
        
        # Compute mean vector for FAISS index
        mean_vec = mem_tokens.mean(axis=0).astype(np.float32)  # FAISS requires float32
        
        # Store full memory tokens and metadata
        self.store[doc_id] = {
            "memory_tokens": mem_tokens.astype(np.float16),  # Keep as float16 for storage
            "metadata": metadata,
        }
        
        # Initialize index if needed
        if self.index is None:
            # Use inner product (cosine similarity after normalization)
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add mean vector to index (will normalize later for cosine similarity)
        # Reshape to (1, dimension) for FAISS
        mean_vec = mean_vec.reshape(1, -1)
        # Normalize for cosine similarity
        faiss.normalize_L2(mean_vec)
        self.index.add(mean_vec)
    
    def build(self) -> None:
        """
        Build the index. Call after adding all documents.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, nothing to build")
            return
        
        self._is_built = True
        logger.info(f"Index built with {self.index.ntotal} documents")
    
    def search(self, query_vec: np.ndarray, k: int) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_vec: Query vector of shape (hidden_size,)
            k: Number of results to return
            
        Returns:
            List of dicts with keys: {doc_id, memory_tokens, metadata, similarity_score}
        """
        if self.index is None or not self._is_built:
            logger.warning("Index not built, returning empty results")
            return []
        
        if len(self.store) == 0:
            return []
        
        try:
            # Normalize query vector for cosine similarity
            query_vec = query_vec.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vec)
            
            # Search
            k = min(k, self.index.ntotal)  # Don't ask for more than available
            distances, indices = self.index.search(query_vec, k)
            
            # Get results
            results = []
            doc_ids = list(self.store.keys())
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # FAISS returns -1 for invalid indices
                    continue
                
                doc_id = doc_ids[idx]
                stored = self.store[doc_id]
                
                results.append({
                    "doc_id": doc_id,
                    "memory_tokens": stored["memory_tokens"],
                    "metadata": stored["metadata"],
                    "similarity_score": float(distance),  # Cosine similarity (higher is better)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def save(self, index_path: Optional[Path] = None, store_path: Optional[Path] = None) -> None:
        """
        Save index and store to disk.
        
        Args:
            index_path: Path to save FAISS index (default: INDEX_PATH)
            store_path: Path to save pickle store (default: STORE_PATH)
        """
        index_path = Path(index_path or INDEX_PATH)
        store_path = Path(store_path or STORE_PATH)
        
        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(index_path))
                logger.info(f"Saved FAISS index to {index_path}")
            
            # Save pickle store
            with open(store_path, "wb") as f:
                pickle.dump(self.store, f)
            logger.info(f"Saved store to {store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    @classmethod
    def load(cls, dimension: int, index_path: Optional[Path] = None, store_path: Optional[Path] = None) -> "LatentIndex":
        """
        Load index and store from disk.
        
        Args:
            dimension: Dimension of embedding vectors
            index_path: Path to FAISS index (default: INDEX_PATH)
            store_path: Path to pickle store (default: STORE_PATH)
            
        Returns:
            Loaded LatentIndex instance
        """
        index_path = Path(index_path or INDEX_PATH)
        store_path = Path(store_path or STORE_PATH)
        
        instance = cls(dimension)
        
        try:
            # Load FAISS index
            if index_path.exists():
                instance.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index from {index_path} ({instance.index.ntotal} vectors)")
            else:
                logger.warning(f"Index file not found: {index_path}")
                return instance
            
            # Load pickle store
            if store_path.exists():
                with open(store_path, "rb") as f:
                    instance.store = pickle.load(f)
                logger.info(f"Loaded store from {store_path} ({len(instance.store)} documents)")
            else:
                logger.warning(f"Store file not found: {store_path}")
                return instance
            
            instance._is_built = True
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise



