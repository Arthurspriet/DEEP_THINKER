"""
Retriever for latent mission memories.

Retrieves similar past missions based on query text.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import (
    LATENT_MEMORY_ENABLED,
    TOP_K_RETRIEVAL,
    INDEX_PATH,
    STORE_PATH,
)
from .compressor import LatentCompressor
from .index import LatentIndex

logger = logging.getLogger(__name__)


@dataclass
class LatentMission:
    """
    A retrieved latent mission memory.
    """
    mission_id: str
    objective: str
    memory_tokens: np.ndarray  # Shape: (MEMORY_TOKENS_PER_DOC, hidden_size)
    similarity_score: float


class LatentMissionRetriever:
    """
    Retrieves similar past missions based on query text.
    
    If LATENT_MEMORY_ENABLED is False, always returns empty list.
    """
    
    def __init__(self):
        """Initialize the retriever."""
        self.compressor: Optional[LatentCompressor] = None
        self.index: Optional[LatentIndex] = None
        
        if not LATENT_MEMORY_ENABLED:
            logger.debug("Latent memory is disabled, retriever will return empty results")
            return
        
        try:
            # Initialize compressor
            self.compressor = LatentCompressor()
            
            # Try to load existing index
            index_path = Path(INDEX_PATH)
            store_path = Path(STORE_PATH)
            
            if index_path.exists() and store_path.exists():
                self.index = LatentIndex.load(
                    dimension=self.compressor.hidden_size,
                    index_path=index_path,
                    store_path=store_path,
                )
                logger.info(f"Loaded latent memory index with {len(self.index.store)} missions")
            else:
                logger.warning(
                    f"Index not found at {index_path} or {store_path}. "
                    "Use build_latent_memory_from_missions() to build the index first."
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            # Continue with None index - retrieve() will return empty list
    
    def retrieve(self, query: str) -> List[LatentMission]:
        """
        Retrieve similar past missions.
        
        Args:
            query: Query text (e.g., mission objective or question)
            
        Returns:
            List of LatentMission objects, sorted by similarity (highest first)
            Returns empty list if disabled or index not available
        """
        if not LATENT_MEMORY_ENABLED:
            return []
        
        if self.compressor is None or self.index is None:
            return []
        
        if not query or not query.strip():
            return []
        
        try:
            # Embed query
            query_vec = self.compressor.embed_query(query)
            
            # Search index
            results = self.index.search(query_vec, k=TOP_K_RETRIEVAL)
            
            # Convert to LatentMission objects
            missions = []
            for result in results:
                missions.append(LatentMission(
                    mission_id=result["doc_id"],
                    objective=result["metadata"].get("objective", "Unknown"),
                    memory_tokens=result["memory_tokens"],
                    similarity_score=result["similarity_score"],
                ))
            
            logger.debug(f"Retrieved {len(missions)} similar missions for query")
            
            return missions
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []



