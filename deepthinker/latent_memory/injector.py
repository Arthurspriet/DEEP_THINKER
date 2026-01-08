"""
Injector for converting retrieved memory tokens into prefix embeddings.

Converts latent mission memories into prefix embeddings that can be
injected into model generation (when using HuggingFace models directly).
"""

import logging
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LatentInjector:
    """
    Converts retrieved memory tokens into prefix embeddings.
    
    Simply concatenates all memory tokens without any learned projections.
    """
    
    def make_prefix(self, memories: List[np.ndarray]) -> torch.Tensor:
        """
        Create prefix embeddings from retrieved memories.
        
        Args:
            memories: List of memory token arrays, each of shape
                     (MEMORY_TOKENS_PER_DOC, hidden_size)
        
        Returns:
            Tensor of shape (total_memory_tokens, hidden_size) as float16
            Returns empty tensor if no memories provided
        """
        if not memories:
            return torch.empty(0, 0, dtype=torch.float16)
        
        try:
            # Concatenate all memory tokens
            all_tokens = np.concatenate(memories, axis=0)  # Shape: (total_tokens, hidden_size)
            
            # Convert to torch tensor (float16)
            prefix = torch.from_numpy(all_tokens.astype(np.float16))
            
            logger.debug(f"Created prefix embeddings of shape {prefix.shape}")
            
            return prefix
            
        except Exception as e:
            logger.error(f"Failed to create prefix embeddings: {e}")
            return torch.empty(0, 0, dtype=torch.float16)



