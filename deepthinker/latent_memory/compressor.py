"""
Latent Compressor for converting text to latent memory tokens.

Uses HuggingFace Transformers to compress documents into fixed-size
latent representations. No training. No gradients.
"""

import logging
import numpy as np
import torch
from typing import Any, Optional

from transformers import AutoModel, AutoTokenizer

from .config import (
    MODEL_NAME,
    DEVICE,
    DTYPE,
    MAX_TOKENS_DOC,
    MAX_TOKENS_QUERY,
    MEMORY_TOKENS_PER_DOC,
)

logger = logging.getLogger(__name__)


class LatentCompressor:
    """
    Compresses text documents into latent memory tokens.
    
    Uses last hidden state from a HuggingFace model, chunk-pooled
    into a fixed number of memory tokens.
    """
    
    def __init__(self):
        """Initialize the compressor with model and tokenizer."""
        try:
            logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DTYPE == "float16" else torch.float32,
                device_map=DEVICE if DEVICE != "cpu" else None,
            )
            
            if DEVICE == "cpu":
                self.model = self.model.to(DEVICE)
            
            self.model.eval()  # No training, no gradients
            
            # Get hidden size
            self.hidden_size = self.model.config.hidden_size
            
            logger.info(f"Model loaded. Hidden size: {self.hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model {MODEL_NAME}: {e}")
            raise
    
    def compress_document(self, text: str) -> np.ndarray:
        """
        Compress a document into memory tokens.
        
        Args:
            text: Document text to compress
            
        Returns:
            Array of shape (MEMORY_TOKENS_PER_DOC, hidden_size) as float16
        """
        if not text or not text.strip():
            # Return zero-filled array if empty
            return np.zeros((MEMORY_TOKENS_PER_DOC, self.hidden_size), dtype=np.float16)
        
        try:
            # Tokenize with truncation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS_DOC,
                padding=False,
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get last hidden layer (index -1)
            hidden_states = outputs.hidden_states[-1]  # Shape: (batch, seq_len, hidden_size)
            hidden_states = hidden_states.squeeze(0)  # Remove batch dim: (seq_len, hidden_size)
            
            seq_len = hidden_states.shape[0]
            
            # Split tokens evenly into MEMORY_TOKENS_PER_DOC chunks
            if seq_len <= MEMORY_TOKENS_PER_DOC:
                # If fewer tokens than memory tokens, pad with mean
                chunks = [hidden_states[i:i+1] if i < seq_len else hidden_states.mean(dim=0, keepdim=True) 
                         for i in range(MEMORY_TOKENS_PER_DOC)]
            else:
                # Split evenly
                chunk_size = seq_len // MEMORY_TOKENS_PER_DOC
                chunks = []
                for i in range(MEMORY_TOKENS_PER_DOC):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < MEMORY_TOKENS_PER_DOC - 1 else seq_len
                    chunk = hidden_states[start_idx:end_idx]
                    chunks.append(chunk)
            
            # Mean-pool each chunk
            memory_tokens = []
            for chunk in chunks:
                pooled = chunk.mean(dim=0)  # Mean over sequence dimension
                memory_tokens.append(pooled.cpu().numpy())
            
            # Stack and convert to float16
            result = np.array(memory_tokens, dtype=np.float16)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compress document: {e}")
            # Return zero-filled array on error
            return np.zeros((MEMORY_TOKENS_PER_DOC, self.hidden_size), dtype=np.float16)
    
    def compress_document_with_logging(
        self,
        text: str,
        mission_id: str = "",
        phase_id: str = "",
        constitution_ledger: Optional["Any"] = None,
    ) -> np.ndarray:
        """
        Compress a document and log to constitution ledger.
        
        Args:
            text: Document text to compress
            mission_id: Mission identifier for logging
            phase_id: Phase identifier for logging
            constitution_ledger: Optional ConstitutionLedger for event logging
            
        Returns:
            Array of shape (MEMORY_TOKENS_PER_DOC, hidden_size) as float16
        """
        # Measure input size
        size_before = len(text) if text else 0
        
        # Perform compression
        result = self.compress_document(text)
        
        # Measure output size (in float16 elements)
        size_after = result.size * 2  # float16 = 2 bytes
        
        # Log to constitution ledger if provided
        if constitution_ledger is not None:
            try:
                from ..constitution.types import CompressionEvent
                constitution_ledger.write_event(CompressionEvent(
                    mission_id=mission_id,
                    phase_id=phase_id,
                    method="latent_compression",
                    size_before=size_before,
                    size_after=size_after,
                    # Note: uncertainty tracking would require additional analysis
                    uncertainty_before=0.0,
                    uncertainty_after=0.0,
                    validated=False,
                ))
            except Exception as e:
                logger.debug(f"[COMPRESSOR] Constitution ledger write failed: {e}")
        
        return result
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query text into a single vector.
        
        Args:
            text: Query text to embed
            
        Returns:
            Array of shape (hidden_size,) as float16
        """
        if not text or not text.strip():
            return np.zeros(self.hidden_size, dtype=np.float16)
        
        try:
            # Tokenize with truncation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS_QUERY,
                padding=False,
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get last hidden layer
            hidden_states = outputs.hidden_states[-1]  # Shape: (batch, seq_len, hidden_size)
            hidden_states = hidden_states.squeeze(0)  # Remove batch dim: (seq_len, hidden_size)
            
            # Mean-pool over all tokens
            query_vec = hidden_states.mean(dim=0).cpu().numpy()
            
            return query_vec.astype(np.float16)
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return np.zeros(self.hidden_size, dtype=np.float16)



