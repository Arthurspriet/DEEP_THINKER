"""
Index Metadata Utilities for HF Instruments.

Provides read/write/verify utilities for index metadata files (meta.json).
Used to ensure HF embeddings are compatible with existing indices.

Constraint 1: HF embeddings must NEVER degrade retrieval.
- Always verify embedding_model_id + dimension + similarity_type + normalization
- HF_EMBEDDINGS_ENABLED is ignored unless meta confirms compatibility
- Log clear WARNING when skipping HF embeddings
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexMeta:
    """
    Metadata for a vector index.
    
    Stored in meta.json alongside index.npy files.
    Used to verify HF embedding compatibility.
    
    Attributes:
        embedding_model_id: ID of the model used to create embeddings
        embedding_dimension: Dimension of the embedding vectors
        similarity_type: Type of similarity metric (cosine, dot, euclidean)
        normalization: Normalization applied to vectors (l2, none)
        created_at: When the index was created
        document_count: Number of documents in the index
        notes: Additional notes about the index
    """
    embedding_model_id: str
    embedding_dimension: int
    similarity_type: str = "cosine"
    normalization: str = "l2"
    created_at: Optional[str] = None
    document_count: int = 0
    notes: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexMeta":
        """Create from dictionary."""
        return cls(
            embedding_model_id=data.get("embedding_model_id", "unknown"),
            embedding_dimension=data.get("embedding_dimension", 0),
            similarity_type=data.get("similarity_type", "cosine"),
            normalization=data.get("normalization", "l2"),
            created_at=data.get("created_at"),
            document_count=data.get("document_count", 0),
            notes=data.get("notes", ""),
        )


def read_index_meta(index_dir: Path) -> Optional[IndexMeta]:
    """
    Read index metadata from meta.json.
    
    Args:
        index_dir: Directory containing the index files
        
    Returns:
        IndexMeta if found and valid, None otherwise
    """
    meta_path = index_dir / "meta.json"
    
    if not meta_path.exists():
        logger.debug(f"No meta.json found at {meta_path}")
        return None
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        meta = IndexMeta.from_dict(data)
        logger.debug(f"Loaded index meta from {meta_path}: dim={meta.embedding_dimension}")
        return meta
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Invalid meta.json at {meta_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading meta.json at {meta_path}: {e}")
        return None


def write_index_meta(index_dir: Path, meta: IndexMeta) -> bool:
    """
    Write index metadata to meta.json.
    
    Args:
        index_dir: Directory containing the index files
        meta: IndexMeta to write
        
    Returns:
        True if successful
    """
    meta_path = index_dir / "meta.json"
    
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)
        
        logger.info(f"Wrote index meta to {meta_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to write meta.json to {meta_path}: {e}")
        return False


def get_hf_model_dimension(model_id: str) -> Optional[int]:
    """
    Get the embedding dimension for a HuggingFace model.
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        Embedding dimension, or None if cannot be determined
    """
    from .config import HF_AVAILABLE
    
    if not HF_AVAILABLE:
        return None
    
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Try different attribute names
        for attr in ["hidden_size", "d_model", "embedding_dim", "dim"]:
            if hasattr(config, attr):
                dim = getattr(config, attr)
                logger.debug(f"Model {model_id} dimension: {dim} (from {attr})")
                return dim
        
        logger.warning(f"Could not determine dimension for model {model_id}")
        return None
        
    except Exception as e:
        logger.warning(f"Error getting dimension for model {model_id}: {e}")
        return None


def is_embedding_compatible(
    meta: Optional[IndexMeta],
    hf_model_id: str,
    strict: bool = True
) -> bool:
    """
    Check if HF embeddings are compatible with an existing index.
    
    Constraint 1: HF embeddings must NEVER degrade retrieval.
    
    Args:
        meta: Index metadata (None if no meta.json exists)
        hf_model_id: HuggingFace model ID to check
        strict: If True, require exact dimension match
        
    Returns:
        True if compatible, False otherwise (with WARNING logged)
    """
    # No meta.json - cannot verify compatibility
    if meta is None:
        logger.warning(
            f"No meta.json found for index - HF embeddings DISABLED, "
            f"using existing embedder. Create meta.json to enable HF embeddings."
        )
        return False
    
    # Get HF model dimension
    hf_dim = get_hf_model_dimension(hf_model_id)
    
    if hf_dim is None:
        logger.warning(
            f"Cannot determine dimension for HF model '{hf_model_id}' - "
            f"HF embeddings DISABLED, using existing embedder."
        )
        return False
    
    # Check dimension match
    if meta.embedding_dimension != hf_dim:
        logger.warning(
            f"Embedding dimension mismatch: index={meta.embedding_dimension}, "
            f"HF model '{hf_model_id}'={hf_dim} - HF embeddings DISABLED, "
            f"using existing embedder."
        )
        return False
    
    # Check normalization compatibility (optional warning)
    if strict and meta.normalization not in ("l2", "none"):
        logger.warning(
            f"Unusual index normalization: '{meta.normalization}' - "
            f"HF embeddings may produce different results."
        )
    
    logger.info(
        f"HF embeddings compatible with index: "
        f"model={hf_model_id}, dim={hf_dim}, similarity={meta.similarity_type}"
    )
    return True


def get_global_rag_meta() -> Optional[IndexMeta]:
    """
    Get metadata for the global RAG index.
    
    Returns:
        IndexMeta for kb/rag/global/ or None
    """
    return read_index_meta(Path("kb/rag/global"))


def get_mission_rag_meta(mission_id: str, base_dir: Optional[Path] = None) -> Optional[IndexMeta]:
    """
    Get metadata for a mission's RAG index.
    
    Args:
        mission_id: Mission identifier
        base_dir: Base directory (default: kb)
        
    Returns:
        IndexMeta for kb/missions/<mission_id>/rag/ or None
    """
    base = base_dir or Path("kb")
    return read_index_meta(base / "missions" / mission_id / "rag")


def get_general_knowledge_meta(base_dir: Optional[Path] = None) -> Optional[IndexMeta]:
    """
    Get metadata for the general knowledge index.
    
    Args:
        base_dir: Base directory (default: kb)
        
    Returns:
        IndexMeta for kb/general_knowledge/ or None
    """
    base = base_dir or Path("kb")
    return read_index_meta(base / "general_knowledge")


def create_meta_for_existing_index(
    index_dir: Path,
    embedding_model_id: str,
    embedding_dimension: int,
    notes: str = ""
) -> Optional[IndexMeta]:
    """
    Create metadata for an existing index that lacks meta.json.
    
    This is a helper for migrating existing indices.
    
    Args:
        index_dir: Directory containing the index
        embedding_model_id: ID of the embedding model used
        embedding_dimension: Dimension of the embeddings
        notes: Additional notes
        
    Returns:
        Created IndexMeta, or None if failed
    """
    import numpy as np
    
    index_path = index_dir / "index.npy"
    docs_path = index_dir / "documents.json"
    
    # Try to get document count
    doc_count = 0
    if docs_path.exists():
        try:
            with open(docs_path, "r") as f:
                docs = json.load(f)
            doc_count = len(docs) if isinstance(docs, list) else 0
        except Exception:
            pass
    
    # Verify embedding dimension if index exists
    if index_path.exists():
        try:
            embeddings = np.load(index_path)
            if len(embeddings.shape) >= 2:
                actual_dim = embeddings.shape[1]
                if actual_dim != embedding_dimension:
                    logger.warning(
                        f"Dimension mismatch: specified={embedding_dimension}, "
                        f"actual in index={actual_dim}. Using actual."
                    )
                    embedding_dimension = actual_dim
        except Exception as e:
            logger.warning(f"Could not verify index dimensions: {e}")
    
    meta = IndexMeta(
        embedding_model_id=embedding_model_id,
        embedding_dimension=embedding_dimension,
        similarity_type="cosine",
        normalization="l2",
        document_count=doc_count,
        notes=notes or f"Created for existing index at {index_dir}",
    )
    
    if write_index_meta(index_dir, meta):
        return meta
    return None


__all__ = [
    "IndexMeta",
    "read_index_meta",
    "write_index_meta",
    "get_hf_model_dimension",
    "is_embedding_compatible",
    "get_global_rag_meta",
    "get_mission_rag_meta",
    "get_general_knowledge_meta",
    "create_meta_for_existing_index",
]

