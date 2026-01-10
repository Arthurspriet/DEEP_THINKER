"""
HuggingFace Instruments Layer for DeepThinker.

Provides optional HF-based enhancements:
- Reranking: Cross-encoder reranking after vector retrieval
- Embeddings: HF embeddings with compatibility verification
- Claim Extraction: Multi-mode claim extraction with fallback

All features are flag-gated and fail gracefully when HF is unavailable.

Usage:
    from deepthinker.hf_instruments import get_reranker, get_config
    
    # Check if HF is available
    from deepthinker.hf_instruments.config import HF_AVAILABLE
    
    # Get reranker (returns None if not available/enabled)
    reranker = get_reranker()
    if reranker:
        results = reranker.rerank(query, passages)
    
    # Extract claims
    from deepthinker.hf_instruments import extract_claims
    result = extract_claims(text, source={"mission_id": "abc"})

Configuration via environment variables:
    DEEPTHINKER_HF_INSTRUMENTS_ENABLED=true
    DEEPTHINKER_HF_RERANKER_ENABLED=true
    DEEPTHINKER_HF_EMBEDDINGS_ENABLED=true
    DEEPTHINKER_HF_CLAIM_EXTRACTOR_ENABLED=true
    DEEPTHINKER_HF_DEVICE=auto
    DEEPTHINKER_HF_RERANK_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
    DEEPTHINKER_HF_EMBED_MODEL_ID=BAAI/bge-small-en-v1.5
"""

# Config
from .config import (
    HF_AVAILABLE,
    HFInstrumentsConfig,
    get_config,
    reload_config,
    set_config,
)

# Manager
from .manager import (
    HFInstrumentManager,
    get_instrument_manager,
    get_reranker,
    get_embedder,
)

# Meta utilities
from .meta import (
    IndexMeta,
    read_index_meta,
    write_index_meta,
    is_embedding_compatible,
    get_global_rag_meta,
    get_mission_rag_meta,
)

# Reranker
from .reranker import CrossEncoderReranker

# Embeddings
from .embeddings import HFEmbedder, CompatibleEmbedder

# Claim extraction
from .claim_extractor import (
    ExtractedClaim,
    ClaimExtractionResult,
    ClaimExtractionPipeline,
    get_claim_pipeline,
    extract_claims,
)


__all__ = [
    # Config
    "HF_AVAILABLE",
    "HFInstrumentsConfig",
    "get_config",
    "reload_config",
    "set_config",
    # Manager
    "HFInstrumentManager",
    "get_instrument_manager",
    "get_reranker",
    "get_embedder",
    # Meta
    "IndexMeta",
    "read_index_meta",
    "write_index_meta",
    "is_embedding_compatible",
    "get_global_rag_meta",
    "get_mission_rag_meta",
    # Reranker
    "CrossEncoderReranker",
    # Embeddings
    "HFEmbedder",
    "CompatibleEmbedder",
    # Claim extraction
    "ExtractedClaim",
    "ClaimExtractionResult",
    "ClaimExtractionPipeline",
    "get_claim_pipeline",
    "extract_claims",
]

