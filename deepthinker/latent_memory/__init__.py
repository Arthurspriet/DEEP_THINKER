"""
Latent Mission Memory Module for DeepThinker.

Provides read-only latent memory system that:
- Reads past mission final outputs
- Compresses them to latent vectors using HuggingFace Transformers
- Stores them in FAISS index
- Retrieves similar past missions
- Optionally injects retrieved memories as prefix embeddings

Disabled by default. Must be explicitly enabled and manually built.
"""

from .config import (
    LATENT_MEMORY_ENABLED,
    MODEL_NAME,
    DEVICE,
    DTYPE,
    MAX_TOKENS_DOC,
    MAX_TOKENS_QUERY,
    MEMORY_TOKENS_PER_DOC,
    TOP_K_RETRIEVAL,
    INDEX_PATH,
    STORE_PATH,
)

from .compressor import LatentCompressor
from .index import LatentIndex
from .retriever import LatentMissionRetriever, LatentMission
from .injector import LatentInjector
from .builder import (
    build_latent_memory_from_missions,
    build_decision_fingerprints,
    load_decision_fingerprints,
    DecisionFingerprint,
)

__all__ = [
    "LATENT_MEMORY_ENABLED",
    "MODEL_NAME",
    "DEVICE",
    "DTYPE",
    "MAX_TOKENS_DOC",
    "MAX_TOKENS_QUERY",
    "MEMORY_TOKENS_PER_DOC",
    "TOP_K_RETRIEVAL",
    "INDEX_PATH",
    "STORE_PATH",
    "LatentCompressor",
    "LatentIndex",
    "LatentMissionRetriever",
    "LatentMission",
    "LatentInjector",
    "build_latent_memory_from_missions",
    # Decision Accountability Layer
    "build_decision_fingerprints",
    "load_decision_fingerprints",
    "DecisionFingerprint",
]



