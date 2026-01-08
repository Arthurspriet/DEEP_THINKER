"""
RAG Store for DeepThinker Memory System.

Provides per-mission and global RAG indexing using the same Ollama
embedding backend as consensus/voting.py (MajorityVoteConsensus).

Storage paths:
- Per-mission: kb/missions/<mission_id>/rag/
- Global: kb/rag/global/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import numpy as np

from .schemas import EvidenceSchema

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Embedding provider using Ollama API.
    
    Reuses the same pattern as MajorityVoteConsensus._get_embedding()
    for consistency across the codebase.
    """
    
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434",
        cache_size: int = 500,
    ):
        """
        Initialize embedding provider.
        
        Args:
            embedding_model: Ollama model for embeddings
            ollama_base_url: Ollama server URL
            cache_size: Maximum cache entries
        """
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text, using cache if available.
        
        Uses centralized model_caller for proper resource management.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or empty list on failure
        """
        from deepthinker.models.model_caller import call_embeddings
        
        # Truncate for cache key and embedding
        text_truncated = text[:2000]
        cache_key = hashlib.md5(text_truncated.encode()).hexdigest()
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Use centralized model_caller for proper resource management
        embedding = call_embeddings(
            text=text_truncated,
            model=self.embedding_model,
            timeout=60.0,
            max_retries=2,
            base_url=self.ollama_base_url,
        )
        
        if embedding:
            # Cache management
            if len(self._cache) >= self._cache_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self._cache.keys())[:100]
                for k in keys_to_remove:
                    del self._cache[k]
            
            self._cache[cache_key] = embedding
        
        return embedding
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class MissionRAGStore:
    """
    Per-mission RAG store for evidence and text retrieval.
    
    Stores evidence with embeddings for semantic search within a mission.
    
    Storage: kb/missions/<mission_id>/rag/
    - index.npy: NumPy array of embeddings
    - documents.json: Document metadata and text
    """
    
    def __init__(
        self,
        mission_id: str,
        base_dir: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize mission RAG store.
        
        Args:
            mission_id: Mission identifier
            base_dir: Base directory for storage
            embedding_fn: Optional custom embedding function
            embedding_model: Ollama embedding model (if embedding_fn not provided)
            ollama_base_url: Ollama server URL
        """
        self.mission_id = mission_id
        self.base_dir = base_dir or Path("kb")
        self._rag_dir = self.base_dir / "missions" / mission_id / "rag"
        
        # Embedding provider
        if embedding_fn:
            self._get_embedding = embedding_fn
        else:
            self._provider = EmbeddingProvider(
                embedding_model=embedding_model,
                ollama_base_url=ollama_base_url,
            )
            self._get_embedding = self._provider.get_embedding
        
        # In-memory storage
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._id_counter = 0
        
        # Load existing data
        self._load()
    
    def add_evidence(self, evidence: EvidenceSchema) -> str:
        """
        Add evidence to the RAG store.
        
        Args:
            evidence: Evidence schema with text and metadata
            
        Returns:
            Evidence ID
        """
        doc_id = evidence.id or self._generate_id()
        
        # Get embedding
        embedding = self._get_embedding(evidence.text)
        if not embedding:
            logger.warning(f"Failed to get embedding for evidence {doc_id}")
            # Still store the document, just without embedding
        
        # Store document
        doc = {
            "id": doc_id,
            "text": evidence.text,
            "mission_id": evidence.mission_id,
            "phase": evidence.phase,
            "artifact_type": evidence.artifact_type,
            "hypothesis_id": evidence.hypothesis_id,
            "tags": evidence.tags,
            "confidence": evidence.confidence,
            "source": evidence.source,
            "created_at": evidence.created_at.isoformat() if evidence.created_at else datetime.utcnow().isoformat(),
            "type": "evidence",
        }
        
        self._add_document(doc, embedding)
        return doc_id
    
    def add_text(
        self,
        text: str,
        phase: str,
        artifact_type: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add arbitrary text to the RAG store.
        
        Args:
            text: Text content
            phase: Phase name
            artifact_type: Type of artifact
            tags: Optional tags for filtering
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = self._generate_id()
        
        # Get embedding
        embedding = self._get_embedding(text)
        
        # Store document
        doc = {
            "id": doc_id,
            "text": text,
            "mission_id": self.mission_id,
            "phase": phase,
            "artifact_type": artifact_type,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
            "type": "text",
            **(metadata or {}),
        }
        
        self._add_document(doc, embedding)
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: int = 6,
        phase_filter: Optional[str] = None,
        artifact_type_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            top_k: Maximum results to return
            phase_filter: Optional phase filter
            artifact_type_filter: Optional artifact type filter
            min_score: Minimum similarity score
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not self._documents or self._embeddings is None or self._embeddings.size == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Compute similarities
        results = []
        for i, doc in enumerate(self._documents):
            # Apply filters
            if phase_filter and doc.get("phase") != phase_filter:
                continue
            if artifact_type_filter and doc.get("artifact_type") != artifact_type_filter:
                continue
            
            # Get embedding for this document
            if i >= len(self._embeddings):
                continue
            
            doc_embedding = self._embeddings[i]
            if doc_embedding.size == 0:
                continue
            
            score = cosine_similarity(query_vec, doc_embedding)
            
            if score >= min_score:
                results.append((doc, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the store."""
        return self._documents.copy()
    
    def get_document_count(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)
    
    def persist(self) -> bool:
        """
        Persist store to disk.
        
        Returns:
            True if successful
        """
        try:
            self._rag_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            docs_path = self._rag_dir / "documents.json"
            with open(docs_path, "w") as f:
                json.dump(self._documents, f, indent=2)
            
            # Save embeddings
            if self._embeddings is not None and self._embeddings.size > 0:
                index_path = self._rag_dir / "index.npy"
                np.save(index_path, self._embeddings)
            
            logger.debug(f"Persisted RAG store for mission {self.mission_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to persist RAG store: {e}")
            return False
    
    def _load(self) -> None:
        """Load store from disk if exists."""
        try:
            docs_path = self._rag_dir / "documents.json"
            index_path = self._rag_dir / "index.npy"
            
            if docs_path.exists():
                with open(docs_path, "r") as f:
                    self._documents = json.load(f)
                
                # Update ID counter
                for doc in self._documents:
                    doc_id = doc.get("id", "")
                    if doc_id.startswith("doc_"):
                        try:
                            num = int(doc_id.split("_")[1])
                            self._id_counter = max(self._id_counter, num)
                        except (ValueError, IndexError):
                            pass
            
            if index_path.exists():
                self._embeddings = np.load(index_path)
            else:
                self._embeddings = np.array([])
                
        except Exception as e:
            logger.warning(f"Failed to load RAG store: {e}")
            self._documents = []
            self._embeddings = np.array([])
    
    def _add_document(self, doc: Dict[str, Any], embedding: List[float]) -> None:
        """Add document with embedding to store."""
        self._documents.append(doc)
        
        if embedding:
            emb_array = np.array(embedding).reshape(1, -1)
            if self._embeddings is None or self._embeddings.size == 0:
                self._embeddings = emb_array
            else:
                # Ensure dimensions match
                if self._embeddings.shape[1] == emb_array.shape[1]:
                    self._embeddings = np.vstack([self._embeddings, emb_array])
                else:
                    logger.warning(f"Embedding dimension mismatch: {self._embeddings.shape[1]} vs {emb_array.shape[1]}")
        else:
            # Add zero embedding as placeholder
            if self._embeddings is not None and self._embeddings.size > 0:
                zero_emb = np.zeros((1, self._embeddings.shape[1]))
                self._embeddings = np.vstack([self._embeddings, zero_emb])
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        self._id_counter += 1
        return f"doc_{self._id_counter}_{self.mission_id[:8]}"


class GlobalRAGStore:
    """
    Global RAG store aggregating knowledge across all missions.
    
    Stores key insights and evidence from completed missions for
    cross-mission retrieval and pattern recognition.
    
    Storage: kb/rag/global/
    - index.npy: NumPy array of embeddings
    - documents.json: Document metadata and text
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize global RAG store.
        
        Args:
            base_dir: Base directory for storage
            embedding_fn: Optional custom embedding function
            embedding_model: Ollama embedding model
            ollama_base_url: Ollama server URL
        """
        self.base_dir = base_dir or Path("kb")
        self._rag_dir = self.base_dir / "rag" / "global"
        
        # Embedding provider
        if embedding_fn:
            self._get_embedding = embedding_fn
        else:
            self._provider = EmbeddingProvider(
                embedding_model=embedding_model,
                ollama_base_url=ollama_base_url,
            )
            self._get_embedding = self._provider.get_embedding
        
        # In-memory storage
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._id_counter = 0
        
        # Load existing data
        self._load()
    
    def add_from_mission(
        self,
        mission_rag: MissionRAGStore,
        min_confidence: float = 0.5,
        max_documents: int = 50,
    ) -> int:
        """
        Add relevant documents from a mission RAG store.
        
        Args:
            mission_rag: Mission RAG store to import from
            min_confidence: Minimum confidence threshold
            max_documents: Maximum documents to import
            
        Returns:
            Number of documents added
        """
        added = 0
        docs = mission_rag.get_all_documents()
        
        # Sort by confidence if available
        docs_with_conf = [
            (d, d.get("confidence", 0.5)) for d in docs
        ]
        docs_with_conf.sort(key=lambda x: x[1], reverse=True)
        
        for doc, conf in docs_with_conf[:max_documents]:
            if conf < min_confidence:
                continue
            
            # Skip if already exists (by text hash)
            text_hash = hashlib.md5(doc.get("text", "").encode()).hexdigest()
            if any(
                hashlib.md5(d.get("text", "").encode()).hexdigest() == text_hash
                for d in self._documents
            ):
                continue
            
            # Add to global store
            self.add_document(
                text=doc.get("text", ""),
                mission_id=doc.get("mission_id", ""),
                phase=doc.get("phase", ""),
                artifact_type=doc.get("artifact_type", "general"),
                tags=doc.get("tags", []),
                confidence=conf,
                source_type=doc.get("type", "evidence"),
            )
            added += 1
        
        return added
    
    def add_document(
        self,
        text: str,
        mission_id: str,
        phase: str = "",
        artifact_type: str = "general",
        tags: Optional[List[str]] = None,
        confidence: float = 0.5,
        source_type: str = "evidence",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a document to the global store.
        
        Args:
            text: Document text
            mission_id: Source mission ID
            phase: Source phase
            artifact_type: Type of artifact
            tags: Tags for filtering
            confidence: Confidence score
            source_type: Type of source document
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = self._generate_id()
        
        # Get embedding
        embedding = self._get_embedding(text)
        
        # Store document
        doc = {
            "id": doc_id,
            "text": text,
            "mission_id": mission_id,
            "phase": phase,
            "artifact_type": artifact_type,
            "tags": tags or [],
            "confidence": confidence,
            "source_type": source_type,
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        
        self._add_document_with_embedding(doc, embedding)
        return doc_id
    
    def add_insight(
        self,
        insight: str,
        mission_id: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Add a key insight to the global store.
        
        Args:
            insight: Insight text
            mission_id: Source mission
            domain: Optional domain classification
            tags: Tags for filtering
            
        Returns:
            Document ID
        """
        return self.add_document(
            text=insight,
            mission_id=mission_id,
            artifact_type="insight",
            tags=tags or [],
            confidence=0.8,
            source_type="insight",
            metadata={"domain": domain} if domain else None,
        )
    
    def search_global(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        artifact_type_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents globally.
        
        Args:
            query: Query text
            top_k: Maximum results
            domain_filter: Optional domain filter
            artifact_type_filter: Optional artifact type filter
            min_score: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        if not self._documents or self._embeddings is None or self._embeddings.size == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Compute similarities
        results = []
        for i, doc in enumerate(self._documents):
            # Apply filters
            if domain_filter and doc.get("domain") != domain_filter:
                continue
            if artifact_type_filter and doc.get("artifact_type") != artifact_type_filter:
                continue
            
            if i >= len(self._embeddings):
                continue
            
            doc_embedding = self._embeddings[i]
            if doc_embedding.size == 0:
                continue
            
            score = cosine_similarity(query_vec, doc_embedding)
            
            if score >= min_score:
                results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_by_mission(self, mission_id: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific mission."""
        return [d for d in self._documents if d.get("mission_id") == mission_id]
    
    def rebuild_index(self) -> bool:
        """
        Rebuild embeddings index from documents.
        
        Useful if embedding model changed or embeddings were corrupted.
        
        Returns:
            True if successful
        """
        try:
            new_embeddings = []
            
            for doc in self._documents:
                text = doc.get("text", "")
                embedding = self._get_embedding(text)
                if embedding:
                    new_embeddings.append(embedding)
                else:
                    # Use zero vector as placeholder
                    if new_embeddings:
                        new_embeddings.append([0.0] * len(new_embeddings[0]))
            
            if new_embeddings:
                self._embeddings = np.array(new_embeddings)
            else:
                self._embeddings = np.array([])
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to rebuild index: {e}")
            return False
    
    def persist(self) -> bool:
        """
        Persist store to disk.
        
        Returns:
            True if successful
        """
        try:
            self._rag_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            docs_path = self._rag_dir / "documents.json"
            with open(docs_path, "w") as f:
                json.dump(self._documents, f, indent=2)
            
            # Save embeddings
            if self._embeddings is not None and self._embeddings.size > 0:
                index_path = self._rag_dir / "index.npy"
                np.save(index_path, self._embeddings)
            
            logger.debug("Persisted global RAG store")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to persist global RAG store: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)
    
    def _load(self) -> None:
        """Load store from disk if exists."""
        try:
            docs_path = self._rag_dir / "documents.json"
            index_path = self._rag_dir / "index.npy"
            
            if docs_path.exists():
                with open(docs_path, "r") as f:
                    self._documents = json.load(f)
                
                # Update ID counter
                for doc in self._documents:
                    doc_id = doc.get("id", "")
                    if doc_id.startswith("global_"):
                        try:
                            num = int(doc_id.split("_")[1])
                            self._id_counter = max(self._id_counter, num)
                        except (ValueError, IndexError):
                            pass
            
            if index_path.exists():
                self._embeddings = np.load(index_path)
            else:
                self._embeddings = np.array([])
                
        except Exception as e:
            logger.warning(f"Failed to load global RAG store: {e}")
            self._documents = []
            self._embeddings = np.array([])
    
    def _add_document_with_embedding(self, doc: Dict[str, Any], embedding: List[float]) -> None:
        """Add document with embedding to store."""
        self._documents.append(doc)
        
        if embedding:
            emb_array = np.array(embedding).reshape(1, -1)
            if self._embeddings is None or self._embeddings.size == 0:
                self._embeddings = emb_array
            else:
                if self._embeddings.shape[1] == emb_array.shape[1]:
                    self._embeddings = np.vstack([self._embeddings, emb_array])
                else:
                    logger.warning(f"Embedding dimension mismatch in global store")
        else:
            if self._embeddings is not None and self._embeddings.size > 0:
                zero_emb = np.zeros((1, self._embeddings.shape[1]))
                self._embeddings = np.vstack([self._embeddings, zero_emb])
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        self._id_counter += 1
        return f"global_{self._id_counter}"

