"""
General Knowledge Store for DeepThinker Memory System.

Provides RAG-based retrieval for static reference data like the CIA World Factbook.
Unlike mission-specific stores, this contains persistent factual knowledge that
can be queried across all missions.

Storage path: kb/general_knowledge/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import numpy as np

from .rag_store import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)


# Category groupings for CIA Factbook data
CIA_CATEGORIES = {
    "introduction": ["Introduction"],
    "geography": ["Geography"],
    "people": ["People and Society"],
    "environment": ["Environment"],
    "government": ["Government"],
    "economy": ["Economy"],
    "energy": ["Energy"],
    "communications": ["Communications"],
    "transportation": ["Transportation"],
    "military": ["Military and Security"],
    "terrorism": ["Terrorism"],
    "transnational": ["Transnational Issues"],
}


def _get_category(field_name: str) -> str:
    """Extract category from a field name like 'Geography: Location'."""
    if ":" in field_name:
        prefix = field_name.split(":")[0].strip()
        for cat_key, prefixes in CIA_CATEGORIES.items():
            if prefix in prefixes:
                return cat_key
    return "other"


def _chunk_country_data(country_name: str, country_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk country data by category for optimal embedding.
    
    Each chunk contains all fields for one category, creating a coherent
    document that can be embedded and retrieved semantically.
    
    Args:
        country_name: Name of the country
        country_data: Dictionary of all fields for this country
        
    Returns:
        List of chunk dictionaries with text, metadata
    """
    chunks = []
    category_fields: Dict[str, List[Tuple[str, str]]] = {}
    
    # Group fields by category
    for field_name, value in country_data.items():
        if field_name == "url":
            continue
        if not isinstance(value, str) or not value.strip():
            continue
            
        category = _get_category(field_name)
        if category not in category_fields:
            category_fields[category] = []
        category_fields[category].append((field_name, value))
    
    # Create chunks for each category
    for category, fields in category_fields.items():
        if not fields:
            continue
            
        # Build chunk text
        text_parts = [f"Country: {country_name}", f"Category: {category.title()}", ""]
        
        for field_name, value in fields:
            # Clean field name (remove category prefix)
            clean_name = field_name
            if ":" in field_name:
                clean_name = field_name.split(":", 1)[1].strip()
            
            # Truncate very long values
            if len(value) > 1500:
                value = value[:1500] + "..."
                
            text_parts.append(f"{clean_name}: {value}")
        
        chunk_text = "\n".join(text_parts)
        
        # Create chunk with metadata
        chunk = {
            "id": f"cia_{country_name.lower().replace(' ', '_')}_{category}",
            "text": chunk_text,
            "country": country_name,
            "category": category,
            "source": "cia_world_factbook",
            "field_count": len(fields),
            "created_at": datetime.utcnow().isoformat(),
        }
        chunks.append(chunk)
    
    return chunks


class GeneralKnowledgeStore:
    """
    RAG store for static general knowledge data.
    
    Designed for reference data like the CIA World Factbook and OWID datasets
    that provides factual information about countries, geography, economics, etc.
    
    Unlike mission-specific stores, this data is:
    - Static (loaded once, updated infrequently)
    - Shared across all missions
    - Searchable by semantic similarity and category/source filters
    
    Storage: kb/general_knowledge/
    - index.npy: NumPy array of embeddings
    - documents.json: Document metadata and text
    
    Supported sources:
    - cia_world_factbook: Country-level factual data
    - owid: Our World in Data datasets (economics, health, environment, etc.)
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize general knowledge store.
        
        Args:
            base_dir: Base directory for storage (default: kb/)
            embedding_fn: Optional custom embedding function
            embedding_model: Ollama embedding model
            ollama_base_url: Ollama server URL
        """
        self.base_dir = base_dir or Path("kb")
        self._store_dir = self.base_dir / "general_knowledge"
        
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
        self._country_index: Dict[str, List[int]] = {}  # country -> doc indices
        self._category_index: Dict[str, List[int]] = {}  # category -> doc indices
        self._source_index: Dict[str, List[int]] = {}  # source (owid, cia) -> doc indices
        self._dataset_index: Dict[str, List[int]] = {}  # dataset_name -> doc indices (for OWID)
        
        # Load existing data
        self._load()
    
    def load_cia_facts(
        self,
        json_path: Path,
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Load and index CIA World Factbook data.
        
        Parses the JSON file, chunks by country+category, generates embeddings,
        and persists to disk.
        
        Args:
            json_path: Path to countries.json
            batch_size: Number of chunks to embed before saving checkpoint
            progress_callback: Optional callback(processed, total) for progress
            
        Returns:
            Number of documents indexed
        """
        logger.info(f"Loading CIA Factbook from {json_path}")
        
        # Load JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create all chunks
        all_chunks = []
        for country_name, country_data in data.items():
            chunks = _chunk_country_data(country_name, country_data)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(data)} countries")
        
        # Clear existing data (but preserve OWID data if present)
        owid_docs = [d for d in self._documents if d.get("source") == "owid"]
        owid_indices = self._source_index.get("owid", [])
        
        self._documents = []
        self._embeddings = None
        self._country_index = {}
        self._category_index = {}
        self._source_index = {}
        self._dataset_index = {}
        
        # Process chunks with embedding
        embeddings_list = []
        for i, chunk in enumerate(all_chunks):
            # Get embedding
            embedding = self._get_embedding(chunk["text"])
            
            if embedding:
                embeddings_list.append(embedding)
            else:
                # Use zero vector as placeholder
                if embeddings_list:
                    embeddings_list.append([0.0] * len(embeddings_list[0]))
                else:
                    logger.warning(f"Failed to get embedding for chunk {i}")
                    continue
            
            # Add document
            doc_idx = len(self._documents)
            self._documents.append(chunk)
            
            # Update indices
            country = chunk["country"]
            category = chunk["category"]
            source = chunk.get("source", "cia_world_factbook")
            
            if country not in self._country_index:
                self._country_index[country] = []
            self._country_index[country].append(doc_idx)
            
            if category not in self._category_index:
                self._category_index[category] = []
            self._category_index[category].append(doc_idx)
            
            if source not in self._source_index:
                self._source_index[source] = []
            self._source_index[source].append(doc_idx)
            
            # Progress callback
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(all_chunks))
            
            # Checkpoint save
            if (i + 1) % batch_size == 0:
                self._embeddings = np.array(embeddings_list)
                self.persist()
                logger.debug(f"Checkpoint: {i + 1}/{len(all_chunks)} chunks indexed")
        
        # Final save
        if embeddings_list:
            self._embeddings = np.array(embeddings_list)
        self.persist()
        
        logger.info(f"Indexed {len(self._documents)} documents from CIA Factbook")
        return len(self._documents)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        country_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            top_k: Maximum results to return
            country_filter: Optional country name filter (CIA data)
            category_filter: Optional category filter (geography, economy, environment, etc.)
            source_filter: Optional source filter ("owid", "cia_world_factbook")
            dataset_filter: Optional OWID dataset name filter
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
        
        # Determine candidate documents based on filters
        candidate_sets = []
        
        if source_filter:
            candidate_sets.append(set(self._source_index.get(source_filter, [])))
        
        if country_filter:
            candidate_sets.append(set(self._country_index.get(country_filter, [])))
        
        if category_filter:
            candidate_sets.append(set(self._category_index.get(category_filter, [])))
        
        if dataset_filter:
            candidate_sets.append(set(self._dataset_index.get(dataset_filter, [])))
        
        # Compute intersection of all filters
        if candidate_sets:
            candidate_indices = list(set.intersection(*candidate_sets))
        else:
            candidate_indices = list(range(len(self._documents)))
        
        # Compute similarities
        results = []
        for idx in candidate_indices:
            if idx >= len(self._embeddings):
                continue
            
            doc_embedding = self._embeddings[idx]
            if doc_embedding.size == 0:
                continue
            
            score = cosine_similarity(query_vec, doc_embedding)
            
            if score >= min_score:
                results.append((self._documents[idx], score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def search_owid(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search specifically in OWID datasets.
        
        Args:
            query: Search query text
            top_k: Maximum results to return
            category_filter: Optional category (environment, health, economy, etc.)
            min_score: Minimum similarity score
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        return self.search(
            query=query,
            top_k=top_k,
            source_filter="owid",
            category_filter=category_filter,
            min_score=min_score,
        )
    
    def get_country_info(
        self,
        country_name: str,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all information for a specific country.
        
        Args:
            country_name: Country name (case-sensitive)
            categories: Optional list of categories to filter
            
        Returns:
            List of document chunks for the country
        """
        if country_name not in self._country_index:
            # Try case-insensitive lookup
            for stored_country in self._country_index:
                if stored_country.lower() == country_name.lower():
                    country_name = stored_country
                    break
            else:
                return []
        
        indices = self._country_index[country_name]
        docs = [self._documents[i] for i in indices]
        
        if categories:
            docs = [d for d in docs if d.get("category") in categories]
        
        return docs
    
    def get_dataset_info(
        self,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific OWID dataset.
        
        Args:
            dataset_name: Dataset name (case-sensitive)
            
        Returns:
            List of document chunks for the dataset
        """
        if dataset_name not in self._dataset_index:
            # Try case-insensitive lookup
            for stored_name in self._dataset_index:
                if stored_name.lower() == dataset_name.lower():
                    dataset_name = stored_name
                    break
            else:
                return []
        
        indices = self._dataset_index[dataset_name]
        return [self._documents[i] for i in indices]
    
    def list_countries(self) -> List[str]:
        """Get list of all indexed countries."""
        return sorted(self._country_index.keys())
    
    def list_categories(self) -> List[str]:
        """Get list of all indexed categories."""
        return sorted(self._category_index.keys())
    
    def list_sources(self) -> List[str]:
        """Get list of all data sources (owid, cia_world_factbook, etc.)."""
        return sorted(self._source_index.keys())
    
    def list_datasets(self, source: Optional[str] = None) -> List[str]:
        """
        Get list of all indexed datasets.
        
        Args:
            source: Optional source filter (e.g., "owid")
            
        Returns:
            List of dataset names
        """
        if source:
            # Filter datasets by source
            source_docs = set(self._source_index.get(source, []))
            datasets = set()
            for name, indices in self._dataset_index.items():
                if any(idx in source_docs for idx in indices):
                    datasets.add(name)
            return sorted(datasets)
        return sorted(self._dataset_index.keys())
    
    def get_document_count(self) -> int:
        """Get total number of indexed documents."""
        return len(self._documents)
    
    def get_document_count_by_source(self, source: str) -> int:
        """Get number of documents from a specific source."""
        return len(self._source_index.get(source, []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        stats = {
            "total_documents": len(self._documents),
            "countries_count": len(self._country_index),
            "categories_count": len(self._category_index),
            "sources_count": len(self._source_index),
            "datasets_count": len(self._dataset_index),
            "embedding_dimensions": self._embeddings.shape[1] if self._embeddings is not None and self._embeddings.size > 0 else 0,
            "storage_path": str(self._store_dir),
        }
        
        # Add per-source counts
        for source, indices in self._source_index.items():
            stats[f"documents_{source}"] = len(indices)
        
        return stats
    
    def persist(self) -> bool:
        """
        Persist store to disk.
        
        Returns:
            True if successful
        """
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents with all indices
            docs_data = {
                "documents": self._documents,
                "country_index": self._country_index,
                "category_index": self._category_index,
                "source_index": self._source_index,
                "dataset_index": self._dataset_index,
            }
            docs_path = self._store_dir / "documents.json"
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self._embeddings is not None and self._embeddings.size > 0:
                index_path = self._store_dir / "index.npy"
                np.save(index_path, self._embeddings)
            
            logger.debug("Persisted general knowledge store")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to persist general knowledge store: {e}")
            return False
    
    def _load(self) -> None:
        """Load store from disk if exists."""
        try:
            docs_path = self._store_dir / "documents.json"
            index_path = self._store_dir / "index.npy"
            
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self._documents = data.get("documents", [])
                self._country_index = data.get("country_index", {})
                self._category_index = data.get("category_index", {})
                self._source_index = data.get("source_index", {})
                self._dataset_index = data.get("dataset_index", {})
                
                logger.debug(f"Loaded {len(self._documents)} documents from general knowledge store")
            
            if index_path.exists():
                self._embeddings = np.load(index_path)
            else:
                self._embeddings = np.array([])
                
        except Exception as e:
            logger.warning(f"Failed to load general knowledge store: {e}")
            self._documents = []
            self._embeddings = np.array([])
            self._country_index = {}
            self._category_index = {}
            self._source_index = {}
            self._dataset_index = {}
    
    def is_loaded(self) -> bool:
        """Check if store has data loaded."""
        return len(self._documents) > 0 and self._embeddings is not None and self._embeddings.size > 0

