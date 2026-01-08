"""
Summary Memory for DeepThinker Memory System.

Stores compressed summaries of completed missions for cross-mission
intelligence and pattern recognition.

Storage: kb/long_memory/summaries.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schemas import MissionSummarySchema

logger = logging.getLogger(__name__)


class SummaryMemory:
    """
    Long-term memory for mission summaries.
    
    Provides:
    - Storage of compressed mission summaries
    - Retrieval by recency, domain, keywords
    - Semantic search over past missions
    - Pattern detection across missions
    
    Storage: kb/long_memory/summaries.json
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        max_summaries: int = 1000,
    ):
        """
        Initialize summary memory.
        
        Args:
            base_dir: Base directory for storage
            embedding_fn: Optional embedding function for semantic search
            max_summaries: Maximum summaries to store (FIFO eviction)
        """
        self.base_dir = base_dir or Path("kb")
        self._storage_path = self.base_dir / "long_memory" / "summaries.json"
        self._embedding_fn = embedding_fn
        self._max_summaries = max_summaries
        
        # In-memory storage
        self._summaries: List[MissionSummarySchema] = []
        self._embeddings: Dict[str, List[float]] = {}  # mission_id -> embedding
        
        # Load existing data
        self._load()
    
    def add_summary(self, summary: MissionSummarySchema) -> bool:
        """
        Add a mission summary to long-term memory.
        
        Args:
            summary: Mission summary schema
            
        Returns:
            True if added successfully
        """
        try:
            # Check for duplicates
            existing_ids = {s.mission_id for s in self._summaries}
            if summary.mission_id in existing_ids:
                # Update existing summary
                self._summaries = [
                    s for s in self._summaries if s.mission_id != summary.mission_id
                ]
            
            # Add summary
            self._summaries.append(summary)
            
            # Generate embedding for semantic search
            if self._embedding_fn:
                try:
                    search_text = summary.to_search_text()
                    embedding = self._embedding_fn(search_text)
                    if embedding:
                        self._embeddings[summary.mission_id] = embedding
                except Exception as e:
                    logger.warning(f"Failed to generate summary embedding: {e}")
            
            # Evict old summaries if needed
            if len(self._summaries) > self._max_summaries:
                # Remove oldest summaries (by created_at)
                self._summaries.sort(key=lambda s: s.created_at or datetime.min)
                evict_count = len(self._summaries) - self._max_summaries
                evicted = self._summaries[:evict_count]
                self._summaries = self._summaries[evict_count:]
                
                # Remove embeddings for evicted summaries
                for s in evicted:
                    self._embeddings.pop(s.mission_id, None)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to add summary: {e}")
            return False
    
    def get_recent(self, n: int = 10) -> List[MissionSummarySchema]:
        """
        Get the most recent mission summaries.
        
        Args:
            n: Number of summaries to return
            
        Returns:
            List of most recent summaries
        """
        # Sort by created_at descending
        sorted_summaries = sorted(
            self._summaries,
            key=lambda s: s.created_at or datetime.min,
            reverse=True
        )
        return sorted_summaries[:n]
    
    def get_by_mission_id(self, mission_id: str) -> Optional[MissionSummarySchema]:
        """
        Get summary by mission ID.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            Summary if found, None otherwise
        """
        for summary in self._summaries:
            if summary.mission_id == mission_id:
                return summary
        return None
    
    def query_by_domain(
        self,
        domain: str,
        limit: int = 10
    ) -> List[MissionSummarySchema]:
        """
        Query summaries by domain.
        
        Args:
            domain: Domain to filter by
            limit: Maximum results
            
        Returns:
            Matching summaries sorted by recency
        """
        matches = [
            s for s in self._summaries
            if s.domain and s.domain.lower() == domain.lower()
        ]
        
        # Sort by recency
        matches.sort(key=lambda s: s.created_at or datetime.min, reverse=True)
        return matches[:limit]
    
    def query_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10,
        match_all: bool = False
    ) -> List[MissionSummarySchema]:
        """
        Query summaries by keywords.
        
        Args:
            keywords: Keywords to search for
            limit: Maximum results
            match_all: If True, all keywords must match; otherwise any
            
        Returns:
            Matching summaries sorted by relevance
        """
        keywords_lower = [k.lower() for k in keywords]
        
        results = []
        for summary in self._summaries:
            # Build searchable text
            search_text = summary.to_search_text().lower()
            summary_keywords = [k.lower() for k in summary.keywords]
            summary_tags = [t.lower() for t in summary.tags]
            
            # Count keyword matches
            matches = 0
            for kw in keywords_lower:
                if (kw in search_text or 
                    kw in summary_keywords or 
                    kw in summary_tags):
                    matches += 1
            
            if match_all:
                if matches == len(keywords_lower):
                    results.append((summary, matches))
            else:
                if matches > 0:
                    results.append((summary, matches))
        
        # Sort by match count, then recency
        results.sort(
            key=lambda x: (x[1], x[0].created_at or datetime.min),
            reverse=True
        )
        
        return [s for s, _ in results[:limit]]
    
    def query_by_objective_similarity(
        self,
        objective: str,
        limit: int = 5,
        min_score: float = 0.4
    ) -> List[Tuple[MissionSummarySchema, float]]:
        """
        Find summaries with similar objectives using semantic search.
        
        Args:
            objective: Objective text to compare
            limit: Maximum results
            min_score: Minimum similarity score
            
        Returns:
            List of (summary, similarity_score) tuples
        """
        if not self._embedding_fn:
            # Fallback to keyword search
            words = objective.lower().split()
            keywords = [w for w in words if len(w) > 3][:5]
            matches = self.query_by_keywords(keywords, limit=limit)
            return [(s, 0.5) for s in matches]
        
        try:
            # Get query embedding
            query_embedding = self._embedding_fn(objective)
            if not query_embedding:
                return []
            
            import numpy as np
            query_vec = np.array(query_embedding)
            
            results = []
            for summary in self._summaries:
                if summary.mission_id not in self._embeddings:
                    continue
                
                summary_vec = np.array(self._embeddings[summary.mission_id])
                
                # Cosine similarity
                norm_q = np.linalg.norm(query_vec)
                norm_s = np.linalg.norm(summary_vec)
                
                if norm_q > 0 and norm_s > 0:
                    score = float(np.dot(query_vec, summary_vec) / (norm_q * norm_s))
                    if score >= min_score:
                        results.append((summary, score))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    def get_insights_for_domain(
        self,
        domain: str,
        limit: int = 20
    ) -> List[str]:
        """
        Get key insights from missions in a domain.
        
        Args:
            domain: Domain to query
            limit: Maximum insights to return
            
        Returns:
            List of insight strings
        """
        summaries = self.query_by_domain(domain, limit=10)
        
        all_insights = []
        for summary in summaries:
            all_insights.extend(summary.key_insights)
        
        return all_insights[:limit]
    
    def get_common_contradictions(
        self,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Get common contradictions across missions.
        
        Args:
            domain: Optional domain filter
            limit: Maximum results
            
        Returns:
            List of (contradiction, count) tuples
        """
        # Count contradiction occurrences
        contradiction_counts: Dict[str, int] = {}
        
        summaries = self._summaries
        if domain:
            summaries = [s for s in summaries if s.domain == domain]
        
        for summary in summaries:
            for contradiction in summary.contradictions:
                # Normalize for counting
                normalized = contradiction.lower().strip()
                contradiction_counts[normalized] = contradiction_counts.get(normalized, 0) + 1
        
        # Sort by count
        sorted_items = sorted(
            contradiction_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_items[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored summaries.
        
        Returns:
            Dictionary with statistics
        """
        if not self._summaries:
            return {
                "total_summaries": 0,
                "domains": [],
                "avg_quality_score": 0,
                "total_insights": 0,
            }
        
        domains = list({s.domain for s in self._summaries if s.domain})
        quality_scores = [
            s.final_quality_score for s in self._summaries
            if s.final_quality_score is not None
        ]
        total_insights = sum(len(s.key_insights) for s in self._summaries)
        
        return {
            "total_summaries": len(self._summaries),
            "domains": domains,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "total_insights": total_insights,
            "oldest_mission": min(
                (s.created_at for s in self._summaries if s.created_at),
                default=None
            ),
            "newest_mission": max(
                (s.created_at for s in self._summaries if s.created_at),
                default=None
            ),
        }
    
    def save(self) -> bool:
        """
        Save summaries to disk.
        
        Returns:
            True if successful
        """
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert summaries to dict
            data = {
                "summaries": [s.model_dump() for s in self._summaries],
                "embeddings": self._embeddings,
                "metadata": {
                    "saved_at": datetime.utcnow().isoformat(),
                    "count": len(self._summaries),
                }
            }
            
            with open(self._storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved {len(self._summaries)} summaries to {self._storage_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save summaries: {e}")
            return False
    
    def _load(self) -> None:
        """Load summaries from disk if exists."""
        try:
            if not self._storage_path.exists():
                return
            
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            
            # Load summaries
            for s_dict in data.get("summaries", []):
                # Handle datetime parsing
                if s_dict.get("created_at") and isinstance(s_dict["created_at"], str):
                    s_dict["created_at"] = datetime.fromisoformat(s_dict["created_at"])
                if s_dict.get("completed_at") and isinstance(s_dict["completed_at"], str):
                    s_dict["completed_at"] = datetime.fromisoformat(s_dict["completed_at"])
                
                summary = MissionSummarySchema(**s_dict)
                self._summaries.append(summary)
            
            # Load embeddings
            self._embeddings = data.get("embeddings", {})
            
            logger.debug(f"Loaded {len(self._summaries)} summaries from {self._storage_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load summaries: {e}")
            self._summaries = []
            self._embeddings = {}
    
    def clear(self) -> None:
        """Clear all summaries from memory (does not delete from disk)."""
        self._summaries = []
        self._embeddings = {}

