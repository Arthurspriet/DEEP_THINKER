"""
Majority Vote Consensus for DeepThinker 2.0.

Uses embedding similarity to cluster semantically similar outputs
and returns the most common semantic meaning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class VoteResult:
    """Result of majority vote consensus."""
    
    winner: str
    winner_model: str
    vote_counts: Dict[str, int]
    cluster_assignments: Dict[str, int]
    confidence: float


class MajorityVoteConsensus:
    """
    Majority vote consensus using semantic similarity clustering.
    
    Groups outputs by semantic similarity using embeddings,
    then selects the output from the largest cluster.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        embedding_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize majority vote consensus.
        
        Args:
            similarity_threshold: Cosine similarity threshold for clustering
            embedding_model: Ollama model for embeddings
            ollama_base_url: Ollama server URL
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        from deepthinker.models.model_caller import call_embeddings
        
        cache_key = text[:500]  # Truncate for cache key
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Use centralized model_caller for proper resource management
        embedding = call_embeddings(
            text=text,
            model=self.embedding_model,
            timeout=60.0,
            max_retries=2,
            base_url=self.ollama_base_url,
        )
        
        if embedding:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        a = np.array(vec1)
        b = np.array(vec2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _cluster_outputs(
        self,
        outputs: Dict[str, str]
    ) -> Dict[int, List[str]]:
        """
        Cluster outputs by semantic similarity.
        
        Args:
            outputs: Dictionary mapping model_name -> output_text
            
        Returns:
            Dictionary mapping cluster_id -> list of model_names
        """
        if not outputs:
            return {}
        
        model_names = list(outputs.keys())
        embeddings = {
            name: self._get_embedding(outputs[name])
            for name in model_names
        }
        
        # Simple agglomerative clustering
        clusters: Dict[int, List[str]] = {}
        assigned: Dict[str, int] = {}
        cluster_id = 0
        
        for model_name in model_names:
            if model_name in assigned:
                continue
            
            # Start new cluster
            clusters[cluster_id] = [model_name]
            assigned[model_name] = cluster_id
            
            # Find similar outputs
            for other_name in model_names:
                if other_name in assigned:
                    continue
                
                similarity = self._cosine_similarity(
                    embeddings[model_name],
                    embeddings[other_name]
                )
                
                if similarity >= self.similarity_threshold:
                    clusters[cluster_id].append(other_name)
                    assigned[other_name] = cluster_id
            
            cluster_id += 1
        
        return clusters
    
    def apply(
        self,
        outputs: Dict[str, Any]
    ) -> VoteResult:
        """
        Apply majority vote consensus to model outputs.
        
        Args:
            outputs: Dictionary mapping model_name -> output
                     (can be ModelOutput objects or strings)
        
        Returns:
            VoteResult with winning output and voting details
        """
        # Extract text outputs
        text_outputs: Dict[str, str] = {}
        for name, output in outputs.items():
            if hasattr(output, 'output'):
                # ModelOutput object
                if output.success and output.output:
                    text_outputs[name] = output.output
            elif isinstance(output, str) and output:
                text_outputs[name] = output
        
        if not text_outputs:
            return VoteResult(
                winner="",
                winner_model="",
                vote_counts={},
                cluster_assignments={},
                confidence=0.0
            )
        
        # Cluster by semantic similarity
        clusters = self._cluster_outputs(text_outputs)
        
        # Find largest cluster
        vote_counts = {
            cluster_id: len(members)
            for cluster_id, members in clusters.items()
        }
        
        winning_cluster = max(vote_counts, key=vote_counts.get)
        winning_models = clusters[winning_cluster]
        
        # Select first model from winning cluster as representative
        winner_model = winning_models[0]
        winner_output = text_outputs[winner_model]
        
        # Build cluster assignments
        cluster_assignments = {}
        for cluster_id, members in clusters.items():
            for member in members:
                cluster_assignments[member] = cluster_id
        
        # Calculate confidence based on vote distribution
        total_votes = sum(vote_counts.values())
        winning_votes = vote_counts[winning_cluster]
        confidence = winning_votes / total_votes if total_votes > 0 else 0.0
        
        return VoteResult(
            winner=winner_output,
            winner_model=winner_model,
            vote_counts=vote_counts,
            cluster_assignments=cluster_assignments,
            confidence=confidence
        )
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()

