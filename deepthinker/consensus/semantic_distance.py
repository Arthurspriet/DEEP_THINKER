"""
Semantic Distance Consensus for DeepThinker 2.0.

Uses embeddings to identify hallucination clusters and picks the response
furthest from the outlier cluster, favoring grounded, consistent outputs.

Performance optimizations (Phase 2):
- SHA256-based cache keys to prevent collisions
- Module-level imports to avoid per-call overhead
- Cache hit/miss counters for monitoring
"""

import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Module-level import for embedding calls (avoids per-call import overhead)
from deepthinker.models.model_caller import call_embeddings

logger = logging.getLogger(__name__)


@dataclass
class SemanticDistanceResult:
    """Result of semantic distance consensus."""
    
    selected_output: str
    selected_model: str
    distance_scores: Dict[str, float]
    outlier_cluster: List[str]
    confidence: float


class SemanticDistanceConsensus:
    """
    Semantic distance consensus that identifies and avoids hallucination clusters.
    
    Process:
    1. Embed all outputs
    2. Compute pairwise cosine similarities
    3. Identify the "consensus cluster" (most similar outputs)
    4. Identify "outlier cluster" (potential hallucinations)
    5. Select output from consensus cluster furthest from outliers
    """
    
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:4b",
        outlier_threshold: float = 0.6,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize semantic distance consensus.
        
        Args:
            embedding_model: Ollama model for embeddings
            outlier_threshold: Similarity threshold below which outputs are outliers
            ollama_base_url: Ollama server URL
        """
        self.embedding_model = embedding_model
        self.outlier_threshold = outlier_threshold
        self.ollama_base_url = ollama_base_url
        self._embedding_cache: Dict[str, List[float]] = {}
        # Cache hit/miss counters for monitoring
        self._cache_hits: int = 0
        self._cache_misses: int = 0
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        # Use SHA256 hash for collision-free cache key (Phase 2.1)
        cache_key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key]
        
        self._cache_misses += 1
        
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
    
    def _compute_similarity_matrix(
        self,
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise similarity matrix.
        
        Args:
            embeddings: Dictionary mapping model_name -> embedding
            
        Returns:
            Nested dictionary of pairwise similarities
        """
        model_names = list(embeddings.keys())
        similarity_matrix: Dict[str, Dict[str, float]] = {}
        
        for name1 in model_names:
            similarity_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    similarity_matrix[name1][name2] = 1.0
                else:
                    sim = self._cosine_similarity(
                        embeddings[name1],
                        embeddings[name2]
                    )
                    similarity_matrix[name1][name2] = sim
        
        return similarity_matrix
    
    def _identify_clusters(
        self,
        similarity_matrix: Dict[str, Dict[str, float]]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify consensus and outlier clusters.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            Tuple of (consensus_cluster, outlier_cluster)
        """
        model_names = list(similarity_matrix.keys())
        
        if len(model_names) <= 2:
            # With 2 or fewer models, all are consensus
            return model_names, []
        
        # Compute average similarity for each model to all others
        avg_similarities = {}
        for name in model_names:
            others = [
                similarity_matrix[name][other]
                for other in model_names
                if other != name
            ]
            avg_similarities[name] = np.mean(others) if others else 0.0
        
        # Models with below-threshold average similarity are outliers
        consensus_cluster = []
        outlier_cluster = []
        
        for name, avg_sim in avg_similarities.items():
            if avg_sim < self.outlier_threshold:
                outlier_cluster.append(name)
            else:
                consensus_cluster.append(name)
        
        # If all are outliers or all are consensus, use similarity ranking
        if not consensus_cluster or not outlier_cluster:
            sorted_by_sim = sorted(
                avg_similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Top half is consensus, bottom half is outliers
            mid = len(sorted_by_sim) // 2
            if mid == 0:
                mid = 1
            
            consensus_cluster = [name for name, _ in sorted_by_sim[:mid]]
            outlier_cluster = [name for name, _ in sorted_by_sim[mid:]]
        
        return consensus_cluster, outlier_cluster
    
    def _select_best_from_consensus(
        self,
        consensus_cluster: List[str],
        outlier_cluster: List[str],
        embeddings: Dict[str, List[float]],
        outputs: Dict[str, str]
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Select best output from consensus cluster.
        
        Picks the output that is most similar to other consensus members
        but furthest from outlier cluster.
        
        Args:
            consensus_cluster: List of consensus model names
            outlier_cluster: List of outlier model names
            embeddings: All embeddings
            outputs: All text outputs
            
        Returns:
            Tuple of (selected_output, selected_model, distance_scores)
        """
        if not consensus_cluster:
            # Fallback: return first output
            first_model = list(outputs.keys())[0]
            return outputs[first_model], first_model, {}
        
        distance_scores = {}
        
        for name in consensus_cluster:
            # Distance from outliers (want high)
            if outlier_cluster:
                outlier_distances = [
                    1.0 - self._cosine_similarity(embeddings[name], embeddings[outlier])
                    for outlier in outlier_cluster
                    if outlier in embeddings and embeddings[outlier]
                ]
                avg_outlier_distance = np.mean(outlier_distances) if outlier_distances else 0.5
            else:
                avg_outlier_distance = 0.5
            
            # Similarity to consensus (want high)
            consensus_similarities = [
                self._cosine_similarity(embeddings[name], embeddings[other])
                for other in consensus_cluster
                if other != name and other in embeddings and embeddings[other]
            ]
            avg_consensus_sim = np.mean(consensus_similarities) if consensus_similarities else 0.5
            
            # Combined score: high consensus similarity + high outlier distance
            distance_scores[name] = avg_consensus_sim * 0.5 + avg_outlier_distance * 0.5
        
        # Select model with highest score
        best_model = max(distance_scores, key=distance_scores.get)
        
        return outputs[best_model], best_model, distance_scores
    
    def apply(
        self,
        outputs: Dict[str, Any]
    ) -> SemanticDistanceResult:
        """
        Apply semantic distance consensus to model outputs.
        
        Args:
            outputs: Dictionary mapping model_name -> output
            
        Returns:
            SemanticDistanceResult with selected output
        """
        # Extract text outputs
        text_outputs: Dict[str, str] = {}
        for name, output in outputs.items():
            if hasattr(output, 'output'):
                if output.success and output.output:
                    text_outputs[name] = output.output
            elif isinstance(output, str) and output:
                text_outputs[name] = output
        
        if not text_outputs:
            return SemanticDistanceResult(
                selected_output="",
                selected_model="",
                distance_scores={},
                outlier_cluster=[],
                confidence=0.0
            )
        
        # Single output - no distance calculation needed
        if len(text_outputs) == 1:
            model_name = list(text_outputs.keys())[0]
            return SemanticDistanceResult(
                selected_output=text_outputs[model_name],
                selected_model=model_name,
                distance_scores={model_name: 1.0},
                outlier_cluster=[],
                confidence=1.0
            )
        
        # Get embeddings
        embeddings = {
            name: self._get_embedding(output)
            for name, output in text_outputs.items()
        }
        
        # Filter out failed embeddings
        valid_embeddings = {
            name: emb for name, emb in embeddings.items()
            if emb
        }
        
        if not valid_embeddings:
            # Fallback: return first output
            first_model = list(text_outputs.keys())[0]
            return SemanticDistanceResult(
                selected_output=text_outputs[first_model],
                selected_model=first_model,
                distance_scores={},
                outlier_cluster=[],
                confidence=0.5
            )
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(valid_embeddings)
        
        # Identify clusters
        consensus_cluster, outlier_cluster = self._identify_clusters(similarity_matrix)
        
        # Select best from consensus
        selected_output, selected_model, distance_scores = self._select_best_from_consensus(
            consensus_cluster,
            outlier_cluster,
            valid_embeddings,
            text_outputs
        )
        
        # Compute confidence based on cluster separation
        if consensus_cluster and outlier_cluster:
            confidence = len(consensus_cluster) / len(text_outputs)
        elif consensus_cluster:
            confidence = 0.8
        else:
            confidence = 0.5
        
        # Log cache statistics
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = self._cache_hits / total_requests
            logger.debug(
                f"[SemanticDistance] Embedding cache: {self._cache_hits} hits, "
                f"{self._cache_misses} misses ({hit_rate:.1%} hit rate)"
            )
        
        return SemanticDistanceResult(
            selected_output=selected_output,
            selected_model=selected_model,
            distance_scores=distance_scores,
            outlier_cluster=outlier_cluster,
            confidence=confidence
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get embedding cache statistics.
        
        Returns:
            Dictionary with cache size, hits, misses, and hit rate
        """
        total = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._embedding_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache and reset counters."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

