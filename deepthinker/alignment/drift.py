"""
Alignment Control Layer - Drift Detection.

Implements embedding-based drift detection with time series analysis.
This module is pure metrics + embeddings - no controller logic.

Key metrics:
- a_t: Goal similarity (cosine of goal vs output embeddings)
- d_t: Drift delta (a_t - a_{t-1})
- s_t: Semantic jump size (L2 distance on normalized vectors)
- D_t^-: Cumulative negative drift
- cusum_neg: One-sided CUSUM for detecting sustained negative drift
"""

import hashlib
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AlignmentConfig, get_alignment_config
from .models import AlignmentPoint, AlignmentTrajectory, NorthStarGoal

logger = logging.getLogger(__name__)


class EmbeddingDriftDetector:
    """
    Detects alignment drift using embedding similarity over time.
    
    Computes alignment metrics at each phase and maintains a trajectory.
    Uses CUSUM algorithm for detecting sustained negative drift.
    
    Usage:
        detector = EmbeddingDriftDetector(config)
        
        # Set up the goal
        detector.set_north_star(north_star)
        
        # After each phase
        point = detector.compute_alignment_point(
            output_text="phase output",
            phase_name="research",
            prev_point=trajectory.last_point(),
        )
        
        if point.triggered:
            # Take corrective action
            pass
    """
    
    def __init__(
        self,
        config: Optional[AlignmentConfig] = None,
    ):
        """
        Initialize the drift detector.
        
        Args:
            config: Alignment configuration (uses global if None)
        """
        self.config = config or get_alignment_config()
        
        # Embedding cache (hash -> embedding)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        
        # North star embedding (cached)
        self._north_star_embedding: Optional[np.ndarray] = None
        self._north_star: Optional[NorthStarGoal] = None
    
    def set_north_star(self, north_star: NorthStarGoal) -> None:
        """
        Set the north star goal and compute its embedding.
        
        Args:
            north_star: The goal to track alignment against
        """
        self._north_star = north_star
        
        # Compute embedding for the goal
        goal_text = north_star.get_full_text()
        embedding = self._get_embedding(goal_text)
        
        if embedding:
            self._north_star_embedding = self._normalize(np.array(embedding))
            north_star.embedding = embedding
            logger.debug(f"[ALIGNMENT] North star embedding computed (dim={len(embedding)})")
        else:
            logger.warning("[ALIGNMENT] Failed to compute north star embedding")
            self._north_star_embedding = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text, using cache if available.
        
        Uses the centralized model_caller for proper resource management.
        
        Args:
            text: Text to embed (truncated to 2000 chars)
            
        Returns:
            Embedding vector, or empty list on failure
        """
        # Truncate text
        text_truncated = text[:2000]
        
        # Check cache
        cache_key = hashlib.sha256(text_truncated.encode('utf-8')).hexdigest()
        
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            from deepthinker.models.model_caller import call_embeddings
            
            embedding = call_embeddings(
                text=text_truncated,
                model=self.config.embedding_model,
                timeout=60.0,
                max_retries=2,
                base_url=self.config.ollama_base_url,
            )
            
            if embedding:
                self._embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.debug(f"[ALIGNMENT] Embedding failed: {e}")
            return []
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Assumes vectors are already normalized.
        
        Args:
            vec1: First vector (normalized)
            vec2: Second vector (normalized)
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        # Handle dimension mismatch (can occur with mixed embedding models)
        if vec1.shape[0] != vec2.shape[0]:
            return 0.0
        
        return float(np.dot(vec1, vec2))
    
    def _euclidean_distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """
        Compute Euclidean distance between two normalized vectors.
        
        For normalized vectors, this gives the semantic jump size.
        Range is [0, 2] where 0 = identical, 2 = opposite.
        
        Args:
            vec1: First vector (normalized)
            vec2: Second vector (normalized)
            
        Returns:
            L2 distance
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        return float(np.linalg.norm(vec1 - vec2))
    
    def compute_alignment_point(
        self,
        output_text: str,
        phase_name: str,
        prev_point: Optional[AlignmentPoint] = None,
    ) -> Optional[AlignmentPoint]:
        """
        Compute alignment metrics for a phase output.
        
        This is the main entry point for drift detection.
        
        Args:
            output_text: The phase output text to analyze
            phase_name: Name of the current phase
            prev_point: Previous alignment point (for delta computation)
            
        Returns:
            AlignmentPoint with all metrics, or None on failure
        """
        if self._north_star_embedding is None:
            logger.warning("[ALIGNMENT] No north star embedding - cannot compute alignment")
            return None
        
        # Get embedding for output
        output_embedding = self._get_embedding(output_text)
        if not output_embedding:
            logger.debug("[ALIGNMENT] Failed to embed output text")
            return None
        
        output_vec = self._normalize(np.array(output_embedding))
        
        # Compute a_t (goal similarity)
        a_t = self._cosine_similarity(self._north_star_embedding, output_vec)
        
        # Get previous values
        prev_a = prev_point.a_t if prev_point else 1.0
        prev_cusum = prev_point.cusum_neg if prev_point else 0.0
        prev_cum_neg = prev_point.cumulative_neg_drift if prev_point else 0.0
        t = (prev_point.t + 1) if prev_point else 0
        
        # Need previous output embedding for s_t
        # For simplicity, we estimate s_t from a_t change when we don't have prev embedding
        # In a more complete implementation, we'd cache the previous embedding
        
        # Compute d_t (drift delta)
        d_t = a_t - prev_a
        
        # Compute s_t (semantic jump size)
        # Approximation: s_t ≈ sqrt(2 * (1 - a_t)) for normalized vectors
        # This is derived from ||x-y||^2 = 2(1 - cos(x,y)) for unit vectors
        if prev_point is not None:
            # Use the change in similarity as proxy for semantic jump
            s_t = abs(d_t) * 2  # Scale factor
        else:
            s_t = 0.0
        
        # Compute cumulative negative drift D_t^-
        neg_drift = max(0.0, -d_t)
        cumulative_neg_drift = prev_cum_neg + neg_drift
        
        # Compute CUSUM for negative drift
        # cusum_neg = max(0, prev_cusum + (-d_t) - k)
        cusum_neg = max(0.0, prev_cusum + neg_drift - self.config.cusum_k)
        
        # Check triggers (two-tier: warning and correction)
        warning, triggered = self._check_triggers(
            a_t=a_t,
            d_t=d_t,
            cusum_neg=cusum_neg,
            t=t,
        )
        
        point = AlignmentPoint(
            t=t,
            a_t=a_t,
            d_t=d_t,
            s_t=s_t,
            cusum_neg=cusum_neg,
            cumulative_neg_drift=cumulative_neg_drift,
            triggered=triggered,
            phase_name=phase_name,
            timestamp_iso=datetime.utcnow().isoformat(),
            output_embedding_norm=float(np.linalg.norm(output_embedding)),
            warning=warning,
        )
        
        logger.debug(
            f"[ALIGNMENT] Point t={t}: a_t={a_t:.3f}, d_t={d_t:.3f}, "
            f"cusum={cusum_neg:.3f}, warning={warning}, triggered={triggered}"
        )
        
        return point
    
    def _check_triggers(
        self,
        a_t: float,
        d_t: float,
        cusum_neg: float,
        t: int,
    ) -> Tuple[bool, bool]:
        """
        Check if alignment triggers should fire.
        
        Implements two-tier threshold system:
        - Warning: a_t below warning_threshold (visibility only)
        - Correction: a_t below correction_threshold (triggers action)
        
        Triggers are soft - they indicate drift is detected,
        not that the mission should stop.
        
        Args:
            a_t: Current goal similarity
            d_t: Current drift delta
            cusum_neg: Current CUSUM statistic
            t: Current timestep
            
        Returns:
            Tuple of (warning, correction) booleans
        """
        # Don't trigger on first few events
        if t < self.config.min_events_before_trigger:
            return False, False
        
        warning = False
        correction = False
        warning_reasons = []
        correction_reasons = []
        
        # Two-tier threshold: warning vs correction based on a_t
        if a_t < self.config.warning_threshold:
            warning = True
            warning_reasons.append(f"low_similarity({a_t:.3f}<{self.config.warning_threshold})")
        
        if a_t < self.config.correction_threshold:
            correction = True
            correction_reasons.append(f"low_similarity({a_t:.3f}<{self.config.correction_threshold})")
        
        # Sharp negative drift triggers correction
        if d_t < self.config.delta_neg_soft:
            correction = True
            correction_reasons.append(f"sharp_drift({d_t:.3f})")
        
        # CUSUM threshold exceeded triggers correction
        if cusum_neg > self.config.cusum_h:
            correction = True
            correction_reasons.append(f"cusum({cusum_neg:.3f})")
        
        # Log warnings and corrections
        if warning and not correction:
            logger.info(f"[ALIGNMENT] Warning state: {warning_reasons}")
        if correction:
            logger.info(f"[ALIGNMENT] Correction triggers fired: {correction_reasons}")
        
        return warning, correction
    
    def create_trajectory(
        self,
        mission_id: str,
        north_star: NorthStarGoal,
    ) -> AlignmentTrajectory:
        """
        Create a new alignment trajectory for a mission.
        
        Args:
            mission_id: Mission ID
            north_star: The north star goal
            
        Returns:
            New AlignmentTrajectory
        """
        self.set_north_star(north_star)
        
        return AlignmentTrajectory(
            mission_id=mission_id,
            north_star=north_star,
            points=[],
            assessments=[],
            actions_taken=[],
        )
    
    def update_trajectory(
        self,
        trajectory: AlignmentTrajectory,
        output_text: str,
        phase_name: str,
    ) -> Tuple[AlignmentTrajectory, Optional[AlignmentPoint]]:
        """
        Update a trajectory with a new phase output.
        
        Convenience method that computes the alignment point
        and adds it to the trajectory.
        
        Args:
            trajectory: Existing trajectory
            output_text: Phase output text
            phase_name: Phase name
            
        Returns:
            Tuple of (updated trajectory, new point or None)
        """
        # Ensure north star is set
        if self._north_star is None or self._north_star.goal_id != trajectory.north_star.goal_id:
            self.set_north_star(trajectory.north_star)
        
        # Compute new point
        prev_point = trajectory.last_point()
        new_point = self.compute_alignment_point(
            output_text=output_text,
            phase_name=phase_name,
            prev_point=prev_point,
        )
        
        if new_point:
            trajectory.add_point(new_point)
        
        return trajectory, new_point
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get embedding cache statistics."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._embedding_cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# Global detector instance (lazy-loaded)
_detector: Optional[EmbeddingDriftDetector] = None


def get_drift_detector(
    config: Optional[AlignmentConfig] = None,
    force_new: bool = False,
) -> EmbeddingDriftDetector:
    """
    Get the global drift detector instance.
    
    Args:
        config: Optional configuration override
        force_new: Force creation of new instance
        
    Returns:
        EmbeddingDriftDetector instance
    """
    global _detector
    
    if _detector is None or force_new:
        _detector = EmbeddingDriftDetector(config)
    
    return _detector

