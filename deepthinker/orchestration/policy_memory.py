"""
Policy Memory for DeepThinker.

Read-only statistical aggregator over historical phase outcomes.
Computes rolling statistics to enable learning queries.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .orchestration_store import OrchestrationStore
from .outcome_logger import PhaseOutcome

logger = logging.getLogger(__name__)


@dataclass
class CouncilPhaseStats:
    """
    Statistical summary for a council-phase combination.
    """
    council_name: str
    phase_type: str
    invocation_count: int
    avg_quality_gain: float
    avg_tokens: int
    avg_wall_time: float
    avg_gpu_seconds: float
    quality_per_token: float
    quality_per_gpu_second: float
    success_rate: float
    
    # Additional metrics
    avg_confidence: float = 0.0
    median_quality: float = 0.0
    stddev_quality: float = 0.0


class PolicyMemory:
    """
    Read-only policy memory that aggregates historical outcomes.
    
    Computes rolling statistics over phase outcomes to answer queries like:
    - Which councils are useful in which phases?
    - What's the expected quality gain and cost?
    - Which councils have negative ROI?
    """
    
    def __init__(
        self,
        store: OrchestrationStore,
        window_size: int = 100,
        refresh_interval_seconds: float = 300.0
    ):
        """
        Initialize policy memory.
        
        Args:
            store: OrchestrationStore to read from
            window_size: Number of recent outcomes to consider per council-phase
            refresh_interval_seconds: How often to refresh statistics
        """
        self._store = store
        self._window_size = window_size
        self._refresh_interval = refresh_interval_seconds
        
        # Cache of computed statistics
        self._cache: Dict[str, CouncilPhaseStats] = {}
        self._last_refresh: Optional[datetime] = None
    
    @staticmethod
    def _safe_int(value: any, default: int = 0) -> int:
        """Safely convert a value to int, handling strings and None."""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))  # Handle "123.45" strings
            except (ValueError, TypeError):
                return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_float(value: any, default: float = 0.0) -> float:
        """Safely convert a value to float, handling strings and None."""
        if value is None:
            return default
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def refresh_statistics(self) -> None:
        """
        Refresh statistics from the store.
        
        Computes rolling statistics for all council-phase combinations.
        """
        try:
            logger.debug("Refreshing policy memory statistics...")
            
            # Group outcomes by council-phase
            by_council_phase: Dict[Tuple[str, str], List[PhaseOutcome]] = {}
            
            for outcome in self._store.read_outcomes():
                for council in outcome.councils_invoked:
                    key = (council, outcome.phase_type)
                    if key not in by_council_phase:
                        by_council_phase[key] = []
                    by_council_phase[key].append(outcome)
            
            # Compute statistics for each council-phase
            new_cache: Dict[str, CouncilPhaseStats] = {}
            
            for (council, phase_type), outcomes in by_council_phase.items():
                # Take most recent outcomes up to window_size
                recent = outcomes[-self._window_size:]
                
                if not recent:
                    continue
                
                stats = self._compute_stats(council, phase_type, recent)
                cache_key = f"{council}:{phase_type}"
                new_cache[cache_key] = stats
            
            self._cache = new_cache
            self._last_refresh = datetime.utcnow()
            
            logger.debug(f"Refreshed statistics for {len(new_cache)} council-phase combinations")
            
        except Exception as e:
            logger.warning(f"Failed to refresh statistics: {e}")
    
    def _compute_stats(
        self,
        council_name: str,
        phase_type: str,
        outcomes: List[PhaseOutcome]
    ) -> CouncilPhaseStats:
        """Compute statistics for a council-phase combination."""
        if not outcomes:
            return CouncilPhaseStats(
                council_name=council_name,
                phase_type=phase_type,
                invocation_count=0,
                avg_quality_gain=0.0,
                avg_tokens=0,
                avg_wall_time=0.0,
                avg_gpu_seconds=0.0,
                quality_per_token=0.0,
                quality_per_gpu_second=0.0,
                success_rate=0.0,
            )
        
        # Filter outcomes that actually used this council
        relevant = [
            o for o in outcomes
            if council_name in o.councils_invoked
        ]
        
        if not relevant:
            return CouncilPhaseStats(
                council_name=council_name,
                phase_type=phase_type,
                invocation_count=0,
                avg_quality_gain=0.0,
                avg_tokens=0,
                avg_wall_time=0.0,
                avg_gpu_seconds=0.0,
                quality_per_token=0.0,
                quality_per_gpu_second=0.0,
                success_rate=0.0,
            )
        
        # Compute averages with type coercion to handle JSON deserialization issues
        quality_scores = [
            self._safe_float(o.quality_score) 
            for o in relevant 
            if o.quality_score is not None
        ]
        tokens = [self._safe_int(o.tokens_consumed) for o in relevant]
        wall_times = [self._safe_float(o.wall_time_seconds) for o in relevant]
        gpu_seconds = [self._safe_float(o.gpu_seconds) for o in relevant]
        confidences = [
            self._safe_float(o.confidence_score) 
            for o in relevant 
            if o.confidence_score is not None
        ]
        
        # Quality gain: difference from baseline (assume 0.5 baseline)
        baseline_quality = 0.5
        quality_gains = [
            (q - baseline_quality) for q in quality_scores
        ] if quality_scores else [0.0]
        
        avg_quality_gain = sum(quality_gains) / len(quality_gains) if quality_gains else 0.0
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0
        avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0
        avg_gpu_seconds = sum(gpu_seconds) / len(gpu_seconds) if gpu_seconds else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Quality per cost ratios
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        quality_per_token = avg_quality / max(avg_tokens, 1)
        quality_per_gpu_second = avg_quality / max(avg_gpu_seconds, 0.1)
        
        # Success rate (missions that succeeded)
        successful = [
            o for o in relevant
            if o.mission_outcome_success is True
        ]
        success_rate = len(successful) / len(relevant) if relevant else 0.0
        
        # Median and stddev
        sorted_qualities = sorted(quality_scores) if quality_scores else []
        median_quality = sorted_qualities[len(sorted_qualities) // 2] if sorted_qualities else 0.0
        
        if len(quality_scores) > 1:
            mean_quality = sum(quality_scores) / len(quality_scores)
            variance = sum((q - mean_quality) ** 2 for q in quality_scores) / len(quality_scores)
            stddev_quality = variance ** 0.5
        else:
            stddev_quality = 0.0
        
        return CouncilPhaseStats(
            council_name=council_name,
            phase_type=phase_type,
            invocation_count=len(relevant),
            avg_quality_gain=avg_quality_gain,
            avg_tokens=int(avg_tokens),
            avg_wall_time=avg_wall_time,
            avg_gpu_seconds=avg_gpu_seconds,
            quality_per_token=quality_per_token,
            quality_per_gpu_second=quality_per_gpu_second,
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            median_quality=median_quality,
            stddev_quality=stddev_quality,
        )
    
    def get_council_stats(
        self,
        council_name: str,
        phase_type: str
    ) -> CouncilPhaseStats:
        """
        Get statistics for a council-phase combination.
        
        Args:
            council_name: Name of the council
            phase_type: Type of phase
            
        Returns:
            CouncilPhaseStats (may be empty if no data)
        """
        self._ensure_fresh()
        
        cache_key = f"{council_name}:{phase_type}"
        return self._cache.get(cache_key, CouncilPhaseStats(
            council_name=council_name,
            phase_type=phase_type,
            invocation_count=0,
            avg_quality_gain=0.0,
            avg_tokens=0,
            avg_wall_time=0.0,
            avg_gpu_seconds=0.0,
            quality_per_token=0.0,
            quality_per_gpu_second=0.0,
            success_rate=0.0,
        ))
    
    def best_council_sequence(
        self,
        phase_type: str,
        max_time_seconds: float
    ) -> List[str]:
        """
        Get best council sequence for a phase type within time budget.
        
        Args:
            phase_type: Type of phase
            max_time_seconds: Maximum time budget
            
        Returns:
            Ordered list of council names
        """
        self._ensure_fresh()
        
        # Find all councils for this phase type
        candidates: List[Tuple[str, CouncilPhaseStats]] = []
        for cache_key, stats in self._cache.items():
            if stats.phase_type == phase_type and stats.invocation_count > 0:
                candidates.append((stats.council_name, stats))
        
        if not candidates:
            return []
        
        # Sort by quality_per_gpu_second (efficiency)
        candidates.sort(key=lambda x: x[1].quality_per_gpu_second, reverse=True)
        
        # Greedily select councils that fit in time budget
        sequence = []
        total_time = 0.0
        
        for council_name, stats in candidates:
            if total_time + stats.avg_wall_time <= max_time_seconds:
                sequence.append(council_name)
                total_time += stats.avg_wall_time
            else:
                break
        
        return sequence
    
    def councils_with_negative_roi(
        self,
        phase_type: str
    ) -> List[str]:
        """
        Get councils with negative ROI for a phase type.
        
        Negative ROI = quality_per_gpu_second < 0.1 or avg_quality_gain < 0
        
        Args:
            phase_type: Type of phase
            
        Returns:
            List of council names with negative ROI
        """
        self._ensure_fresh()
        
        negative_roi = []
        
        for cache_key, stats in self._cache.items():
            if stats.phase_type != phase_type:
                continue
            
            if stats.invocation_count < 10:
                continue  # Not enough data
            
            # Negative ROI criteria
            if (stats.quality_per_gpu_second < 0.1 or
                stats.avg_quality_gain < 0.0 or
                (stats.success_rate < 0.3 and stats.invocation_count > 20)):
                negative_roi.append(stats.council_name)
        
        return negative_roi
    
    def expected_quality_gain(
        self,
        council: str,
        phase_type: str
    ) -> float:
        """
        Get expected quality gain from invoking a council in a phase.
        
        Args:
            council: Council name
            phase_type: Phase type
            
        Returns:
            Expected quality gain (may be negative)
        """
        stats = self.get_council_stats(council, phase_type)
        return stats.avg_quality_gain
    
    def expected_cost(
        self,
        council: str,
        phase_type: str
    ) -> Tuple[int, float]:
        """
        Get expected cost (tokens, time) from invoking a council in a phase.
        
        Args:
            council: Council name
            phase_type: Phase type
            
        Returns:
            Tuple of (expected_tokens, expected_time_seconds)
        """
        stats = self.get_council_stats(council, phase_type)
        return (stats.avg_tokens, stats.avg_wall_time)
    
    def _ensure_fresh(self) -> None:
        """Ensure statistics are fresh (refresh if needed)."""
        if self._last_refresh is None:
            self.refresh_statistics()
            return
        
        elapsed = (datetime.utcnow() - self._last_refresh).total_seconds()
        if elapsed > self._refresh_interval:
            self.refresh_statistics()

