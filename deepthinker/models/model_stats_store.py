"""
Model Statistics Store for DeepThinker.

Aggregates model performance data from phase_outcomes.jsonl to provide
historical performance grades and statistics for model selection.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """
    Aggregated statistics for a single model.
    
    Attributes:
        model_name: Name of the model (e.g., "gemma3:27b")
        total_uses: Total number of phase executions using this model
        total_missions: Number of unique missions this model participated in
        avg_quality_score: Average quality score (0-10, None if no scores)
        avg_wall_time_seconds: Average wall time per phase execution
        avg_gpu_seconds: Average GPU seconds per phase execution
        success_count: Number of phases linked to successful missions
        failure_count: Number of phases linked to failed missions
        unknown_outcome_count: Number of phases with unlinked outcomes
        tier_distribution: Dict of tier -> count (how often used in each tier)
        phase_type_distribution: Dict of phase_type -> count
    """
    model_name: str
    total_uses: int = 0
    total_missions: int = 0
    avg_quality_score: Optional[float] = None
    avg_wall_time_seconds: float = 0.0
    avg_gpu_seconds: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    unknown_outcome_count: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    phase_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    def compute_grade(self) -> str:
        """
        Compute performance grade based on statistics.
        
        Returns:
            Grade label: "excellent", "good", "fair", "poor", or "unknown"
        """
        if self.total_uses < 3:
            return "unknown"
        
        # Calculate success rate
        total_with_outcome = self.success_count + self.failure_count
        if total_with_outcome == 0:
            success_rate = 0.5  # Neutral if no outcomes linked
        else:
            success_rate = self.success_count / total_with_outcome
        
        # Calculate quality component
        if self.avg_quality_score is not None:
            quality_score = self.avg_quality_score / 10.0  # Normalize to 0-1
        else:
            quality_score = 0.5  # Neutral if no quality scores
        
        # Combined score: 60% quality, 40% success rate
        combined_score = quality_score * 0.6 + success_rate * 0.4
        
        if combined_score >= 0.8:
            return "excellent"
        elif combined_score >= 0.65:
            return "good"
        elif combined_score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "model_name": self.model_name,
            "total_uses": self.total_uses,
            "total_missions": self.total_missions,
            "avg_quality_score": self.avg_quality_score,
            "avg_wall_time_seconds": round(self.avg_wall_time_seconds, 2),
            "avg_gpu_seconds": round(self.avg_gpu_seconds, 2),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "unknown_outcome_count": self.unknown_outcome_count,
            "performance_grade": self.compute_grade(),
            "tier_distribution": self.tier_distribution,
            "phase_type_distribution": self.phase_type_distribution,
        }


class ModelStatsStore:
    """
    Store for aggregating and caching model performance statistics.
    
    Reads from kb/orchestration/phase_outcomes.jsonl and computes
    per-model statistics with TTL-based caching.
    """
    
    def __init__(
        self,
        outcomes_file: Optional[Path] = None,
        cache_ttl_seconds: float = 60.0
    ):
        """
        Initialize the stats store.
        
        Args:
            outcomes_file: Path to phase_outcomes.jsonl (default: kb/orchestration/phase_outcomes.jsonl)
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        if outcomes_file is None:
            outcomes_file = Path("kb/orchestration/phase_outcomes.jsonl")
        
        self.outcomes_file = Path(outcomes_file)
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Cache state
        self._cache: Dict[str, ModelStats] = {}
        self._cache_timestamp: float = 0.0
        self._file_mtime: float = 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        # Check TTL
        if time.time() - self._cache_timestamp > self.cache_ttl_seconds:
            return False
        
        # Check if file has been modified
        if self.outcomes_file.exists():
            current_mtime = self.outcomes_file.stat().st_mtime
            if current_mtime > self._file_mtime:
                return False
        
        return True
    
    def _refresh_cache(self) -> None:
        """Refresh the cache by re-reading outcomes file."""
        self._cache.clear()
        
        if not self.outcomes_file.exists():
            logger.debug(f"Outcomes file not found: {self.outcomes_file}")
            self._cache_timestamp = time.time()
            return
        
        # Temporary accumulators
        model_data: Dict[str, Dict[str, Any]] = {}
        
        try:
            with open(self.outcomes_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        self._process_outcome(data, model_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
            
            # Convert accumulators to ModelStats
            for model_name, acc in model_data.items():
                stats = ModelStats(model_name=model_name)
                stats.total_uses = acc["total_uses"]
                stats.total_missions = len(acc["missions"])
                
                # Compute averages
                if acc["quality_scores"]:
                    stats.avg_quality_score = sum(acc["quality_scores"]) / len(acc["quality_scores"])
                
                if acc["wall_times"]:
                    stats.avg_wall_time_seconds = sum(acc["wall_times"]) / len(acc["wall_times"])
                
                if acc["gpu_seconds"]:
                    stats.avg_gpu_seconds = sum(acc["gpu_seconds"]) / len(acc["gpu_seconds"])
                
                stats.success_count = acc["success_count"]
                stats.failure_count = acc["failure_count"]
                stats.unknown_outcome_count = acc["unknown_outcome_count"]
                stats.tier_distribution = acc["tier_distribution"]
                stats.phase_type_distribution = acc["phase_type_distribution"]
                
                self._cache[model_name] = stats
            
            # Update cache metadata
            self._cache_timestamp = time.time()
            if self.outcomes_file.exists():
                self._file_mtime = self.outcomes_file.stat().st_mtime
            
            logger.debug(f"Refreshed model stats cache: {len(self._cache)} models")
            
        except Exception as e:
            logger.warning(f"Failed to refresh model stats cache: {e}")
            self._cache_timestamp = time.time()
    
    def _process_outcome(
        self,
        data: Dict[str, Any],
        model_data: Dict[str, Dict[str, Any]]
    ) -> None:
        """Process a single phase outcome and accumulate stats."""
        models_used = data.get("models_used", [])
        mission_id = data.get("mission_id", "")
        phase_type = data.get("phase_type", "unknown")
        quality_score = data.get("quality_score")
        wall_time = data.get("wall_time_seconds", 0.0)
        gpu_seconds = data.get("gpu_seconds", 0.0)
        mission_success = data.get("mission_outcome_success")
        
        for model_entry in models_used:
            # Handle both tuple and list formats
            if isinstance(model_entry, (list, tuple)) and len(model_entry) >= 1:
                model_name = model_entry[0]
                tier = model_entry[1] if len(model_entry) > 1 else "unknown"
            else:
                continue
            
            # Initialize accumulator if needed
            if model_name not in model_data:
                model_data[model_name] = {
                    "total_uses": 0,
                    "missions": set(),
                    "quality_scores": [],
                    "wall_times": [],
                    "gpu_seconds": [],
                    "success_count": 0,
                    "failure_count": 0,
                    "unknown_outcome_count": 0,
                    "tier_distribution": {},
                    "phase_type_distribution": {},
                }
            
            acc = model_data[model_name]
            acc["total_uses"] += 1
            acc["missions"].add(mission_id)
            
            # Quality score (only if not None)
            if quality_score is not None:
                acc["quality_scores"].append(quality_score)
            
            # Resource usage
            if wall_time > 0:
                acc["wall_times"].append(wall_time)
            if gpu_seconds > 0:
                acc["gpu_seconds"].append(gpu_seconds)
            
            # Mission outcome
            if mission_success is True:
                acc["success_count"] += 1
            elif mission_success is False:
                acc["failure_count"] += 1
            else:
                acc["unknown_outcome_count"] += 1
            
            # Tier distribution
            acc["tier_distribution"][tier] = acc["tier_distribution"].get(tier, 0) + 1
            
            # Phase type distribution
            acc["phase_type_distribution"][phase_type] = (
                acc["phase_type_distribution"].get(phase_type, 0) + 1
            )
    
    def get_stats(self, model_name: str) -> Optional[ModelStats]:
        """
        Get statistics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelStats or None if model not found
        """
        if not self._is_cache_valid():
            self._refresh_cache()
        
        return self._cache.get(model_name)
    
    def get_all_stats(self) -> Dict[str, ModelStats]:
        """
        Get statistics for all models.
        
        Returns:
            Dict mapping model name to ModelStats
        """
        if not self._is_cache_valid():
            self._refresh_cache()
        
        return dict(self._cache)
    
    def get_grade(self, model_name: str) -> str:
        """
        Get performance grade for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Grade label: "excellent", "good", "fair", "poor", or "unknown"
        """
        stats = self.get_stats(model_name)
        if stats is None:
            return "unknown"
        return stats.compute_grade()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all model statistics.
        
        Returns:
            Dict with summary information
        """
        if not self._is_cache_valid():
            self._refresh_cache()
        
        return {
            "total_models": len(self._cache),
            "cache_age_seconds": round(time.time() - self._cache_timestamp, 1),
            "outcomes_file": str(self.outcomes_file),
            "models": {
                name: stats.to_dict()
                for name, stats in self._cache.items()
            }
        }
    
    def invalidate_cache(self) -> None:
        """Force cache invalidation."""
        self._cache_timestamp = 0.0


# Global instance
_stats_store: Optional[ModelStatsStore] = None


def get_model_stats_store() -> ModelStatsStore:
    """Get the global model stats store instance."""
    global _stats_store
    if _stats_store is None:
        _stats_store = ModelStatsStore()
    return _stats_store

