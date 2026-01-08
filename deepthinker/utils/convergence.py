"""
Convergence Tracking Utilities for DeepThinker 2.0.

Provides time-based and quality-based convergence detection
for iterative council workflows.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any


class StoppingReason(Enum):
    """Reasons for stopping iteration."""
    CONTINUING = "continuing"
    TIME_EXHAUSTED = "time_exhausted"
    CONVERGENCE_REACHED = "convergence_reached"
    QUALITY_THRESHOLD_MET = "quality_threshold_met"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    FORCED_STOP = "forced_stop"


@dataclass
class ConvergenceStatus:
    """
    Status of convergence tracking.
    
    Attributes:
        should_continue: Whether iteration should continue
        reason: Reason for the decision
        iterations_completed: Number of iterations completed
        quality_score: Latest quality score
        quality_delta: Change from previous iteration
        consecutive_small_deltas: Count of consecutive small improvements
        time_elapsed_seconds: Total time elapsed
        time_remaining_seconds: Estimated time remaining
        convergence_score: Overall convergence score (0-1, higher = more converged)
    """
    
    should_continue: bool
    reason: StoppingReason
    iterations_completed: int = 0
    quality_score: float = 0.0
    quality_delta: float = 0.0
    consecutive_small_deltas: int = 0
    time_elapsed_seconds: float = 0.0
    time_remaining_seconds: float = 0.0
    convergence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_continue": self.should_continue,
            "reason": self.reason.value,
            "iterations_completed": self.iterations_completed,
            "quality_score": self.quality_score,
            "quality_delta": self.quality_delta,
            "consecutive_small_deltas": self.consecutive_small_deltas,
            "time_elapsed_seconds": self.time_elapsed_seconds,
            "time_remaining_seconds": self.time_remaining_seconds,
            "convergence_score": self.convergence_score,
        }


@dataclass
class IterationRecord:
    """Record of a single iteration."""
    
    iteration: int
    quality_score: float
    timestamp: datetime
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConvergenceTracker:
    """
    Tracks convergence for iterative workflows.
    
    Supports three stopping modes:
    1. Time exhaustion: Stop when time budget is nearly depleted
    2. Convergence threshold: Stop when improvement plateaus
    3. Max iterations: Stop after maximum iteration count
    
    Convergence detection uses:
    - Consecutive small improvement tracking (improvement < threshold for N iterations)
    - Quality score plateau detection
    - Time-based urgency scaling
    """
    
    def __init__(
        self,
        time_budget_seconds: float = 360.0,  # 6 minutes default
        min_iteration_time_seconds: float = 30.0,  # Minimum time for one iteration
        convergence_threshold: float = 0.02,  # Quality improvement threshold
        consecutive_threshold_count: int = 2,  # Consecutive small improvements to trigger
        max_iterations: int = 10,
        quality_threshold: float = 8.0,  # Quality score to stop early
    ):
        """
        Initialize convergence tracker.
        
        Args:
            time_budget_seconds: Total time budget
            min_iteration_time_seconds: Minimum time needed for one iteration
            convergence_threshold: Improvement below this is considered "small"
            consecutive_threshold_count: Number of consecutive small improvements to converge
            max_iterations: Maximum iterations allowed
            quality_threshold: Quality score that triggers early stop
        """
        self.time_budget_seconds = time_budget_seconds
        self.min_iteration_time_seconds = min_iteration_time_seconds
        self.convergence_threshold = convergence_threshold
        self.consecutive_threshold_count = consecutive_threshold_count
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        
        # State
        self._start_time: Optional[datetime] = None
        self._history: List[IterationRecord] = []
        self._consecutive_small_deltas: int = 0
        self._forced_stop: bool = False
    
    def start(self) -> None:
        """Start or reset the convergence tracker."""
        self._start_time = datetime.now()
        self._history = []
        self._consecutive_small_deltas = 0
        self._forced_stop = False
    
    def record_iteration(
        self,
        quality_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConvergenceStatus:
        """
        Record an iteration and check convergence.
        
        Args:
            quality_score: Quality score for this iteration
            metadata: Optional metadata to store
            
        Returns:
            ConvergenceStatus with continue/stop decision
        """
        if self._start_time is None:
            self.start()
        
        now = datetime.now()
        iteration = len(self._history) + 1
        
        # Calculate duration
        if self._history:
            prev_time = self._history[-1].timestamp
            duration = (now - prev_time).total_seconds()
        else:
            duration = (now - self._start_time).total_seconds()
        
        # Record iteration
        record = IterationRecord(
            iteration=iteration,
            quality_score=quality_score,
            timestamp=now,
            duration_seconds=duration,
            metadata=metadata or {}
        )
        self._history.append(record)
        
        # Check convergence
        return self.check_status()
    
    def check_status(self) -> ConvergenceStatus:
        """
        Check current convergence status.
        
        Returns:
            ConvergenceStatus with current state
        """
        if not self._history:
            return ConvergenceStatus(
                should_continue=True,
                reason=StoppingReason.CONTINUING
            )
        
        latest = self._history[-1]
        iterations = len(self._history)
        
        # Calculate time metrics
        elapsed = self._get_elapsed_seconds()
        remaining = max(0, self.time_budget_seconds - elapsed)
        
        # Calculate quality delta
        quality_delta = 0.0
        if len(self._history) >= 2:
            prev = self._history[-2]
            quality_delta = latest.quality_score - prev.quality_score
            
            # Track consecutive small deltas
            if abs(quality_delta) < self.convergence_threshold:
                self._consecutive_small_deltas += 1
            else:
                self._consecutive_small_deltas = 0
        
        # Calculate convergence score (higher = more converged)
        convergence_score = self._calculate_convergence_score()
        
        # Build base status
        status = ConvergenceStatus(
            should_continue=True,
            reason=StoppingReason.CONTINUING,
            iterations_completed=iterations,
            quality_score=latest.quality_score,
            quality_delta=quality_delta,
            consecutive_small_deltas=self._consecutive_small_deltas,
            time_elapsed_seconds=elapsed,
            time_remaining_seconds=remaining,
            convergence_score=convergence_score
        )
        
        # Check stopping conditions
        
        # 1. Forced stop
        if self._forced_stop:
            status.should_continue = False
            status.reason = StoppingReason.FORCED_STOP
            return status
        
        # 2. Quality threshold met
        if latest.quality_score >= self.quality_threshold:
            status.should_continue = False
            status.reason = StoppingReason.QUALITY_THRESHOLD_MET
            return status
        
        # 3. Max iterations reached
        if iterations >= self.max_iterations:
            status.should_continue = False
            status.reason = StoppingReason.MAX_ITERATIONS_REACHED
            return status
        
        # 4. Time exhausted
        if remaining < self.min_iteration_time_seconds:
            status.should_continue = False
            status.reason = StoppingReason.TIME_EXHAUSTED
            return status
        
        # 5. Convergence reached (consecutive small improvements)
        if self._consecutive_small_deltas >= self.consecutive_threshold_count:
            status.should_continue = False
            status.reason = StoppingReason.CONVERGENCE_REACHED
            return status
        
        return status
    
    def force_stop(self) -> None:
        """Force the tracker to indicate stopping."""
        self._forced_stop = True
    
    def get_history(self) -> List[IterationRecord]:
        """Get iteration history."""
        return list(self._history)
    
    def get_best_iteration(self) -> Optional[IterationRecord]:
        """Get the iteration with the highest quality score."""
        if not self._history:
            return None
        return max(self._history, key=lambda x: x.quality_score)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the convergence tracking."""
        if not self._history:
            return {
                "iterations": 0,
                "best_score": 0.0,
                "final_score": 0.0,
                "improvement": 0.0,
                "time_used_seconds": 0.0,
                "converged": False,
                "convergence_reason": None
            }
        
        status = self.check_status()
        best = self.get_best_iteration()
        
        return {
            "iterations": len(self._history),
            "best_score": best.quality_score if best else 0.0,
            "best_iteration": best.iteration if best else 0,
            "final_score": self._history[-1].quality_score,
            "improvement": self._history[-1].quality_score - self._history[0].quality_score,
            "time_used_seconds": status.time_elapsed_seconds,
            "time_remaining_seconds": status.time_remaining_seconds,
            "converged": not status.should_continue,
            "convergence_reason": status.reason.value,
            "convergence_score": status.convergence_score,
            "scores_per_iteration": [r.quality_score for r in self._history]
        }
    
    def _get_elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()
    
    def _calculate_convergence_score(self) -> float:
        """
        Calculate overall convergence score.
        
        Higher score (closer to 1.0) means more converged.
        Based on:
        - Quality score plateau
        - Consecutive small improvements
        - Score variance in recent iterations
        """
        if len(self._history) < 2:
            return 0.0
        
        # Factor 1: Consecutive small deltas
        consecutive_factor = min(1.0, self._consecutive_small_deltas / self.consecutive_threshold_count)
        
        # Factor 2: Recent score variance (low variance = high convergence)
        recent = [r.quality_score for r in self._history[-3:]]
        if len(recent) >= 2:
            import statistics
            try:
                variance = statistics.variance(recent)
                variance_factor = max(0.0, 1.0 - variance * 10)  # Scale variance
            except statistics.StatisticsError:
                variance_factor = 0.5
        else:
            variance_factor = 0.0
        
        # Factor 3: Improvement rate slowdown
        if len(self._history) >= 3:
            early_improvement = self._history[1].quality_score - self._history[0].quality_score
            recent_improvement = self._history[-1].quality_score - self._history[-2].quality_score
            
            if abs(early_improvement) > 0.01:
                slowdown_ratio = abs(recent_improvement) / abs(early_improvement)
                slowdown_factor = max(0.0, 1.0 - slowdown_ratio)
            else:
                slowdown_factor = 0.5
        else:
            slowdown_factor = 0.0
        
        # Weighted combination
        convergence_score = (
            0.5 * consecutive_factor +
            0.3 * variance_factor +
            0.2 * slowdown_factor
        )
        
        return min(1.0, max(0.0, convergence_score))
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        elapsed = self._get_elapsed_seconds()
        return max(0, self.time_budget_seconds - elapsed)
    
    def time_exhausted(self) -> bool:
        """Check if time is exhausted."""
        return self.time_remaining() < self.min_iteration_time_seconds
    
    def estimate_iterations_remaining(self) -> int:
        """
        Estimate how many more iterations can fit in remaining time.
        
        Based on average iteration duration.
        """
        if not self._history:
            return self.max_iterations
        
        # Calculate average iteration duration
        durations = [r.duration_seconds for r in self._history if r.duration_seconds > 0]
        if not durations:
            avg_duration = self.min_iteration_time_seconds
        else:
            avg_duration = sum(durations) / len(durations)
        
        remaining_time = self.time_remaining()
        estimated = int(remaining_time / max(avg_duration, self.min_iteration_time_seconds))
        
        # Cap by max iterations remaining
        iterations_done = len(self._history)
        max_remaining = self.max_iterations - iterations_done
        
        return min(estimated, max_remaining)

