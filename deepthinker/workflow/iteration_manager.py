"""
Iteration Manager for DeepThinker 2.0.

Manages council iteration loops, disagreement detection, and convergence checking.

Enhanced for autonomous execution:
- Time-based stopping
- Convergence-based stopping with consecutive improvement tracking
- Multi-view disagreement tracking
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Verbose logging integration
try:
    from ..cli import verbose_logger
    VERBOSE_LOGGER_AVAILABLE = True
except ImportError:
    VERBOSE_LOGGER_AVAILABLE = False
    verbose_logger = None

# Convergence utilities
try:
    from ..utils.convergence import ConvergenceTracker, StoppingReason
    CONVERGENCE_AVAILABLE = True
except ImportError:
    CONVERGENCE_AVAILABLE = False


@dataclass
class IterationResult:
    """Result from a single iteration."""
    
    iteration: int
    code: str
    quality_score: float
    passed: bool
    council_outputs: Dict[str, Any] = field(default_factory=dict)
    arbiter_decision: Optional[Any] = None
    convergence_score: float = 0.0
    duration_seconds: float = 0.0
    timestamp: Optional[datetime] = None
    
    # Multi-view outputs
    optimist_output: Optional[Any] = None
    skeptic_output: Optional[Any] = None
    multiview_agreement: float = 0.0  # 0-1, how much optimist and skeptic agree


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence detection."""
    
    score_improvement: float
    score_variance: float
    code_similarity: float
    is_converged: bool
    reason: str
    consecutive_small_improvements: int = 0
    time_elapsed_seconds: float = 0.0
    time_remaining_seconds: float = 0.0


class IterationManager:
    """
    Manages iteration loops for council-based workflow.
    
    Handles:
    - Council iteration loops
    - Disagreement detection between councils
    - Convergence checking (quality, time, consecutive improvement)
    - Early termination decisions
    - Multi-view disagreement tracking
    
    Stopping modes:
    1. Time exhaustion: Stop when time budget nearly depleted
    2. Convergence threshold: Stop when improvement < 0.02 for 2 consecutive iterations
    3. Quality threshold: Stop when quality exceeds threshold
    4. Max iterations: Stop after maximum iteration count
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        quality_threshold: float = 7.0,
        convergence_threshold: float = 0.02,
        min_iterations: int = 2,
        time_budget_seconds: Optional[float] = None,
        min_iteration_time_seconds: float = 30.0,
        consecutive_convergence_count: int = 2
    ):
        """
        Initialize iteration manager.
        
        Args:
            max_iterations: Maximum iterations allowed
            quality_threshold: Score threshold for early termination
            convergence_threshold: Improvement below this is considered "converged"
            min_iterations: Minimum iterations before early termination
            time_budget_seconds: Total time budget (None = no time limit)
            min_iteration_time_seconds: Minimum time needed for one iteration
            consecutive_convergence_count: Number of consecutive small improvements to trigger convergence
        """
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.time_budget_seconds = time_budget_seconds
        self.min_iteration_time_seconds = min_iteration_time_seconds
        self.consecutive_convergence_count = consecutive_convergence_count
        
        self.iteration_history: List[IterationResult] = []
        self.current_iteration = 0
        
        # Time tracking
        self._start_time: Optional[datetime] = None
        self._iteration_start_time: Optional[datetime] = None
        
        # Convergence tracking
        self._consecutive_small_improvements: int = 0
    
    def start(self) -> None:
        """Start or reset the iteration manager for a new workflow."""
        self._start_time = datetime.now()
        self.iteration_history = []
        self.current_iteration = 0
        self._consecutive_small_improvements = 0
    
    def start_iteration(self) -> int:
        """
        Start a new iteration.
        
        Returns:
            Current iteration number (1-indexed)
        """
        if self._start_time is None:
            self.start()
        
        self.current_iteration += 1
        self._iteration_start_time = datetime.now()
        return self.current_iteration
    
    def record_iteration(
        self,
        code: str,
        quality_score: float,
        passed: bool,
        council_outputs: Dict[str, Any],
        arbiter_decision: Optional[Any] = None,
        optimist_output: Optional[Any] = None,
        skeptic_output: Optional[Any] = None
    ) -> IterationResult:
        """
        Record the results of an iteration.
        
        Args:
            code: Generated code
            quality_score: Quality score from evaluator
            passed: Whether evaluation passed
            council_outputs: Outputs from each council
            arbiter_decision: Final arbiter decision
            optimist_output: Output from OptimistCouncil
            skeptic_output: Output from SkepticCouncil
            
        Returns:
            IterationResult with convergence score
        """
        now = datetime.now()
        
        # Calculate iteration duration
        duration = 0.0
        if self._iteration_start_time:
            duration = (now - self._iteration_start_time).total_seconds()
        
        # Track consecutive small improvements
        if len(self.iteration_history) >= 1:
            prev_score = self.iteration_history[-1].quality_score
            improvement = quality_score - prev_score
            if abs(improvement) < self.convergence_threshold:
                self._consecutive_small_improvements += 1
            else:
                self._consecutive_small_improvements = 0
        
        # Calculate convergence score
        convergence_score = self._calculate_convergence()
        
        # Calculate multi-view agreement
        multiview_agreement = self._calculate_multiview_agreement(optimist_output, skeptic_output)
        
        result = IterationResult(
            iteration=self.current_iteration,
            code=code,
            quality_score=quality_score,
            passed=passed,
            council_outputs=council_outputs,
            arbiter_decision=arbiter_decision,
            convergence_score=convergence_score,
            duration_seconds=duration,
            timestamp=now,
            optimist_output=optimist_output,
            skeptic_output=skeptic_output,
            multiview_agreement=multiview_agreement
        )
        
        self.iteration_history.append(result)
        
        # Verbose logging: iteration summary
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            notes = []
            if convergence_score > 0:
                notes.append(f"Convergence: {convergence_score:.2f}")
            if self._consecutive_small_improvements > 0:
                notes.append(f"Small improvements: {self._consecutive_small_improvements}")
            if duration > 0:
                notes.append(f"Duration: {duration:.1f}s")
            
            verbose_logger.log_iteration_summary({
                "iteration": self.current_iteration,
                "quality_score": quality_score,
                "passed": passed,
                "notes": " | ".join(notes) if notes else ""
            })
        
        return result
    
    def _calculate_multiview_agreement(
        self,
        optimist_output: Optional[Any],
        skeptic_output: Optional[Any]
    ) -> float:
        """Calculate agreement between optimist and skeptic perspectives."""
        if optimist_output is None or skeptic_output is None:
            return 0.0
        
        # Extract confidence from both
        opt_conf = getattr(optimist_output, 'confidence', 0.5)
        skep_conf = getattr(skeptic_output, 'confidence', 0.5)
        
        # Agreement is higher when confidences are similar
        # (both confident or both uncertain suggests agreement)
        conf_diff = abs(opt_conf - skep_conf)
        agreement = 1.0 - conf_diff
        
        return agreement
    
    def should_continue(self) -> Tuple[bool, str]:
        """
        Determine if iteration should continue.
        
        Checks stopping conditions in order:
        1. Time exhaustion
        2. Max iterations reached
        3. Quality threshold met
        4. Convergence reached (consecutive small improvements)
        
        Returns:
            Tuple of (should_continue, reason)
        """
        # Check time exhaustion first (highest priority)
        if self.time_exhausted():
            remaining = self.time_remaining()
            return False, f"Time exhausted ({remaining:.0f}s remaining)"
        
        # Check max iterations
        if self.current_iteration >= self.max_iterations:
            return False, "Maximum iterations reached"
        
        # Need history to make decisions
        if not self.iteration_history:
            return True, "No iterations completed yet"
        
        latest = self.iteration_history[-1]
        
        # Check quality threshold
        if latest.quality_score >= self.quality_threshold:
            return False, f"Quality threshold reached ({latest.quality_score:.1f} >= {self.quality_threshold})"
        
        # Check minimum iterations
        if self.current_iteration < self.min_iterations:
            return True, f"Minimum iterations not reached ({self.current_iteration} < {self.min_iterations})"
        
        # Check convergence (consecutive small improvements)
        if self._consecutive_small_improvements >= self.consecutive_convergence_count:
            return False, f"Converged: {self._consecutive_small_improvements} consecutive small improvements"
        
        # Check full convergence metrics
        convergence = self.check_convergence()
        if convergence.is_converged:
            return False, f"Converged: {convergence.reason}"
        
        return True, "Continuing iteration"
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        if self.time_budget_seconds is None:
            return float('inf')
        
        if self._start_time is None:
            return self.time_budget_seconds
        
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return max(0, self.time_budget_seconds - elapsed)
    
    def time_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()
    
    def time_exhausted(self) -> bool:
        """Check if time budget is exhausted."""
        if self.time_budget_seconds is None:
            return False
        return self.time_remaining() < self.min_iteration_time_seconds
    
    def estimate_iterations_remaining(self) -> int:
        """
        Estimate how many more iterations can fit in remaining time.
        
        Based on average iteration duration.
        """
        if not self.iteration_history:
            return self.max_iterations - self.current_iteration
        
        # Calculate average iteration duration
        durations = [r.duration_seconds for r in self.iteration_history if r.duration_seconds > 0]
        if not durations:
            avg_duration = self.min_iteration_time_seconds
        else:
            avg_duration = sum(durations) / len(durations)
        
        remaining_time = self.time_remaining()
        if remaining_time == float('inf'):
            return self.max_iterations - self.current_iteration
        
        estimated = int(remaining_time / max(avg_duration, self.min_iteration_time_seconds))
        max_remaining = self.max_iterations - self.current_iteration
        
        return min(estimated, max_remaining)
    
    def check_convergence(self) -> ConvergenceMetrics:
        """
        Check if iterations have converged.
        
        Returns:
            ConvergenceMetrics with convergence status
        """
        elapsed = self.time_elapsed()
        remaining = self.time_remaining()
        
        if len(self.iteration_history) < 2:
            return ConvergenceMetrics(
                score_improvement=0.0,
                score_variance=0.0,
                code_similarity=0.0,
                is_converged=False,
                reason="Not enough iterations for convergence check",
                consecutive_small_improvements=0,
                time_elapsed_seconds=elapsed,
                time_remaining_seconds=remaining
            )
        
        # Calculate score improvement
        recent_scores = [r.quality_score for r in self.iteration_history[-3:]]
        score_improvement = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0.0
        
        # Calculate score variance
        import statistics
        score_variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.0
        
        # Check code similarity (simple length-based heuristic)
        recent_codes = [r.code for r in self.iteration_history[-2:]]
        if len(recent_codes) == 2 and recent_codes[0] and recent_codes[1]:
            len_diff = abs(len(recent_codes[0]) - len(recent_codes[1]))
            avg_len = (len(recent_codes[0]) + len(recent_codes[1])) / 2
            code_similarity = 1.0 - (len_diff / max(avg_len, 1))
        else:
            code_similarity = 0.0
        
        # Determine convergence
        is_converged = False
        reason = ""
        
        # Check consecutive small improvements (primary convergence criterion)
        if self._consecutive_small_improvements >= self.consecutive_convergence_count:
            is_converged = True
            reason = f"{self._consecutive_small_improvements} consecutive improvements < {self.convergence_threshold}"
        elif abs(score_improvement) < self.convergence_threshold and len(self.iteration_history) >= 3:
            is_converged = True
            reason = f"Score improvement below threshold ({score_improvement:.3f} < {self.convergence_threshold})"
        elif score_variance < 0.005 and len(self.iteration_history) >= 3:
            is_converged = True
            reason = f"Score variance too low ({score_variance:.4f})"
        elif code_similarity > 0.98:
            is_converged = True
            reason = f"Code similarity too high ({code_similarity:.3f})"
        
        return ConvergenceMetrics(
            score_improvement=score_improvement,
            score_variance=score_variance,
            code_similarity=code_similarity,
            is_converged=is_converged,
            reason=reason,
            consecutive_small_improvements=self._consecutive_small_improvements,
            time_elapsed_seconds=elapsed,
            time_remaining_seconds=remaining
        )
    
    def detect_council_disagreement(
        self,
        council_outputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Detect disagreement between council outputs.
        
        Args:
            council_outputs: Dictionary of council name -> output
            
        Returns:
            Dictionary of council pairs -> disagreement score (0-1)
        """
        disagreements = {}
        council_names = list(council_outputs.keys())
        
        for i, name1 in enumerate(council_names):
            for name2 in council_names[i+1:]:
                # Simple string-based disagreement detection
                output1 = str(council_outputs[name1])
                output2 = str(council_outputs[name2])
                
                # Use length difference as a proxy for disagreement
                len_diff = abs(len(output1) - len(output2))
                avg_len = (len(output1) + len(output2)) / 2
                
                disagreement = min(1.0, len_diff / max(avg_len, 1))
                disagreements[f"{name1}_vs_{name2}"] = disagreement
        
        return disagreements
    
    def get_best_iteration(self) -> Optional[IterationResult]:
        """
        Get the best iteration result based on quality score.
        
        Returns:
            Best IterationResult or None if no iterations
        """
        if not self.iteration_history:
            return None
        
        return max(self.iteration_history, key=lambda x: x.quality_score)
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all iterations.
        
        Returns:
            Summary dictionary
        """
        if not self.iteration_history:
            return {
                "total_iterations": 0,
                "best_score": 0.0,
                "final_score": 0.0,
                "improvement": 0.0,
                "passed": False,
                "time_elapsed_seconds": 0.0,
                "converged": False,
                "convergence_reason": None
            }
        
        scores = [r.quality_score for r in self.iteration_history]
        best = self.get_best_iteration()
        convergence = self.check_convergence()
        should_continue, reason = self.should_continue()
        
        # Calculate total duration
        total_duration = sum(r.duration_seconds for r in self.iteration_history)
        
        # Get multi-view data
        multiview_agreements = [r.multiview_agreement for r in self.iteration_history if r.multiview_agreement > 0]
        avg_multiview_agreement = sum(multiview_agreements) / len(multiview_agreements) if multiview_agreements else 0.0
        
        return {
            "total_iterations": len(self.iteration_history),
            "best_score": max(scores),
            "best_iteration": best.iteration if best else 0,
            "final_score": scores[-1],
            "improvement": scores[-1] - scores[0],
            "passed": self.iteration_history[-1].passed,
            "scores_per_iteration": scores,
            "time_elapsed_seconds": self.time_elapsed(),
            "time_remaining_seconds": self.time_remaining(),
            "total_duration_seconds": total_duration,
            "avg_iteration_duration": total_duration / len(self.iteration_history),
            "converged": not should_continue,
            "convergence_reason": reason,
            "consecutive_small_improvements": self._consecutive_small_improvements,
            "convergence_score": convergence.code_similarity,
            "avg_multiview_agreement": avg_multiview_agreement
        }
    
    def _calculate_convergence(self) -> float:
        """Calculate convergence score based on recent history."""
        if len(self.iteration_history) < 2:
            return 0.0
        
        recent = self.iteration_history[-3:]
        scores = [r.quality_score for r in recent]
        
        if len(scores) < 2:
            return 0.0
        
        # Factor 1: Consecutive small improvements
        consecutive_factor = min(1.0, self._consecutive_small_improvements / self.consecutive_convergence_count)
        
        # Factor 2: Score variance (low = more converged)
        import statistics
        try:
            variance = statistics.variance(scores)
            variance_factor = max(0.0, 1.0 - variance * 10)
        except statistics.StatisticsError:
            variance_factor = 0.5
        
        # Factor 3: Improvement rate slowdown
        if len(self.iteration_history) >= 3:
            early_improvement = self.iteration_history[1].quality_score - self.iteration_history[0].quality_score
            recent_improvement = self.iteration_history[-1].quality_score - self.iteration_history[-2].quality_score
            
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
        
        return max(0.0, min(1.0, convergence_score))
    
    def reset(self) -> None:
        """Reset iteration manager for a new workflow."""
        self.iteration_history = []
        self.current_iteration = 0
        self._start_time = None
        self._iteration_start_time = None
        self._consecutive_small_improvements = 0
    
    def set_time_budget(self, seconds: float) -> None:
        """
        Set or update the time budget.
        
        Args:
            seconds: Time budget in seconds
        """
        self.time_budget_seconds = seconds
    
    def log_final_summary(self) -> None:
        """Log the final iteration summary table."""
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_iteration_table(self.iteration_history)
            
            # Also log convergence evolution if available
            if hasattr(verbose_logger, 'log_convergence_evolution'):
                summary = self.get_iteration_summary()
                verbose_logger.log_convergence_evolution(summary)

