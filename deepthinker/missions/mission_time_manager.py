"""
Mission Time Manager for DeepThinker 2.0.

Provides accurate wall-clock time tracking, phase-time vs council-time
separation, reserved synthesis time, and grace periods for final deliverables.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from ..orchestration.phase_time_allocator import PhaseTimeBudget, TimeAllocation

logger = logging.getLogger(__name__)


@dataclass
class PhaseTimeRecord:
    """
    Tracks time spent on a single phase.
    
    Attributes:
        phase_name: Name of the phase
        wall_start: Wall-clock start time (time.time())
        wall_end: Wall-clock end time
        council_time_seconds: Cumulative time spent in council executions
        iteration_count: Number of iterations in this phase
    """
    phase_name: str
    wall_start: float = 0.0
    wall_end: float = 0.0
    council_time_seconds: float = 0.0
    iteration_count: int = 0
    
    @property
    def wall_time_seconds(self) -> float:
        """Get total wall-clock time for this phase."""
        if self.wall_end > 0:
            return self.wall_end - self.wall_start
        elif self.wall_start > 0:
            return time.time() - self.wall_start
        return 0.0
    
    @property
    def overhead_seconds(self) -> float:
        """Get non-council overhead time (wall - council)."""
        return max(0.0, self.wall_time_seconds - self.council_time_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase_name": self.phase_name,
            "wall_time_seconds": self.wall_time_seconds,
            "council_time_seconds": self.council_time_seconds,
            "overhead_seconds": self.overhead_seconds,
            "iteration_count": self.iteration_count,
        }


@dataclass
class MissionTimeManager:
    """
    Manages time tracking for mission execution.
    
    Features:
    - Separate wall-clock vs council execution time tracking
    - Reserved time for synthesis phase
    - Grace period to ensure final deliverables are produced
    - Phase abort decisions based on remaining time
    
    Usage:
        time_manager = MissionTimeManager(
            total_budget_seconds=600,
            reserved_synthesis_time=120,
            grace_period_seconds=60
        )
        time_manager.start_mission()
        
        time_manager.start_phase("recon")
        # ... execute phase ...
        time_manager.end_phase("recon")
        
        if time_manager.should_abort_phase("deep_analysis"):
            # skip to synthesis
    """
    
    total_budget_seconds: float
    reserved_synthesis_time: float = 120.0
    grace_period_seconds: float = 60.0
    target_utilization: float = 0.8  # Target 80% budget utilization
    
    mission_start: float = field(default=0.0, init=False)
    phase_times: Dict[str, PhaseTimeRecord] = field(default_factory=dict, init=False)
    _current_phase: Optional[str] = field(default=None, init=False)
    _current_council_start: float = field(default=0.0, init=False)
    
    # Phase reservations
    _phase_reservations: Dict[str, float] = field(default_factory=dict, init=False)
    
    # Phase time budgets (from PhaseTimeAllocator)
    _phase_budgets: Dict[str, "PhaseTimeBudget"] = field(default_factory=dict, init=False)
    _time_allocation: Optional["TimeAllocation"] = field(default=None, init=False)
    
    def start_mission(self) -> None:
        """Start the mission timer."""
        self.mission_start = time.time()
        logger.debug(f"Mission started at {self.mission_start}")
    
    @property
    def wall_elapsed(self) -> float:
        """Get wall-clock time elapsed since mission start."""
        if self.mission_start <= 0:
            return 0.0
        return time.time() - self.mission_start
    
    @property
    def wall_remaining(self) -> float:
        """Get wall-clock time remaining in budget."""
        return max(0.0, self.total_budget_seconds - self.wall_elapsed)
    
    @property
    def effective_remaining(self) -> float:
        """
        Get effective remaining time (excluding reserved synthesis time).
        
        This is the time available for non-synthesis phases.
        """
        return max(0.0, self.wall_remaining - self.reserved_synthesis_time)
    
    @property
    def in_grace_period(self) -> bool:
        """Check if we're in the grace period (past budget but within grace)."""
        over_budget = self.wall_elapsed - self.total_budget_seconds
        return 0 < over_budget <= self.grace_period_seconds
    
    @property
    def past_grace_period(self) -> bool:
        """Check if we've exceeded even the grace period."""
        over_budget = self.wall_elapsed - self.total_budget_seconds
        return over_budget > self.grace_period_seconds
    
    def reserve_time_for(self, phase: str, min_seconds: float) -> None:
        """
        Reserve minimum time for a specific phase.
        
        Args:
            phase: Phase name to reserve time for
            min_seconds: Minimum seconds to reserve
        """
        self._phase_reservations[phase] = min_seconds
        logger.debug(f"Reserved {min_seconds}s for phase '{phase}'")
    
    def get_reserved_time(self, phase: str) -> float:
        """Get reserved time for a phase."""
        return self._phase_reservations.get(phase, 0.0)
    
    def total_reserved_time(self) -> float:
        """Get total reserved time across all phases."""
        return sum(self._phase_reservations.values()) + self.reserved_synthesis_time
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start tracking a new phase.
        
        Args:
            phase_name: Name of the phase to start
        """
        if phase_name not in self.phase_times:
            self.phase_times[phase_name] = PhaseTimeRecord(phase_name=phase_name)
        
        self.phase_times[phase_name].wall_start = time.time()
        self._current_phase = phase_name
        
        logger.debug(f"Started phase '{phase_name}' at wall_elapsed={self.wall_elapsed:.1f}s")
    
    def end_phase(self, phase_name: str) -> PhaseTimeRecord:
        """
        End tracking for a phase.
        
        Args:
            phase_name: Name of the phase to end
            
        Returns:
            PhaseTimeRecord with final timings
        """
        if phase_name in self.phase_times:
            self.phase_times[phase_name].wall_end = time.time()
            
            if self._current_phase == phase_name:
                self._current_phase = None
            
            record = self.phase_times[phase_name]
            logger.debug(
                f"Ended phase '{phase_name}': "
                f"wall={record.wall_time_seconds:.1f}s, "
                f"council={record.council_time_seconds:.1f}s"
            )
            return record
        
        return PhaseTimeRecord(phase_name=phase_name)
    
    def start_council_execution(self) -> None:
        """Mark the start of a council execution within current phase."""
        self._current_council_start = time.time()
    
    def end_council_execution(self, phase_name: Optional[str] = None) -> float:
        """
        Mark the end of a council execution and accumulate time.
        
        Args:
            phase_name: Phase to attribute time to (uses current if None)
            
        Returns:
            Duration of this council execution
        """
        if self._current_council_start <= 0:
            return 0.0
        
        duration = time.time() - self._current_council_start
        self._current_council_start = 0.0
        
        target_phase = phase_name or self._current_phase
        if target_phase and target_phase in self.phase_times:
            self.phase_times[target_phase].council_time_seconds += duration
        
        return duration
    
    def record_council_time(self, phase_name: str, duration_seconds: float) -> None:
        """
        Directly record council execution time for a phase.
        
        Args:
            phase_name: Phase to attribute time to
            duration_seconds: Duration to add
        """
        if phase_name not in self.phase_times:
            self.phase_times[phase_name] = PhaseTimeRecord(phase_name=phase_name)
        
        self.phase_times[phase_name].council_time_seconds += duration_seconds
    
    def increment_iteration(self, phase_name: str) -> int:
        """
        Increment iteration count for a phase.
        
        Args:
            phase_name: Phase to increment
            
        Returns:
            New iteration count
        """
        if phase_name not in self.phase_times:
            self.phase_times[phase_name] = PhaseTimeRecord(phase_name=phase_name)
        
        self.phase_times[phase_name].iteration_count += 1
        return self.phase_times[phase_name].iteration_count
    
    def should_abort_phase(self, phase_name: str) -> Tuple[bool, str]:
        """
        Determine if a phase should be aborted due to time constraints.
        
        Args:
            phase_name: Name of the phase to check
            
        Returns:
            Tuple of (should_abort, reason)
        """
        # Never abort synthesis during grace period
        if "synthesis" in phase_name.lower():
            if self.past_grace_period:
                return True, "Past grace period - must abort even synthesis"
            return False, "Synthesis allowed during grace period"
        
        # Check if we've exceeded budget + grace
        if self.past_grace_period:
            return True, f"Past grace period ({self.grace_period_seconds}s)"
        
        # Check if remaining time is enough for this phase + synthesis
        reserved = self.get_reserved_time(phase_name)
        min_needed = reserved + self.reserved_synthesis_time
        
        if self.wall_remaining < min_needed:
            return True, (
                f"Insufficient time: {self.wall_remaining:.1f}s remaining, "
                f"need {min_needed:.1f}s (phase={reserved:.1f}s + synthesis={self.reserved_synthesis_time:.1f}s)"
            )
        
        # Check effective remaining time (excluding synthesis reserve)
        if self.effective_remaining <= 0:
            return True, "No effective time remaining (synthesis reserve reached)"
        
        return False, f"Time available: {self.effective_remaining:.1f}s effective remaining"
    
    def can_run_synthesis(self) -> Tuple[bool, str]:
        """
        Check if synthesis can and should run.
        
        Returns:
            Tuple of (can_run, reason)
        """
        # Always allow synthesis if within grace period
        if self.wall_remaining > 0:
            return True, f"Within budget: {self.wall_remaining:.1f}s remaining"
        
        if self.in_grace_period:
            over_by = self.wall_elapsed - self.total_budget_seconds
            return True, f"In grace period: {over_by:.1f}s over budget"
        
        if self.past_grace_period:
            over_by = self.wall_elapsed - self.total_budget_seconds
            return False, f"Past grace period: {over_by:.1f}s over budget (grace={self.grace_period_seconds}s)"
        
        return True, "Time available for synthesis"
    
    def should_use_lightweight_synthesis(self) -> bool:
        """
        Determine if lightweight synthesis should be used due to time pressure.
        
        Returns:
            True if time is critically low and lightweight synthesis is needed
        """
        # Use lightweight if in grace period or very little time left
        if self.in_grace_period:
            return True
        
        # Also use lightweight if less than 30s remaining
        return self.wall_remaining < 30.0
    
    def get_phase_time_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of time spent per phase.
        
        Returns:
            Dictionary mapping phase_name -> {wall, council, overhead}
        """
        return {
            name: record.to_dict()
            for name, record in self.phase_times.items()
        }
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Get ordered timeline of phases with timing info.
        
        Returns:
            List of phase timing dictionaries in execution order
        """
        # Sort by wall_start time
        sorted_phases = sorted(
            self.phase_times.values(),
            key=lambda r: r.wall_start if r.wall_start > 0 else float('inf')
        )
        
        return [record.to_dict() for record in sorted_phases]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current time manager status.
        
        Returns:
            Status dictionary with all timing information
        """
        total_council_time = sum(
            r.council_time_seconds for r in self.phase_times.values()
        )
        
        return {
            "mission_started": self.mission_start > 0,
            "total_budget_seconds": self.total_budget_seconds,
            "wall_elapsed_seconds": self.wall_elapsed,
            "wall_remaining_seconds": self.wall_remaining,
            "effective_remaining_seconds": self.effective_remaining,
            "total_council_time_seconds": total_council_time,
            "reserved_synthesis_time": self.reserved_synthesis_time,
            "grace_period_seconds": self.grace_period_seconds,
            "in_grace_period": self.in_grace_period,
            "past_grace_period": self.past_grace_period,
            "current_phase": self._current_phase,
            "phases_completed": len([
                r for r in self.phase_times.values() if r.wall_end > 0
            ]),
            "phase_times": self.get_phase_time_summary(),
        }
    
    def format_timeline_string(self) -> str:
        """
        Format a human-readable timeline string.
        
        Returns:
            Formatted timeline string for CLI display
        """
        lines = ["Timeline:"]
        
        for record in sorted(
            self.phase_times.values(),
            key=lambda r: r.wall_start if r.wall_start > 0 else float('inf')
        ):
            wall = record.wall_time_seconds
            council = record.council_time_seconds
            lines.append(
                f"  - {record.phase_name}: {wall:.0f}s wall, {council:.0f}s council"
            )
        
        # Add totals
        total_wall = self.wall_elapsed
        total_council = sum(r.council_time_seconds for r in self.phase_times.values())
        lines.append(f"  Total: {total_wall:.0f}s wall, {total_council:.0f}s council")
        
        return "\n".join(lines)
    
    # ========== Phase Budget Methods ==========
    
    def set_time_allocation(self, allocation: "TimeAllocation") -> None:
        """
        Set the time allocation for all phases.
        
        Args:
            allocation: TimeAllocation from PhaseTimeAllocator
        """
        self._time_allocation = allocation
        self._phase_budgets = allocation.phase_budgets.copy()
        logger.debug(
            f"Set time allocation: {len(self._phase_budgets)} phases, "
            f"scale={allocation.scale_factor:.2f}, "
            f"target_util={allocation.target_utilization:.0%}"
        )
    
    def get_phase_budget(self, phase_name: str) -> Optional["PhaseTimeBudget"]:
        """
        Get the time budget for a specific phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            PhaseTimeBudget or None if not allocated
        """
        return self._phase_budgets.get(phase_name)
    
    def get_phase_time_remaining(self, phase_name: str) -> float:
        """
        Get remaining time in a phase's budget.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Remaining seconds (inf if no budget set)
        """
        budget = self._phase_budgets.get(phase_name)
        if budget is None:
            return float('inf')
        
        # Get time used from phase_times record
        record = self.phase_times.get(phase_name)
        if record is None:
            return budget.allocated_seconds
        
        used = record.wall_time_seconds
        return max(0.0, budget.max_seconds - used)
    
    def should_deepen_phase(
        self,
        phase_name: str,
        convergence_score: float,
        deepening_rounds_done: int,
        min_iteration_seconds: float = 30.0,
        max_deepening_rounds: int = 2,  # Reduced from 3 to prevent runaway deepening
        convergence_threshold: float = 0.7,
        max_phase_time_pct: float = 0.8  # Force completion after using 80% of phase budget
    ) -> Tuple[bool, str]:
        """
        Determine if a phase should be deepened with more iterations.
        
        Deepening is allowed when:
        - Quality hasn't plateaued (convergence < threshold)
        - Max deepening rounds not reached
        - Time budget allows another iteration
        - Overall mission time allows
        - Phase hasn't used more than max_phase_time_pct of its allocated time
        
        Args:
            phase_name: Name of the phase
            convergence_score: Current convergence/plateau score (0-1)
            deepening_rounds_done: Number of deepening rounds already done
            min_iteration_seconds: Minimum time needed for one iteration
            max_deepening_rounds: Maximum deepening rounds allowed
            convergence_threshold: Score above which quality has plateaued
            max_phase_time_pct: Force phase completion after this percentage of budget
            
        Returns:
            Tuple of (should_deepen, reason)
        """
        # Check convergence (quality plateau)
        if convergence_score >= convergence_threshold:
            return False, f"Quality plateaued (convergence={convergence_score:.2f} >= {convergence_threshold})"
        
        # Check max deepening rounds
        if deepening_rounds_done >= max_deepening_rounds:
            return False, f"Max deepening rounds reached ({max_deepening_rounds})"
        
        # Check overall mission time
        if self.effective_remaining < min_iteration_seconds:
            return False, f"Insufficient mission time ({self.effective_remaining:.0f}s < {min_iteration_seconds}s)"
        
        # Check phase budget if set
        budget = self._phase_budgets.get(phase_name)
        if budget is not None:
            record = self.phase_times.get(phase_name)
            time_used = record.wall_time_seconds if record else 0.0
            
            # NEW: Hard time-based forced completion
            # If phase has used more than 80% of its allocated time, force completion
            time_pct_used = time_used / budget.allocated_seconds if budget.allocated_seconds > 0 else 1.0
            if time_pct_used >= max_phase_time_pct:
                return False, (
                    f"Time budget exhausted ({time_pct_used*100:.0f}% used >= {max_phase_time_pct*100:.0f}% limit) - "
                    "forcing phase completion"
                )
            
            if not budget.can_deepen(time_used, min_iteration_seconds):
                remaining = budget.max_seconds - time_used
                return False, f"Phase budget exhausted ({remaining:.0f}s < {min_iteration_seconds}s)"
        
        # Deepening is allowed
        phase_remaining = self.get_phase_time_remaining(phase_name)
        return True, f"Deepening allowed: {phase_remaining:.0f}s phase budget, convergence={convergence_score:.2f}"
    
    def get_utilization(self) -> float:
        """
        Get current budget utilization ratio.
        
        Returns:
            Fraction of total budget used (0-1+)
        """
        if self.total_budget_seconds <= 0:
            return 0.0
        return self.wall_elapsed / self.total_budget_seconds
    
    def utilization_target_met(self) -> bool:
        """
        Check if utilization has met the target.
        
        Returns:
            True if utilization >= target_utilization
        """
        return self.get_utilization() >= self.target_utilization
    
    def time_to_target_utilization(self) -> float:
        """
        Get time needed to reach target utilization.
        
        Returns:
            Seconds remaining to reach target (negative if exceeded)
        """
        target_time = self.total_budget_seconds * self.target_utilization
        return target_time - self.wall_elapsed
    
    def get_utilization_status(self) -> Dict[str, Any]:
        """
        Get detailed utilization status.
        
        Returns:
            Dictionary with utilization metrics
        """
        utilization = self.get_utilization()
        target_time = self.total_budget_seconds * self.target_utilization
        
        return {
            "current_utilization": utilization,
            "target_utilization": self.target_utilization,
            "target_met": utilization >= self.target_utilization,
            "time_used_seconds": self.wall_elapsed,
            "target_time_seconds": target_time,
            "time_to_target_seconds": target_time - self.wall_elapsed,
            "budget_remaining_seconds": self.wall_remaining,
        }


def create_time_manager_from_constraints(
    time_budget_minutes: float,
    reserved_synthesis_minutes: float = 2.0,
    grace_period_seconds: float = 60.0
) -> MissionTimeManager:
    """
    Factory function to create MissionTimeManager from mission constraints.
    
    Args:
        time_budget_minutes: Total time budget in minutes
        reserved_synthesis_minutes: Minutes to reserve for synthesis
        grace_period_seconds: Grace period in seconds
        
    Returns:
        Configured MissionTimeManager instance
    """
    return MissionTimeManager(
        total_budget_seconds=time_budget_minutes * 60.0,
        reserved_synthesis_time=reserved_synthesis_minutes * 60.0,
        grace_period_seconds=grace_period_seconds
    )

