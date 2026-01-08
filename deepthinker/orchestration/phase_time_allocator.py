"""
Phase Time Allocator for DeepThinker.

Allocates time budgets to phases upfront using historical PolicyMemory data.
Ensures target utilization (default 80%) of mission time budget by scaling
phase allocations proportionally.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .policy_memory import PolicyMemory

logger = logging.getLogger(__name__)

# Default time estimates per phase type when no historical data available
DEFAULT_PHASE_TIMES: Dict[str, float] = {
    "reconnaissance": 60.0,
    "recon": 60.0,
    "research": 90.0,
    "analysis": 120.0,
    "planning": 60.0,
    "design": 90.0,
    "deep_analysis": 180.0,
    "implementation": 120.0,
    "coding": 120.0,
    "testing": 90.0,
    "simulation": 90.0,
    "synthesis": 120.0,
    "review": 60.0,
}

# Phase priority weights (higher = more important, gets more slack time)
PHASE_PRIORITIES: Dict[str, float] = {
    "reconnaissance": 0.5,
    "recon": 0.5,
    "research": 0.8,
    "analysis": 0.9,
    "planning": 0.6,
    "design": 0.7,
    "deep_analysis": 1.0,
    "implementation": 0.8,
    "coding": 0.8,
    "testing": 0.7,
    "simulation": 0.7,
    "synthesis": 0.9,
    "review": 0.5,
}


@dataclass
class PhaseTimeBudget:
    """
    Time budget allocation for a single phase.
    
    Attributes:
        phase_name: Name of the phase
        phase_type: Type classification (research, analysis, etc.)
        allocated_seconds: Target time to spend on this phase
        min_seconds: Minimum time (from historical average)
        max_seconds: Maximum time allowed (for deepening)
        priority: Priority weight (0-1, higher = more important)
        estimated_from_history: Whether estimate came from historical data
    """
    phase_name: str
    phase_type: str
    allocated_seconds: float
    min_seconds: float
    max_seconds: float
    priority: float = 0.5
    estimated_from_history: bool = False
    
    @property
    def slack_seconds(self) -> float:
        """Get available slack time for deepening."""
        return max(0.0, self.max_seconds - self.allocated_seconds)
    
    def can_deepen(self, time_used: float, min_iteration_time: float = 30.0) -> bool:
        """
        Check if phase can be deepened with more iterations.
        
        Args:
            time_used: Time already used in this phase
            min_iteration_time: Minimum time needed for one iteration
            
        Returns:
            True if there's enough budget for another iteration
        """
        remaining = self.max_seconds - time_used
        return remaining >= min_iteration_time


@dataclass
class TimeAllocation:
    """
    Complete time allocation for a mission.
    
    Attributes:
        phase_budgets: Per-phase time budgets
        total_budget_seconds: Total mission time budget
        reserved_synthesis_seconds: Time reserved for synthesis
        target_utilization: Target utilization ratio (e.g., 0.8)
        predicted_total_seconds: Sum of predicted phase times
        scale_factor: Factor applied to reach target utilization
    """
    phase_budgets: Dict[str, PhaseTimeBudget] = field(default_factory=dict)
    total_budget_seconds: float = 0.0
    reserved_synthesis_seconds: float = 120.0
    target_utilization: float = 0.8
    predicted_total_seconds: float = 0.0
    scale_factor: float = 1.0
    
    def get_budget(self, phase_name: str) -> Optional[PhaseTimeBudget]:
        """Get budget for a specific phase."""
        return self.phase_budgets.get(phase_name)
    
    def total_allocated(self) -> float:
        """Get total allocated time across all phases."""
        return sum(b.allocated_seconds for b in self.phase_budgets.values())
    
    def utilization_target_met(self) -> bool:
        """Check if allocation meets target utilization."""
        if self.total_budget_seconds <= 0:
            return True
        return (self.total_allocated() / self.total_budget_seconds) >= self.target_utilization


class PhaseTimeAllocator:
    """
    Allocates time budgets to phases using historical data.
    
    Uses PolicyMemory to predict phase durations, then scales
    allocations to meet target utilization (default 80%).
    
    Features:
    - Historical data lookup from PolicyMemory
    - Default estimates when no history available
    - Priority-weighted slack distribution
    - Reserved synthesis time handling
    """
    
    def __init__(
        self,
        policy_memory: Optional[PolicyMemory] = None,
        default_phase_times: Optional[Dict[str, float]] = None,
        min_iteration_seconds: float = 30.0,
        max_deepening_rounds: int = 3
    ):
        """
        Initialize phase time allocator.
        
        Args:
            policy_memory: PolicyMemory for historical data lookup
            default_phase_times: Override default phase time estimates
            min_iteration_seconds: Minimum time for one iteration
            max_deepening_rounds: Maximum deepening rounds per phase
        """
        self._policy_memory = policy_memory
        self._default_times = default_phase_times or DEFAULT_PHASE_TIMES
        self._min_iteration_seconds = min_iteration_seconds
        self._max_deepening_rounds = max_deepening_rounds
    
    def allocate(
        self,
        phase_names: List[str],
        total_budget_seconds: float,
        reserved_synthesis_seconds: float = 120.0,
        target_utilization: float = 0.8
    ) -> TimeAllocation:
        """
        Allocate time budgets to phases.
        
        Args:
            phase_names: List of phase names in execution order
            total_budget_seconds: Total mission time budget
            reserved_synthesis_seconds: Time to reserve for synthesis
            target_utilization: Target fraction of budget to use (0.8 = 80%)
            
        Returns:
            TimeAllocation with per-phase budgets
        """
        if not phase_names:
            return TimeAllocation(
                total_budget_seconds=total_budget_seconds,
                reserved_synthesis_seconds=reserved_synthesis_seconds,
                target_utilization=target_utilization
            )
        
        # Step 1: Get predicted times per phase
        predictions = self._predict_phase_times(phase_names)
        
        # Step 2: Calculate available budget (minus synthesis reserve)
        available_budget = total_budget_seconds - reserved_synthesis_seconds
        target_time = available_budget * target_utilization
        
        # Step 3: Sum predictions
        predicted_total = sum(p[0] for p in predictions.values())
        
        # Step 4: Calculate scale factor to hit target utilization
        if predicted_total > 0:
            scale_factor = max(1.0, target_time / predicted_total)
        else:
            scale_factor = 1.0
        
        logger.debug(
            f"Time allocation: predicted={predicted_total:.0f}s, "
            f"target={target_time:.0f}s, scale={scale_factor:.2f}"
        )
        
        # Step 5: Allocate scaled budgets per phase
        phase_budgets: Dict[str, PhaseTimeBudget] = {}
        
        for phase_name in phase_names:
            phase_type = self._infer_phase_type(phase_name)
            predicted_time, from_history = predictions.get(phase_name, (60.0, False))
            priority = PHASE_PRIORITIES.get(phase_type, 0.5)
            
            # Min time is the predicted/historical time
            min_seconds = predicted_time
            
            # Allocated time is scaled up toward target
            allocated_seconds = predicted_time * scale_factor
            
            # Max time allows for deepening (priority-weighted extra slack)
            extra_slack = (allocated_seconds - min_seconds) * (1 + priority)
            max_seconds = allocated_seconds + extra_slack
            
            # Cap max at reasonable limit (3x min or remaining budget)
            max_seconds = min(max_seconds, min_seconds * 3, available_budget * 0.5)
            
            phase_budgets[phase_name] = PhaseTimeBudget(
                phase_name=phase_name,
                phase_type=phase_type,
                allocated_seconds=allocated_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                priority=priority,
                estimated_from_history=from_history
            )
            
            logger.debug(
                f"Phase '{phase_name}' ({phase_type}): "
                f"allocated={allocated_seconds:.0f}s, "
                f"min={min_seconds:.0f}s, max={max_seconds:.0f}s"
            )
        
        return TimeAllocation(
            phase_budgets=phase_budgets,
            total_budget_seconds=total_budget_seconds,
            reserved_synthesis_seconds=reserved_synthesis_seconds,
            target_utilization=target_utilization,
            predicted_total_seconds=predicted_total,
            scale_factor=scale_factor
        )
    
    def _predict_phase_times(
        self,
        phase_names: List[str]
    ) -> Dict[str, Tuple[float, bool]]:
        """
        Predict time for each phase using historical data or defaults.
        
        Args:
            phase_names: List of phase names
            
        Returns:
            Dict mapping phase_name -> (predicted_seconds, from_history)
        """
        predictions: Dict[str, Tuple[float, bool]] = {}
        
        for phase_name in phase_names:
            phase_type = self._infer_phase_type(phase_name)
            
            # Try historical data first
            from_history = False
            predicted_time = None
            
            if self._policy_memory is not None:
                # Look up councils typically used in this phase type
                # and get their avg_wall_time
                council_names = self._get_councils_for_phase_type(phase_type)
                
                total_time = 0.0
                count = 0
                for council in council_names:
                    stats = self._policy_memory.get_council_stats(council, phase_type)
                    if stats.invocation_count > 0 and stats.avg_wall_time > 0:
                        total_time += stats.avg_wall_time
                        count += 1
                
                if count > 0:
                    predicted_time = total_time / count
                    from_history = True
            
            # Fall back to defaults
            if predicted_time is None:
                predicted_time = self._default_times.get(phase_type, 60.0)
            
            predictions[phase_name] = (predicted_time, from_history)
        
        return predictions
    
    def _infer_phase_type(self, phase_name: str) -> str:
        """Infer phase type from phase name."""
        name_lower = phase_name.lower()
        
        # Check for keywords
        type_keywords = {
            "reconnaissance": ["recon", "reconnaissance"],
            "research": ["research", "gather", "context", "sources"],
            "analysis": ["analysis", "analyze", "investigate"],
            "planning": ["plan", "strategy"],
            "design": ["design", "architecture"],
            "deep_analysis": ["deep", "thorough"],
            "implementation": ["implement", "coding", "code", "build", "develop"],
            "testing": ["test", "validation", "verify"],
            "simulation": ["simulat", "scenario"],
            "synthesis": ["synthesis", "synthesize", "report", "consolidat"],
            "review": ["review", "final"],
        }
        
        for phase_type, keywords in type_keywords.items():
            if any(kw in name_lower for kw in keywords):
                return phase_type
        
        return "default"
    
    def _get_councils_for_phase_type(self, phase_type: str) -> List[str]:
        """Get council names typically used for a phase type."""
        phase_to_councils = {
            "reconnaissance": ["research"],
            "research": ["research"],
            "analysis": ["research", "planner"],
            "planning": ["planner"],
            "design": ["planner", "coder"],
            "deep_analysis": ["research", "evaluator"],
            "implementation": ["coder"],
            "testing": ["evaluator", "simulation"],
            "simulation": ["simulation"],
            "synthesis": ["planner"],
            "review": ["evaluator"],
        }
        return phase_to_councils.get(phase_type, ["research"])
    
    def should_deepen_phase(
        self,
        phase_name: str,
        allocation: TimeAllocation,
        time_used_seconds: float,
        convergence_score: float,
        deepening_rounds_done: int,
        convergence_threshold: float = 0.7
    ) -> Tuple[bool, str]:
        """
        Determine if a phase should be deepened with more iterations.
        
        Args:
            phase_name: Name of the phase
            allocation: Current time allocation
            time_used_seconds: Time already used in this phase
            convergence_score: Current convergence/plateau score (0-1)
            deepening_rounds_done: Number of deepening rounds already done
            convergence_threshold: Score above which quality has plateaued
            
        Returns:
            Tuple of (should_deepen, reason)
        """
        budget = allocation.get_budget(phase_name)
        if budget is None:
            return False, "No budget allocated for phase"
        
        # Check convergence (quality plateau)
        if convergence_score >= convergence_threshold:
            return False, f"Quality plateaued (convergence={convergence_score:.2f})"
        
        # Check max deepening rounds
        if deepening_rounds_done >= self._max_deepening_rounds:
            return False, f"Max deepening rounds reached ({self._max_deepening_rounds})"
        
        # Check if time budget allows another iteration
        if not budget.can_deepen(time_used_seconds, self._min_iteration_seconds):
            remaining = budget.max_seconds - time_used_seconds
            return False, f"Insufficient time ({remaining:.0f}s < {self._min_iteration_seconds}s)"
        
        # Deepen!
        remaining = budget.max_seconds - time_used_seconds
        return True, f"Deepening: {remaining:.0f}s available, convergence={convergence_score:.2f}"


def create_allocator_from_store(
    orchestration_store: Optional["OrchestrationStore"] = None,
    **kwargs
) -> PhaseTimeAllocator:
    """
    Factory function to create PhaseTimeAllocator with PolicyMemory.
    
    Args:
        orchestration_store: OrchestrationStore for PolicyMemory
        **kwargs: Additional arguments for PhaseTimeAllocator
        
    Returns:
        Configured PhaseTimeAllocator instance
    """
    policy_memory = None
    if orchestration_store is not None:
        policy_memory = PolicyMemory(orchestration_store)
    
    return PhaseTimeAllocator(policy_memory=policy_memory, **kwargs)

