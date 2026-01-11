"""
Bio Prior Context and Metrics.

Defines BioPriorContext which captures the current state of reasoning
for the bio prior engine to evaluate.

The context is built from mission state with best-effort metric extraction.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState
    from ..meta.supervisor import MissionMetrics

logger = logging.getLogger(__name__)


# Fixed window for recent metrics computation (determinism)
RECENT_WINDOW_STEPS: int = 3


@dataclass
class BioPriorContext:
    """
    Context for bio prior evaluation.
    
    Built from mission state with best-effort metric extraction.
    Missing metrics are set to None and logged in trace.
    
    Attributes:
        phase: Current phase name
        step_index: Current step index within phase
        time_remaining_s: Time remaining in seconds (if available)
        evidence_new_count_recent: New evidence in last RECENT_WINDOW_STEPS steps
        contradiction_rate: Current contradiction rate (0-1)
        uncertainty_trend: Trend of uncertainty (positive = increasing)
        drift_score: Current drift/alignment score
        plan_branching_factor: Number of active branches in plan
        last_step_evidence_delta: Evidence change in last step
        recent_window_steps: Window size used (for traceability)
    """
    phase: str
    step_index: int
    time_remaining_s: Optional[float] = None
    evidence_new_count_recent: int = 0
    contradiction_rate: Optional[float] = None
    uncertainty_trend: Optional[float] = None
    drift_score: Optional[float] = None
    plan_branching_factor: Optional[float] = None
    last_step_evidence_delta: Optional[int] = None
    
    # Traceability
    recent_window_steps: int = RECENT_WINDOW_STEPS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_missing_metrics(self) -> List[str]:
        """Get list of metrics that are None (unavailable)."""
        missing = []
        if self.time_remaining_s is None:
            missing.append("time_remaining_s")
        if self.contradiction_rate is None:
            missing.append("contradiction_rate")
        if self.uncertainty_trend is None:
            missing.append("uncertainty_trend")
        if self.drift_score is None:
            missing.append("drift_score")
        if self.plan_branching_factor is None:
            missing.append("plan_branching_factor")
        if self.last_step_evidence_delta is None:
            missing.append("last_step_evidence_delta")
        return missing
    
    @property
    def is_early_phase(self) -> bool:
        """Check if we're in early phase (step_index < 3)."""
        return self.step_index < 3
    
    @property
    def is_late_phase(self) -> bool:
        """Check if we're in late phase (low time or high step index)."""
        if self.time_remaining_s is not None and self.time_remaining_s < 60:
            return True
        return self.step_index > 10
    
    @property
    def has_stagnation_signal(self) -> bool:
        """Check if evidence production has stagnated."""
        return self.evidence_new_count_recent == 0
    
    @property
    def has_high_contradiction(self) -> bool:
        """Check if contradiction rate is high (> 0.2)."""
        return self.contradiction_rate is not None and self.contradiction_rate > 0.2
    
    @property
    def has_high_drift(self) -> bool:
        """Check if drift score is high (> 0.3)."""
        return self.drift_score is not None and self.drift_score > 0.3
    
    @property
    def has_high_branching(self) -> bool:
        """Check if branching factor is high (> 3)."""
        return self.plan_branching_factor is not None and self.plan_branching_factor > 3


def build_context(
    state: "MissionState",
    metrics: Optional["MissionMetrics"] = None,
    claim_graph: Optional[Any] = None,
    alignment_trajectory: Optional[List[Dict]] = None,
) -> BioPriorContext:
    """
    Build BioPriorContext from mission state.
    
    Best-effort extraction of available metrics.
    Missing metrics are set to None.
    
    Args:
        state: Current mission state
        metrics: Optional mission metrics from supervisor
        claim_graph: Optional claim graph for contradiction rate
        alignment_trajectory: Optional alignment trajectory for drift
        
    Returns:
        BioPriorContext with available metrics
    """
    # Get current phase info
    current_phase = state.current_phase()
    phase_name = current_phase.name if current_phase else "unknown"
    step_index = state.iteration_count
    
    # Time remaining
    time_remaining_s: Optional[float] = None
    try:
        time_remaining_s = state.remaining_time().total_seconds()
    except Exception:
        pass
    
    # Evidence count from recent steps
    evidence_new_count_recent = _compute_recent_evidence(state, RECENT_WINDOW_STEPS)
    
    # Contradiction rate from claim graph or metrics
    contradiction_rate: Optional[float] = None
    if claim_graph is not None:
        try:
            contradiction_rate = 1.0 - claim_graph.compute_consistency_score()
        except Exception:
            pass
    
    # Uncertainty trend from metrics history
    uncertainty_trend: Optional[float] = None
    if metrics is not None:
        try:
            uncertainty_trend = _compute_uncertainty_trend(state, metrics)
        except Exception:
            pass
    
    # Drift score from alignment trajectory
    drift_score: Optional[float] = None
    if alignment_trajectory and len(alignment_trajectory) > 0:
        try:
            # Get latest alignment point
            latest = alignment_trajectory[-1]
            # Drift score is inverse of alignment score
            a_t = latest.get("a_t", 1.0)
            drift_score = 1.0 - a_t
        except Exception:
            pass
    elif hasattr(state, "alignment_trajectory") and state.alignment_trajectory:
        try:
            latest = state.alignment_trajectory[-1]
            a_t = latest.get("a_t", 1.0)
            drift_score = 1.0 - a_t
        except Exception:
            pass
    
    # Plan branching factor
    plan_branching_factor: Optional[float] = None
    if hasattr(state, "updated_plan") and state.updated_plan:
        try:
            plan = state.updated_plan
            if isinstance(plan, dict):
                subgoals = plan.get("new_subgoals", [])
                plan_branching_factor = float(len(subgoals)) if subgoals else 1.0
        except Exception:
            pass
    
    # Last step evidence delta
    last_step_evidence_delta: Optional[int] = None
    if hasattr(state, "context_evolution_log") and state.context_evolution_log:
        try:
            if len(state.context_evolution_log) >= 2:
                latest = state.context_evolution_log[-1]
                previous = state.context_evolution_log[-2]
                latest_knowledge = latest.get("prior_knowledge_size", 0)
                previous_knowledge = previous.get("prior_knowledge_size", 0)
                last_step_evidence_delta = latest_knowledge - previous_knowledge
            elif len(state.context_evolution_log) == 1:
                last_step_evidence_delta = state.context_evolution_log[-1].get(
                    "web_searches", 0
                )
        except Exception:
            pass
    
    ctx = BioPriorContext(
        phase=phase_name,
        step_index=step_index,
        time_remaining_s=time_remaining_s,
        evidence_new_count_recent=evidence_new_count_recent,
        contradiction_rate=contradiction_rate,
        uncertainty_trend=uncertainty_trend,
        drift_score=drift_score,
        plan_branching_factor=plan_branching_factor,
        last_step_evidence_delta=last_step_evidence_delta,
        recent_window_steps=RECENT_WINDOW_STEPS,
    )
    
    # Log missing metrics
    missing = ctx.get_missing_metrics()
    if missing:
        logger.debug(f"[BIO_PRIORS] Context missing metrics: {missing}")
    
    return ctx


def _compute_recent_evidence(state: "MissionState", window: int) -> int:
    """Compute evidence added in recent window steps."""
    total = 0
    
    if hasattr(state, "context_evolution_log") and state.context_evolution_log:
        # Get last N entries
        recent = state.context_evolution_log[-window:]
        for entry in recent:
            total += entry.get("web_searches", 0)
    
    # Also check web_search_history
    if hasattr(state, "web_search_history") and state.web_search_history:
        recent = state.web_search_history[-window:]
        for entry in recent:
            total += entry.get("results_count", 0)
    
    return total


def _compute_uncertainty_trend(
    state: "MissionState",
    metrics: "MissionMetrics",
) -> Optional[float]:
    """Compute trend of uncertainty over recent iterations."""
    # If we have metrics history, compute trend
    if hasattr(state, "context_evolution_log") and len(state.context_evolution_log) >= 2:
        recent = state.context_evolution_log[-RECENT_WINDOW_STEPS:]
        if len(recent) >= 2:
            # Use questions count as proxy for uncertainty
            first_questions = recent[0].get("questions_count", 0)
            last_questions = recent[-1].get("questions_count", 0)
            
            if first_questions > 0:
                # Positive trend means uncertainty is increasing
                return (last_questions - first_questions) / first_questions
    
    # Fall back to metrics uncertainty
    if hasattr(metrics, "avg_uncertainty"):
        return metrics.avg_uncertainty - 0.5  # Center around 0.5
    
    return None



