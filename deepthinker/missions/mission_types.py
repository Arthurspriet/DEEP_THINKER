"""
Mission Data Types for DeepThinker 2.0.

Defines the core data structures for long-running, time-bounded missions.
Now includes step-level execution tracking via the Step Engine.

Enhanced with effort-based execution control:
- EffortLevel enum determines depth of analysis
- EFFORT_PRESETS provide default configurations per effort level
- Multi-round phase and council execution support
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Union, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from ..outputs.output_types import OutputArtifact
    from ..steps.step_types import StepDefinition
    from .evidence_recovery import EvidenceRecoveryState
    from ..epistemics.focus_area_manager import FocusAreaManager


class EffortLevel(str, Enum):
    """
    Effort level for mission execution.
    
    Determines the depth of analysis, number of rounds, model selection,
    and quality thresholds based on time budget.
    """
    QUICK = "quick"        # < 5 min: Fast, single-round execution
    STANDARD = "standard"  # 5-30 min: Balanced multi-round execution
    DEEP = "deep"          # 30-120 min: Thorough analysis with larger models
    MARATHON = "marathon"  # > 120 min: Exhaustive analysis until time runs out


class MissionFailureReason(str, Enum):
    """
    Standardized failure reason codes for mission and phase failures.
    
    Used to provide queryable, structured failure information via API.
    """
    MODEL_REFUSAL = "model_refusal"      # Model refused to follow instructions
    BUDGET_EXCEEDED = "budget_exceeded"  # Resource budget (tokens/time/iterations) exceeded
    GOVERNANCE_BLOCK = "governance_block"  # Governance rules blocked execution
    TOOL_FAILURE = "tool_failure"        # Tool/subprocess execution failed
    TIMEOUT = "timeout"                  # Operation timed out
    COUNCIL_FAILURE = "council_failure"  # Council execution failed (all members)
    UNKNOWN_ERROR = "unknown_error"      # Uncategorized error


# Effort-based preset configurations
EFFORT_PRESETS: Dict[EffortLevel, Dict[str, Any]] = {
    EffortLevel.QUICK: {
        "max_recon_rounds": 1,
        "max_analysis_rounds": 1,
        "max_deep_rounds": 1,
        "max_branches": 1,
        "max_council_rounds": 1,
        "max_web_docs_per_query": 3,
        "min_quality_to_stop": 0.70,
    },
    EffortLevel.STANDARD: {
        "max_recon_rounds": 2,
        "max_analysis_rounds": 2,
        "max_deep_rounds": 2,
        "max_branches": 2,
        "max_council_rounds": 2,
        "max_web_docs_per_query": 5,
        "min_quality_to_stop": 0.80,
    },
    EffortLevel.DEEP: {
        "max_recon_rounds": 3,
        "max_analysis_rounds": 4,
        "max_deep_rounds": 3,
        "max_branches": 3,
        "max_council_rounds": 3,
        "max_web_docs_per_query": 8,
        "min_quality_to_stop": 0.85,
    },
    EffortLevel.MARATHON: {
        "max_recon_rounds": 5,
        "max_analysis_rounds": 6,
        "max_deep_rounds": 5,
        "max_branches": 4,
        "max_council_rounds": 4,
        "max_web_docs_per_query": 10,
        "min_quality_to_stop": 0.90,
    },
}


@dataclass
class ConvergenceState:
    """
    Tracks mission convergence status for iterative execution.
    
    Attributes:
        convergence_reached: Whether convergence criteria are met
        min_iterations: Minimum iterations before allowing convergence
        min_time_guard_seconds: Minimum time remaining to continue
        multiview_disagreement: Disagreement score between optimist/skeptic (0-1)
        evaluator_has_questions: Whether evaluator has unresolved questions
        evaluator_has_missing_info: Whether evaluator reports missing info
        confidence_score: Latest evaluator confidence score
        last_quality_score: Latest quality score from evaluator
        pending_subgoals: Number of unaddressed subgoals from planner
    """
    convergence_reached: bool = False
    min_iterations: int = 2
    min_time_guard_seconds: float = 60.0
    multiview_disagreement: float = 0.0
    evaluator_has_questions: bool = False
    evaluator_has_missing_info: bool = False
    confidence_score: float = 0.5
    last_quality_score: float = 0.0
    pending_subgoals: int = 0
    
    def can_converge(self, iteration: int, time_remaining_seconds: float) -> bool:
        """
        Check if convergence is allowed based on all criteria.
        
        Returns True only if:
        - Minimum iterations completed
        - Sufficient time remaining OR convergence criteria fully met
        - No unresolved evaluator issues
        - Acceptable multiview agreement
        - No pending subgoals
        - Adequate confidence
        """
        # Must complete minimum iterations
        if iteration < self.min_iterations:
            return False
        
        # Check evaluator criteria
        if self.evaluator_has_questions:
            return False
        if self.evaluator_has_missing_info:
            return False
        
        # Check confidence threshold
        if self.confidence_score < 0.70:
            return False
        
        # Check multiview agreement
        if self.multiview_disagreement > 0.25:
            return False
        
        # Check pending subgoals
        if self.pending_subgoals > 0:
            return False
        
        return True
    
    def should_force_continue(self, time_remaining_seconds: float, total_time_seconds: float) -> bool:
        """
        Check if another iteration should be forced.
        
        Returns True if:
        - Plenty of time remaining (> 25% of budget)
        - Low confidence (< 0.7)
        """
        if time_remaining_seconds <= self.min_time_guard_seconds:
            return False
        
        time_ratio = time_remaining_seconds / total_time_seconds if total_time_seconds > 0 else 0
        
        if time_ratio > 0.25 and self.confidence_score < 0.70:
            return True
        
        return False
    
    def update_from_evaluator(self, evaluator_output: Any) -> None:
        """Update state from evaluator output."""
        if evaluator_output is None:
            return
        
        if hasattr(evaluator_output, 'questions'):
            self.evaluator_has_questions = bool(evaluator_output.questions)
        if hasattr(evaluator_output, 'missing_info'):
            self.evaluator_has_missing_info = bool(evaluator_output.missing_info)
        if hasattr(evaluator_output, 'confidence_score'):
            self.confidence_score = evaluator_output.confidence_score
        if hasattr(evaluator_output, 'quality_score'):
            self.last_quality_score = evaluator_output.quality_score
    
    def update_from_plan(self, workflow_plan: Any) -> None:
        """Update state from updated workflow plan."""
        if workflow_plan is None:
            return
        
        if hasattr(workflow_plan, 'new_subgoals'):
            self.pending_subgoals = len(workflow_plan.new_subgoals)


def infer_effort_level(time_budget_minutes: int) -> EffortLevel:
    """
    Infer effort level from time budget.
    
    Args:
        time_budget_minutes: Total time budget in minutes
        
    Returns:
        Appropriate EffortLevel for the given time budget
    """
    if time_budget_minutes < 5:
        return EffortLevel.QUICK
    elif time_budget_minutes < 30:
        return EffortLevel.STANDARD
    elif time_budget_minutes < 120:
        return EffortLevel.DEEP
    else:
        return EffortLevel.MARATHON


def build_constraints_from_time_budget(
    time_budget_minutes: int,
    allow_code: bool = True,
    allow_internet: bool = True,
    notes: Optional[str] = None,
    **overrides
) -> "MissionConstraints":
    """
    Build MissionConstraints from time budget with effort-based defaults.
    
    Automatically infers effort level and applies appropriate presets,
    while allowing user overrides for specific parameters.
    
    Args:
        time_budget_minutes: Total time budget in minutes
        allow_code: Whether code execution is allowed
        allow_internet: Whether internet access is allowed
        notes: Additional notes/constraints
        **overrides: Override specific constraint values
        
    Returns:
        Populated MissionConstraints instance
    """
    effort = infer_effort_level(time_budget_minutes)
    preset = EFFORT_PRESETS[effort].copy()
    
    # Apply user overrides (they take precedence over presets)
    for key, value in overrides.items():
        if value is not None:
            preset[key] = value
    
    return MissionConstraints(
        time_budget_minutes=time_budget_minutes,
        effort=effort,
        max_iterations=preset.pop("max_iterations", 100),
        max_recon_rounds=preset.get("max_recon_rounds", 1),
        max_analysis_rounds=preset.get("max_analysis_rounds", 1),
        max_deep_rounds=preset.get("max_deep_rounds", 1),
        max_branches=preset.get("max_branches", 1),
        max_council_rounds=preset.get("max_council_rounds", 1),
        max_web_docs_per_query=preset.get("max_web_docs_per_query", 5),
        min_quality_to_stop=preset.get("min_quality_to_stop", 0.85),
        enable_code_execution=allow_code,
        enable_internet=allow_internet,
        allow_code_execution=allow_code,
        allow_internet=allow_internet,
        notes=notes,
    )


@dataclass
class MissionPreferences:
    """
    Mission preference vector for resource-aware orchestration.
    
    Attributes:
        cost_sensitivity: 0=ignore cost, 1=cost-obsessed
        latency_sensitivity: 0=ignore time, 1=time-obsessed
        quality_priority: 0=good-enough, 1=perfection
        forbidden_tool_classes: Tool classes to avoid (e.g., ["web"])
    """
    cost_sensitivity: float = 0.5  # 0=ignore cost, 1=cost-obsessed
    latency_sensitivity: float = 0.5  # 0=ignore time, 1=time-obsessed
    quality_priority: float = 0.5  # 0=good-enough, 1=perfection
    forbidden_tool_classes: List[str] = field(default_factory=list)  # e.g., ["web"]


@dataclass
class MissionConstraints:
    """
    Constraints for a mission execution.
    
    Attributes:
        time_budget_minutes: Total time allowed for the mission
        effort: Effort level (QUICK, STANDARD, DEEP, MARATHON)
        max_iterations: Maximum iterations per phase
        max_recon_rounds: Max rounds for reconnaissance/research phases
        max_analysis_rounds: Max rounds for analysis/design phases
        max_deep_rounds: Max rounds for implementation phases
        max_branches: Max parallel branches to explore
        max_council_rounds: Max rounds per council execution
        max_web_docs_per_query: Max documents to retrieve per web query
        min_quality_to_stop: Quality threshold for early stopping (0-1)
        quality_safety_margin: Safety margin for quality checks
        enable_code_execution: Whether code execution is enabled
        enable_internet: Whether internet access is enabled
        allow_internet: Whether web search/research is allowed (backward compat)
        allow_code_execution: Whether code can be executed (backward compat)
        notes: Optional notes or additional constraints
        preferences: Mission preference vector for resource-aware orchestration
    """
    time_budget_minutes: int
    effort: EffortLevel = EffortLevel.STANDARD
    max_iterations: int = 100
    max_recon_rounds: int = 1
    max_analysis_rounds: int = 1
    max_deep_rounds: int = 1
    max_branches: int = 1
    max_council_rounds: int = 1
    max_web_docs_per_query: int = 5
    min_quality_to_stop: float = 0.85
    quality_safety_margin: float = 0.05
    enable_code_execution: bool = True
    enable_internet: bool = True
    allow_internet: bool = True
    allow_code_execution: bool = True
    notes: Optional[str] = None
    preferences: MissionPreferences = field(default_factory=MissionPreferences)
    
    # Epistemic Hardening config flags (enabled by default for epistemic grounding)
    enable_evidence_recovery: bool = True
    enable_split_recon: bool = True
    enable_evidence_budgets: bool = True
    enable_focus_area_freeze: bool = True
    enable_claim_registry: bool = True
    enable_step_tier_policy: bool = True
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary for serialization."""
        return {
            "time_budget_minutes": self.time_budget_minutes,
            "effort": self.effort.value if isinstance(self.effort, EffortLevel) else self.effort,
            "max_iterations": self.max_iterations,
            "max_recon_rounds": self.max_recon_rounds,
            "max_analysis_rounds": self.max_analysis_rounds,
            "max_deep_rounds": self.max_deep_rounds,
            "max_branches": self.max_branches,
            "max_council_rounds": self.max_council_rounds,
            "max_web_docs_per_query": self.max_web_docs_per_query,
            "min_quality_to_stop": self.min_quality_to_stop,
            "quality_safety_margin": self.quality_safety_margin,
            "enable_code_execution": self.enable_code_execution,
            "enable_internet": self.enable_internet,
            "allow_internet": self.allow_internet,
            "allow_code_execution": self.allow_code_execution,
            "notes": self.notes,
            "preferences": {
                "cost_sensitivity": self.preferences.cost_sensitivity,
                "latency_sensitivity": self.preferences.latency_sensitivity,
                "quality_priority": self.preferences.quality_priority,
                "forbidden_tool_classes": self.preferences.forbidden_tool_classes,
            },
            # Epistemic Hardening config flags
            "enable_evidence_recovery": self.enable_evidence_recovery,
            "enable_split_recon": self.enable_split_recon,
            "enable_evidence_budgets": self.enable_evidence_budgets,
            "enable_focus_area_freeze": self.enable_focus_area_freeze,
            "enable_claim_registry": self.enable_claim_registry,
            "enable_step_tier_policy": self.enable_step_tier_policy,
        }


@dataclass
class MissionPhase:
    """
    A single phase within a mission.
    
    Phases contain steps - the atomic units of work executed by the Step Engine.
    
    Attributes:
        name: Short identifier for the phase
        description: Detailed description of what this phase accomplishes
        status: Current status (pending, running, completed, skipped, failed)
        started_at: When the phase began execution
        ended_at: When the phase completed
        iterations: Number of iterations performed in this phase
        artifacts: Results produced by this phase (e.g., reports, code, evaluations)
        steps: List of step definitions to execute within this phase
        time_budget_seconds: Allocated time budget for this phase
        time_used_seconds: Cumulative time spent on this phase
        deepening_rounds: Number of deepening rounds performed (beyond normal iterations)
        convergence_score: Last recorded convergence/plateau score (0-1)
        enrichment_passes: Number of depth enrichment passes performed
        depth_achieved: Final computed depth score (0-1)
        termination_reason: Why this phase ended (for attribution)
    """
    name: str
    description: str
    status: str = "pending"  # pending | running | completed | skipped | failed
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    iterations: int = 0
    artifacts: Dict[str, str] = field(default_factory=dict)
    steps: List["StepDefinition"] = field(default_factory=list)
    # Time budgeting fields (new)
    time_budget_seconds: Optional[float] = None
    time_used_seconds: float = 0.0
    deepening_rounds: int = 0
    convergence_score: float = 0.0
    # Depth control fields
    enrichment_passes: int = 0
    depth_achieved: float = 0.0
    termination_reason: str = ""
    
    def mark_running(self) -> None:
        """Mark this phase as running."""
        self.status = "running"
        self.started_at = datetime.utcnow()
    
    def mark_completed(self) -> None:
        """Mark this phase as completed."""
        self.status = "completed"
        self.ended_at = datetime.utcnow()
    
    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark this phase as failed."""
        self.status = "failed"
        self.ended_at = datetime.utcnow()
        if error:
            self.artifacts["error"] = error
    
    def mark_skipped(self, reason: Optional[str] = None) -> None:
        """Mark this phase as skipped."""
        self.status = "skipped"
        self.ended_at = datetime.utcnow()
        if reason:
            self.artifacts["skip_reason"] = reason
    
    def mark_completed_degraded(self, reason: str) -> None:
        """
        Mark this phase as completed with degraded quality.
        
        Model-Aware Phase Stabilization: Used when a phase cannot fully
        succeed but has produced partial usable output. This allows
        subsequent phases to proceed with available context rather
        than failing the entire mission.
        
        Args:
            reason: Explanation of why this is a degraded completion
        """
        self.status = "completed_degraded"
        self.ended_at = datetime.utcnow()
        self.artifacts["_degraded"] = True
        self.artifacts["_degraded_reason"] = reason
    
    def is_degraded(self) -> bool:
        """Check if this phase completed with degraded quality."""
        return self.status == "completed_degraded" or self.artifacts.get("_degraded", False)
    
    def is_terminal(self) -> bool:
        """Check if phase has reached a terminal state (completed, failed, skipped, or degraded)."""
        return self.status in ("completed", "completed_degraded", "failed", "skipped")
    
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds, if phase has ended."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None
    
    def update_time_used(self) -> None:
        """Update time_used_seconds from started_at to now."""
        if self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            self.time_used_seconds = elapsed
    
    def time_budget_remaining(self) -> float:
        """Get remaining time in phase budget."""
        if self.time_budget_seconds is None:
            return float('inf')
        return max(0.0, self.time_budget_seconds - self.time_used_seconds)
    
    def time_budget_utilization(self) -> float:
        """Get fraction of time budget used (0-1+)."""
        if self.time_budget_seconds is None or self.time_budget_seconds <= 0:
            return 0.0
        return self.time_used_seconds / self.time_budget_seconds
    
    def can_deepen(self, min_iteration_seconds: float = 30.0, max_deepening_rounds: int = 3) -> bool:
        """
        Check if phase can be deepened with more iterations.
        
        Args:
            min_iteration_seconds: Minimum time needed for one iteration
            max_deepening_rounds: Maximum deepening rounds allowed
            
        Returns:
            True if deepening is possible
        """
        # Check max deepening rounds
        if self.deepening_rounds >= max_deepening_rounds:
            return False
        
        # Check time budget
        if self.time_budget_remaining() < min_iteration_seconds:
            return False
        
        return True
    
    def should_deepen(
        self,
        convergence_threshold: float = 0.7,
        min_iteration_seconds: float = 30.0,
        max_deepening_rounds: int = 3
    ) -> tuple:
        """
        Determine if phase should be deepened.
        
        Args:
            convergence_threshold: Convergence score above which to stop
            min_iteration_seconds: Minimum time needed for one iteration
            max_deepening_rounds: Maximum deepening rounds allowed
            
        Returns:
            Tuple of (should_deepen: bool, reason: str)
        """
        # Check convergence (quality plateau)
        if self.convergence_score >= convergence_threshold:
            return False, f"Quality plateaued (convergence={self.convergence_score:.2f})"
        
        # Check max deepening rounds
        if self.deepening_rounds >= max_deepening_rounds:
            return False, f"Max deepening rounds reached ({max_deepening_rounds})"
        
        # Check time budget
        remaining = self.time_budget_remaining()
        if remaining < min_iteration_seconds:
            return False, f"Insufficient time ({remaining:.0f}s < {min_iteration_seconds}s)"
        
        # Deepening is possible
        return True, f"Deepening: {remaining:.0f}s remaining, convergence={self.convergence_score:.2f}"
    
    def get_pending_steps(self) -> List["StepDefinition"]:
        """Get all steps that still need to be executed."""
        return [s for s in self.steps if s.status in ("pending", "running")]
    
    def get_completed_steps(self) -> List["StepDefinition"]:
        """Get all successfully completed steps."""
        return [s for s in self.steps if s.status == "completed"]
    
    def all_steps_complete(self) -> bool:
        """Check if all steps are in a terminal state."""
        if not self.steps:
            return True
        return all(s.status in ("completed", "failed", "skipped") for s in self.steps)
    
    def get_step_summary(self) -> str:
        """Generate a summary of step outputs for context passing."""
        summaries = []
        for step in self.steps:
            if step.last_result and step.status == "completed":
                output = step.last_result.output
                # Truncate long outputs
                truncated = output[:1000] + "..." if len(output) > 1000 else output
                summaries.append(f"### {step.name}\n{truncated}")
        return "\n\n".join(summaries) if summaries else ""
    
    def get_step_artifacts(self) -> Dict[str, str]:
        """Collect all artifacts from completed steps."""
        all_artifacts: Dict[str, str] = {}
        for step in self.steps:
            if step.last_result and step.last_result.artifacts:
                for name, content in step.last_result.artifacts.items():
                    # Prefix with step name to avoid collisions
                    key = f"{step.name}_{name}"
                    all_artifacts[key] = content
        return all_artifacts


@dataclass
class MissionState:
    """
    Full state of a mission.
    
    Attributes:
        mission_id: Unique identifier for this mission
        objective: The primary goal/objective of the mission
        constraints: Execution constraints (time, tools, etc.)
        created_at: When the mission was created
        deadline_at: When the mission must complete
        current_phase_index: Index of the currently active phase
        phases: List of mission phases
        status: Overall mission status
        logs: Execution logs and messages
        final_artifacts: Consolidated artifacts from all phases
        event_logs: Structured event logs for tracking decisions
        context_summaries: Summarized context per phase for later consumption
        recent_outputs: Recent outputs for loop detection
        iteration_count: Number of mission-level iterations completed
    """
    mission_id: str
    objective: str
    constraints: MissionConstraints
    created_at: datetime
    deadline_at: datetime
    current_phase_index: int = 0
    phases: List[MissionPhase] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | aborted | failed | expired
    # Failure tracking for queryable failure information
    failure_reason: Optional[str] = None  # MissionFailureReason value when status is "failed"
    failure_details: Dict[str, Any] = field(default_factory=dict)  # Structured failure context
    logs: List[str] = field(default_factory=list)
    final_artifacts: Dict[str, str] = field(default_factory=dict)
    event_logs: List[Dict[str, Any]] = field(default_factory=list)
    output_deliverables: List["OutputArtifact"] = field(default_factory=list)
    # Multi-round execution tracking
    phase_rounds: Dict[str, int] = field(default_factory=dict)
    council_rounds: Dict[str, int] = field(default_factory=dict)
    work_summary: Dict[str, Any] = field(default_factory=dict)
    # Meta-cognition state
    hypotheses: Dict[str, Any] = field(default_factory=dict)
    updated_plan: Dict[str, Any] = field(default_factory=dict)
    next_actions: List[str] = field(default_factory=list)
    meta_traces: Dict[str, Any] = field(default_factory=dict)
    # Context and iteration tracking (new in 2.0)
    context_summaries: Dict[str, str] = field(default_factory=dict)
    recent_outputs: List[str] = field(default_factory=list)
    iteration_count: int = 0
    # Context evolution tracking (new for iterative reasoning)
    context_evolution_log: List[Dict[str, Any]] = field(default_factory=list)
    focus_areas_history: List[List[str]] = field(default_factory=list)
    questions_history: List[List[str]] = field(default_factory=list)
    web_search_history: List[Dict[str, Any]] = field(default_factory=list)
    # Wall-clock time tracking (new for accurate time management)
    wall_clock_start: Optional[float] = None
    phase_wall_times: Dict[str, float] = field(default_factory=dict)
    phase_council_times: Dict[str, float] = field(default_factory=dict)
    total_web_searches: int = 0
    memory_items_used: int = 0
    # Execution escalation tracking
    execution_escalations: List[Dict[str, Any]] = field(default_factory=list)
    execution_metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # ==========================================================================
    # Decision Accountability Layer (Horizon 3)
    # ==========================================================================
    # List of decision_ids emitted during this mission (for linkage tracking)
    decision_records: List[str] = field(default_factory=list)
    # Last model selection decision ID (for causal linking)
    last_model_decision_id: Optional[str] = None
    # Last governance decision ID (for retry escalation linking)
    last_governance_decision_id: Optional[str] = None
    
    # ==========================================================================
    # Alignment Control Layer (Hybrid Alignment Control)
    # ==========================================================================
    # NorthStarGoal stored as dict for serialization
    alignment_north_star: Optional[Dict[str, Any]] = None
    # AlignmentTrajectory points stored as list of dicts
    alignment_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Controller state for escalation tracking
    alignment_controller_state: Dict[str, Any] = field(default_factory=dict)
    # Pending user event for drift confirmation (UserDriftEvent as dict)
    pending_user_event: Optional[Dict[str, Any]] = None
    
    # ==========================================================================
    # Epistemic Telemetry (DeepThinker 2.0 Epistemic Hardening)
    # ==========================================================================
    epistemic_telemetry: Dict[str, Any] = field(default_factory=lambda: {
        "epistemic_risk": 0.0,
        "grounded_claim_ratio": 0.0,
        "sources_per_phase": {},
        "web_searches_used_in_claims": 0,
        "phase_contamination_scores": {},
        "scenario_distinctness_score": 0.0,
        "claim_validation_results": {},
        "ungrounded_facts_count": 0,
        "speculative_density": 0.0,
    })
    
    # ==========================================================================
    # Evidence Recovery State (Epistemic Hardening Phase 1)
    # ==========================================================================
    # Lazy import to avoid circular dependency - will be set by orchestrator
    evidence_recovery_state: Optional["EvidenceRecoveryState"] = None
    
    # ==========================================================================
    # Focus Area Manager (Epistemic Hardening Phase 4)
    # ==========================================================================
    # Manages focus areas with strict limits to prevent conceptual sprawl
    focus_area_manager: Optional["FocusAreaManager"] = None
    
    def update_epistemic_telemetry(
        self,
        epistemic_risk: Optional[float] = None,
        grounded_claim_ratio: Optional[float] = None,
        phase_name: Optional[str] = None,
        source_count: Optional[int] = None,
        web_searches_in_claims: Optional[int] = None,
        contamination_score: Optional[float] = None,
        scenario_distinctness: Optional[float] = None,
        ungrounded_facts: Optional[int] = None,
        speculative_density: Optional[float] = None,
    ) -> None:
        """
        Update epistemic telemetry with new values.
        
        Args:
            epistemic_risk: Overall epistemic risk score (0-1)
            grounded_claim_ratio: Ratio of grounded claims (0-1)
            phase_name: Phase to update source count for
            source_count: Number of sources used in phase
            web_searches_in_claims: Number of web searches backing claims
            contamination_score: Phase contamination score
            scenario_distinctness: Scenario distinctness score
            ungrounded_facts: Count of ungrounded factual claims
            speculative_density: Fraction of speculative content
        """
        if epistemic_risk is not None:
            self.epistemic_telemetry["epistemic_risk"] = epistemic_risk
        
        if grounded_claim_ratio is not None:
            self.epistemic_telemetry["grounded_claim_ratio"] = grounded_claim_ratio
        
        if phase_name and source_count is not None:
            self.epistemic_telemetry["sources_per_phase"][phase_name] = source_count
        
        if web_searches_in_claims is not None:
            self.epistemic_telemetry["web_searches_used_in_claims"] = web_searches_in_claims
        
        if phase_name and contamination_score is not None:
            self.epistemic_telemetry["phase_contamination_scores"][phase_name] = contamination_score
        
        if scenario_distinctness is not None:
            self.epistemic_telemetry["scenario_distinctness_score"] = scenario_distinctness
        
        if ungrounded_facts is not None:
            self.epistemic_telemetry["ungrounded_facts_count"] = ungrounded_facts
        
        if speculative_density is not None:
            self.epistemic_telemetry["speculative_density"] = speculative_density
    
    def get_epistemic_summary(self) -> Dict[str, Any]:
        """Get a summary of epistemic quality for reporting."""
        return {
            "epistemic_risk": self.epistemic_telemetry.get("epistemic_risk", 0.0),
            "grounded_ratio": self.epistemic_telemetry.get("grounded_claim_ratio", 0.0),
            "total_sources": sum(self.epistemic_telemetry.get("sources_per_phase", {}).values()),
            "web_searches_in_claims": self.epistemic_telemetry.get("web_searches_used_in_claims", 0),
            "avg_contamination": (
                sum(self.epistemic_telemetry.get("phase_contamination_scores", {}).values()) /
                max(1, len(self.epistemic_telemetry.get("phase_contamination_scores", {})))
            ),
            "scenario_distinctness": self.epistemic_telemetry.get("scenario_distinctness_score", 0.0),
            "quality_grade": self._compute_epistemic_grade(),
        }
    
    def _compute_epistemic_grade(self) -> str:
        """Compute letter grade for epistemic quality."""
        risk = self.epistemic_telemetry.get("epistemic_risk", 0.5)
        grounded = self.epistemic_telemetry.get("grounded_claim_ratio", 0.5)
        
        # Combined score (lower risk + higher grounded = better)
        score = (1 - risk) * 0.5 + grounded * 0.5
        
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def log(self, message: str) -> None:
        """Add a timestamped log entry."""
        timestamp = datetime.utcnow().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        phase_name: Optional[str] = None
    ) -> None:
        """
        Add a structured event log entry.
        
        Args:
            event_type: Type of event (e.g., "supervisor_decision", "phase_start")
            data: Event data dictionary
            phase_name: Optional phase name for context
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "phase": phase_name,
            **data
        }
        self.event_logs.append(entry)
        
        # Also add to text logs for backwards compatibility
        self.logs.append(f"[{entry['timestamp']}] EVENT:{event_type} - {json.dumps(data)}")
    
    def log_supervisor_decision(
        self,
        phase_name: str,
        models: List[str],
        temperature: float,
        parallelism: int,
        downgraded: bool,
        reason: str,
        estimated_vram: int = 0,
        wait_for_capacity: bool = False,
        max_wait_minutes: float = 0.0,
        fallback_models: Optional[List[str]] = None,
        phase_importance: float = 0.5
    ) -> None:
        """
        Log a supervisor decision event.
        
        Args:
            phase_name: Name of the phase
            models: List of models selected
            temperature: Temperature setting
            parallelism: Parallelism setting
            downgraded: Whether configuration was downgraded
            reason: Reason for the decision
            estimated_vram: Estimated VRAM usage
            wait_for_capacity: Whether to wait for GPU capacity
            max_wait_minutes: Maximum wait time before fallback
            fallback_models: Models to use if wait times out
            phase_importance: Importance weight of the phase (0-1)
        """
        self.log_event(
            event_type="supervisor_decision",
            data={
                "models_used": models,
                "temperature": temperature,
                "parallelism": parallelism,
                "downgraded": downgraded,
                "reason": reason,
                "estimated_vram": estimated_vram,
                "wait_for_capacity": wait_for_capacity,
                "max_wait_minutes": max_wait_minutes,
                "fallback_models": fallback_models or [],
                "phase_importance": phase_importance
            },
            phase_name=phase_name
        )
    
    def log_gpu_status(
        self,
        total_mem: int,
        used_mem: int,
        free_mem: int,
        utilization: int,
        pressure: str
    ) -> None:
        """
        Log GPU status event.
        
        Args:
            total_mem: Total GPU memory in MB
            used_mem: Used GPU memory in MB
            free_mem: Free GPU memory in MB
            utilization: GPU utilization percentage
            pressure: Resource pressure level
        """
        self.log_event(
            event_type="gpu_status",
            data={
                "total_mem": total_mem,
                "used_mem": used_mem,
                "free_mem": free_mem,
                "utilization": utilization,
                "pressure": pressure
            }
        )
    
    def log_council_execution(
        self,
        council_name: str,
        phase_name: str,
        models_used: List[str],
        success: bool,
        duration_s: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a council execution event.
        
        Args:
            council_name: Name of the council
            phase_name: Name of the phase
            models_used: Models that were used
            success: Whether execution succeeded
            duration_s: Duration in seconds
            error: Error message if failed
        """
        self.log_event(
            event_type="council_execution",
            data={
                "council_name": council_name,
                "models_used": models_used,
                "success": success,
                "duration_s": duration_s,
                "error": error
            },
            phase_name=phase_name
        )
    
    def log_step_execution(
        self,
        phase_name: str,
        step_name: str,
        step_type: str,
        chosen_model: str,
        status: str,
        attempts: int,
        duration_s: Optional[float] = None,
        pivot_suggestion: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a step execution event.
        
        Captures detailed information about each step executed by the Step Engine.
        
        Args:
            phase_name: Name of the phase containing the step
            step_name: Name of the step
            step_type: Type of step (research, coding, etc.)
            chosen_model: Model used for execution
            status: Final status (completed, failed, skipped)
            attempts: Number of execution attempts
            duration_s: Duration in seconds
            pivot_suggestion: Any suggested plan changes
            error: Error message if failed
        """
        self.log_event(
            event_type="step_execution",
            data={
                "mission_id": self.mission_id,
                "step_name": step_name,
                "step_type": step_type,
                "chosen_model": chosen_model,
                "status": status,
                "attempts": attempts,
                "duration_s": duration_s,
                "pivot_suggestion": pivot_suggestion,
                "error": error
            },
            phase_name=phase_name
        )
    
    def log_context_evolution(
        self,
        iteration: int,
        focus_areas: List[str],
        unresolved_questions: List[str],
        data_needs: List[str],
        web_searches: int,
        prior_knowledge_size: int,
        delta_summary: str
    ) -> None:
        """
        Log context evolution for visibility into iteration progress.
        
        This enables tracking of:
        - focus_areas changing across iterations
        - unresolved_questions reducing
        - web_searches_performed > 0
        
        Args:
            iteration: Current iteration number
            focus_areas: Current focus areas
            unresolved_questions: Current unresolved questions
            data_needs: Current data needs
            web_searches: Number of web searches performed this iteration
            prior_knowledge_size: Size of accumulated prior knowledge
            delta_summary: Summary of what changed
        """
        evolution_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": iteration,
            "focus_areas": focus_areas[:5],
            "focus_areas_count": len(focus_areas),
            "unresolved_questions": unresolved_questions[:5],
            "questions_count": len(unresolved_questions),
            "data_needs": data_needs[:5],
            "data_needs_count": len(data_needs),
            "web_searches": web_searches,
            "prior_knowledge_size": prior_knowledge_size,
            "delta_summary": delta_summary
        }
        
        self.context_evolution_log.append(evolution_entry)
        
        # Track history for visualization
        self.focus_areas_history.append(focus_areas[:5])
        self.questions_history.append(unresolved_questions[:5])
        
        self.log_event(
            event_type="context_evolution",
            data=evolution_entry
        )
        
        # Also log as text for easy viewing
        self.log(
            f"Context evolution #{iteration}: "
            f"{len(focus_areas)} focus areas, "
            f"{len(unresolved_questions)} questions, "
            f"{web_searches} web searches, "
            f"{prior_knowledge_size} chars knowledge | "
            f"{delta_summary}"
        )
    
    def log_multiview_disagreement(
        self,
        phase_name: str,
        agreement_score: float,
        contested_risks: List[str],
        contested_opportunities: List[str],
        unresolved_from_disagreement: List[str]
    ) -> None:
        """
        Log multi-view disagreement details.
        
        Args:
            phase_name: Name of the phase
            agreement_score: Agreement score between Optimist and Skeptic
            contested_risks: Risks that Skeptic raised but Optimist dismissed
            contested_opportunities: Opportunities Optimist saw but Skeptic doubted
            unresolved_from_disagreement: Questions derived from disagreement
        """
        self.log_event(
            event_type="multiview_disagreement",
            data={
                "agreement_score": agreement_score,
                "contested_risks_count": len(contested_risks),
                "contested_opportunities_count": len(contested_opportunities),
                "unresolved_count": len(unresolved_from_disagreement),
                "contested_risks": contested_risks[:3],
                "contested_opportunities": contested_opportunities[:3],
                "unresolved_questions": unresolved_from_disagreement[:3]
            },
            phase_name=phase_name
        )
        
        self.log(
            f"Multi-view disagreement in {phase_name}: "
            f"agreement={agreement_score:.1%}, "
            f"{len(contested_risks)} contested risks, "
            f"{len(contested_opportunities)} contested opportunities"
        )
    
    def log_escalation(
        self,
        from_profile: str,
        to_profile: str,
        reason: str,
        provenance: Optional[Any] = None,
        human_approved: bool = False
    ) -> None:
        """
        Log execution profile escalation event.
        
        Args:
            from_profile: Source execution profile
            to_profile: Target execution profile
            reason: Reason for escalation
            provenance: Code provenance (if available)
            human_approved: Whether human approved the escalation
        """
        escalation_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_profile": from_profile,
            "to_profile": to_profile,
            "reason": reason,
            "human_approved": human_approved,
            "provenance": provenance.to_dict() if provenance and hasattr(provenance, 'to_dict') else None
        }
        self.execution_escalations.append(escalation_entry)
        
        self.log_event(
            event_type="execution_escalation",
            data=escalation_entry
        )
        
        self.log(
            f"Execution escalation: {from_profile} → {to_profile} "
            f"(reason: {reason}, approved: {human_approved})"
        )
    
    def log_execution_metrics(
        self,
        profile_name: str,
        metrics: Any
    ) -> None:
        """
        Log execution metrics for observability.
        
        Args:
            profile_name: Execution profile used
            metrics: ExecutionMetrics object or dict
        """
        if hasattr(metrics, 'to_dict'):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "profile": profile_name,
            **metrics_dict
        }
        self.execution_metrics_history.append(entry)
        
        self.log_event(
            event_type="execution_metrics",
            data=entry
        )
    
    def log_web_search(
        self,
        phase_name: str,
        queries: List[str],
        results_count: int,
        triggered_by: str
    ) -> None:
        """
        Log web search activity.
        
        Args:
            phase_name: Name of the phase
            queries: Search queries executed
            results_count: Number of results returned
            triggered_by: What triggered the search (data_needs, questions, etc.)
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase_name,
            "queries": queries,
            "results_count": results_count,
            "triggered_by": triggered_by
        }
        self.web_search_history.append(entry)
        
        self.log_event(
            event_type="web_search",
            data=entry,
            phase_name=phase_name
        )
        
        self.log(
            f"Web search in {phase_name}: {len(queries)} queries, "
            f"{results_count} results (triggered by {triggered_by})"
        )
    
    def get_context_evolution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of context evolution for reporting.
        
        Returns:
            Dictionary with evolution statistics
        """
        if not self.context_evolution_log:
            return {"status": "no evolution logged"}
        
        first = self.context_evolution_log[0]
        last = self.context_evolution_log[-1]
        
        # Phase 6.1: Use SearchTriggerManager as single source of truth if available
        # If search_trigger_manager is passed, use it instead of summing context_evolution_log
        total_web_searches = self.total_web_searches  # Use stored value (updated from SearchTriggerManager)
        
        # Track question resolution
        first_questions = set(first.get("unresolved_questions", []))
        last_questions = set(last.get("unresolved_questions", []))
        resolved_questions = first_questions - last_questions
        
        return {
            "total_iterations": len(self.context_evolution_log),
            "initial_focus_areas": first.get("focus_areas_count", 0),
            "final_focus_areas": last.get("focus_areas_count", 0),
            "initial_questions": first.get("questions_count", 0),
            "final_questions": last.get("questions_count", 0),
            "questions_resolved": len(resolved_questions),
            "total_web_searches": total_web_searches,
            "prior_knowledge_growth": last.get("prior_knowledge_size", 0) - first.get("prior_knowledge_size", 0),
            "evolution_log": self.context_evolution_log
        }
    
    def get_step_executions(self) -> List[Dict[str, Any]]:
        """Get all step execution events."""
        return self.get_events_by_type("step_execution")
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """
        Get all events of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            List of matching event entries
        """
        return [
            event for event in self.event_logs
            if event.get("event") == event_type
        ]
    
    def get_supervisor_decisions(self) -> List[Dict[str, Any]]:
        """Get all supervisor decision events."""
        return self.get_events_by_type("supervisor_decision")
    
    # ==========================================================================
    # Decision Accountability Layer Methods
    # ==========================================================================
    
    def track_decision(self, decision_id: str) -> None:
        """
        Track a decision ID for this mission.
        
        Args:
            decision_id: UUID of the emitted decision
        """
        self.decision_records.append(decision_id)
    
    def set_last_model_decision(self, decision_id: str) -> None:
        """
        Set the last model selection decision ID.
        
        Used for causal linking of subsequent decisions.
        
        Args:
            decision_id: UUID of the model selection decision
        """
        self.last_model_decision_id = decision_id
        self.track_decision(decision_id)
    
    def set_last_governance_decision(self, decision_id: str) -> None:
        """
        Set the last governance decision ID.
        
        Used for causal linking of retry escalations.
        
        Args:
            decision_id: UUID of the governance decision
        """
        self.last_governance_decision_id = decision_id
        self.track_decision(decision_id)
    
    def get_decision_count(self) -> int:
        """Get total number of decisions tracked for this mission."""
        return len(self.decision_records)
    
    def current_phase(self) -> Optional[MissionPhase]:
        """Get the current phase, or None if all phases complete."""
        if self.current_phase_index < len(self.phases):
            return self.phases[self.current_phase_index]
        return None
    
    def advance_phase(self) -> None:
        """Advance to the next phase."""
        self.current_phase_index += 1
    
    def remaining_time(self) -> timedelta:
        """Get remaining time until deadline."""
        now = datetime.utcnow()
        if now >= self.deadline_at:
            return timedelta(seconds=0)
        return self.deadline_at - now
    
    def remaining_minutes(self) -> float:
        """Get remaining time in minutes."""
        return self.remaining_time().total_seconds() / 60.0
    
    def is_expired(self) -> bool:
        """Check if the mission has exceeded its deadline."""
        return datetime.utcnow() >= self.deadline_at
    
    def is_terminal(self) -> bool:
        """Check if the mission is in a terminal state."""
        return self.status in ("completed", "aborted", "failed", "expired")
    
    def set_failed(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark the mission as failed with a structured reason code.
        
        Args:
            reason: MissionFailureReason value (e.g., "council_failure", "governance_block")
            details: Optional structured details about the failure
            error_message: Optional human-readable error message for logs
        """
        self.status = "failed"
        self.failure_reason = reason
        if details:
            self.failure_details = details
        if error_message:
            self.log(f"Mission failed: {error_message}")
            self.failure_details["error_message"] = error_message
    
    def summary(self) -> str:
        """Generate a human-readable summary of mission state."""
        lines = [
            f"Mission: {self.mission_id}",
            f"Objective: {self.objective}",
            f"Status: {self.status}",
            f"Created: {self.created_at.isoformat()}",
            f"Deadline: {self.deadline_at.isoformat()}",
            f"Remaining: {self.remaining_minutes():.1f} minutes",
            f"Phases: {len(self.phases)}",
            f"Current Phase: {self.current_phase_index + 1}/{len(self.phases)}",
            "",
            "Phase Summary:"
        ]
        
        for i, phase in enumerate(self.phases):
            marker = ">" if i == self.current_phase_index else " "
            status_icon = {
                "pending": "○",
                "running": "◐",
                "completed": "●",
                "skipped": "◌",
                "failed": "✗"
            }.get(phase.status, "?")
            step_info = ""
            if phase.steps:
                completed = len([s for s in phase.steps if s.status == "completed"])
                total = len(phase.steps)
                step_info = f" [{completed}/{total} steps]"
            lines.append(f"  {marker} {status_icon} {phase.name}: {phase.status}{step_info}")
        
        # Add supervisor decisions summary
        decisions = self.get_supervisor_decisions()
        if decisions:
            lines.append("")
            lines.append("Supervisor Decisions:")
            for d in decisions[-5:]:  # Last 5 decisions
                downgrade_mark = " [DOWNGRADED]" if d.get("downgraded") else ""
                lines.append(f"  - {d.get('phase', 'unknown')}: {d.get('models_used', [])}{downgrade_mark}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert mission state to dictionary for serialization.
        
        Returns:
            Dictionary representation of mission state
        """
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "constraints": self.constraints.as_dict(),
            "created_at": self.created_at.isoformat(),
            "deadline_at": self.deadline_at.isoformat(),
            "current_phase_index": self.current_phase_index,
            "phases": [
                {
                    "name": p.name,
                    "description": p.description,
                    "status": p.status,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "ended_at": p.ended_at.isoformat() if p.ended_at else None,
                    "iterations": p.iterations,
                    "artifacts": p.artifacts,
                    "steps": [s.to_dict() for s in p.steps] if p.steps else []
                }
                for p in self.phases
            ],
            "status": self.status,
            "failure_reason": self.failure_reason,
            "failure_details": self.failure_details,
            "logs": self.logs,
            "event_logs": self.event_logs,
            "final_artifacts": self.final_artifacts,
            "output_deliverables": [
                od.to_dict() if hasattr(od, 'to_dict') else od
                for od in self.output_deliverables
            ],
            "phase_rounds": self.phase_rounds,
            "council_rounds": self.council_rounds,
            "work_summary": self.work_summary,
            "hypotheses": self.hypotheses,
            "updated_plan": self.updated_plan,
            "next_actions": self.next_actions,
            "meta_traces": self.meta_traces,
            # New fields in 2.0
            "context_summaries": self.context_summaries,
            "recent_outputs": self.recent_outputs[-10:] if self.recent_outputs else [],  # Limit size
            "iteration_count": self.iteration_count,
            # Context evolution tracking
            "context_evolution_log": self.context_evolution_log,
            "context_evolution_summary": self.get_context_evolution_summary(),
            "focus_areas_history": self.focus_areas_history[-10:] if self.focus_areas_history else [],
            "questions_history": self.questions_history[-10:] if self.questions_history else [],
            "web_search_history": self.web_search_history,
            # Epistemic Hardening state
            "evidence_recovery_state": (
                self.evidence_recovery_state.to_dict() 
                if self.evidence_recovery_state else None
            ),
            "focus_area_manager": (
                self.focus_area_manager.to_dict()
                if self.focus_area_manager else None
            ),
            # Alignment Control Layer
            "alignment_north_star": self.alignment_north_star,
            "alignment_trajectory": self.alignment_trajectory,
            "alignment_controller_state": self.alignment_controller_state,
            "pending_user_event": self.pending_user_event,
        }
