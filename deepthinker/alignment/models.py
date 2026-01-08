"""
Alignment Control Layer - Core Data Models.

Defines the data structures for the Hybrid Alignment Control system:
- NorthStarGoal: Immutable goal specification for drift detection
- AlignmentPoint: Single timestep measurement in alignment trajectory
- AlignmentTrajectory: Time series of alignment measurements
- AlignmentAssessment: LLM evaluator output
- AlignmentAction: Soft corrective actions

These models are designed to be serializable for persistence and
backward-compatible with existing MissionState.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import uuid


class AlignmentAction(str, Enum):
    """
    Soft corrective actions that can be applied when drift is detected.
    
    These actions apply pressure to the mission without hard-stopping it.
    They escalate gradually based on repeated drift detection.
    """
    REANCHOR_INTERNAL = "reanchor_internal"
    INCREASE_SKEPTIC_WEIGHT = "increase_skeptic_weight"
    PRUNE_OR_PARK_FOCUS_AREAS = "prune_focus_areas"
    SWITCH_DEEPEN_MODE_TO_EVIDENCE = "switch_to_evidence"
    TRIGGER_USER_EVENT_DRIFT_CONFIRMATION = "user_event"


@dataclass
class NorthStarGoal:
    """
    Immutable goal specification that anchors the mission.
    
    Created once at mission start and used as the reference point
    for all drift detection. The embedding is computed once and cached.
    
    Attributes:
        goal_id: Unique identifier for this goal
        intent_summary: Core mission objective (usually from mission.objective)
        success_criteria: What constitutes mission success
        forbidden_outcomes: Outcomes that must be avoided
        priority_axes: Key dimensions to optimize (e.g., {"cost": "minimize"})
        scope_boundaries: Explicit scope limits
        created_at_iso: ISO timestamp of creation
        embedding: Cached embedding vector (computed once)
    """
    goal_id: str
    intent_summary: str
    success_criteria: List[str] = field(default_factory=list)
    forbidden_outcomes: List[str] = field(default_factory=list)
    priority_axes: Dict[str, str] = field(default_factory=dict)
    scope_boundaries: List[str] = field(default_factory=list)
    created_at_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    embedding: Optional[List[float]] = None
    
    @classmethod
    def from_mission_objective(cls, objective: str, mission_id: str) -> "NorthStarGoal":
        """
        Create a NorthStarGoal from a mission objective.
        
        This is the default initialization path when no explicit
        goal calibration is provided.
        
        Args:
            objective: Mission objective string
            mission_id: Mission ID for goal_id generation
            
        Returns:
            NorthStarGoal with intent_summary populated
        """
        return cls(
            goal_id=f"ns_{mission_id}_{uuid.uuid4().hex[:8]}",
            intent_summary=objective,
            success_criteria=[],
            forbidden_outcomes=[],
            priority_axes={},
            scope_boundaries=[],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal_id": self.goal_id,
            "intent_summary": self.intent_summary,
            "success_criteria": self.success_criteria,
            "forbidden_outcomes": self.forbidden_outcomes,
            "priority_axes": self.priority_axes,
            "scope_boundaries": self.scope_boundaries,
            "created_at_iso": self.created_at_iso,
            # Exclude embedding from serialization (too large)
            "has_embedding": self.embedding is not None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NorthStarGoal":
        """Create from dictionary."""
        return cls(
            goal_id=data.get("goal_id", ""),
            intent_summary=data.get("intent_summary", ""),
            success_criteria=data.get("success_criteria", []),
            forbidden_outcomes=data.get("forbidden_outcomes", []),
            priority_axes=data.get("priority_axes", {}),
            scope_boundaries=data.get("scope_boundaries", []),
            created_at_iso=data.get("created_at_iso", datetime.utcnow().isoformat()),
            embedding=None,  # Embedding must be recomputed
        )
    
    def get_full_text(self) -> str:
        """
        Get full text representation for embedding.
        
        Concatenates all goal components into a single string
        suitable for embedding computation.
        """
        parts = [self.intent_summary]
        
        if self.success_criteria:
            parts.append("Success criteria: " + "; ".join(self.success_criteria))
        
        if self.forbidden_outcomes:
            parts.append("Forbidden: " + "; ".join(self.forbidden_outcomes))
        
        if self.priority_axes:
            axes = [f"{k}: {v}" for k, v in self.priority_axes.items()]
            parts.append("Priorities: " + "; ".join(axes))
        
        if self.scope_boundaries:
            parts.append("Scope: " + "; ".join(self.scope_boundaries))
        
        return "\n".join(parts)


@dataclass
class AlignmentPoint:
    """
    Single measurement point in the alignment trajectory.
    
    Represents alignment state at a specific timestep (phase).
    
    Attributes:
        t: Timestep index (phase index)
        a_t: Goal similarity - cos(goal_embedding, output_embedding)
        d_t: Drift delta - (a_t - a_{t-1})
        s_t: Semantic jump size - ||x_t - x_{t-1}|| on normalized vectors
        cusum_neg: Cumulative negative drift (CUSUM statistic)
        cumulative_neg_drift: Sum of max(0, -d_t) over all t
        warning: Whether this point is in warning state (a_t < warning_threshold)
        triggered: Whether this point exceeded correction thresholds (a_t < correction_threshold)
        phase_name: Name of the phase at this timestep
        timestamp_iso: ISO timestamp of measurement
        output_embedding_norm: L2 norm of output embedding (for debugging)
    """
    t: int
    a_t: float
    d_t: float
    s_t: float
    cusum_neg: float
    cumulative_neg_drift: float
    triggered: bool
    phase_name: str
    timestamp_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    output_embedding_norm: float = 0.0
    # Two-tier threshold system: warning is visibility-only, triggered means correction
    warning: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignmentPoint":
        """Create from dictionary."""
        return cls(
            t=data.get("t", 0),
            a_t=data.get("a_t", 0.0),
            d_t=data.get("d_t", 0.0),
            s_t=data.get("s_t", 0.0),
            cusum_neg=data.get("cusum_neg", 0.0),
            cumulative_neg_drift=data.get("cumulative_neg_drift", 0.0),
            triggered=data.get("triggered", False),
            phase_name=data.get("phase_name", ""),
            timestamp_iso=data.get("timestamp_iso", datetime.utcnow().isoformat()),
            output_embedding_norm=data.get("output_embedding_norm", 0.0),
            warning=data.get("warning", False),
        )
    
    @classmethod
    def initial(cls, phase_name: str = "initial") -> "AlignmentPoint":
        """Create initial alignment point (t=0)."""
        return cls(
            t=0,
            a_t=1.0,  # Perfect alignment at start
            d_t=0.0,
            s_t=0.0,
            cusum_neg=0.0,
            cumulative_neg_drift=0.0,
            triggered=False,
            phase_name=phase_name,
            warning=False,
        )


@dataclass
class AlignmentTrajectory:
    """
    Time series of alignment measurements for a mission.
    
    Tracks the evolution of goal alignment over the mission lifecycle.
    
    Attributes:
        mission_id: Mission this trajectory belongs to
        north_star: The NorthStarGoal being tracked against
        points: List of AlignmentPoints in chronological order
        assessments: List of LLM assessments (when evaluator runs)
        actions_taken: Log of actions applied
    """
    mission_id: str
    north_star: NorthStarGoal
    points: List[AlignmentPoint] = field(default_factory=list)
    assessments: List["AlignmentAssessment"] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_point(self, point: AlignmentPoint) -> None:
        """Add a new alignment point to the trajectory."""
        self.points.append(point)
    
    def last_point(self) -> Optional[AlignmentPoint]:
        """Get the most recent alignment point."""
        return self.points[-1] if self.points else None
    
    def get_trigger_count(self) -> int:
        """Count how many points triggered drift detection."""
        return sum(1 for p in self.points if p.triggered)
    
    def get_consecutive_triggers(self) -> int:
        """Count consecutive triggers from the end."""
        count = 0
        for point in reversed(self.points):
            if point.triggered:
                count += 1
            else:
                break
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mission_id": self.mission_id,
            "north_star": self.north_star.to_dict(),
            "points": [p.to_dict() for p in self.points],
            "assessments": [a.to_dict() for a in self.assessments],
            "actions_taken": self.actions_taken,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignmentTrajectory":
        """Create from dictionary."""
        return cls(
            mission_id=data.get("mission_id", ""),
            north_star=NorthStarGoal.from_dict(data.get("north_star", {})),
            points=[AlignmentPoint.from_dict(p) for p in data.get("points", [])],
            assessments=[AlignmentAssessment.from_dict(a) for a in data.get("assessments", [])],
            actions_taken=data.get("actions_taken", []),
        )


@dataclass
class AlignmentAssessment:
    """
    LLM evaluator output for alignment assessment.
    
    Generated when drift detector triggers and evaluator is enabled.
    Provides qualitative interpretation of the quantitative metrics.
    
    Attributes:
        perceived_alignment: Overall alignment level
        drift_risk: Risk of continued drift
        dominant_drift_vector: Description of main drift direction
        neglected_axes: Priority axes being neglected
        suggested_correction: Suggested corrective action
        timestamp_iso: When assessment was made
        metrics_snapshot: Metrics at time of assessment
    """
    perceived_alignment: Literal["high", "medium", "low"]
    drift_risk: Literal["none", "emerging", "severe"]
    dominant_drift_vector: str
    neglected_axes: List[str]
    suggested_correction: str
    timestamp_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignmentAssessment":
        """Create from dictionary."""
        return cls(
            perceived_alignment=data.get("perceived_alignment", "medium"),
            drift_risk=data.get("drift_risk", "none"),
            dominant_drift_vector=data.get("dominant_drift_vector", ""),
            neglected_axes=data.get("neglected_axes", []),
            suggested_correction=data.get("suggested_correction", ""),
            timestamp_iso=data.get("timestamp_iso", datetime.utcnow().isoformat()),
            metrics_snapshot=data.get("metrics_snapshot", {}),
        )
    
    @classmethod
    def fallback(cls, reason: str = "evaluation_failed") -> "AlignmentAssessment":
        """Create a fallback assessment when LLM evaluation fails."""
        return cls(
            perceived_alignment="medium",
            drift_risk="emerging",
            dominant_drift_vector=f"Unable to assess: {reason}",
            neglected_axes=[],
            suggested_correction="Continue with caution, re-anchor to original objective",
        )


@dataclass
class ControllerState:
    """
    Persistent state for the alignment controller.
    
    Tracks escalation state across phase boundaries.
    
    Attributes:
        consecutive_triggers: Number of consecutive triggered points
        last_action_timestamps: When each action type was last applied
        last_reanchor_t: Timestep of last re-anchor action
        total_actions_taken: Total count of actions taken
        implicit_goal_snapshot: Snapshot of goal at last re-anchor
        last_injection_t: Timestep of last prompt injection (for cooldown)
        injection_count_this_mission: Total prompt injections this mission (for limit)
    """
    consecutive_triggers: int = 0
    last_action_timestamps: Dict[str, str] = field(default_factory=dict)
    last_reanchor_t: int = -1
    total_actions_taken: int = 0
    implicit_goal_snapshot: Optional[str] = None
    last_injection_t: int = -1
    injection_count_this_mission: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControllerState":
        """Create from dictionary."""
        return cls(
            consecutive_triggers=data.get("consecutive_triggers", 0),
            last_action_timestamps=data.get("last_action_timestamps", {}),
            last_reanchor_t=data.get("last_reanchor_t", -1),
            total_actions_taken=data.get("total_actions_taken", 0),
            implicit_goal_snapshot=data.get("implicit_goal_snapshot"),
            last_injection_t=data.get("last_injection_t", -1),
            injection_count_this_mission=data.get("injection_count_this_mission", 0),
        )
    
    def record_action(self, action: AlignmentAction) -> None:
        """Record that an action was taken."""
        self.last_action_timestamps[action.value] = datetime.utcnow().isoformat()
        self.total_actions_taken += 1
    
    def record_injection(self, t: int) -> None:
        """Record that a prompt injection occurred."""
        self.last_injection_t = t
        self.injection_count_this_mission += 1
    
    def can_inject(self, t: int, min_phases: int, max_injections: int) -> bool:
        """Check if injection is allowed based on cooldown and limits."""
        # Check mission-wide limit
        if self.injection_count_this_mission >= max_injections:
            return False
        # Check cooldown between injections
        if self.last_injection_t >= 0 and (t - self.last_injection_t) < min_phases:
            return False
        return True
    
    def increment_triggers(self) -> None:
        """Increment consecutive trigger count."""
        self.consecutive_triggers += 1
    
    def reset_triggers(self) -> None:
        """Reset consecutive trigger count."""
        self.consecutive_triggers = 0


@dataclass
class UserDriftEvent:
    """
    Structured user event for drift confirmation.
    
    Presented to user when drift persists despite corrective actions.
    Contains structured choices for user to select from.
    
    Attributes:
        event_id: Unique event identifier
        event_type: Type of event ("drift_confirmation")
        created_at_iso: When event was created
        drift_summary: Summary of detected drift
        current_trajectory_summary: Brief trajectory description
        choices: List of structured choices for user
        selected_choice: User's selection (None until resolved)
        resolved_at_iso: When user resolved the event
    """
    event_id: str
    event_type: str = "drift_confirmation"
    created_at_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    drift_summary: str = ""
    current_trajectory_summary: str = ""
    choices: List[Dict[str, str]] = field(default_factory=list)
    selected_choice: Optional[str] = None
    resolved_at_iso: Optional[str] = None
    
    @classmethod
    def create_drift_confirmation(
        cls,
        mission_id: str,
        drift_summary: str,
        trajectory_summary: str,
    ) -> "UserDriftEvent":
        """Create a drift confirmation event with standard choices."""
        return cls(
            event_id=f"drift_{mission_id}_{uuid.uuid4().hex[:8]}",
            event_type="drift_confirmation",
            drift_summary=drift_summary,
            current_trajectory_summary=trajectory_summary,
            choices=[
                {
                    "id": "continue_as_is",
                    "label": "Continue as-is",
                    "description": "The current direction is acceptable despite detected drift",
                },
                {
                    "id": "reanchor_strong",
                    "label": "Re-anchor to original goal",
                    "description": "Strongly redirect back to the original mission objective",
                },
                {
                    "id": "adjust_goal",
                    "label": "Adjust goal (new input required)",
                    "description": "Update the mission objective to reflect evolved understanding",
                },
                {
                    "id": "abort_mission",
                    "label": "Abort mission",
                    "description": "Stop the mission due to unrecoverable drift",
                },
            ],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserDriftEvent":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", ""),
            event_type=data.get("event_type", "drift_confirmation"),
            created_at_iso=data.get("created_at_iso", datetime.utcnow().isoformat()),
            drift_summary=data.get("drift_summary", ""),
            current_trajectory_summary=data.get("current_trajectory_summary", ""),
            choices=data.get("choices", []),
            selected_choice=data.get("selected_choice"),
            resolved_at_iso=data.get("resolved_at_iso"),
        )

