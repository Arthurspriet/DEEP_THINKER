"""
Alignment Control Layer - Soft Action Controller.

Implements a PID-like controller that applies soft corrective pressure
based on alignment metrics and LLM assessments.

Key principles:
- Never hard-stops mission
- Escalates gradually based on consecutive triggers
- Actions are suggestions/pressure, not commands
- Integrates with existing mission knobs (councils, deepening, focus)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import AlignmentConfig, get_alignment_config
from .models import (
    AlignmentAction,
    AlignmentAssessment,
    AlignmentPoint,
    AlignmentTrajectory,
    ControllerState,
    UserDriftEvent,
)

logger = logging.getLogger(__name__)


class AlignmentController:
    """
    Soft action controller for alignment corrections.
    
    Maintains state across phase boundaries and applies graduated
    pressure based on drift severity and persistence.
    
    Escalation Ladder:
    1. First trigger: REANCHOR_INTERNAL
    2. Second consecutive: INCREASE_SKEPTIC_WEIGHT
    3. Third consecutive: SWITCH_DEEPEN_MODE_TO_EVIDENCE
    4. Fourth+: PRUNE_OR_PARK_FOCUS_AREAS
    5. Fifth+ or severe: TRIGGER_USER_EVENT_DRIFT_CONFIRMATION
    
    Usage:
        controller = AlignmentController(config)
        
        actions = controller.decide(
            point=alignment_point,
            trajectory=trajectory,
            assessment=optional_llm_assessment,
        )
        
        for action in actions:
            # Apply action to mission state
            pass
    """
    
    def __init__(
        self,
        config: Optional[AlignmentConfig] = None,
        state: Optional[ControllerState] = None,
    ):
        """
        Initialize the controller.
        
        Args:
            config: Alignment configuration (uses global if None)
            state: Existing controller state (for persistence)
        """
        self.config = config or get_alignment_config()
        self.state = state or ControllerState()
    
    def decide(
        self,
        point: AlignmentPoint,
        trajectory: AlignmentTrajectory,
        assessment: Optional[AlignmentAssessment] = None,
    ) -> List[AlignmentAction]:
        """
        Decide which corrective actions to take.
        
        This is the main entry point. Returns a list of actions
        to apply based on current alignment state.
        
        Args:
            point: Current alignment point
            trajectory: Full alignment trajectory
            assessment: Optional LLM assessment
            
        Returns:
            List of actions to apply (may be empty)
        """
        actions: List[AlignmentAction] = []
        
        # If not triggered, reset consecutive count and return
        if not point.triggered:
            self.state.reset_triggers()
            logger.debug("[ALIGNMENT] No trigger - resetting consecutive count")
            return actions
        
        # Increment consecutive triggers
        self.state.increment_triggers()
        consecutive = self.state.consecutive_triggers
        
        logger.info(
            f"[ALIGNMENT] Trigger #{consecutive} at t={point.t}, "
            f"a_t={point.a_t:.3f}, cusum={point.cusum_neg:.3f}"
        )
        
        # Determine severity
        severity = self._assess_severity(point, assessment)
        
        # Get actions based on escalation ladder
        ladder_actions = self._get_ladder_actions(
            consecutive=consecutive,
            severity=severity,
            point=point,
        )
        
        # Apply action limits
        actions = ladder_actions[:self.config.max_actions_per_phase]
        
        # Record actions
        for action in actions:
            self.state.record_action(action)
            trajectory.actions_taken.append({
                "action": action.value,
                "t": point.t,
                "phase": point.phase_name,
                "timestamp": datetime.utcnow().isoformat(),
                "consecutive_triggers": consecutive,
                "severity": severity,
            })
        
        if actions:
            logger.info(f"[ALIGNMENT] Actions decided: {[a.value for a in actions]}")
        
        return actions
    
    def _assess_severity(
        self,
        point: AlignmentPoint,
        assessment: Optional[AlignmentAssessment],
    ) -> str:
        """
        Assess severity of current drift.
        
        Returns "low", "medium", or "severe".
        """
        # Start with medium
        severity = "medium"
        
        # Check metrics
        if point.a_t < 0.3:
            severity = "severe"
        elif point.cusum_neg > self.config.cusum_h * 1.5:
            severity = "severe"
        elif point.a_t > 0.6 and point.cusum_neg < self.config.cusum_h * 0.7:
            severity = "low"
        
        # Override with LLM assessment if available
        if assessment is not None:
            if assessment.drift_risk == "severe":
                severity = "severe"
            elif assessment.drift_risk == "none" and severity != "severe":
                severity = "low"
        
        return severity
    
    def _get_ladder_actions(
        self,
        consecutive: int,
        severity: str,
        point: AlignmentPoint,
    ) -> List[AlignmentAction]:
        """
        Get actions based on escalation ladder.
        
        Higher consecutive counts and severity lead to stronger actions.
        """
        actions: List[AlignmentAction] = []
        ladder = self.config.escalation_ladder
        
        # Check re-anchor cooldown
        can_reanchor = (
            point.t - self.state.last_reanchor_t >= self.config.reanchor_cooldown_phases
        )
        
        # Level 1: Re-anchor (always first response if cooldown allows)
        if consecutive >= ladder.get("reanchor_internal", 1) and can_reanchor:
            actions.append(AlignmentAction.REANCHOR_INTERNAL)
            self.state.last_reanchor_t = point.t
        
        # Level 2: Increase skeptic weight
        if consecutive >= ladder.get("increase_skeptic_weight", 2):
            actions.append(AlignmentAction.INCREASE_SKEPTIC_WEIGHT)
        
        # Level 3: Switch to evidence mode
        if consecutive >= ladder.get("switch_to_evidence", 3):
            actions.append(AlignmentAction.SWITCH_DEEPEN_MODE_TO_EVIDENCE)
        
        # Level 4: Prune focus areas
        if consecutive >= ladder.get("prune_focus_areas", 4) or severity == "severe":
            actions.append(AlignmentAction.PRUNE_OR_PARK_FOCUS_AREAS)
        
        # Level 5: User event (last resort)
        if consecutive >= ladder.get("user_event", 5) or (
            severity == "severe" and consecutive >= 3
        ):
            # Only add if threshold met
            if consecutive >= self.config.user_event_threshold:
                actions.append(AlignmentAction.TRIGGER_USER_EVENT_DRIFT_CONFIRMATION)
        
        return actions
    
    def create_user_event(
        self,
        mission_id: str,
        trajectory: AlignmentTrajectory,
    ) -> UserDriftEvent:
        """
        Create a user drift confirmation event.
        
        Called when the user event action is triggered.
        
        Args:
            mission_id: Mission ID
            trajectory: Current trajectory
            
        Returns:
            UserDriftEvent for user interaction
        """
        # Build drift summary
        last_point = trajectory.last_point()
        if last_point:
            drift_summary = (
                f"Alignment has declined to {last_point.a_t:.1%} "
                f"(was 100% at start). Cumulative drift: {last_point.cusum_neg:.3f}. "
                f"Triggers fired {trajectory.get_trigger_count()} times."
            )
        else:
            drift_summary = "Unable to compute drift metrics."
        
        # Build trajectory summary
        if trajectory.points:
            recent_points = trajectory.points[-5:]
            trajectory_summary = ", ".join(
                f"t{p.t}:{p.a_t:.1%}" for p in recent_points
            )
        else:
            trajectory_summary = "No trajectory data."
        
        event = UserDriftEvent.create_drift_confirmation(
            mission_id=mission_id,
            drift_summary=drift_summary,
            trajectory_summary=f"Recent alignment: [{trajectory_summary}]",
        )
        
        logger.info(f"[ALIGNMENT] Created user event: {event.event_id}")
        
        return event
    
    def get_state(self) -> ControllerState:
        """Get the current controller state."""
        return self.state
    
    def set_state(self, state: ControllerState) -> None:
        """Set the controller state (for persistence)."""
        self.state = state
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.state = ControllerState()


def apply_action(
    action: AlignmentAction,
    mission_state: Any,
    controller: AlignmentController,
    trajectory: AlignmentTrajectory,
) -> bool:
    """
    Apply an alignment action to the mission state.
    
    This function modifies the mission state based on the action.
    Uses existing mission state knobs where possible.
    
    Args:
        action: Action to apply
        mission_state: MissionState object to modify
        controller: Controller for user event creation
        trajectory: Current trajectory
        
    Returns:
        True if action was applied successfully
    """
    try:
        if action == AlignmentAction.REANCHOR_INTERNAL:
            config = controller.config
            point = trajectory.last_point()
            current_t = point.t if point else 0
            
            # Check injection safety limits before applying prompt injection
            if config.inject_reanchor_prompt:
                if not controller.state.can_inject(
                    current_t,
                    config.min_phases_between_injections,
                    config.max_injections_per_mission,
                ):
                    logger.info(
                        f"[ALIGNMENT] Skipping prompt injection due to cooldown/limit "
                        f"(last_injection_t={controller.state.last_injection_t}, "
                        f"injection_count={controller.state.injection_count_this_mission})"
                    )
                    # Still apply the non-injection parts of reanchor
                    if not hasattr(mission_state, "alignment_controller_state"):
                        mission_state.alignment_controller_state = {}
                    mission_state.alignment_controller_state["implicit_goal_snapshot"] = (
                        trajectory.north_star.intent_summary
                    )
                    mission_state.alignment_controller_state["reanchor_timestamp"] = (
                        datetime.utcnow().isoformat()
                    )
                    if hasattr(mission_state, "log"):
                        mission_state.log(
                            f"[ALIGNMENT] Re-anchor (no prompt - limit reached): '{trajectory.north_star.intent_summary[:100]}...'"
                        )
                    logger.info("[ALIGNMENT] Applied: REANCHOR_INTERNAL (without prompt injection)")
                    return True
            
            # Store implicit goal snapshot for re-anchoring
            if not hasattr(mission_state, "alignment_controller_state"):
                mission_state.alignment_controller_state = {}
            
            mission_state.alignment_controller_state["implicit_goal_snapshot"] = (
                trajectory.north_star.intent_summary
            )
            mission_state.alignment_controller_state["reanchor_timestamp"] = (
                datetime.utcnow().isoformat()
            )
            
            # Inject re-anchor prompt for next council/step execution (Gap 2)
            if config.inject_reanchor_prompt:
                # Format the prompt template with the objective
                reanchor_prompt = config.reanchor_prompt_template.format(
                    objective=trajectory.north_star.intent_summary[:200]
                )
                mission_state.alignment_controller_state["reanchor_prompt"] = reanchor_prompt
                # Set expiry to next phase only (single-use)
                mission_state.alignment_controller_state["reanchor_prompt_phase"] = (
                    point.phase_name if point else ""
                )
                
                # Record injection for tracking
                controller.state.record_injection(current_t)
                
                logger.debug(f"[ALIGNMENT] Injected re-anchor prompt: {reanchor_prompt[:100]}...")
            
            # Log for observability
            if hasattr(mission_state, "log"):
                mission_state.log(
                    f"[ALIGNMENT] Re-anchor: refreshed goal snapshot to '{trajectory.north_star.intent_summary[:100]}...'"
                )
            
            logger.info("[ALIGNMENT] Applied: REANCHOR_INTERNAL")
            return True
        
        elif action == AlignmentAction.INCREASE_SKEPTIC_WEIGHT:
            # Set skeptic weight boost in work_summary
            if not hasattr(mission_state, "work_summary"):
                mission_state.work_summary = {}
            
            current_boost = mission_state.work_summary.get("skeptic_weight_boost", 0.0)
            mission_state.work_summary["skeptic_weight_boost"] = min(0.5, current_boost + 0.2)
            
            if hasattr(mission_state, "log"):
                mission_state.log(
                    f"[ALIGNMENT] Increased skeptic weight to {mission_state.work_summary['skeptic_weight_boost']}"
                )
            
            logger.info("[ALIGNMENT] Applied: INCREASE_SKEPTIC_WEIGHT")
            return True
        
        elif action == AlignmentAction.SWITCH_DEEPEN_MODE_TO_EVIDENCE:
            # Set deepen mode to evidence
            if not hasattr(mission_state, "work_summary"):
                mission_state.work_summary = {}
            
            mission_state.work_summary["deepen_mode"] = "evidence"
            mission_state.work_summary["deepen_mode_reason"] = "alignment_drift"
            
            if hasattr(mission_state, "log"):
                mission_state.log("[ALIGNMENT] Switched deepen mode to 'evidence'")
            
            logger.info("[ALIGNMENT] Applied: SWITCH_DEEPEN_MODE_TO_EVIDENCE")
            return True
        
        elif action == AlignmentAction.PRUNE_OR_PARK_FOCUS_AREAS:
            # Mark some focus areas as parked
            # This is a signal for the orchestrator to reduce scope
            if not hasattr(mission_state, "work_summary"):
                mission_state.work_summary = {}
            
            mission_state.work_summary["alignment_prune_requested"] = True
            mission_state.work_summary["alignment_prune_reason"] = (
                f"Drift detected at t={trajectory.last_point().t if trajectory.last_point() else 0}"
            )
            
            if hasattr(mission_state, "log"):
                mission_state.log("[ALIGNMENT] Requested focus area pruning")
            
            logger.info("[ALIGNMENT] Applied: PRUNE_OR_PARK_FOCUS_AREAS")
            return True
        
        elif action == AlignmentAction.TRIGGER_USER_EVENT_DRIFT_CONFIRMATION:
            # Create and store user event
            event = controller.create_user_event(
                mission_id=mission_state.mission_id,
                trajectory=trajectory,
            )
            
            if hasattr(mission_state, "pending_user_event"):
                mission_state.pending_user_event = event.to_dict()
            else:
                # Store in work_summary as fallback
                if not hasattr(mission_state, "work_summary"):
                    mission_state.work_summary = {}
                mission_state.work_summary["pending_user_event"] = event.to_dict()
            
            if hasattr(mission_state, "log"):
                mission_state.log(f"[ALIGNMENT] User confirmation requested: {event.event_id}")
            
            logger.info("[ALIGNMENT] Applied: TRIGGER_USER_EVENT_DRIFT_CONFIRMATION")
            return True
        
        else:
            logger.warning(f"[ALIGNMENT] Unknown action: {action}")
            return False
        
    except Exception as e:
        logger.warning(f"[ALIGNMENT] Failed to apply action {action}: {e}")
        return False


# Global controller instance (not used - controllers should be per-mission)
# Use get_controller_for_mission() pattern if needed

