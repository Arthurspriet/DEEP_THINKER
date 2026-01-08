"""
Alignment Control Layer - Integration.

Main integration point for the alignment subsystem.
Provides the hook that is called from MissionOrchestrator at phase end.

This module:
1. Extracts artifact text from phase results
2. Updates the alignment trajectory
3. Runs evaluator if triggered
4. Runs controller to get actions
5. Applies actions to mission state
6. Persists logs if enabled
7. Publishes SSE events for real-time alignment monitoring (Gap 3)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import AlignmentConfig, get_alignment_config
from .controller import AlignmentController, apply_action
from .drift import EmbeddingDriftDetector, get_drift_detector
from .evaluator import AlignmentEvaluator, get_alignment_evaluator
from .models import (
    AlignmentAction,
    AlignmentAssessment,
    AlignmentPoint,
    AlignmentTrajectory,
    ControllerState,
    NorthStarGoal,
)
from .persist import AlignmentLogStore, get_log_store, log_alignment_event

# SSE integration (Gap 3)
try:
    from api.sse import sse_manager
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    sse_manager = None


def _publish_sse_event(coro):
    """
    Helper to publish SSE events from sync code.
    Safely schedules the coroutine if an event loop is running.
    """
    if not SSE_AVAILABLE or sse_manager is None:
        coro.close()
        return
    try:
        import asyncio
        asyncio.get_running_loop()
        asyncio.create_task(coro)
    except RuntimeError:
        coro.close()


if TYPE_CHECKING:
    from deepthinker.missions.mission_types import MissionPhase, MissionState

logger = logging.getLogger(__name__)


class AlignmentManager:
    """
    Manages alignment control for a mission.
    
    Coordinates drift detection, evaluation, control, and persistence.
    One instance per mission, created lazily on first alignment check.
    
    Usage:
        manager = AlignmentManager.for_mission(mission_state)
        actions = manager.check_phase(phase, phase_result)
    """
    
    def __init__(
        self,
        mission_id: str,
        objective: str,
        config: Optional[AlignmentConfig] = None,
    ):
        """
        Initialize the alignment manager.
        
        Args:
            mission_id: Mission ID
            objective: Mission objective (for NorthStarGoal)
            config: Alignment configuration
        """
        self.mission_id = mission_id
        self.config = config or get_alignment_config()
        
        # Create components
        self.detector = EmbeddingDriftDetector(self.config)
        self.evaluator = AlignmentEvaluator(self.config) if self.config.run_evaluator_on_trigger else None
        self.controller = AlignmentController(self.config)
        self.log_store = get_log_store(self.config)
        
        # Create north star from objective
        self.north_star = NorthStarGoal.from_mission_objective(
            objective=objective,
            mission_id=mission_id,
        )
        
        # Create trajectory
        self.trajectory = self.detector.create_trajectory(
            mission_id=mission_id,
            north_star=self.north_star,
        )
        
        # Initialize log
        self.log_store.initialize(mission_id, self.north_star)
        
        logger.info(f"[ALIGNMENT] Manager initialized for mission {mission_id}")
    
    def check_phase(
        self,
        phase: "MissionPhase",
        phase_result: Dict[str, Any],
        mission_state: "MissionState",
    ) -> List[AlignmentAction]:
        """
        Check alignment after a phase completes.
        
        This is the main entry point called by the orchestrator.
        
        Args:
            phase: The completed phase
            phase_result: Phase artifacts/results
            mission_state: Current mission state
            
        Returns:
            List of actions applied
        """
        # Extract artifact text
        artifact_text = self._extract_artifact_text(phase_result)
        
        if not artifact_text:
            logger.debug(f"[ALIGNMENT] No artifact text for phase {phase.name}")
            return []
        
        # Update trajectory
        self.trajectory, point = self.detector.update_trajectory(
            trajectory=self.trajectory,
            output_text=artifact_text,
            phase_name=phase.name,
        )
        
        if point is None:
            logger.debug(f"[ALIGNMENT] Failed to compute alignment point")
            return []
        
        # Log point
        self.log_store.add_point(self.mission_id, point)
        log_alignment_event(
            mission_id=self.mission_id,
            event_type="point",
            data={
                "t": point.t,
                "a_t": f"{point.a_t:.3f}",
                "d_t": f"{point.d_t:.3f}",
                "cusum": f"{point.cusum_neg:.3f}",
                "warning": point.warning,
                "triggered": point.triggered,
                "phase": phase.name,
            },
        )
        
        # Publish SSE event for real-time alignment monitoring (Gap 3)
        if SSE_AVAILABLE and sse_manager:
            _publish_sse_event(sse_manager.publish_alignment_update(
                mission_id=self.mission_id,
                phase_name=phase.name,
                alignment_score=point.a_t,
                warning=point.warning,
                correction=point.triggered,
                action_applied=None,  # Will be updated below if action applied
                cusum_neg=point.cusum_neg,
                drift_delta=point.d_t,
                consecutive_triggers=self.controller.state.consecutive_triggers,
            ))
            
            # Publish warning event if warning threshold crossed but not triggered
            if point.warning and not point.triggered:
                _publish_sse_event(sse_manager.publish_alignment_warning(
                    mission_id=self.mission_id,
                    phase_name=phase.name,
                    alignment_score=point.a_t,
                    message=f"Alignment score {point.a_t:.1%} below warning threshold",
                    threshold=self.config.warning_threshold,
                ))
        
        # If not triggered, no actions needed
        if not point.triggered:
            return []
        
        # Run evaluator if enabled
        assessment: Optional[AlignmentAssessment] = None
        if self.evaluator is not None:
            try:
                assessment = self.evaluator.evaluate(
                    north_star=self.north_star,
                    current_output=artifact_text,
                    point=point,
                )
                self.trajectory.assessments.append(assessment)
                self.log_store.add_assessment(self.mission_id, assessment)
                
                log_alignment_event(
                    mission_id=self.mission_id,
                    event_type="assessment",
                    data={
                        "alignment": assessment.perceived_alignment,
                        "risk": assessment.drift_risk,
                        "phase": phase.name,
                    },
                )
            except Exception as e:
                logger.warning(f"[ALIGNMENT] Evaluator failed: {e}")
                assessment = AlignmentAssessment.fallback(str(e))
        
        # Run controller
        actions = self.controller.decide(
            point=point,
            trajectory=self.trajectory,
            assessment=assessment,
        )
        
        # Apply actions
        applied_actions: List[AlignmentAction] = []
        for action in actions:
            success = apply_action(
                action=action,
                mission_state=mission_state,
                controller=self.controller,
                trajectory=self.trajectory,
            )
            
            if success:
                applied_actions.append(action)
                
                # Capture injected prompt for audit logging (if this was a reanchor action)
                injected_prompt = None
                if action == AlignmentAction.REANCHOR_INTERNAL:
                    controller_state = getattr(mission_state, "alignment_controller_state", {})
                    if isinstance(controller_state, dict):
                        injected_prompt = controller_state.get("reanchor_prompt")
                
                self.log_store.add_action(
                    mission_id=self.mission_id,
                    action=action,
                    point=point,
                    metadata={"assessment": assessment.to_dict() if assessment else None},
                    injected_prompt=injected_prompt,
                )
                
                log_alignment_event(
                    mission_id=self.mission_id,
                    event_type="action",
                    data={
                        "action": action.value,
                        "t": point.t,
                        "phase": phase.name,
                    },
                )
                
                # Publish SSE correction event (Gap 3)
                if SSE_AVAILABLE and sse_manager:
                    _publish_sse_event(sse_manager.publish_alignment_correction(
                        mission_id=self.mission_id,
                        phase_name=phase.name,
                        action=action.value,
                        reason=f"Alignment drift detected (a_t={point.a_t:.2f})",
                        alignment_score=point.a_t,
                        consecutive_triggers=self.controller.state.consecutive_triggers,
                    ))
        
        # Update mission state with trajectory
        self._sync_to_mission_state(mission_state)
        
        # Save logs
        self.log_store.save(self.mission_id)
        
        return applied_actions
    
    def _extract_artifact_text(self, phase_result: Dict[str, Any]) -> str:
        """
        Extract text content from phase result for embedding.
        
        Tries various common artifact patterns.
        """
        if not phase_result:
            return ""
        
        # Try common artifact keys
        text_keys = [
            "summary",
            "output",
            "research_output",
            "analysis",
            "findings",
            "report",
            "content",
            "text",
            "phase_summary",
            "claim_bundle_summary",
        ]
        
        for key in text_keys:
            if key in phase_result:
                value = phase_result[key]
                if isinstance(value, str) and len(value) > 50:
                    return value
        
        # Try to concatenate all string values
        texts = []
        for key, value in phase_result.items():
            if isinstance(value, str) and len(value) > 20:
                texts.append(f"{key}: {value}")
        
        if texts:
            return "\n\n".join(texts)
        
        # Last resort: convert to string
        try:
            import json
            return json.dumps(phase_result, default=str)[:3000]
        except Exception:
            return str(phase_result)[:3000]
    
    def _sync_to_mission_state(self, mission_state: "MissionState") -> None:
        """
        Sync alignment data to mission state for persistence.
        """
        # Store trajectory
        if hasattr(mission_state, "alignment_trajectory"):
            mission_state.alignment_trajectory = [
                p.to_dict() for p in self.trajectory.points
            ]
        
        # Store north star
        if hasattr(mission_state, "alignment_north_star"):
            mission_state.alignment_north_star = self.north_star.to_dict()
        
        # Store controller state
        if hasattr(mission_state, "alignment_controller_state"):
            state_dict = self.controller.get_state().to_dict()
            if isinstance(mission_state.alignment_controller_state, dict):
                mission_state.alignment_controller_state.update(state_dict)
            else:
                mission_state.alignment_controller_state = state_dict
    
    def finalize(self) -> None:
        """
        Finalize alignment tracking at mission end.
        
        Saves final logs and summary.
        """
        self.log_store.update_from_trajectory(
            mission_id=self.mission_id,
            trajectory=self.trajectory,
            controller_state=self.controller.get_state(),
        )
        self.log_store.save(self.mission_id)
        
        logger.info(
            f"[ALIGNMENT] Mission {self.mission_id} finalized: "
            f"{len(self.trajectory.points)} points, "
            f"{self.trajectory.get_trigger_count()} triggers, "
            f"{len(self.trajectory.actions_taken)} actions"
        )


# Per-mission manager cache
_managers: Dict[str, AlignmentManager] = {}


def get_alignment_manager(
    mission_state: "MissionState",
    config: Optional[AlignmentConfig] = None,
) -> Optional[AlignmentManager]:
    """
    Get or create alignment manager for a mission.
    
    Returns None if alignment is disabled.
    
    Args:
        mission_state: Mission state
        config: Optional configuration override
        
    Returns:
        AlignmentManager or None
    """
    config = config or get_alignment_config(mission_state.constraints)
    
    if not config.enabled:
        return None
    
    mission_id = mission_state.mission_id
    
    if mission_id not in _managers:
        _managers[mission_id] = AlignmentManager(
            mission_id=mission_id,
            objective=mission_state.objective,
            config=config,
        )
    
    return _managers[mission_id]


def run_alignment_check(
    mission_state: "MissionState",
    phase: "MissionPhase",
    phase_result: Dict[str, Any],
) -> List[AlignmentAction]:
    """
    Main entry point for alignment checks.
    
    Called by MissionOrchestrator at the end of each phase.
    Handles all errors gracefully - never crashes the mission.
    
    Args:
        mission_state: Current mission state
        phase: Completed phase
        phase_result: Phase artifacts/results
        
    Returns:
        List of actions applied (empty if disabled or error)
    """
    try:
        # Get configuration (may be None if not in constraints)
        config = get_alignment_config(
            getattr(mission_state, "constraints", None)
        )
        
        if not config.enabled:
            return []
        
        # Get or create manager
        manager = get_alignment_manager(mission_state, config)
        
        if manager is None:
            return []
        
        # Run check
        actions = manager.check_phase(
            phase=phase,
            phase_result=phase_result,
            mission_state=mission_state,
        )
        
        return actions
        
    except Exception as e:
        # Silent failure - alignment errors never crash mission
        logger.debug(f"[ALIGNMENT] Check failed (non-fatal): {e}")
        return []


def finalize_alignment(mission_state: "MissionState") -> None:
    """
    Finalize alignment tracking for a mission.
    
    Called when mission completes.
    
    Args:
        mission_state: Mission state
    """
    try:
        mission_id = mission_state.mission_id
        
        if mission_id in _managers:
            _managers[mission_id].finalize()
            del _managers[mission_id]
            
    except Exception as e:
        logger.debug(f"[ALIGNMENT] Finalize failed (non-fatal): {e}")


def cleanup_alignment(mission_id: str) -> None:
    """
    Clean up alignment resources for a mission.
    
    Called when mission is aborted or cleaned up.
    
    Args:
        mission_id: Mission ID
    """
    try:
        if mission_id in _managers:
            del _managers[mission_id]
    except Exception:
        pass

