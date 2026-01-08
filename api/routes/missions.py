"""
Mission API Routes.

Provides CRUD operations, status, logs, artifacts, and SSE events for missions.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deepthinker.missions import MissionStore, MissionOrchestrator
from deepthinker.missions.mission_types import build_constraints_from_time_budget, MissionFailureReason
from deepthinker.councils.planner_council.planner_council import PlannerCouncil
from deepthinker.councils.researcher_council.researcher_council import ResearcherCouncil
from deepthinker.councils.coder_council.coder_council import CoderCouncil
from deepthinker.councils.evaluator_council.evaluator_council import EvaluatorCouncil
from deepthinker.councils.simulation_council.simulation_council import SimulationCouncil
from deepthinker.arbiter.arbiter import Arbiter

from ..sse import sse_manager, SSEEvent

router = APIRouter(prefix="/api/missions", tags=["missions"])

# Initialize store
_store = MissionStore()


def _get_orchestrator() -> MissionOrchestrator:
    """Create a MissionOrchestrator with all councils."""
    ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    
    return MissionOrchestrator(
        planner_council=PlannerCouncil(ollama_base_url=ollama_url),
        researcher_council=ResearcherCouncil(ollama_base_url=ollama_url),
        coder_council=CoderCouncil(ollama_base_url=ollama_url),
        evaluator_council=EvaluatorCouncil(ollama_base_url=ollama_url),
        simulation_council=SimulationCouncil(ollama_base_url=ollama_url),
        arbiter=Arbiter(ollama_base_url=ollama_url),
        store=_store
    )


# Request/Response Models
class CreateMissionRequest(BaseModel):
    """Request body for creating a new mission."""
    objective: str = Field(..., description="The mission objective")
    time_budget_minutes: int = Field(..., ge=1, le=1440, description="Time budget in minutes")
    allow_internet: bool = Field(default=True, description="Allow web research")
    allow_code_execution: bool = Field(default=True, description="Allow code execution")
    max_iterations: int = Field(default=100, ge=1, le=1000, description="Max iterations per phase")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class MissionSummary(BaseModel):
    """Summary of a mission for list views."""
    mission_id: str
    objective: str
    status: str
    created_at: str
    remaining_minutes: float
    total_phases: int
    completed_phases: int
    current_phase_index: int
    progress_percent: float
    # Failure information (populated when status is "failed")
    failure_reason: Optional[str] = None
    failure_details: Optional[Dict[str, Any]] = None


class PhaseDetail(BaseModel):
    """Details of a single phase."""
    name: str
    description: str
    status: str
    started_at: Optional[str]
    ended_at: Optional[str]
    iterations: int
    artifacts: Dict[str, str]
    duration_seconds: Optional[float]


class MissionDetail(BaseModel):
    """Full mission details."""
    mission_id: str
    objective: str
    status: str
    created_at: str
    deadline_at: str
    remaining_minutes: float
    current_phase_index: int
    phases: List[PhaseDetail]
    logs: List[str]
    final_artifacts: Dict[str, str]
    constraints: Dict[str, Any]
    work_summary: Dict[str, Any] = {}
    phase_rounds: Dict[str, int] = {}
    council_rounds: Dict[str, int] = {}
    # Failure information (populated when status is "failed")
    failure_reason: Optional[str] = None
    failure_details: Optional[Dict[str, Any]] = None


class ArtifactInfo(BaseModel):
    """Information about an artifact."""
    name: str
    phase: str
    content: str
    type: str


class DeliverableInfo(BaseModel):
    """Information about a deliverable."""
    path: str
    format: str
    description: Optional[str]


# Background task for running missions with SSE events
async def run_mission_with_events(mission_id: str):
    """
    Run a mission and publish SSE events.
    
    Ensures deterministic state transitions:
    - Sets status="running" BEFORE execution starts
    - Persists failure state on ANY exception
    - SSE publishing failures never propagate to crash the backend
    """
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    
    state = None
    orchestrator = None
    
    try:
        # Load state and transition to RUNNING immediately
        state = _store.load(mission_id)
        state.status = "running"
        state.log("Mission execution starting")
        _store.save(state)
        
        orchestrator = _get_orchestrator()
        
        # Publish mission started (SSE errors should not crash execution)
        try:
            await sse_manager.publish_log_added(mission_id, f"Mission started: {state.objective}")
        except Exception as sse_err:
            logger.warning(f"SSE publish failed (non-fatal): {sse_err}")
        
        # Run with heartbeat that publishes events
        def heartbeat(s):
            phase = s.current_phase()
            if phase and phase.status == "running":
                try:
                    asyncio.create_task(
                        sse_manager.publish_log_added(
                            mission_id, 
                            f"Phase '{phase.name}' in progress ({s.remaining_minutes():.1f}m remaining)"
                        )
                    )
                except Exception:
                    pass  # SSE heartbeat failures are non-fatal
        
        # Execute the mission
        final_state = orchestrator.run_until_complete_or_timeout(
            mission_id,
            heartbeat_callback=heartbeat
        )
        
        # Publish completion (SSE errors should not affect final state)
        try:
            await sse_manager.publish_mission_completed(
                mission_id,
                final_state.status,
                final_state.final_artifacts
            )
        except Exception as sse_err:
            logger.warning(f"SSE completion publish failed (non-fatal): {sse_err}")
        
    except Exception as e:
        # Capture full traceback for diagnostics
        error_trace = traceback.format_exc()
        error_msg = str(e)
        logger.error(f"Mission {mission_id} failed with error: {error_msg}\n{error_trace}")
        
        # Persist failed state to store - this is critical
        try:
            if state is None:
                state = _store.load(mission_id)
            
            state.set_failed(
                reason=MissionFailureReason.UNKNOWN_ERROR.value,
                details={
                    "error_type": type(e).__name__,
                    "traceback": error_trace[:2000],  # Limit size
                },
                error_message=error_msg
            )
            _store.save(state)
        except Exception as persist_err:
            logger.error(f"Failed to persist error state for mission {mission_id}: {persist_err}")
        
        # Publish SSE events (failures here are non-fatal)
        try:
            await sse_manager.publish_log_added(mission_id, f"Error: {error_msg}", "error")
            await sse_manager.publish_mission_completed(mission_id, "failed", {})
        except Exception:
            pass  # SSE failures should never propagate


@router.get("", response_model=List[MissionSummary])
async def list_missions(
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List all missions with summary information."""
    missions = _store.list_missions_with_status()
    
    if status:
        missions = [m for m in missions if m["status"] == status]
    
    result = []
    for m in missions:
        try:
            state = _store.load(m["mission_id"])
            total_phases = len(state.phases)
            completed_phases = len([p for p in state.phases if p.status == "completed"])
            progress = (completed_phases / total_phases * 100) if total_phases > 0 else 0
            
            result.append(MissionSummary(
                mission_id=state.mission_id,
                objective=state.objective,
                status=state.status,
                created_at=state.created_at.isoformat(),
                remaining_minutes=state.remaining_minutes(),
                total_phases=total_phases,
                completed_phases=completed_phases,
                current_phase_index=state.current_phase_index,
                progress_percent=progress,
                failure_reason=state.failure_reason,
                failure_details=state.failure_details if state.failure_details else None
            ))
        except FileNotFoundError:
            # Mission file was deleted between list and load
            continue
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error loading mission {m['mission_id']}: {e}")
            continue
    
    return result


@router.post("", response_model=MissionSummary)
async def create_mission(
    request: CreateMissionRequest,
    background_tasks: BackgroundTasks
):
    """Create and start a new mission."""
    constraints = build_constraints_from_time_budget(
        time_budget_minutes=request.time_budget_minutes,
        allow_code=request.allow_code_execution,
        allow_internet=request.allow_internet,
        notes=request.notes,
        max_iterations=request.max_iterations,
    )
    
    orchestrator = _get_orchestrator()
    state = orchestrator.create_mission(request.objective, constraints)
    
    # Start mission in background
    background_tasks.add_task(run_mission_with_events, state.mission_id)
    
    total_phases = len(state.phases)
    
    return MissionSummary(
        mission_id=state.mission_id,
        objective=state.objective,
        status=state.status,
        created_at=state.created_at.isoformat(),
        remaining_minutes=state.remaining_minutes(),
        total_phases=total_phases,
        completed_phases=0,
        current_phase_index=0,
        progress_percent=0.0
    )


@router.get("/{mission_id}", response_model=MissionDetail)
async def get_mission(mission_id: str):
    """Get full details of a mission."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    
    phases = []
    for p in state.phases:
        phases.append(PhaseDetail(
            name=p.name,
            description=p.description,
            status=p.status,
            started_at=p.started_at.isoformat() if p.started_at else None,
            ended_at=p.ended_at.isoformat() if p.ended_at else None,
            iterations=p.iterations,
            artifacts=p.artifacts,
            duration_seconds=p.duration_seconds()
        ))
    
    return MissionDetail(
        mission_id=state.mission_id,
        objective=state.objective,
        status=state.status,
        created_at=state.created_at.isoformat(),
        deadline_at=state.deadline_at.isoformat(),
        remaining_minutes=state.remaining_minutes(),
        current_phase_index=state.current_phase_index,
        phases=phases,
        logs=state.logs[-100:],  # Last 100 logs
        final_artifacts=state.final_artifacts,
        constraints=state.constraints.as_dict(),
        work_summary=state.work_summary,
        phase_rounds=state.phase_rounds,
        council_rounds=state.council_rounds,
        failure_reason=state.failure_reason,
        failure_details=state.failure_details if state.failure_details else None
    )


@router.get("/{mission_id}/status")
async def get_mission_status(mission_id: str):
    """Get quick status of a mission."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    current_phase = state.current_phase()
    
    return {
        "mission_id": state.mission_id,
        "status": state.status,
        "remaining_minutes": state.remaining_minutes(),
        "current_phase": current_phase.name if current_phase else None,
        "current_phase_index": state.current_phase_index,
        "total_phases": len(state.phases),
        "is_expired": state.is_expired(),
        "is_terminal": state.is_terminal(),
        # Failure information (populated when status is "failed")
        "failure_reason": state.failure_reason,
        "failure_details": state.failure_details if state.failure_details else None,
    }


@router.post("/{mission_id}/resume")
async def resume_mission(
    mission_id: str,
    background_tasks: BackgroundTasks
):
    """Resume a paused or pending mission."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    
    if state.is_terminal():
        raise HTTPException(status_code=400, detail=f"Cannot resume mission with status: {state.status}")
    
    # Resume in background
    background_tasks.add_task(run_mission_with_events, mission_id)
    
    return {"message": "Mission resumed", "mission_id": mission_id}


@router.post("/{mission_id}/abort")
async def abort_mission(
    mission_id: str,
    reason: Optional[str] = Query(default="User requested abort")
):
    """Abort a running mission."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    orchestrator = _get_orchestrator()
    state = orchestrator.abort_mission(mission_id, reason)
    
    # Publish abort event
    await sse_manager.publish_mission_completed(mission_id, "aborted", state.final_artifacts)
    
    return {"message": "Mission aborted", "mission_id": mission_id, "status": state.status}


@router.get("/{mission_id}/logs")
async def get_mission_logs(
    mission_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Get mission execution logs."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    logs = state.logs[offset:offset + limit]
    
    return {
        "mission_id": mission_id,
        "total": len(state.logs),
        "offset": offset,
        "limit": limit,
        "logs": logs
    }


@router.get("/{mission_id}/artifacts", response_model=List[ArtifactInfo])
async def get_mission_artifacts(
    mission_id: str,
    phase: Optional[str] = Query(None, description="Filter by phase name")
):
    """Get all artifacts from a mission."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    artifacts = []
    
    # Collect artifacts from phases
    for p in state.phases:
        if phase and p.name != phase:
            continue
        for name, content in p.artifacts.items():
            artifact_type = "text"
            if name.endswith((".py", ".js", ".ts")):
                artifact_type = "code"
            elif name.endswith((".md", ".txt")):
                artifact_type = "document"
            elif name.endswith(".json"):
                artifact_type = "data"
            
            artifacts.append(ArtifactInfo(
                name=name,
                phase=p.name,
                content=content[:5000] if len(content) > 5000 else content,
                type=artifact_type
            ))
    
    # Add final artifacts
    if not phase:
        for name, content in state.final_artifacts.items():
            artifacts.append(ArtifactInfo(
                name=name,
                phase="final",
                content=content[:5000] if len(content) > 5000 else content,
                type="final"
            ))
    
    return artifacts


@router.get("/{mission_id}/deliverables", response_model=List[DeliverableInfo])
async def get_mission_deliverables(mission_id: str):
    """Get final deliverables (output files) from a mission."""
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    
    return [
        DeliverableInfo(
            path=d.path if hasattr(d, 'path') else str(d),
            format=d.format.value if hasattr(d, 'format') and hasattr(d.format, 'value') else "unknown",
            description=d.description if hasattr(d, 'description') else None
        )
        for d in state.output_deliverables
    ]


@router.get("/{mission_id}/events")
async def mission_events(mission_id: str):
    """
    SSE endpoint for real-time mission events.
    
    Events: phase_started, phase_completed, council_started, council_completed,
            artifact_generated, log_added, mission_completed, alignment_update
    """
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    return StreamingResponse(
        sse_manager.subscribe(mission_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =========================================================================
# ALIGNMENT CONTROL LAYER ENDPOINT (Gap 4)
# =========================================================================

class AlignmentPointInfo(BaseModel):
    """Alignment measurement at a single timestep."""
    t: int
    phase_name: str
    alignment_score: float
    drift_delta: float
    cusum_neg: float
    warning: bool
    triggered: bool
    timestamp: str


class AlignmentActionInfo(BaseModel):
    """Record of an applied corrective action."""
    action: str
    t: int
    phase_name: str
    timestamp: str
    consecutive_triggers: int


class AlignmentResponse(BaseModel):
    """Full alignment data for a mission."""
    mission_id: str
    enabled: bool
    north_star: Optional[Dict[str, Any]] = None
    trajectory: List[AlignmentPointInfo] = []
    actions_taken: List[AlignmentActionInfo] = []
    summary: Dict[str, Any] = {}


@router.get("/{mission_id}/alignment", response_model=AlignmentResponse)
async def get_mission_alignment(mission_id: str):
    """
    Get alignment control data for a mission.
    
    Returns the north star goal, alignment trajectory (score over time),
    and any corrective actions applied.
    
    Alignment Control Layer (Gap 4): API endpoint for frontend to fetch
    alignment history and display alignment monitoring dashboard.
    """
    if not _store.exists(mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    
    state = _store.load(mission_id)
    
    # Check if alignment is enabled
    try:
        from deepthinker.alignment import get_alignment_config
        config = get_alignment_config(getattr(state, "constraints", None))
        enabled = config.enabled
    except Exception:
        enabled = False
    
    # Get trajectory from mission state
    trajectory_data = getattr(state, "alignment_trajectory", []) or []
    trajectory = []
    for p in trajectory_data:
        if isinstance(p, dict):
            trajectory.append(AlignmentPointInfo(
                t=p.get("t", 0),
                phase_name=p.get("phase_name", ""),
                alignment_score=p.get("a_t", 0.0),
                drift_delta=p.get("d_t", 0.0),
                cusum_neg=p.get("cusum_neg", 0.0),
                warning=p.get("warning", False),
                triggered=p.get("triggered", False),
                timestamp=p.get("timestamp_iso", ""),
            ))
    
    # Get north star
    north_star = getattr(state, "alignment_north_star", None)
    
    # Try to load from alignment log store for more complete data
    actions_taken = []
    try:
        from deepthinker.alignment.persist import get_log_store, get_alignment_config
        log_store = get_log_store(get_alignment_config())
        log_data = log_store.load(mission_id)
        
        if log_data:
            # Use log store data if available (more complete)
            if not north_star and "north_star" in log_data:
                north_star = log_data["north_star"]
            
            # Use trajectory from log store if mission state is empty
            if not trajectory and "trajectory" in log_data:
                for p in log_data["trajectory"]:
                    if isinstance(p, dict):
                        trajectory.append(AlignmentPointInfo(
                            t=p.get("t", 0),
                            phase_name=p.get("phase_name", ""),
                            alignment_score=p.get("a_t", 0.0),
                            drift_delta=p.get("d_t", 0.0),
                            cusum_neg=p.get("cusum_neg", 0.0),
                            warning=p.get("warning", False),
                            triggered=p.get("triggered", False),
                            timestamp=p.get("timestamp_iso", ""),
                        ))
            
            # Get actions from log
            for action_record in log_data.get("actions", []):
                actions_taken.append(AlignmentActionInfo(
                    action=action_record.get("action", ""),
                    t=action_record.get("t", 0),
                    phase_name=action_record.get("phase", ""),
                    timestamp=action_record.get("timestamp", ""),
                    consecutive_triggers=action_record.get("consecutive_triggers", 0),
                ))
    except Exception:
        pass  # Silent failure, use mission state data
    
    # Build summary
    summary = {}
    if trajectory:
        latest = trajectory[-1]
        summary["current_alignment"] = latest.alignment_score
        summary["total_points"] = len(trajectory)
        summary["trigger_count"] = sum(1 for p in trajectory if p.triggered)
        summary["warning_count"] = sum(1 for p in trajectory if p.warning)
        summary["actions_count"] = len(actions_taken)
        
        # Determine overall status
        if latest.triggered:
            summary["status"] = "correction"
        elif latest.warning:
            summary["status"] = "warning"
        else:
            summary["status"] = "healthy"
    else:
        summary["status"] = "no_data"
    
    return AlignmentResponse(
        mission_id=mission_id,
        enabled=enabled,
        north_star=north_star,
        trajectory=trajectory,
        actions_taken=actions_taken,
        summary=summary,
    )

