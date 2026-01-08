"""
Workflow API Routes (Legacy Mode).

Provides endpoints for running single workflows without the mission system.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deepthinker.execution import (
    run_deepthinker_workflow,
    IterationConfig,
    DataConfig,
    ResearchConfig,
    PlanningConfig
)
from deepthinker.models import AgentModelConfig

from ..sse import SSEManager, SSEEvent

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# Workflow-specific SSE manager
_workflow_sse = SSEManager()

# In-memory workflow state (for demo purposes)
_workflows: Dict[str, Dict[str, Any]] = {}


class WorkflowRequest(BaseModel):
    """Request body for running a workflow."""
    objective: str = Field(..., description="The workflow objective")
    model: str = Field(default="deepseek-r1:8b", description="Model to use")
    max_iterations: int = Field(default=3, ge=1, le=10)
    quality_threshold: float = Field(default=7.0, ge=0, le=10)
    enable_research: bool = Field(default=True)
    enable_planning: bool = Field(default=True)
    context: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response for workflow operations."""
    workflow_id: str
    status: str
    objective: str
    started_at: str


async def run_workflow_task(workflow_id: str, request: WorkflowRequest):
    """Background task to run a workflow."""
    try:
        _workflows[workflow_id]["status"] = "running"
        
        await _workflow_sse.publish(workflow_id, SSEEvent(
            event_type="workflow_started",
            data={"workflow_id": workflow_id, "objective": request.objective}
        ))
        
        # Configure workflow
        iteration_config = IterationConfig(
            max_iterations=request.max_iterations,
            quality_threshold=request.quality_threshold,
            enabled=True
        )
        
        research_config = ResearchConfig(enabled=request.enable_research)
        planning_config = PlanningConfig(enabled=request.enable_planning)
        agent_model_config = AgentModelConfig()
        
        # Run workflow
        result = run_deepthinker_workflow(
            objective=request.objective,
            context=request.context,
            model_name=request.model,
            iteration_config=iteration_config,
            research_config=research_config,
            planning_config=planning_config,
            agent_model_config=agent_model_config,
            verbose=False
        )
        
        _workflows[workflow_id].update({
            "status": "completed",
            "result": {
                "final_code": result.get("final_code", ""),
                "quality_score": result.get("quality_score", 0),
                "iterations_completed": result.get("iterations_completed", 0),
                "passed": result.get("final_evaluation", {}).passed if hasattr(result.get("final_evaluation", {}), "passed") else False
            }
        })
        
        await _workflow_sse.publish(workflow_id, SSEEvent(
            event_type="workflow_completed",
            data={
                "workflow_id": workflow_id,
                "status": "completed",
                "quality_score": result.get("quality_score", 0)
            }
        ))
        
    except Exception as e:
        _workflows[workflow_id].update({
            "status": "failed",
            "error": str(e)
        })
        
        await _workflow_sse.publish(workflow_id, SSEEvent(
            event_type="workflow_failed",
            data={"workflow_id": workflow_id, "error": str(e)}
        ))


@router.post("/run", response_model=WorkflowResponse)
async def run_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks
):
    """Start a new workflow execution."""
    from datetime import datetime
    
    workflow_id = str(uuid4())
    
    _workflows[workflow_id] = {
        "workflow_id": workflow_id,
        "status": "pending",
        "objective": request.objective,
        "started_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(run_workflow_task, workflow_id, request)
    
    return WorkflowResponse(
        workflow_id=workflow_id,
        status="pending",
        objective=request.objective,
        started_at=_workflows[workflow_id]["started_at"]
    )


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow status and results."""
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return _workflows[workflow_id]


@router.get("/{workflow_id}/events")
async def workflow_events(workflow_id: str):
    """SSE endpoint for workflow events."""
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return StreamingResponse(
        _workflow_sse.subscribe(workflow_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

