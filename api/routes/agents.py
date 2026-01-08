"""
Agent Monitoring API Routes.

Provides endpoints for monitoring agent status, traces, and metrics.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from deepthinker.missions import MissionStore

router = APIRouter(prefix="/api/agents", tags=["agents"])

_store = MissionStore()


class AgentStatus(BaseModel):
    """Status of a single agent."""
    name: str
    status: str  # idle, running, error
    current_mission: Optional[str]
    current_phase: Optional[str]
    last_active: Optional[str]
    model: str


class AgentTrace(BaseModel):
    """A single trace entry from an agent."""
    timestamp: str
    event_type: str
    mission_id: Optional[str]
    phase: Optional[str]
    data: Dict[str, Any]


class AgentMetrics(BaseModel):
    """Aggregated metrics for an agent."""
    name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_s: float
    total_tokens_used: int


# Agent configurations
AGENT_CONFIGS = {
    "planner": {
        "name": "Planner Council",
        "default_model": "cogito:14b",
        "description": "Strategic planning and workflow design"
    },
    "researcher": {
        "name": "Researcher Council",
        "default_model": "gemma3:12b",
        "description": "Web research and information gathering"
    },
    "coder": {
        "name": "Coder Council", 
        "default_model": "deepseek-r1:8b",
        "description": "Code generation and implementation"
    },
    "evaluator": {
        "name": "Evaluator Council",
        "default_model": "gemma3:27b",
        "description": "Quality evaluation and scoring"
    },
    "simulator": {
        "name": "Simulator Council",
        "default_model": "mistral:instruct",
        "description": "Simulation and testing"
    },
    "executor": {
        "name": "Executor Agent",
        "default_model": "llama3.2:3b",
        "description": "Code execution and verification"
    }
}


def _get_agent_status_from_missions() -> Dict[str, Dict[str, Any]]:
    """Get agent status by analyzing running missions."""
    agent_status = {}
    
    for name in AGENT_CONFIGS:
        agent_status[name] = {
            "name": name,
            "status": "idle",
            "current_mission": None,
            "current_phase": None,
            "last_active": None,
            "model": AGENT_CONFIGS[name]["default_model"]
        }
    
    # Check running missions for active agents
    try:
        missions = _store.list_missions_with_status()
        for m in missions:
            if m["status"] == "running":
                state = _store.load(m["mission_id"])
                phase = state.current_phase()
                if phase:
                    # Determine which agent is likely running based on phase name
                    phase_lower = phase.name.lower()
                    
                    if any(kw in phase_lower for kw in ["research", "recon", "gather"]):
                        agent_name = "researcher"
                    elif any(kw in phase_lower for kw in ["plan", "design", "architect"]):
                        agent_name = "planner"
                    elif any(kw in phase_lower for kw in ["code", "implement", "build"]):
                        agent_name = "coder"
                    elif any(kw in phase_lower for kw in ["eval", "review", "quality"]):
                        agent_name = "evaluator"
                    elif any(kw in phase_lower for kw in ["test", "simul", "valid"]):
                        agent_name = "simulator"
                    else:
                        agent_name = "planner"
                    
                    agent_status[agent_name].update({
                        "status": "running",
                        "current_mission": state.mission_id,
                        "current_phase": phase.name,
                        "last_active": datetime.utcnow().isoformat()
                    })
    except FileNotFoundError:
        # Mission store not initialized yet
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error fetching agent status from missions: {e}")
    
    return agent_status


@router.get("/status", response_model=List[AgentStatus])
async def get_agents_status():
    """Get status of all agents."""
    status = _get_agent_status_from_missions()
    
    return [
        AgentStatus(**s) for s in status.values()
    ]


@router.get("/{agent_name}/traces", response_model=List[AgentTrace])
async def get_agent_traces(
    agent_name: str,
    limit: int = Query(default=50, ge=1, le=500),
    mission_id: Optional[str] = Query(default=None)
):
    """Get execution traces for an agent."""
    if agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    traces = []
    
    # Collect traces from mission event logs
    try:
        missions = _store.list_missions()
        
        for mid in missions:
            if mission_id and mid != mission_id:
                continue
            
            try:
                state = _store.load(mid)
                
                # Filter events related to this agent
                for event in state.event_logs:
                    event_type = event.get("event", "")
                    
                    # Check if event is related to this agent
                    is_related = False
                    if event_type == "council_execution":
                        council = event.get("council_name", "").lower()
                        if agent_name in council:
                            is_related = True
                    elif event_type == "step_execution":
                        step_type = event.get("step_type", "").lower()
                        if agent_name in step_type:
                            is_related = True
                    elif event_type == "supervisor_decision":
                        # Supervisor decisions are relevant to all agents
                        is_related = True
                    
                    if is_related:
                        traces.append(AgentTrace(
                            timestamp=event.get("timestamp", ""),
                            event_type=event_type,
                            mission_id=mid,
                            phase=event.get("phase"),
                            data={k: v for k, v in event.items() if k not in ["timestamp", "event", "phase"]}
                        ))
            except (FileNotFoundError, KeyError) as e:
                # Skip corrupted or missing mission files
                continue
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Error loading mission {mid} for traces: {e}")
                continue
        
        # Sort by timestamp descending and limit
        traces.sort(key=lambda x: x.timestamp, reverse=True)
        traces = traces[:limit]
        
    except FileNotFoundError:
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error fetching agent traces: {e}")
    
    return traces


@router.get("/metrics", response_model=List[AgentMetrics])
async def get_agent_metrics():
    """Get aggregated metrics for all agents."""
    metrics = {
        name: {
            "name": name,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration_s": 0.0,
            "total_tokens_used": 0
        }
        for name in AGENT_CONFIGS
    }
    
    # Aggregate from mission event logs
    try:
        missions = _store.list_missions()
        
        for mid in missions:
            try:
                state = _store.load(mid)
                
                for event in state.event_logs:
                    if event.get("event") == "council_execution":
                        council = event.get("council_name", "").lower()
                        
                        for agent_name in AGENT_CONFIGS:
                            if agent_name in council:
                                metrics[agent_name]["total_executions"] += 1
                                if event.get("success"):
                                    metrics[agent_name]["successful_executions"] += 1
                                else:
                                    metrics[agent_name]["failed_executions"] += 1
                                
                                duration = event.get("duration_s")
                                if duration:
                                    metrics[agent_name]["total_duration_s"] += duration
                                break
                                
            except (FileNotFoundError, KeyError):
                # Skip corrupted or missing mission files
                continue
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Error loading mission {mid} for metrics: {e}")
                continue
                
    except FileNotFoundError:
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error fetching agent metrics: {e}")
    
    # Calculate averages
    result = []
    for name, m in metrics.items():
        avg_duration = 0.0
        if m["total_executions"] > 0:
            avg_duration = m["total_duration_s"] / m["total_executions"]
        
        result.append(AgentMetrics(
            name=name,
            total_executions=m["total_executions"],
            successful_executions=m["successful_executions"],
            failed_executions=m["failed_executions"],
            average_duration_s=round(avg_duration, 2),
            total_tokens_used=m["total_tokens_used"]
        ))
    
    return result

