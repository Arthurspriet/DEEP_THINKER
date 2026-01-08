"""
Configuration API Routes.

Provides endpoints for available models, councils, and agent configurations.
"""

import os
import requests
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/config", tags=["config"])


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    size: Optional[str]
    family: Optional[str]
    parameter_size: Optional[str]
    quantization: Optional[str]


class CouncilConfig(BaseModel):
    """Configuration for a council."""
    name: str
    description: str
    default_models: List[str]
    consensus_type: str
    capabilities: List[str]


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    description: str
    default_model: str
    role: str
    capabilities: List[str]


# Council configurations
COUNCIL_CONFIGS = [
    CouncilConfig(
        name="planner",
        description="Strategic planning council for workflow design and objective analysis",
        default_models=["cogito:14b", "qwen2.5:14b"],
        consensus_type="weighted_blend",
        capabilities=["objective_analysis", "workflow_design", "resource_planning", "risk_assessment"]
    ),
    CouncilConfig(
        name="researcher",
        description="Research council for information gathering and web search",
        default_models=["gemma3:12b", "llama3.1:8b"],
        consensus_type="semantic_distance",
        capabilities=["web_search", "document_analysis", "fact_extraction", "source_validation"]
    ),
    CouncilConfig(
        name="coder",
        description="Coding council for code generation and implementation",
        default_models=["deepseek-r1:8b", "qwen2.5-coder:7b"],
        consensus_type="critique_exchange",
        capabilities=["code_generation", "refactoring", "debugging", "optimization"]
    ),
    CouncilConfig(
        name="evaluator",
        description="Evaluation council for quality assessment and scoring",
        default_models=["gemma3:27b", "llama3.1:70b"],
        consensus_type="majority_vote",
        capabilities=["quality_scoring", "code_review", "requirements_validation", "risk_evaluation"]
    ),
    CouncilConfig(
        name="simulator",
        description="Simulation council for testing and validation",
        default_models=["mistral:instruct", "llama3.2:3b"],
        consensus_type="weighted_blend",
        capabilities=["scenario_testing", "edge_case_analysis", "performance_testing", "integration_testing"]
    )
]

# Agent configurations
AGENT_CONFIGS = [
    AgentConfig(
        name="meta_planner",
        description="High-level planning agent that designs mission phases",
        default_model="cogito:14b",
        role="orchestrator",
        capabilities=["mission_design", "phase_planning", "resource_allocation"]
    ),
    AgentConfig(
        name="planner",
        description="Strategic planner for detailed workflow design",
        default_model="cogito:14b",
        role="planner",
        capabilities=["task_decomposition", "dependency_analysis", "timeline_planning"]
    ),
    AgentConfig(
        name="researcher",
        description="Web research and information gathering agent",
        default_model="gemma3:12b",
        role="researcher",
        capabilities=["web_search", "summarization", "fact_checking"]
    ),
    AgentConfig(
        name="coder",
        description="Code generation and implementation agent",
        default_model="deepseek-r1:8b",
        role="implementer",
        capabilities=["code_writing", "debugging", "testing"]
    ),
    AgentConfig(
        name="evaluator",
        description="Quality evaluation and scoring agent",
        default_model="gemma3:27b",
        role="evaluator",
        capabilities=["quality_assessment", "feedback_generation", "scoring"]
    ),
    AgentConfig(
        name="simulator",
        description="Simulation and scenario testing agent",
        default_model="mistral:instruct",
        role="tester",
        capabilities=["scenario_generation", "simulation", "validation"]
    ),
    AgentConfig(
        name="executor",
        description="Code execution and verification agent",
        default_model="llama3.2:3b",
        role="executor",
        capabilities=["code_execution", "output_verification", "error_handling"]
    )
]


def _fetch_ollama_models() -> List[ModelInfo]:
    """Fetch available models from Ollama."""
    base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        models = []
        
        for model in data.get("models", []):
            name = model.get("name", "")
            details = model.get("details", {})
            
            models.append(ModelInfo(
                name=name,
                size=_format_size(model.get("size")),
                family=details.get("family"),
                parameter_size=details.get("parameter_size"),
                quantization=details.get("quantization_level")
            ))
        
        return models
        
    except requests.exceptions.ConnectionError as e:
        import logging
        logging.getLogger(__name__).warning(f"Cannot connect to Ollama at {ollama_url}: {e}")
        return []
    except requests.exceptions.Timeout:
        import logging
        logging.getLogger(__name__).warning(f"Timeout connecting to Ollama at {ollama_url}")
        return []
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error fetching Ollama models: {e}")
        return []


def _format_size(size_bytes: Optional[int]) -> Optional[str]:
    """Format size in bytes to human readable."""
    if not size_bytes:
        return None
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} TB"


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available Ollama models."""
    return _fetch_ollama_models()


@router.get("/councils", response_model=List[CouncilConfig])
async def get_council_configs():
    """Get configuration for all councils."""
    return COUNCIL_CONFIGS


@router.get("/agents", response_model=List[AgentConfig])
async def get_agent_configs():
    """Get configuration for all agents."""
    return AGENT_CONFIGS

