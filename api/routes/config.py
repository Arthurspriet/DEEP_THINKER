"""
Configuration API Routes.

Provides endpoints for available models, councils, and agent configurations.
Includes dynamic model discovery and performance tracking.
"""

import logging
import os
import requests
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/config", tags=["config"])
logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    """Basic information about an available model (legacy format)."""
    name: str
    size: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None


class EnhancedModelInfo(BaseModel):
    """
    Enhanced model information with capabilities and performance data.
    
    Includes:
    - Static metadata (tier, capabilities, VRAM)
    - Dynamic discovery status
    - Historical performance grades from missions
    """
    name: str
    # Basic info
    size: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    # Registry metadata
    tier: str = "medium"  # reasoning, large, medium, small, embedding
    vram_mb: int = 5000
    capabilities: Dict[str, float] = {}
    default_temperature: float = 0.5
    max_tokens: int = 4096
    is_available: bool = True
    is_known: bool = False  # True if in static registry, False if auto-discovered
    # Performance data from missions
    performance_grade: str = "unknown"  # excellent, good, fair, poor, unknown
    total_missions: int = 0
    avg_quality_score: Optional[float] = None
    success_rate: Optional[float] = None


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


def _format_size(size_bytes: Optional[int]) -> Optional[str]:
    """Format size in bytes to human readable."""
    if not size_bytes:
        return None
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} TB"


def _fetch_ollama_raw() -> Dict[str, Dict[str, Any]]:
    """Fetch raw model data from Ollama for size information."""
    base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        result = {}
        for model in data.get("models", []):
            name = model.get("name", "")
            if name:
                result[name] = {
                    "size": _format_size(model.get("size")),
                }
        return result
        
    except Exception as e:
        logger.warning(f"Error fetching Ollama raw data: {e}")
        return {}


def _get_enhanced_models() -> List[EnhancedModelInfo]:
    """
    Get enhanced model list with registry metadata and performance grades.
    
    Returns:
        List of EnhancedModelInfo with full metadata
    """
    from deepthinker.models.model_registry import get_model_registry
    from deepthinker.models.model_stats_store import get_model_stats_store
    
    # Get registry with auto-discovery
    ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    registry = get_model_registry(auto_discover=True)
    registry.ollama_base_url = ollama_url
    registry.discover_ollama_models()
    
    # Get performance stats
    stats_store = get_model_stats_store()
    
    # Get raw Ollama data for size info
    ollama_raw = _fetch_ollama_raw()
    
    # Build enhanced model list
    models = []
    all_models = registry.get_all_models()
    
    for name, info in all_models.items():
        # Get performance stats
        stats = stats_store.get_stats(name)
        
        # Calculate success rate
        success_rate = None
        if stats and (stats.success_count + stats.failure_count) > 0:
            success_rate = stats.success_count / (stats.success_count + stats.failure_count)
        
        # Get tier value
        tier_value = info.tier.value if hasattr(info.tier, 'value') else str(info.tier)
        
        models.append(EnhancedModelInfo(
            name=name,
            size=ollama_raw.get(name, {}).get("size"),
            family=info.family,
            parameter_size=info.parameter_size,
            quantization=info.quantization,
            tier=tier_value,
            vram_mb=info.vram_mb,
            capabilities=info.capabilities,
            default_temperature=info.default_temperature,
            max_tokens=info.max_tokens,
            is_available=info.is_available,
            is_known=info.is_known,
            performance_grade=stats.compute_grade() if stats else "unknown",
            total_missions=stats.total_missions if stats else 0,
            avg_quality_score=stats.avg_quality_score if stats else None,
            success_rate=success_rate,
        ))
    
    # Sort: available first, then by tier (reasoning > large > medium > small)
    tier_order = {"reasoning": 0, "large": 1, "medium": 2, "small": 3, "embedding": 4}
    models.sort(key=lambda m: (
        0 if m.is_available else 1,
        tier_order.get(m.tier, 5),
        m.name
    ))
    
    return models


@router.get("/models", response_model=List[EnhancedModelInfo])
async def get_available_models():
    """
    Get list of available models with enhanced metadata.
    
    Returns models from Ollama with:
    - Registry metadata (tier, capabilities, VRAM)
    - Performance grades from mission history
    - Discovery status (known vs auto-discovered)
    """
    try:
        return _get_enhanced_models()
    except Exception as e:
        logger.error(f"Error getting enhanced models: {e}")
        # Fallback to basic model list
        return []


@router.get("/models/simple", response_model=List[ModelInfo])
async def get_available_models_simple():
    """
    Get simple list of available Ollama models (legacy endpoint).
    
    Returns basic model info without performance tracking.
    """
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
        
    except Exception as e:
        logger.warning(f"Error fetching Ollama models: {e}")
        return []


@router.get("/models/stats")
async def get_model_stats():
    """
    Get detailed statistics for all models.
    
    Returns performance data aggregated from mission history.
    """
    from deepthinker.models.model_stats_store import get_model_stats_store
    
    stats_store = get_model_stats_store()
    return stats_store.get_summary()


@router.post("/models/refresh")
async def refresh_models():
    """
    Force refresh of model discovery.
    
    Re-scans Ollama for available models and updates registry.
    """
    from deepthinker.models.model_registry import get_model_registry
    from deepthinker.models.model_stats_store import get_model_stats_store
    
    ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    registry = get_model_registry()
    registry.ollama_base_url = ollama_url
    
    newly_discovered = registry.discover_ollama_models()
    
    # Also invalidate stats cache
    stats_store = get_model_stats_store()
    stats_store.invalidate_cache()
    
    return {
        "status": "ok",
        "newly_discovered": newly_discovered,
        "total_models": len(registry.get_all_models())
    }


@router.get("/councils", response_model=List[CouncilConfig])
async def get_council_configs():
    """Get configuration for all councils."""
    return COUNCIL_CONFIGS


@router.get("/agents", response_model=List[AgentConfig])
async def get_agent_configs():
    """Get configuration for all agents."""
    return AGENT_CONFIGS

