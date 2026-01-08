"""
DeepThinker 2.0 - Council-Based Multi-LLM Orchestration Engine.

A modular framework for complex, multi-phase AI tasks using council-based
consensus, multiple LLMs, and non-deterministic workflow orchestration.

Key Components:
- MetaPlanner: Highest-level strategist for workflow configuration
- Councils: Multi-LLM collaborative decision-making groups
- Consensus Engines: Voting, blending, critique, and semantic distance
- Arbiter: Final decision-maker for conflict resolution
- Workflow Runner: Council-based workflow orchestration
- Model Supervisor: Dynamic model selection based on GPU resources
- GPU Resource Manager: GPU capacity tracking and queuing

Quick Start:
    from deepthinker import run_council_workflow
    
    result = run_council_workflow(
        objective="Create a machine learning classifier",
        verbose=True
    )
    print(result.final_code)
"""

__version__ = "2.0.0"

# Core workflow
from .workflow import (
    CouncilWorkflowRunner,
    run_council_workflow,
    CouncilStateManager,
    council_state_manager,
    IterationManager,
)

# Meta-planner and Arbiter
from .meta_planner import MetaPlanner
from .arbiter import Arbiter

# Councils
from .councils import (
    BaseCouncil,
    PlannerCouncil,
    ResearcherCouncil,
    CoderCouncil,
    EvaluatorCouncil,
    SimulationCouncil,
)

# Consensus engines
from .consensus import (
    MajorityVoteConsensus,
    WeightedBlendConsensus,
    CritiqueConsensus,
    SemanticDistanceConsensus,
)

# Model configuration
from .models import (
    OllamaLoader,
    AgentModelConfig,
    CouncilConfig,
    ModelPool,
    DEFAULT_COUNCIL_CONFIG,
)

# Legacy agent creation (for backwards compatibility)
from .agents import (
    create_planner_agent,
    create_coder_agent,
    create_evaluator_agent,
    create_simulator_agent,
    create_websearch_agent,
    create_executor_agent,
)

# Security
from .security import SecurityScanner, SandboxExecutor

# GPU Resource Management and Model Supervision
from .resources import GPUResourceManager, GPUResourceStats
from .supervisor import ModelSupervisor, SupervisorDecision

__all__ = [
    # Version
    "__version__",
    # Core workflow
    "CouncilWorkflowRunner",
    "run_council_workflow",
    "CouncilStateManager",
    "council_state_manager",
    "IterationManager",
    # Meta-planner and Arbiter
    "MetaPlanner",
    "Arbiter",
    # Councils
    "BaseCouncil",
    "PlannerCouncil",
    "ResearcherCouncil",
    "CoderCouncil",
    "EvaluatorCouncil",
    "SimulationCouncil",
    # Consensus
    "MajorityVoteConsensus",
    "WeightedBlendConsensus",
    "CritiqueConsensus",
    "SemanticDistanceConsensus",
    # Models
    "OllamaLoader",
    "AgentModelConfig",
    "CouncilConfig",
    "ModelPool",
    "DEFAULT_COUNCIL_CONFIG",
    # Legacy agents
    "create_planner_agent",
    "create_coder_agent",
    "create_evaluator_agent",
    "create_simulator_agent",
    "create_websearch_agent",
    "create_executor_agent",
    # Security
    "SecurityScanner",
    "SandboxExecutor",
    # GPU Resource Management and Model Supervision
    "GPUResourceManager",
    "GPUResourceStats",
    "ModelSupervisor",
    "SupervisorDecision",
]
