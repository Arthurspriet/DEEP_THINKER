"""
Cognitive Spine Core Module for DeepThinker 2.0.

The Cognitive Spine is the central unifying layer that enforces:
- Schema coherence across all councils and phases
- Predictable output structures via contracts
- Resource allocation discipline (tokens, depth, iterations)
- Phase boundary validation
- Consensus engine availability
- Memory compression between phases

DeepThinker 2.0 Additions:
- PhaseValidator: Enforces hard phase contracts
- SafetyCoreRegistry: Centralized safety module management
"""

from .cognitive_spine import (
    CognitiveSpine,
    ResourceBudget,
    ValidationResult,
    MemorySlot,
    SchemaVersion,
)

from .phase_validator import (
    PhaseValidator,
    ValidationResult as PhaseValidationResult,
    get_phase_validator,
)

from .safety_registry import (
    SafetyCoreRegistry,
    SafetyModuleUnavailableError,
    ModuleCategory,
    ImportPriority,
    ModuleSpec,
    safety,
    initialize_safety,
    is_safety_available,
    get_safety_module,
    require_safety_module,
)

from .config_manager import (
    ConfigManager,
    DeepThinkerConfig,
    ModelConfig,
    IterationConfig as UnifiedIterationConfig,
    ResearchConfig as UnifiedResearchConfig,
    PlanningConfig as UnifiedPlanningConfig,
    ExecutionConfig,
    GovernanceConfig as UnifiedGovernanceConfig,
    MemoryConfig,
    SafetyConfig,
    ConfigSection,
    config_manager,
    get_config,
    initialize_config,
    get_model,
    get_temperature,
    set_config,
    update_config,
)

from .state_layer import (
    StateLayer,
    StateBackend,
    TransientBackend,
    PersistentBackend,
    SessionState,
    SessionMode,
    SessionStatus,
    IterationRecord,
    PhaseRecord,
    PhaseStatus,
    state_layer,
    initialize_state_layer,
    get_state_layer,
)

from .council_bridge import (
    AgentBridge,
    SimpleAgentBridge,
    CouncilAgentBridge,
    BridgeResult,
    BridgeFactory,
    get_bridge_factory,
    create_bridged_agent,
)

__all__ = [
    "CognitiveSpine",
    "ResourceBudget",
    "ValidationResult",
    "MemorySlot",
    "SchemaVersion",
    # Phase validation (new in 2.0)
    "PhaseValidator",
    "PhaseValidationResult",
    "get_phase_validator",
    # Safety Core Registry (new)
    "SafetyCoreRegistry",
    "SafetyModuleUnavailableError",
    "ModuleCategory",
    "ImportPriority",
    "ModuleSpec",
    "safety",
    "initialize_safety",
    "is_safety_available",
    "get_safety_module",
    "require_safety_module",
    # Configuration Manager (new)
    "ConfigManager",
    "DeepThinkerConfig",
    "ModelConfig",
    "UnifiedIterationConfig",
    "UnifiedResearchConfig",
    "UnifiedPlanningConfig",
    "ExecutionConfig",
    "UnifiedGovernanceConfig",
    "MemoryConfig",
    "SafetyConfig",
    "ConfigSection",
    "config_manager",
    "get_config",
    "initialize_config",
    "get_model",
    "get_temperature",
    "set_config",
    "update_config",
    # State Layer (new)
    "StateLayer",
    "StateBackend",
    "TransientBackend",
    "PersistentBackend",
    "SessionState",
    "SessionMode",
    "SessionStatus",
    "IterationRecord",
    "PhaseRecord",
    "PhaseStatus",
    "state_layer",
    "initialize_state_layer",
    "get_state_layer",
    # Council-Agent Bridge (new)
    "AgentBridge",
    "SimpleAgentBridge",
    "CouncilAgentBridge",
    "BridgeResult",
    "BridgeFactory",
    "get_bridge_factory",
    "create_bridged_agent",
]

