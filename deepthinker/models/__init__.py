"""
Model loading and management for DeepThinker 2.0.

Provides:
- OllamaLoader for model creation
- AgentModelConfig for agent-specific models
- Council model configuration for multi-LLM councils
- ModelPool for concurrent multi-model execution
- LiteLLM monitoring integration
- Centralized model caller with resource management
- Model capabilities registry for dynamic council generation

DeepThinker 2.0 Additions:
- ModelRegistry: Central registry of models with tier categorization
- ModelSelector: Phase-aware, intelligent model selection
"""

from .ollama_loader import OllamaLoader, AgentModelConfig
from .litellm_monitor import (
    LiteLLMMonitor,
    enable_monitoring,
    get_monitoring_stats,
    print_monitoring_summary
)
from .council_model_config import (
    CouncilConfig,
    CouncilModelPool,
    DEFAULT_COUNCIL_CONFIG,
    PLANNER_MODELS,
    CODER_MODELS,
    EVALUATOR_MODELS,
    SIMULATION_MODELS,
    RESEARCHER_MODELS,
    ARBITER_MODEL,
    META_PLANNER_MODEL,
    EMBEDDING_MODEL,
)
from .model_pool import ModelPool, ModelOutput
from .model_caller import (
    call_model,
    call_model_async,
    call_embeddings,
    call_embeddings_async,
    cleanup_resources,
    count_open_sockets,
    ModelInvocationError,
)

# Model registry and selector (new in 2.0)
from .model_registry import (
    ModelRegistry,
    ModelInfo,
    ModelTier,
    MODEL_REGISTRY,
    MODEL_TIERS,
    PHASE_MODEL_PREFERENCES,
    get_model_registry,
)
from .model_selector import (
    ModelSelector,
    ModelSelection,
    get_model_selector,
)

# Model capabilities registry for dynamic council generation
try:
    from .model_capabilities import (
        ModelCapability,
        ModelCapabilitiesRegistry,
        get_model_capabilities,
        select_models_for_role,
        get_registry,
    )
    CAPABILITIES_AVAILABLE = True
except ImportError:
    CAPABILITIES_AVAILABLE = False
    ModelCapability = None
    ModelCapabilitiesRegistry = None
    get_model_capabilities = None
    select_models_for_role = None
    get_registry = None

# Step Tier Policy (Epistemic Hardening Phase 7)
try:
    from .step_tier_policy import (
        StepType,
        TierDecision,
        STEP_TIER_REQUIREMENTS,
        TRUTH_CRITICAL_STEPS,
        evaluate_tier_policy,
        infer_step_type,
        is_truth_critical,
        get_minimum_tier,
        get_allowed_tiers,
        get_preferred_tier,
        can_use_model,
        get_tier_policy_summary,
    )
    STEP_TIER_POLICY_AVAILABLE = True
except ImportError:
    STEP_TIER_POLICY_AVAILABLE = False
    StepType = None
    TierDecision = None
    STEP_TIER_REQUIREMENTS = None
    TRUTH_CRITICAL_STEPS = None
    evaluate_tier_policy = None
    infer_step_type = None
    is_truth_critical = None
    get_minimum_tier = None
    get_allowed_tiers = None
    get_preferred_tier = None
    can_use_model = None
    get_tier_policy_summary = None

__all__ = [
    # Legacy exports
    "OllamaLoader",
    "AgentModelConfig",
    "LiteLLMMonitor",
    "enable_monitoring",
    "get_monitoring_stats",
    "print_monitoring_summary",
    # DeepThinker 2.0 exports
    "CouncilConfig",
    "CouncilModelPool",
    "DEFAULT_COUNCIL_CONFIG",
    "PLANNER_MODELS",
    "CODER_MODELS",
    "EVALUATOR_MODELS",
    "SIMULATION_MODELS",
    "RESEARCHER_MODELS",
    "ARBITER_MODEL",
    "META_PLANNER_MODEL",
    "EMBEDDING_MODEL",
    "ModelPool",
    "ModelOutput",
    # Model caller exports
    "call_model",
    "call_model_async",
    "call_embeddings",
    "call_embeddings_async",
    "cleanup_resources",
    "count_open_sockets",
    "ModelInvocationError",
    # Model registry and selector (new in 2.0)
    "ModelRegistry",
    "ModelInfo",
    "ModelTier",
    "MODEL_REGISTRY",
    "MODEL_TIERS",
    "PHASE_MODEL_PREFERENCES",
    "get_model_registry",
    "ModelSelector",
    "ModelSelection",
    "get_model_selector",
    # Model capabilities exports
    "ModelCapability",
    "ModelCapabilitiesRegistry",
    "get_model_capabilities",
    "select_models_for_role",
    "get_registry",
    "CAPABILITIES_AVAILABLE",
    # Step Tier Policy exports (Epistemic Hardening Phase 7)
    "StepType",
    "TierDecision",
    "STEP_TIER_REQUIREMENTS",
    "TRUTH_CRITICAL_STEPS",
    "evaluate_tier_policy",
    "infer_step_type",
    "is_truth_critical",
    "get_minimum_tier",
    "get_allowed_tiers",
    "get_preferred_tier",
    "can_use_model",
    "get_tier_policy_summary",
    "STEP_TIER_POLICY_AVAILABLE",
]
