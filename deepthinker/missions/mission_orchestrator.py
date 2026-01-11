"""
Mission Orchestrator for DeepThinker 2.0.

Coordinates long-running, time-bounded missions using the council architecture.
Manages phase execution, constraint enforcement, checkpointing, and resource optimization.

New in 2.0:
- GPU resource management integration
- Model supervisor for dynamic model selection
- Per-phase resource optimization
- Structured event logging
- Meta-cognition engine integration

Step Engine Integration:
- Phases now contain discrete steps executed by single models
- Councils are used for strategy and reflection
- StepExecutor handles individual step execution

CognitiveSpine Integration:
- Centralized schema validation for all contexts
- Resource budget enforcement
- Memory compression between phases
- Fallback policies for error recovery

SafetyCore Integration:
- Centralized safety module registry
- Explicit availability tracking
- Graceful degradation with logging
"""

import asyncio
import threading
import uuid
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

from .mission_types import MissionState, MissionPhase, MissionConstraints, ConvergenceState, MissionFailureReason
from .mission_store import MissionStore
from .mission_time_manager import MissionTimeManager, create_time_manager_from_constraints

import logging
_import_logger = logging.getLogger(__name__)
_orchestrator_logger = logging.getLogger(f"{__name__}.orchestrator")

# =============================================================================
# SafetyCore Integration
# =============================================================================
# The SafetyCore registry provides centralized safety module management.
# It replaces scattered try/except ImportError blocks with a unified system.

try:
    from ..core.safety_registry import (
        safety,
        SafetyModuleUnavailableError,
        ModuleCategory,
    )
    SAFETY_CORE_AVAILABLE = True
    # Initialize safety registry to log availability
    _safety_status = safety.initialize(strict_mode=False)
    _orchestrator_logger.info(
        f"SafetyCore initialized: {sum(_safety_status.values())}/{len(_safety_status)} modules available"
    )
except ImportError:
    SAFETY_CORE_AVAILABLE = False
    safety = None
    _orchestrator_logger.warning(
        "SafetyCore not available - falling back to legacy import pattern"
    )


def _get_safety_module(name: str, export: str = None):
    """
    Helper to get a safety module using SafetyCore if available.
    Falls back to None if SafetyCore is not available.
    """
    if SAFETY_CORE_AVAILABLE and safety is not None:
        return safety.get(name, export, warn_if_missing=False)
    return None


# Track unavailable components for summary logging (legacy pattern)
_UNAVAILABLE_COMPONENTS: list = []

def _log_import_failure(component: str, error: Exception) -> None:
    """Log an import failure and track for summary."""
    _UNAVAILABLE_COMPONENTS.append(component)
    _import_logger.debug(f"Optional component '{component}' unavailable: {error}")

def get_unavailable_components() -> list:
    """Return list of components that failed to import."""
    return list(_UNAVAILABLE_COMPONENTS)

def log_component_summary() -> None:
    """Log a summary of unavailable components (call once at orchestrator init)."""
    # If SafetyCore is available, use its reporting
    if SAFETY_CORE_AVAILABLE and safety is not None:
        report = safety.get_status_report()
        unavailable = report.get("total_unavailable", 0)
        if unavailable > 0:
            _import_logger.info(
                f"SafetyCore: {unavailable} modules unavailable (see startup log for details)"
            )
        return
    
    # Legacy pattern
    if _UNAVAILABLE_COMPONENTS:
        _import_logger.warning(
            f"MissionOrchestrator: {len(_UNAVAILABLE_COMPONENTS)} optional components unavailable: "
            f"{', '.join(_UNAVAILABLE_COMPONENTS[:10])}"
            + (f" and {len(_UNAVAILABLE_COMPONENTS) - 10} more..." if len(_UNAVAILABLE_COMPONENTS) > 10 else "")
        )

# CognitiveSpine integration
try:
    from ..core.cognitive_spine import CognitiveSpine
    COGNITIVE_SPINE_AVAILABLE = True
except ImportError as e:
    COGNITIVE_SPINE_AVAILABLE = False
    CognitiveSpine = None
    _log_import_failure("CognitiveSpine", e)

# Verbose logging integration
try:
    from ..cli import verbose_logger
    VERBOSE_LOGGER_AVAILABLE = True
except ImportError as e:
    VERBOSE_LOGGER_AVAILABLE = False
    verbose_logger = None
    _log_import_failure("verbose_logger", e)

# SSE event publishing integration
try:
    from api.sse import sse_manager
    SSE_AVAILABLE = True
except ImportError as e:
    SSE_AVAILABLE = False
    sse_manager = None
    _log_import_failure("SSE", e)


def _publish_sse_event(coro):
    """
    Helper to publish SSE events from sync code.
    Safely schedules the coroutine if an event loop is running.
    """
    if not SSE_AVAILABLE or sse_manager is None:
        coro.close()  # Clean up the coroutine
        return
    try:
        import asyncio
        asyncio.get_running_loop()
        asyncio.create_task(coro)
    except RuntimeError:
        # No event loop running - close coroutine to avoid warning
        coro.close()

from ..councils.planner_council.planner_council import PlannerCouncil, PlannerContext
from ..councils.researcher_council.researcher_council import ResearcherCouncil, ResearchContext
from ..councils.coder_council.coder_council import CoderCouncil, CoderContext
from ..councils.evaluator_council.evaluator_council import EvaluatorCouncil, EvaluatorContext
from ..councils.simulation_council.simulation_council import SimulationCouncil, SimulationContext
from ..arbiter.arbiter import Arbiter, CouncilOutput
from ..outputs.output_manager import OutputManager
from ..outputs.output_types import OutputArtifact

# Orchestration learning layer
try:
    from ..orchestration import OrchestrationStore, PhaseOutcome
    ORCHESTRATION_AVAILABLE = True
except ImportError as e:
    ORCHESTRATION_AVAILABLE = False
    OrchestrationStore = None
    PhaseOutcome = None
    _log_import_failure("OrchestrationStore", e)

# Phase time allocator for proactive time budgeting
try:
    from ..orchestration.phase_time_allocator import (
        PhaseTimeAllocator,
        TimeAllocation,
        create_allocator_from_store,
    )
    PHASE_TIME_ALLOCATOR_AVAILABLE = True
except ImportError as e:
    PHASE_TIME_ALLOCATOR_AVAILABLE = False
    PhaseTimeAllocator = None
    TimeAllocation = None
    create_allocator_from_store = None
    _log_import_failure("PhaseTimeAllocator", e)

# Multi-view councils
try:
    from ..councils.multi_view import (
        OptimistCouncil, SkepticCouncil,
        extract_disagreements, MultiViewDisagreement
    )
    MULTIVIEW_COUNCILS_AVAILABLE = True
except ImportError as e:
    MULTIVIEW_COUNCILS_AVAILABLE = False
    OptimistCouncil = None
    SkepticCouncil = None
    extract_disagreements = None
    MultiViewDisagreement = None
    _log_import_failure("MultiViewCouncils", e)

# Dynamic Council Generator
try:
    from ..councils.dynamic_council_factory import DynamicCouncilFactory, CouncilDefinition
    DYNAMIC_COUNCIL_AVAILABLE = True
except ImportError as e:
    DYNAMIC_COUNCIL_AVAILABLE = False
    DynamicCouncilFactory = None
    CouncilDefinition = None
    _log_import_failure("DynamicCouncilFactory", e)

# Iteration context management
try:
    from ..workflow.iteration_context import (
        IterationContextManager,
        IterationState,
        ContextDelta
    )
    ITERATION_CONTEXT_AVAILABLE = True
except ImportError as e:
    ITERATION_CONTEXT_AVAILABLE = False
    IterationContextManager = None
    IterationState = None
    ContextDelta = None
    _log_import_failure("IterationContextManager", e)

# Enhanced research evaluation
try:
    from ..councils.evaluator_council.evaluator_council import (
        ResearchEvaluation, ResearchEvaluationContext
    )
    RESEARCH_EVALUATION_AVAILABLE = True
except ImportError as e:
    RESEARCH_EVALUATION_AVAILABLE = False
    ResearchEvaluation = None
    ResearchEvaluationContext = None
    _log_import_failure("ResearchEvaluation", e)

# Synthesis context
try:
    from ..councils.planner_council.planner_council import (
        SynthesisContext, SynthesisResult
    )
    SYNTHESIS_CONTEXT_AVAILABLE = True
except ImportError as e:
    SYNTHESIS_CONTEXT_AVAILABLE = False
    SynthesisContext = None
    SynthesisResult = None
    _log_import_failure("SynthesisContext", e)

# Convergence tracking
try:
    from ..utils.convergence import ConvergenceTracker
    CONVERGENCE_AVAILABLE = True
except ImportError as e:
    CONVERGENCE_AVAILABLE = False
    _log_import_failure("ConvergenceTracker", e)

# Step Engine imports
from ..steps.step_types import StepDefinition, StepExecutionContext, StepResult
from ..steps.step_executor import StepExecutor

# Optional imports for supervisor and GPU manager
try:
    from ..resources.gpu_manager import GPUResourceManager
    GPU_MANAGER_AVAILABLE = True
except ImportError as e:
    GPU_MANAGER_AVAILABLE = False
    _log_import_failure("GPUResourceManager", e)

try:
    from ..supervisor.model_supervisor import (
        ModelSupervisor, 
        SupervisorDecision,
        GovernanceEscalationSignal,
        PHASE_IMPORTANCE,
    )
    SUPERVISOR_AVAILABLE = True
except ImportError as e:
    SUPERVISOR_AVAILABLE = False
    GovernanceEscalationSignal = None
    PHASE_IMPORTANCE = {
        "synthesis": 1.0,
        "implementation": 0.9,
        "design": 0.8,
        "testing": 0.7,
        "research": 0.6,
    }
    _log_import_failure("ModelSupervisor", e)

# Meta-cognition engine import
try:
    from ..meta.meta_controller import MetaController
    META_CONTROLLER_AVAILABLE = True
except ImportError as e:
    META_CONTROLLER_AVAILABLE = False
    _log_import_failure("MetaController", e)

# ReasoningSupervisor import
try:
    from ..meta.supervisor import (
        ReasoningSupervisor,
        PhaseMetrics,
        MissionMetrics,
        DepthContract,
        DeepeningPlan,
        LoopDetection,
    )
    REASONING_SUPERVISOR_AVAILABLE = True
except ImportError as e:
    REASONING_SUPERVISOR_AVAILABLE = False
    ReasoningSupervisor = None
    _log_import_failure("ReasoningSupervisor", e)

# Depth Evaluator import for depth control
try:
    from ..meta.depth_evaluator import (
        compute_depth_score,
        get_depth_target,
        extract_depth_indicators,
        select_enrichment_type,
        get_enrichment_prompt,
        compute_depth_gap,
    )
    DEPTH_EVALUATOR_AVAILABLE = True
except ImportError as e:
    DEPTH_EVALUATOR_AVAILABLE = False
    compute_depth_score = None
    get_depth_target = None
    extract_depth_indicators = None
    select_enrichment_type = None
    get_enrichment_prompt = None
    compute_depth_gap = None
    _log_import_failure("DepthEvaluator", e)

# SSE manager for real-time updates (note: may be imported earlier, skip logging if already attempted)
try:
    from api.sse import sse_manager
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    # Already logged above if failed

# Cost/Time Predictor for shadow mode prediction logging
try:
    from ..resources import COST_PREDICTOR_AVAILABLE
    if COST_PREDICTOR_AVAILABLE:
        from ..resources.cost_time_predictor import (
            CostTimePredictor,
            PhaseContext as PredictorPhaseContext,
            ExecutionPlan as PredictorExecutionPlan,
            SystemState as PredictorSystemState,
            CostTimePrediction,
            EvaluationLogger,
            PREDICTOR_CONFIG,
        )
except ImportError as e:
    COST_PREDICTOR_AVAILABLE = False
    _log_import_failure("CostTimePredictor", e)

# Phase Risk Predictor for shadow mode risk prediction logging
try:
    from ..resources import RISK_PREDICTOR_AVAILABLE
    if RISK_PREDICTOR_AVAILABLE:
        from ..resources.phase_risk_predictor import (
            PhaseRiskPredictor,
            PhaseRiskContext,
            PhaseRiskExecutionPlan,
            PhaseRiskSystemState,
            PhaseRiskPrediction,
            PhaseRiskEvaluationLogger,
            PREDICTOR_CONFIG as RISK_PREDICTOR_CONFIG,
        )
except ImportError as e:
    RISK_PREDICTOR_AVAILABLE = False
    _log_import_failure("PhaseRiskPredictor", e)

# Web Search Predictor for shadow mode search necessity prediction logging
try:
    from ..resources import WEB_SEARCH_PREDICTOR_AVAILABLE
    if WEB_SEARCH_PREDICTOR_AVAILABLE:
        from ..resources.web_search_predictor import (
            WebSearchPredictor,
            WebSearchContext,
            WebSearchExecutionPlan,
            WebSearchSystemState,
            WebSearchPrediction,
            WebSearchEvaluationLogger,
            PREDICTOR_CONFIG as WEB_SEARCH_PREDICTOR_CONFIG,
            analyze_content,
        )
except ImportError as e:
    WEB_SEARCH_PREDICTOR_AVAILABLE = False
    _log_import_failure("WebSearchPredictor", e)

# Memory system import
try:
    from ..memory import MemoryManager, MEMORY_SYSTEM_AVAILABLE
except ImportError as e:
    MEMORY_SYSTEM_AVAILABLE = False
    MemoryManager = None
    _log_import_failure("MemoryManager", e)

# =============================================================================
# DeepThinker 2.0 Structural Hardening Components
# =============================================================================

# Phase specification and validation
try:
    from ..schemas.phase_spec import (
        PhaseSpec,
        get_phase_spec,
        infer_phase_type,
        RECONNAISSANCE_PHASE,
        ANALYSIS_PHASE,
        DEEP_ANALYSIS_PHASE,
        SYNTHESIS_PHASE,
    )
    PHASE_SPEC_AVAILABLE = True
except ImportError as e:
    PHASE_SPEC_AVAILABLE = False
    PhaseSpec = None
    get_phase_spec = None
    _log_import_failure("PhaseSpec", e)

# Phase validator for contract enforcement
try:
    from ..core.phase_validator import PhaseValidator, get_phase_validator
    PHASE_VALIDATOR_AVAILABLE = True
except ImportError as e:
    PHASE_VALIDATOR_AVAILABLE = False
    PhaseValidator = None
    get_phase_validator = None
    _log_import_failure("PhaseValidator", e)

# Memory guard for memory discipline
try:
    from ..memory.memory_guard import (
        MemoryGuard,
        MemoryWriteRequest,
        get_memory_guard,
    )
    MEMORY_GUARD_AVAILABLE = True
except ImportError as e:
    MEMORY_GUARD_AVAILABLE = False
    MemoryGuard = None
    get_memory_guard = None
    _log_import_failure("MemoryGuard", e)

# Model selector for phase-aware model selection
try:
    from ..models.model_selector import ModelSelector, get_model_selector
    MODEL_SELECTOR_AVAILABLE = True
except ImportError as e:
    MODEL_SELECTOR_AVAILABLE = False
    ModelSelector = None
    get_model_selector = None
    _log_import_failure("ModelSelector", e)

# Consensus policy engine for conditional consensus
try:
    from ..consensus.policy_engine import (
        ConsensusPolicyEngine,
        get_consensus_policy_engine,
    )
    CONSENSUS_POLICY_AVAILABLE = True
except ImportError as e:
    CONSENSUS_POLICY_AVAILABLE = False
    ConsensusPolicyEngine = None
    get_consensus_policy_engine = None
    _log_import_failure("ConsensusPolicyEngine", e)

# Strict convergence check
try:
    from ..meta.supervisor import ConvergenceResult
    CONVERGENCE_RESULT_AVAILABLE = True
except ImportError as e:
    CONVERGENCE_RESULT_AVAILABLE = False
    ConvergenceResult = None
    _log_import_failure("ConvergenceResult", e)

# =============================================================================
# Epistemic Hardening Components (DeepThinker 2.0)
# =============================================================================

# Claim validator for evidence enforcement
try:
    from ..epistemics import (
        Claim,
        ClaimType,
        ClaimValidator,
        ClaimValidationResult,
        EpistemicRiskScore,
        Source,
        get_claim_validator,
    )
    CLAIM_VALIDATOR_AVAILABLE = True
except ImportError as e:
    CLAIM_VALIDATOR_AVAILABLE = False
    ClaimValidator = None
    get_claim_validator = None
    EpistemicRiskScore = None
    _log_import_failure("ClaimValidator", e)

# Scenario modeling for structured scenario analysis
try:
    from ..scenarios import (
        Scenario,
        ScenarioFactory,
        ScenarioParser,
        ScenarioSet,
        get_scenario_factory,
    )
    SCENARIO_MODEL_AVAILABLE = True
except ImportError as e:
    SCENARIO_MODEL_AVAILABLE = False
    ScenarioFactory = None
    get_scenario_factory = None
    _log_import_failure("ScenarioFactory", e)

# Phase guard for phase purity enforcement
try:
    from ..phases import (
        PhaseGuard,
        PhaseContamination,
        PhaseContract,
        PHASE_CONTRACTS,
        get_phase_guard,
    )
    PHASE_GUARD_AVAILABLE = True
except ImportError as e:
    PHASE_GUARD_AVAILABLE = False
    PhaseGuard = None
    get_phase_guard = None
    _log_import_failure("PhaseGuard", e)

# Normative Control Layer (Governance)
# Note: SafetyCore can also be used via safety.get("governance", "NormativeController")
try:
    from ..governance import (
        NormativeController,
        NormativeVerdict,
        VerdictStatus,
        RecommendedAction,
    )
    GOVERNANCE_AVAILABLE = True
except ImportError as e:
    GOVERNANCE_AVAILABLE = False
    NormativeController = None
    NormativeVerdict = None
    VerdictStatus = None
    RecommendedAction = None
    _log_import_failure("NormativeController", e)

# Verify with SafetyCore if available (provides centralized logging)
if SAFETY_CORE_AVAILABLE and safety is not None:
    _governance_via_safety = safety.is_available("governance")
    if _governance_via_safety != GOVERNANCE_AVAILABLE:
        _orchestrator_logger.debug(
            f"Governance availability mismatch: legacy={GOVERNANCE_AVAILABLE}, safety_core={_governance_via_safety}"
        )

# Decision Accountability Layer
try:
    from ..decisions import (
        DecisionEmitter,
        DecisionStore,
        DecisionType,
        OutcomeCause,
    )
    DECISION_ACCOUNTABILITY_AVAILABLE = True
except ImportError as e:
    DECISION_ACCOUNTABILITY_AVAILABLE = False
    DecisionEmitter = None
    DecisionStore = None
    DecisionType = None
    OutcomeCause = None
    _log_import_failure("DecisionEmitter", e)

# Proof-Carrying Reasoning (PCR) - Proof Packet v1
try:
    from ..proofs import (
        ProofPacketBuilder,
        ProofStore,
        ProofPacket,
        generate_blinded_view,
    )
    from ..epistemics.contradiction_detector import get_contradiction_detector
    PROOF_PACKETS_AVAILABLE = True
except ImportError as e:
    PROOF_PACKETS_AVAILABLE = False
    ProofPacketBuilder = None
    ProofStore = None
    ProofPacket = None
    generate_blinded_view = None
    get_contradiction_detector = None
    _log_import_failure("ProofPackets", e)

# Web search gate for mandatory search enforcement
try:
    from ..councils.researcher_council.researcher_council import (
        WebSearchGate,
        WebSearchGateResult,
        get_web_search_gate,
    )
    WEB_SEARCH_GATE_AVAILABLE = True
except ImportError as e:
    WEB_SEARCH_GATE_AVAILABLE = False
    WebSearchGate = None
    get_web_search_gate = None
    _log_import_failure("WebSearchGate", e)

# Sprint 1-2: Metrics Integration (Scorecard, Policy, Router, Bandit)
try:
    from ..metrics import (
        get_metrics_config,
        get_metrics_hook,
        MetricsOrchestrationHook,
        PhaseMetricsContext,
    )
    from ..policy import PolicyAction
    METRICS_INTEGRATION_AVAILABLE = True
except ImportError as e:
    METRICS_INTEGRATION_AVAILABLE = False
    get_metrics_config = None
    get_metrics_hook = None
    MetricsOrchestrationHook = None
    PhaseMetricsContext = None
    PolicyAction = None
    _log_import_failure("MetricsIntegration", e)

# Bio Priors - Biological strategy patterns as soft background priors
try:
    from ..bio_priors import (
        BioPriorConfig,
        BioPriorEngine,
        BioPriorOutput,
        BioPriorContext,
        build_context as build_bio_context,
        apply_bio_pressures_to_deepening_plan,
        get_bio_prior_config,
    )
    from ..bio_priors.integration import compute_would_apply_diff
    from ..constitution.types import PriorInfluenceEvent, ConstitutionEventType
    BIO_PRIORS_AVAILABLE = True
except ImportError as e:
    BIO_PRIORS_AVAILABLE = False
    BioPriorConfig = None
    BioPriorEngine = None
    BioPriorOutput = None
    BioPriorContext = None
    build_bio_context = None
    apply_bio_pressures_to_deepening_plan = None
    get_bio_prior_config = None
    compute_would_apply_diff = None
    PriorInfluenceEvent = None
    _log_import_failure("BioPriors", e)

_orchestrator_logger = logging.getLogger(__name__)

# Minimum time (in minutes) required to attempt a phase
MIN_PHASE_TIME_MINUTES = 0.5

# Keywords for mapping phase names to councils
PHASE_KEYWORDS = {
    "research": ["recon", "context", "situation", "research", "gather", "sources", "analysis", "investigate"],
    "design": ["design", "architecture", "plan", "strategy", "approach", "requirements"],
    "implementation": ["implementation", "coding", "build", "develop", "code", "implement", "create"],
    "testing": ["testing", "simulation", "validation", "test", "verify", "edge", "stress"],
    "synthesis": ["synthesis", "report", "conclusion", "summary", "final", "consolidate", "review"]
}


class MissionOrchestrator:
    """
    Orchestrates long-running missions using the council architecture.
    
    Responsibilities:
    - Initialize missions with phases planned by the planner council
    - Execute phases by delegating to appropriate councils
    - Enforce time budget and constraints
    - Checkpoint state regularly for resumability
    - Manage GPU resources and model selection via supervisor
    """
    
    def __init__(
        self,
        planner_council: PlannerCouncil,
        researcher_council: ResearcherCouncil,
        coder_council: CoderCouncil,
        evaluator_council: EvaluatorCouncil,
        simulation_council: SimulationCouncil,
        arbiter: Arbiter,
        store: MissionStore,
        output_manager: Optional[OutputManager] = None,
        gpu_manager: Optional["GPUResourceManager"] = None,
        supervisor: Optional["ModelSupervisor"] = None,
        enable_supervision: bool = True,
        step_executor: Optional[StepExecutor] = None,
        enable_step_execution: bool = True,
        enable_step_reflection: bool = True,
        enable_multiview: bool = True,
        enable_phase_deepening: bool = True,
        deepening_time_threshold_minutes: float = 2.0,
        optimist_council: Optional["OptimistCouncil"] = None,
        skeptic_council: Optional["SkepticCouncil"] = None,
        enable_iterative_execution: bool = True,
        max_mission_iterations: int = 5,
        min_iteration_time_minutes: float = 1.0,
        enable_dynamic_councils: bool = True,
        orchestration_store: Optional["OrchestrationStore"] = None,
        enable_orchestration_logging: bool = True,
    ):
        """
        Initialize the mission orchestrator.
        
        Args:
            planner_council: Council for planning and strategy
            researcher_council: Council for research and information gathering
            coder_council: Council for code generation
            evaluator_council: Council for evaluation and quality assessment
            simulation_council: Council for testing and simulation
            arbiter: Final decision maker for reconciling outputs
            store: Persistence layer for mission state
            gpu_manager: Optional GPU resource manager
            supervisor: Optional model supervisor
            enable_supervision: Whether to use supervisor for model selection
            step_executor: Optional StepExecutor for step-based execution
            enable_step_execution: Whether to use step-based execution (vs legacy council-only)
            enable_step_reflection: Whether to use evaluator reflection for steps
            enable_multiview: Whether to use optimist/skeptic councils
            enable_phase_deepening: Whether to run deepening loops after phases
            deepening_time_threshold_minutes: Minimum time remaining to run deepening
            optimist_council: Optional OptimistCouncil for multi-view reasoning
            skeptic_council: Optional SkepticCouncil for multi-view reasoning
            enable_iterative_execution: Whether to run multiple mission iterations
            max_mission_iterations: Maximum number of mission-level iterations
            min_iteration_time_minutes: Minimum time required to start a new iteration
        """
        self.planner = planner_council
        self.researcher = researcher_council
        self.coder = coder_council
        self.evaluator = evaluator_council
        self.simulator = simulation_council
        self.arbiter = arbiter
        self.store = store
        
        # Multi-view councils
        self.enable_multiview = enable_multiview and MULTIVIEW_COUNCILS_AVAILABLE
        self.optimist = optimist_council
        self.skeptic = skeptic_council
        
        # Phase deepening configuration
        self.enable_phase_deepening = enable_phase_deepening
        self.deepening_time_threshold_minutes = deepening_time_threshold_minutes
        
        # Council output history for enhanced context accumulation
        self._council_output_history: Dict[str, List[Any]] = {
            "planner": [],
            "researcher": [],
            "coder": [],
            "evaluator": [],
            "simulation": [],
            "optimist": [],
            "skeptic": [],
        }
        
        # Multi-view tracking
        self._multiview_disagreements: List[Dict[str, Any]] = []
        
        # Output manager - create default if not provided
        self.output_manager = output_manager if output_manager is not None else OutputManager()
        
        # GPU and supervisor integration
        self.enable_supervision = enable_supervision
        
        # Initialize GPU manager
        if gpu_manager is not None:
            self.gpu_manager = gpu_manager
        elif GPU_MANAGER_AVAILABLE and enable_supervision:
            self.gpu_manager = GPUResourceManager()
        else:
            self.gpu_manager = None
        
        # Phase 3: RAM monitoring integration
        try:
            from ..monitoring.ram_monitor import RAMMonitor
            self.ram_monitor = RAMMonitor()
            # Integrate RAM monitor into GPU manager if available
            if self.gpu_manager:
                self.gpu_manager.ram_monitor = self.ram_monitor
        except Exception:
            self.ram_monitor = None
        
        # Initialize supervisor
        if supervisor is not None:
            self.supervisor = supervisor
        elif SUPERVISOR_AVAILABLE and enable_supervision:
            self.supervisor = ModelSupervisor()
        else:
            self.supervisor = None
        
        # Dynamic Council Generator
        self.enable_dynamic_councils = enable_dynamic_councils and DYNAMIC_COUNCIL_AVAILABLE
        self.dynamic_council_factory: Optional["DynamicCouncilFactory"] = None
        if self.enable_dynamic_councils:
            self.dynamic_council_factory = DynamicCouncilFactory()
            import logging
            logging.getLogger(__name__).info(
                "DynamicCouncilFactory initialized - councils will be configured dynamically"
            )
        
        # Step Engine configuration
        self.enable_step_execution = enable_step_execution
        self.enable_step_reflection = enable_step_reflection
        self.step_executor = step_executor
        
        # Inject GPU manager and supervisor into councils
        self._setup_councils()
        
        # Setup step executor if enabled
        self._setup_step_executor()
        
        # Meta-cognition engine
        self.meta: Optional["MetaController"] = None
        self._setup_meta_controller()
        
        # Iterative execution configuration
        self.enable_iterative_execution = enable_iterative_execution
        self.max_mission_iterations = max_mission_iterations
        self.min_iteration_time_minutes = min_iteration_time_minutes
        
        # ReasoningSupervisor (accessed via MetaController or directly)
        self.reasoning_supervisor: Optional["ReasoningSupervisor"] = None
        self._setup_reasoning_supervisor()
        
        # CognitiveSpine for unified validation, consensus, and resource management
        self.cognitive_spine: Optional["CognitiveSpine"] = None
        self._setup_cognitive_spine()
        
        # Iteration context manager for stateful context evolution
        self._iteration_context_manager: Optional["IterationContextManager"] = None
        if ITERATION_CONTEXT_AVAILABLE:
            self._iteration_context_manager = IterationContextManager()
        
        # Memory system (initialized per-mission)
        self.enable_memory = MEMORY_SYSTEM_AVAILABLE
        self.memory: Optional["MemoryManager"] = None
        
        # Orchestration learning layer
        self.enable_orchestration_logging = enable_orchestration_logging and ORCHESTRATION_AVAILABLE
        if orchestration_store is not None:
            self.orchestration_store = orchestration_store
        elif self.enable_orchestration_logging:
            from pathlib import Path
            self.orchestration_store = OrchestrationStore(base_dir=Path("kb/orchestration"))
        else:
            self.orchestration_store = None
        
        # Phase time allocator for proactive time budgeting
        self._phase_time_allocator: Optional["PhaseTimeAllocator"] = None
        self._time_allocation: Optional["TimeAllocation"] = None
        self._mission_time_manager: Optional["MissionTimeManager"] = None
        if PHASE_TIME_ALLOCATOR_AVAILABLE and self.orchestration_store is not None:
            self._phase_time_allocator = create_allocator_from_store(self.orchestration_store)
            _orchestrator_logger.debug("PhaseTimeAllocator initialized with PolicyMemory")
        elif PHASE_TIME_ALLOCATOR_AVAILABLE:
            self._phase_time_allocator = PhaseTimeAllocator()
            _orchestrator_logger.debug("PhaseTimeAllocator initialized without PolicyMemory")
        
        # Deepening configuration
        self._min_deepening_iteration_seconds = 30.0
        self._max_deepening_rounds = 2  # Reduced from 3 to prevent runaway phase deepening
        self._convergence_threshold_for_deepening = 0.7
        self._target_utilization = 0.8
        
        # Depth control configuration
        self.enable_depth_control = True
        self.max_enrichment_passes = 2
        self.min_enrichment_time_seconds = 30.0
        self.depth_gap_threshold = 0.1  # Stop enrichment if gap <= this
        
        # Convergence tracking for iterative execution
        self._convergence_state: Optional[ConvergenceState] = None
        self._last_evaluator_output: Any = None
        self._last_multiview_disagreement: float = 0.0
        self._current_subgoals: List[str] = []
        
        # Phase failure tracking to prevent infinite loops
        self._phase_failure_counts: Dict[str, int] = {}
        self._max_consecutive_failures = 3
        
        # Semantic distance consensus for multiview disagreement
        try:
            from ..consensus.semantic_distance import SemanticDistanceConsensus
            self._semantic_consensus = SemanticDistanceConsensus()
            self._has_semantic_consensus = True
        except ImportError:
            self._semantic_consensus = None
            self._has_semantic_consensus = False
        
        # Search trigger manager for objective-aware web search
        try:
            from ..tools.search_triggers import SearchTriggerManager
            self._search_trigger_manager = SearchTriggerManager(
                enable_search=True,
                global_quota=10,
            )
            self._has_search_triggers = True
        except ImportError:
            self._search_trigger_manager = None
            self._has_search_triggers = False
        
        # =====================================================================
        # DeepThinker 2.0 Structural Hardening Components
        # =====================================================================
        
        # Phase validator for enforcing phase contracts
        self._phase_validator: Optional["PhaseValidator"] = None
        if PHASE_VALIDATOR_AVAILABLE:
            self._phase_validator = get_phase_validator(strict_mode=False)
            _orchestrator_logger.debug("PhaseValidator initialized")
        
        # Memory guard for memory discipline
        self._memory_guard: Optional["MemoryGuard"] = None
        if MEMORY_GUARD_AVAILABLE:
            self._memory_guard = get_memory_guard(strict_mode=False)
            _orchestrator_logger.debug("MemoryGuard initialized")
        
        # Model selector for phase-aware model selection
        self._model_selector: Optional["ModelSelector"] = None
        if MODEL_SELECTOR_AVAILABLE:
            self._model_selector = get_model_selector()
            _orchestrator_logger.debug("ModelSelector initialized")
        
        # Consensus policy engine for conditional consensus
        self._consensus_policy: Optional["ConsensusPolicyEngine"] = None
        if CONSENSUS_POLICY_AVAILABLE:
            self._consensus_policy = get_consensus_policy_engine()
            _orchestrator_logger.debug("ConsensusPolicyEngine initialized")
        
        # =====================================================================
        # Epistemic Hardening Components
        # =====================================================================
        
        # Claim validator for evidence enforcement
        self._claim_validator: Optional["ClaimValidator"] = None
        if CLAIM_VALIDATOR_AVAILABLE:
            self._claim_validator = get_claim_validator(min_grounded_ratio=0.6)
            _orchestrator_logger.debug("ClaimValidator initialized")
        
        # Scenario factory for structured scenario modeling
        self._scenario_factory: Optional["ScenarioFactory"] = None
        if SCENARIO_MODEL_AVAILABLE:
            self._scenario_factory = get_scenario_factory()
            _orchestrator_logger.debug("ScenarioFactory initialized")
        
        # Phase guard for phase purity enforcement
        self._phase_guard: Optional["PhaseGuard"] = None
        if PHASE_GUARD_AVAILABLE:
            self._phase_guard = get_phase_guard(strict_mode=False)
            _orchestrator_logger.debug("PhaseGuard initialized")
        
        # Web search gate for mandatory search enforcement
        self._web_search_gate: Optional["WebSearchGate"] = None
        if WEB_SEARCH_GATE_AVAILABLE:
            self._web_search_gate = get_web_search_gate()
            _orchestrator_logger.debug("WebSearchGate initialized")
        
        # Normative Control Layer (Governance)
        self._normative_controller: Optional["NormativeController"] = None
        if GOVERNANCE_AVAILABLE:
            self._normative_controller = NormativeController(gpu_manager=self.gpu_manager)
            _orchestrator_logger.debug("NormativeController initialized")
        
        # Decision Accountability Layer
        self._decision_store: Optional["DecisionStore"] = None
        self._decision_emitter: Optional["DecisionEmitter"] = None
        self._enable_decision_accountability = False
        if DECISION_ACCOUNTABILITY_AVAILABLE:
            try:
                self._decision_store = DecisionStore()
                self._decision_emitter = DecisionEmitter(store=self._decision_store, enabled=True)
                self._enable_decision_accountability = True
                _orchestrator_logger.debug("DecisionAccountabilityLayer initialized")
                
                # Wire emitter to supervisor and governance controller
                if self.supervisor:
                    self.supervisor.set_decision_emitter(self._decision_emitter)
                if self._normative_controller:
                    self._normative_controller.set_decision_emitter(self._decision_emitter)
            except Exception as e:
                _orchestrator_logger.debug(f"DecisionAccountabilityLayer init failed: {e}")
        
        # Proof-Carrying Reasoning (PCR) - Proof Packet Layer
        self._proof_store: Optional["ProofStore"] = None
        self._proof_builder: Optional["ProofPacketBuilder"] = None
        self._enable_proof_packets = False
        if PROOF_PACKETS_AVAILABLE:
            try:
                self._proof_store = ProofStore()
                self._proof_builder = ProofPacketBuilder(
                    decision_store=self._decision_store,
                    contradiction_detector=get_contradiction_detector() if get_contradiction_detector else None,
                )
                self._enable_proof_packets = True
                _orchestrator_logger.debug("ProofPacketLayer initialized")
                # Wire proof components to arbiter
                if hasattr(self.arbiter, 'set_proof_components'):
                    self.arbiter.set_proof_components(
                        proof_builder=self._proof_builder,
                        proof_store=self._proof_store,
                    )
            except Exception as e:
                _orchestrator_logger.debug(f"ProofPacketLayer init failed: {e}")
        
        # Cost/Time Predictor for shadow mode prediction logging
        self._cost_predictor: Optional["CostTimePredictor"] = None
        self._cost_eval_logger: Optional["EvaluationLogger"] = None
        self._enable_cost_prediction = False
        if COST_PREDICTOR_AVAILABLE:
            try:
                if PREDICTOR_CONFIG.get("enabled", False):
                    self._cost_predictor = CostTimePredictor()
                    self._cost_eval_logger = EvaluationLogger()
                    self._enable_cost_prediction = True
                    _orchestrator_logger.debug(
                        f"CostTimePredictor initialized (mode={PREDICTOR_CONFIG.get('mode', 'shadow')})"
                    )
            except Exception as e:
                _orchestrator_logger.debug(f"CostTimePredictor init failed: {e}")
        
        # Phase Risk Predictor for shadow mode risk prediction logging
        self._risk_predictor: Optional["PhaseRiskPredictor"] = None
        self._risk_eval_logger: Optional["PhaseRiskEvaluationLogger"] = None
        self._enable_risk_prediction = False
        if RISK_PREDICTOR_AVAILABLE:
            try:
                if RISK_PREDICTOR_CONFIG.get("enabled", False):
                    self._risk_predictor = PhaseRiskPredictor()
                    self._risk_eval_logger = PhaseRiskEvaluationLogger()
                    self._enable_risk_prediction = True
                    _orchestrator_logger.debug(
                        f"PhaseRiskPredictor initialized (mode={RISK_PREDICTOR_CONFIG.get('mode', 'shadow')})"
                    )
            except Exception as e:
                _orchestrator_logger.debug(f"PhaseRiskPredictor init failed: {e}")
        
        # Web Search Predictor for shadow mode search necessity prediction logging
        self._web_search_predictor: Optional["WebSearchPredictor"] = None
        self._web_search_eval_logger: Optional["WebSearchEvaluationLogger"] = None
        self._enable_web_search_prediction = False
        if WEB_SEARCH_PREDICTOR_AVAILABLE:
            try:
                if WEB_SEARCH_PREDICTOR_CONFIG.get("enabled", False):
                    self._web_search_predictor = WebSearchPredictor()
                    self._web_search_eval_logger = WebSearchEvaluationLogger()
                    self._enable_web_search_prediction = True
                    _orchestrator_logger.debug(
                        f"WebSearchPredictor initialized (mode={WEB_SEARCH_PREDICTOR_CONFIG.get('mode', 'shadow')})"
                    )
            except Exception as e:
                _orchestrator_logger.debug(f"WebSearchPredictor init failed: {e}")
        
        # Enable/disable strict phase enforcement (opt-in)
        self._strict_phase_enforcement = False
        
        # DeepThinker 2.0: Strict convergence (prioritizes unresolved questions)
        self._use_strict_convergence = True  # ON by default for 2.0
        
        # Phase 3.1: Model prefetching for next phase
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_models: List[str] = []
        self._enable_prefetch = True
        
        # Sprint 1-2: Metrics Integration Hook
        self._metrics_hook: Optional["MetricsOrchestrationHook"] = None
        self._metrics_config = None
        self._recent_scores: List[float] = []  # For router features
        if METRICS_INTEGRATION_AVAILABLE:
            try:
                self._metrics_config = get_metrics_config()
                if self._metrics_config.is_any_enabled():
                    self._metrics_hook = get_metrics_hook(self._metrics_config)
                    _orchestrator_logger.debug(
                        f"MetricsIntegration initialized "
                        f"(scorecard={self._metrics_config.scorecard_enabled}, "
                        f"policy={self._metrics_config.scorecard_policy_enabled}, "
                        f"router={self._metrics_config.learning_router_enabled}, "
                        f"bandit={self._metrics_config.bandit_enabled})"
                    )
            except Exception as e:
                _orchestrator_logger.debug(f"MetricsIntegration init failed: {e}")
        
        # Cognitive Constitution v1 Integration
        self._constitution_engine = None
        self._constitution_ctx = None
        self._learning_blocked = False
        try:
            from ..constitution import get_engine, get_constitution_config, ConstitutionFlags
            constitution_config = get_constitution_config()
            if constitution_config.is_enabled:
                self._constitution_engine = get_engine(state.mission_id, constitution_config)
                _orchestrator_logger.debug(
                    f"ConstitutionEngine initialized (mode={constitution_config.mode.value})"
                )
        except ImportError:
            pass
        except Exception as e:
            _orchestrator_logger.debug(f"ConstitutionEngine init failed: {e}")
        
        # =====================================================================
        # Bio Priors - Biological strategy patterns as soft background priors
        # =====================================================================
        self._bio_prior_config: Optional["BioPriorConfig"] = None
        self._bio_prior_engine: Optional["BioPriorEngine"] = None
        self._enable_bio_priors = False
        if BIO_PRIORS_AVAILABLE:
            try:
                self._bio_prior_config = get_bio_prior_config()
                if self._bio_prior_config.is_active:
                    self._bio_prior_engine = BioPriorEngine(config=self._bio_prior_config)
                    self._enable_bio_priors = True
                    _orchestrator_logger.info(
                        f"BioPriorEngine initialized (mode={self._bio_prior_config.mode}, "
                        f"topk={self._bio_prior_config.topk})"
                    )
                else:
                    _orchestrator_logger.debug("BioPriors disabled by config")
            except Exception as e:
                _orchestrator_logger.debug(f"BioPriorEngine init failed: {e}")
        
        # Log summary of unavailable components (once per session)
        log_component_summary()
    
    def enable_strict_phases(self, strict: bool = True) -> None:
        """
        Enable or disable strict phase contract enforcement.
        
        When enabled:
        - Councils are validated against phase specs before execution
        - Forbidden artifacts are blocked from output
        - Memory writes are validated against phase policies
        
        Args:
            strict: Whether to enable strict enforcement
        """
        self._strict_phase_enforcement = strict
        if self._phase_validator:
            self._phase_validator.strict_mode = strict
        if self._memory_guard:
            self._memory_guard.strict_mode = strict
        _orchestrator_logger.info(f"Strict phase enforcement: {'enabled' if strict else 'disabled'}")
    
    # =========================================================================
    # Phase 3.1: Model Prefetching
    # =========================================================================
    
    def _prefetch_next_phase_models(
        self,
        state: MissionState,
        current_phase_idx: int
    ) -> Optional[threading.Thread]:
        """
        Start background thread to warm models for next phase.
        
        Phase 3.1: Prefetches models while current phase is executing to
        reduce model loading time for the next phase.
        
        Args:
            state: Current mission state
            current_phase_idx: Index of currently executing phase
            
        Returns:
            Thread handle for the prefetch operation, or None if no prefetch needed
        """
        if not self._enable_prefetch:
            return None
        
        # Get next phase
        next_idx = current_phase_idx + 1
        if next_idx >= len(state.phases):
            return None
        
        next_phase = state.phases[next_idx]
        
        # Skip if next phase is already completed or skipped
        if next_phase.status in ("completed", "skipped", "failed"):
            return None
        
        # Determine models for next phase
        try:
            # Get supervisor decision for next phase to determine models
            next_phase_type = self._classify_phase(next_phase)
            models_to_prefetch = []
            
            if self.enable_supervision and self.supervisor is not None:
                # Get anticipated decision
                decision = self._get_supervisor_decision(state, next_phase)
                if decision and decision.models:
                    models_to_prefetch = decision.models
            
            # Fallback: use phase-type heuristics
            if not models_to_prefetch:
                if next_phase_type == "research":
                    models_to_prefetch = ["gemma3:12b", "llama3.2:3b"]
                elif next_phase_type == "design":
                    models_to_prefetch = ["gemma3:27b"]
                elif next_phase_type == "synthesis":
                    models_to_prefetch = ["gemma3:27b"]
                else:
                    models_to_prefetch = ["gemma3:12b"]
            
            self._prefetch_models = models_to_prefetch
            
            # Start prefetch thread
            def prefetch_work():
                """Background prefetch: warm model cache via ollama."""
                import subprocess
                for model in models_to_prefetch:
                    try:
                        # Use ollama show to trigger model loading without generation
                        subprocess.run(
                            ["ollama", "show", model],
                            capture_output=True,
                            timeout=30
                        )
                        _orchestrator_logger.debug(f"[PREFETCH] Warmed model: {model}")
                    except Exception as e:
                        _orchestrator_logger.debug(f"[PREFETCH] Failed to warm {model}: {e}")
            
            thread = threading.Thread(target=prefetch_work, daemon=True)
            thread.start()
            
            _orchestrator_logger.info(
                f"[PREFETCH] Started prefetch for phase '{next_phase.name}': {models_to_prefetch}"
            )
            
            return thread
            
        except Exception as e:
            _orchestrator_logger.debug(f"[PREFETCH] Failed to start prefetch: {e}")
            return None
    
    def _wait_for_prefetch(self, timeout: float = 5.0) -> None:
        """
        Wait for prefetch thread to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            _orchestrator_logger.debug(f"[PREFETCH] Waiting for prefetch completion (max {timeout}s)")
            self._prefetch_thread.join(timeout=timeout)
            if self._prefetch_thread.is_alive():
                _orchestrator_logger.debug("[PREFETCH] Prefetch still running, proceeding anyway")
        self._prefetch_thread = None
    
    def _get_phase_spec(self, phase_name: str) -> Optional["PhaseSpec"]:
        """
        Get the PhaseSpec for a phase by name.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            PhaseSpec or None if not available
        """
        if not PHASE_SPEC_AVAILABLE or get_phase_spec is None:
            return None
        
        return get_phase_spec(phase_name, strict=False)
    
    def _validate_council_for_phase(
        self,
        phase_spec: "PhaseSpec",
        council_name: str
    ) -> bool:
        """
        Validate if a council is allowed in the current phase.
        
        Args:
            phase_spec: Current phase specification
            council_name: Name of the council to validate
            
        Returns:
            True if allowed, False if blocked
        """
        if phase_spec is None or self._phase_validator is None:
            return True  # Permissive if no validator
        
        result = self._phase_validator.validate_council_for_phase(phase_spec, council_name)
        
        if not result.is_valid:
            _orchestrator_logger.warning(
                f"Council '{council_name}' blocked in phase '{phase_spec.name}': {result.errors}"
            )
            if self._strict_phase_enforcement:
                return False
        
        return True
    
    def _validate_phase_output(
        self,
        phase_spec: "PhaseSpec",
        output: Any
    ) -> Any:
        """
        Validate and potentially strip forbidden content from phase output.
        
        Args:
            phase_spec: Current phase specification
            output: Phase output to validate
            
        Returns:
            Validated/sanitized output
        """
        if phase_spec is None or self._phase_validator is None:
            return output
        
        # Validate output content
        result = self._phase_validator.validate_output_content(phase_spec, output)
        if result.warnings:
            for warning in result.warnings:
                _orchestrator_logger.debug(f"Phase output warning: {warning}")
        
        # Block forbidden content if strict mode
        if self._strict_phase_enforcement:
            output, blocked = self._phase_validator.block_forbidden_output(phase_spec, output)
            if blocked:
                _orchestrator_logger.info(
                    f"Blocked forbidden artifacts in phase '{phase_spec.name}': {blocked}"
                )
        
        return output
    
    def _validate_synthesis_output(self, output: Any) -> bool:
        """
        DeepThinker 2.0: Validate synthesis output for quality and correctness.
        
        Detects:
        - Empty or too-short outputs
        - Model confusion patterns ("please provide", etc.)
        - Role refusal patterns
        
        Args:
            output: The synthesis output to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        if not output:
            return False
        
        output_str = str(output)
        
        # Check minimum length
        if len(output_str.strip()) < 100:
            _orchestrator_logger.warning(
                f"Synthesis validation failed: output too short ({len(output_str)} chars)"
            )
            return False
        
        # Check for consensus failure patterns
        failure_patterns = [
            "please provide",
            "i need the text",
            "once you provide",
            "model outputs from",
            "provide the outputs",
            "waiting for",
        ]
        
        output_lower = output_str.lower()[:300]
        for pattern in failure_patterns:
            if pattern in output_lower:
                _orchestrator_logger.warning(
                    f"Synthesis validation failed: found pattern '{pattern}'"
                )
                return False
        
        # Check for role refusal patterns
        refusal_patterns = [
            "i can't provide",
            "i cannot provide",
            "i'm unable to",
            "i won't",
        ]
        
        for pattern in refusal_patterns:
            if pattern in output_lower:
                _orchestrator_logger.warning(
                    f"Synthesis validation failed: found refusal pattern '{pattern}'"
                )
                return False
        
        # Check for model self-description patterns
        # (model describing itself instead of synthesizing content)
        self_description_patterns = [
            "the model is",
            "the models referenced",
            "available in both",
            "parameter sizes",
            "billion parameters",
            "llama3",
            "gemma",
            "cogito",
            "deepseek",
            "mistral",
            "model_0",
            "model_1",
        ]
        
        for pattern in self_description_patterns:
            if pattern in output_lower:
                _orchestrator_logger.warning(
                    f"Synthesis validation failed: model self-description '{pattern}'"
                )
                return False
        
        return True
    
    def _deduplicate_list(self, items: List[str]) -> List[str]:
        """
        Deduplicate a list of strings while preserving order.
        
        Uses case-insensitive comparison and normalizes whitespace.
        
        Args:
            items: List of strings to deduplicate
            
        Returns:
            Deduplicated list with order preserved
        """
        if not items:
            return []
        
        seen = set()
        unique = []
        
        for item in items:
            # Normalize: lowercase and strip whitespace
            normalized = item.strip().lower()
            # Skip very short items (likely noise)
            if len(normalized) < 5:
                continue
            if normalized not in seen:
                seen.add(normalized)
                unique.append(item.strip())  # Keep original casing
        
        return unique
    
    def _setup_councils(self) -> None:
        """Set up councils with GPU manager and supervisor."""
        councils = [
            self.planner,
            self.researcher,
            self.coder,
            self.evaluator,
            self.simulator
        ]
        
        # Add multi-view councils if available
        if self.optimist is not None:
            councils.append(self.optimist)
        if self.skeptic is not None:
            councils.append(self.skeptic)
        
        for council in councils:
            if council is None:
                continue
            if self.gpu_manager is not None and hasattr(council, 'gpu_manager'):
                council.gpu_manager = self.gpu_manager
            if self.supervisor is not None and hasattr(council, 'supervisor'):
                council.supervisor = self.supervisor
    
    def _setup_multiview_councils(self) -> None:
        """Set up multi-view councils if enabled."""
        if not self.enable_multiview:
            return
        
        if not MULTIVIEW_COUNCILS_AVAILABLE:
            return
        
        # Create multi-view councils if not provided
        if self.optimist is None:
            try:
                model_pool = self.evaluator.model_pool
                self.optimist = OptimistCouncil(model_pool=model_pool)
            except Exception:
                pass
        
        if self.skeptic is None:
            try:
                model_pool = self.evaluator.model_pool
                self.skeptic = SkepticCouncil(model_pool=model_pool)
            except Exception:
                pass
    
    def _setup_step_executor(self) -> None:
        """Set up the step executor for step-based execution."""
        if not self.enable_step_execution:
            return
        
        if self.step_executor is not None:
            return
        
        # Create step executor using the planner's model pool
        # This ensures it uses the same Ollama connection
        model_pool = self.planner.model_pool
        
        self.step_executor = StepExecutor(
            model_pool=model_pool,
            evaluator_council=self.evaluator if self.enable_step_reflection else None,
            arbiter=self.arbiter,
            enable_reflection=self.enable_step_reflection,
            reflection_threshold=6.0,
        )
    
    def _setup_meta_controller(self) -> None:
        """Set up the meta-cognition controller."""
        if not META_CONTROLLER_AVAILABLE:
            return
        
        try:
            # Use the planner's model pool for meta-cognition
            model_pool = self.planner.model_pool
            self.meta = MetaController(
                model_pool=model_pool,
                enable_multiview=self.enable_multiview,
                enable_supervisor=True
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to initialize meta-cognition: {e}")
    
    def _setup_reasoning_supervisor(self) -> None:
        """Set up the ReasoningSupervisor."""
        if not REASONING_SUPERVISOR_AVAILABLE:
            return
        
        try:
            # Get from MetaController if available
            if self.meta and hasattr(self.meta, 'supervisor'):
                self.reasoning_supervisor = self.meta.supervisor
            else:
                # Create standalone instance
                model_pool = self.planner.model_pool
                self.reasoning_supervisor = ReasoningSupervisor(model_pool=model_pool)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to initialize ReasoningSupervisor: {e}")
    
    def _setup_cognitive_spine(self) -> None:
        """
        Set up the CognitiveSpine for unified validation and consensus.
        
        The CognitiveSpine provides:
        - Context schema validation for all councils
        - Centralized consensus engine provisioning
        - Resource budget tracking
        - Memory compression between phases
        - Fallback policy coordination with ReasoningSupervisor
        """
        if not COGNITIVE_SPINE_AVAILABLE:
            import logging
            logging.getLogger(__name__).info(
                "CognitiveSpine not available - councils will use standalone consensus"
            )
            return
        
        try:
            # Get verbose logger if available
            vlogger = None
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger is not None:
                vlogger = verbose_logger
            
            # Determine base URL from planner council
            base_url = "http://localhost:11434"
            if hasattr(self.planner, 'model_pool') and hasattr(self.planner.model_pool, 'base_url'):
                base_url = self.planner.model_pool.base_url
            
            # Create CognitiveSpine
            self.cognitive_spine = CognitiveSpine(
                ollama_base_url=base_url,
                max_output_chars=10000,
                max_tokens=50000,
                enable_compression=True,
                verbose_logger=vlogger
            )
            
            # Inject into all councils
            for council in [
                self.planner, self.researcher, self.coder,
                self.evaluator, self.simulator
            ]:
                if council is not None and hasattr(council, 'set_cognitive_spine'):
                    council.set_cognitive_spine(self.cognitive_spine)
            
            # Inject into multi-view councils if available
            if self.optimist is not None and hasattr(self.optimist, 'set_cognitive_spine'):
                self.optimist.set_cognitive_spine(self.cognitive_spine)
            if self.skeptic is not None and hasattr(self.skeptic, 'set_cognitive_spine'):
                self.skeptic.set_cognitive_spine(self.cognitive_spine)
            
            # Connect with ReasoningSupervisor
            if self.reasoning_supervisor is not None:
                self.reasoning_supervisor.set_cognitive_spine(self.cognitive_spine)
            
            import logging
            logging.getLogger(__name__).info("CognitiveSpine initialized and injected into councils")
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to initialize CognitiveSpine: {e}")
            self.cognitive_spine = None
    
    def _setup_memory(self, state: MissionState) -> None:
        """
        Set up the memory system for a mission.
        
        Initializes MemoryManager with mission context and retrieves
        relevant past insights if available.
        
        Args:
            state: Mission state to set up memory for
        """
        if not self.enable_memory or not MEMORY_SYSTEM_AVAILABLE:
            return
        
        try:
            import os
            from pathlib import Path
            
            # Get base directory from environment or default
            base_dir = Path(os.getenv("DEEPTHINKER_KB_DIR", "kb"))
            
            # Initialize memory manager
            self.memory = MemoryManager(
                mission_id=state.mission_id,
                objective=state.objective,
                mission_type=self._infer_mission_type(state.objective),
                time_budget_minutes=state.constraints.time_budget_minutes,
                base_dir=base_dir,
                embedding_model="qwen3-embedding:4b",
                ollama_base_url=self.planner.model_pool.base_url if hasattr(self.planner.model_pool, 'base_url') else "http://localhost:11434",
            )
            
            # Retrieve past insights for mission context
            past_insights = self.memory.retrieve_relevant_past_insights(state.objective, limit=3)
            if past_insights:
                state.log(f"Retrieved {len(past_insights)} relevant past mission insights")
                
                # Seed initial hypotheses from past missions
                seeded = self.memory.seed_initial_hypotheses(state.objective, limit=2)
                if seeded:
                    state.log(f"Seeded {len(seeded)} hypotheses from past missions")
            
            # NEW: Retrieve comprehensive knowledge context (RAG integration)
            knowledge_summary = self.memory.reason_over(objective=state.objective, limit=10)
            
            # Always initialize knowledge_context on state
            state.knowledge_context = {
                "formatted": "",
                "prior_knowledge": [],
                "known_gaps": [],
                "sources": [],
                "items_count": 0,
            }
            
            if knowledge_summary.get("used_in_prompt"):
                formatted_knowledge = self.memory.format_for_prompt(knowledge_summary)
                
                # Store knowledge context in mission state for use by councils
                state.knowledge_context = {
                    "formatted": formatted_knowledge,
                    "prior_knowledge": knowledge_summary.get("prior_knowledge", []),
                    "known_gaps": knowledge_summary.get("known_gaps", []),
                    "sources": knowledge_summary.get("memory_sources", []),
                    "items_count": knowledge_summary.get("memory_used_count", 0),
                }
                
                state.log(
                    f"Knowledge context: {knowledge_summary.get('memory_used_count', 0)} items "
                    f"from RAG (sources: {', '.join(knowledge_summary.get('memory_sources', [])[:3])})"
                )
            else:
                # Log why knowledge was not populated
                state.log(f"No relevant knowledge found for objective (used_in_prompt=False)")
            
            state.log("Memory system initialized")
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to initialize memory system: {e}")
            self.memory = None
    
    def _setup_memory_deferred(
        self,
        mission_id: str,
        objective: str,
        constraints: MissionConstraints
    ) -> Optional[Dict[str, Any]]:
        """
        Set up memory system without requiring MissionState.
        
        This is a deferred version of _setup_memory that can run in parallel
        before the state is created. Returns a result dict that can be applied
        to the state later via _apply_memory_result.
        
        Args:
            mission_id: The mission ID
            objective: The mission objective
            constraints: Mission constraints
            
        Returns:
            Dict with memory initialization results, or None if setup failed
        """
        if not self.enable_memory or not MEMORY_SYSTEM_AVAILABLE:
            return None
        
        try:
            import os
            from pathlib import Path
            
            # Get base directory from environment or default
            base_dir = Path(os.getenv("DEEPTHINKER_KB_DIR", "kb"))
            
            # Initialize memory manager
            self.memory = MemoryManager(
                mission_id=mission_id,
                objective=objective,
                mission_type=self._infer_mission_type(objective),
                time_budget_minutes=constraints.time_budget_minutes,
                base_dir=base_dir,
                embedding_model="qwen3-embedding:4b",
                ollama_base_url=self.planner.model_pool.base_url if hasattr(self.planner.model_pool, 'base_url') else "http://localhost:11434",
            )
            
            result = {
                "past_insights": [],
                "seeded_hypotheses": [],
                "knowledge_context": {
                    "formatted": "",
                    "prior_knowledge": [],
                    "known_gaps": [],
                    "sources": [],
                    "items_count": 0,
                },
                "logs": [],
            }
            
            # Retrieve past insights for mission context
            past_insights = self.memory.retrieve_relevant_past_insights(objective, limit=3)
            if past_insights:
                result["past_insights"] = past_insights
                result["logs"].append(f"Retrieved {len(past_insights)} relevant past mission insights")
                
                # Seed initial hypotheses from past missions
                seeded = self.memory.seed_initial_hypotheses(objective, limit=2)
                if seeded:
                    result["seeded_hypotheses"] = seeded
                    result["logs"].append(f"Seeded {len(seeded)} hypotheses from past missions")
            
            # Retrieve comprehensive knowledge context (RAG integration)
            knowledge_summary = self.memory.reason_over(objective=objective, limit=10)
            
            if knowledge_summary.get("used_in_prompt"):
                formatted_knowledge = self.memory.format_for_prompt(knowledge_summary)
                
                result["knowledge_context"] = {
                    "formatted": formatted_knowledge,
                    "prior_knowledge": knowledge_summary.get("prior_knowledge", []),
                    "known_gaps": knowledge_summary.get("known_gaps", []),
                    "sources": knowledge_summary.get("memory_sources", []),
                    "items_count": knowledge_summary.get("memory_used_count", 0),
                }
                
                result["logs"].append(
                    f"Knowledge context: {knowledge_summary.get('memory_used_count', 0)} items "
                    f"from RAG (sources: {', '.join(knowledge_summary.get('memory_sources', [])[:3])})"
                )
            else:
                result["logs"].append("No relevant knowledge found for objective (used_in_prompt=False)")
            
            result["logs"].append("Memory system initialized")
            return result
            
        except Exception as e:
            _orchestrator_logger.warning(f"Failed to initialize memory system (deferred): {e}")
            self.memory = None
            return None
    
    def _apply_memory_result(self, state: MissionState, memory_result: Dict[str, Any]) -> None:
        """
        Apply the result of deferred memory setup to the mission state.
        
        Args:
            state: The mission state to update
            memory_result: Result dict from _setup_memory_deferred
        """
        # Apply knowledge context
        state.knowledge_context = memory_result.get("knowledge_context", {
            "formatted": "",
            "prior_knowledge": [],
            "known_gaps": [],
            "sources": [],
            "items_count": 0,
        })
        
        # Apply logs
        for log_msg in memory_result.get("logs", []):
            state.log(log_msg)
    
    def _run_shadow_retrieval_deferred(
        self,
        mission_id: str,
        objective: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run shadow latent memory retrieval without requiring MissionState.
        
        This is a deferred version that can run in parallel before the state
        is created.
        
        Args:
            mission_id: The mission ID
            objective: The mission objective
            
        Returns:
            Dict with retrieval results, or None if retrieval failed
        """
        try:
            from deepthinker.latent_memory.shadow import run_shadow_latent_retrieval, persist_shadow_log
            
            result = run_shadow_latent_retrieval(mission_id, objective)
            if result:
                persist_shadow_log(result)
                return result
            return None
        except Exception:
            # Silent fail - latent memory may not be available
            return None
    
    def _infer_mission_type(self, objective: str) -> str:
        """Infer mission type from objective text."""
        objective_lower = objective.lower()
        
        coding_keywords = ["implement", "build", "create", "develop", "code", "application", "app", "system", "function", "class"]
        research_keywords = ["research", "analyze", "investigate", "study", "explore", "understand"]
        strategic_keywords = ["strategy", "plan", "decision", "evaluate", "compare", "assess", "recommend"]
        
        if any(kw in objective_lower for kw in coding_keywords):
            return "coding"
        elif any(kw in objective_lower for kw in research_keywords):
            return "research"
        elif any(kw in objective_lower for kw in strategic_keywords):
            return "strategic"
        else:
            return "general"
    
    def _now(self) -> datetime:
        """Get current UTC time."""
        return datetime.utcnow()
    
    def _get_council_config(self, phase: MissionPhase) -> Dict[str, Any]:
        """
        Get council configuration for a phase.
        
        Args:
            phase: Mission phase
            
        Returns:
            Configuration dictionary
        """
        phase_type = self._classify_phase(phase)
        return {
            "phase_type": phase_type,
            "phase_name": phase.name,
            "council_name": f"{phase_type}_council"
        }
    
    def _apply_dynamic_council_config(
        self,
        state: MissionState,
        phase: MissionPhase,
        difficulty: Optional[float] = None,
        uncertainty: Optional[float] = None,
    ) -> None:
        """
        Apply dynamic council configurations for a phase.
        
        Builds and applies CouncilDefinitions to all relevant councils
        based on the current mission/phase context.
        
        Args:
            state: Current mission state
            phase: Current phase
            difficulty: Optional difficulty score
            uncertainty: Optional uncertainty score
        """
        if not self.enable_dynamic_councils or self.dynamic_council_factory is None:
            return
        
        phase_type = self._classify_phase(phase)
        
        # Map phase types to council types
        council_mapping = {
            "research": ("researcher", self.researcher),
            "design": ("planner", self.planner),
            "implementation": ("coder", self.coder),
            "testing": ("simulation", self.simulator),
            "synthesis": ("planner", self.planner),
        }
        
        # Always apply to evaluator as it's used in all phases
        councils_to_configure = [("evaluator", self.evaluator)]
        
        # Add the phase-specific council
        if phase_type in council_mapping:
            council_type, council = council_mapping[phase_type]
            councils_to_configure.append((council_type, council))
        
        # Apply configurations
        for council_type, council in councils_to_configure:
            if council is not None:
                definition = self._build_council_definition(
                    council_type=council_type,
                    state=state,
                    phase=phase,
                    difficulty=difficulty,
                    uncertainty=uncertainty,
                )
                if definition:
                    council.apply_council_definition(definition)
    
    def _build_council_definition(
        self,
        council_type: str,
        state: MissionState,
        phase: MissionPhase,
        difficulty: Optional[float] = None,
        uncertainty: Optional[float] = None,
    ) -> Optional["CouncilDefinition"]:
        """
        Build a dynamic council definition for a phase.
        
        Uses the DynamicCouncilFactory to select models, temperatures,
        personas, and consensus type based on mission/phase context.
        
        Args:
            council_type: Type of council (planner, researcher, etc.)
            state: Current mission state
            phase: Current phase
            difficulty: Optional difficulty score
            uncertainty: Optional uncertainty score
            
        Returns:
            CouncilDefinition if dynamic councils enabled, None otherwise
        """
        if not self.enable_dynamic_councils or self.dynamic_council_factory is None:
            return None
        
        try:
            # Get available VRAM if GPU manager is available
            available_vram = None
            if self.gpu_manager is not None:
                try:
                    stats = self.gpu_manager.get_stats()
                    available_vram = max(0, stats.free_mem - 2000)  # 2GB safety margin
                except Exception:
                    pass
            
            # Build the council definition
            definition = self.dynamic_council_factory.build_council_definition(
                council_type=council_type,
                phase=phase.name,
                mission_objective=state.objective,
                difficulty=difficulty,
                uncertainty=uncertainty,
                time_budget=state.remaining_minutes(),
                available_vram=available_vram,
            )
            
            if definition:
                # Log the dynamic configuration
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[DynamicCouncil] Built {council_type} for '{phase.name}': "
                    f"models={[m for m, _, _ in definition.models]}, "
                    f"consensus={definition.consensus_type}"
                )
            
            return definition
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to build dynamic council definition for {council_type}: {e}. "
                f"Falling back to static configuration."
            )
            return None
    
    def _get_supervisor_decision(
        self,
        state: MissionState,
        phase: MissionPhase
    ) -> Optional["SupervisorDecision"]:
        """
        Get supervisor decision for a phase.
        
        The supervisor now considers:
        - GPU resources and current pressure
        - Phase importance (synthesis/implementation get priority)
        - Remaining mission time
        - Whether to wait for capacity vs downgrade immediately
        - Escalation signal from governance (if retry)
        
        Args:
            state: Current mission state
            phase: Phase to execute
            
        Returns:
            SupervisorDecision or None if supervision not available
        """
        if self.supervisor is None or self.gpu_manager is None:
            return None
        
        try:
            gpu_stats = self.gpu_manager.get_stats()
            council_config = self._get_council_config(phase)
            
            # Add remaining time info to council config
            council_config["remaining_minutes"] = state.remaining_minutes()
            council_config["time_budget_minutes"] = state.constraints.time_budget_minutes
            
            # Model-Aware Phase Stabilization: Build escalation signal if this is a retry
            escalation_signal = None
            if SUPERVISOR_AVAILABLE and GovernanceEscalationSignal is not None:
                escalation_data = phase.artifacts.get("_escalation_signal")
                if escalation_data:
                    escalation_signal = GovernanceEscalationSignal(
                        violation_types=escalation_data.get("violation_types", []),
                        failed_models=escalation_data.get("failed_models", []),
                        retry_count=escalation_data.get("retry_count", 0),
                        phase_importance=escalation_data.get("phase_importance", 0.5),
                        aggregate_severity=escalation_data.get("aggregate_severity", 0.0),
                    )
            
            decision = self.supervisor.decide(
                mission_state=state,
                phase=phase,
                gpu_stats=gpu_stats,
                council_config=council_config,
                escalation_signal=escalation_signal
            )
            
            # Log GPU status
            state.log_gpu_status(
                total_mem=gpu_stats.total_mem,
                used_mem=gpu_stats.used_mem,
                free_mem=gpu_stats.free_mem,
                utilization=gpu_stats.utilization,
                pressure=self.gpu_manager.get_resource_pressure()
            )
            
            # Log supervisor decision with enhanced info
            state.log_supervisor_decision(
                phase_name=phase.name,
                models=decision.models,
                temperature=decision.temperature,
                parallelism=decision.parallelism,
                downgraded=decision.downgraded,
                reason=decision.reason,
                estimated_vram=decision.estimated_vram,
                wait_for_capacity=decision.wait_for_capacity,
                max_wait_minutes=decision.max_wait_minutes,
                fallback_models=decision.fallback_models,
                phase_importance=decision.phase_importance
            )
            
            # Log model selection panel
            gpu_stats_dict = {
                'available_gpus': gpu_stats.available_gpus if hasattr(gpu_stats, 'available_gpus') else 0,
                'utilization_percent': gpu_stats.utilization if hasattr(gpu_stats, 'utilization') else 0,
                'vram_used_mb': gpu_stats.used_mem if hasattr(gpu_stats, 'used_mem') else 0,
                'vram_total_mb': gpu_stats.total_mem if hasattr(gpu_stats, 'total_mem') else 0,
            }
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_model_selection_panel(
                    decision=decision,
                    phase_name=phase.name,
                    time_remaining=state.remaining_minutes(),
                    total_time=state.constraints.time_budget_minutes,
                    gpu_stats=gpu_stats_dict
                )
            
            # Publish SSE events for frontend
            if SSE_AVAILABLE and sse_manager:
                # Legacy model_selection event (for backward compatibility)
                _publish_sse_event(sse_manager.publish_model_selection(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    models=decision.models,
                    reason=decision.reason,
                    downgraded=decision.downgraded,
                    wait_for_capacity=decision.wait_for_capacity,
                    fallback_models=decision.fallback_models,
                    phase_importance=decision.phase_importance,
                    estimated_vram=decision.estimated_vram,
                    time_remaining=state.remaining_minutes(),
                    total_time=state.constraints.time_budget_minutes,
                    gpu_stats=gpu_stats_dict
                ))
                
                # New supervisor_decision event (richer data for new frontend)
                gpu_pressure = "normal"
                if gpu_stats_dict.get('utilization_percent', 0) > 80:
                    gpu_pressure = "high"
                elif gpu_stats_dict.get('utilization_percent', 0) > 50:
                    gpu_pressure = "medium"
                elif gpu_stats_dict.get('utilization_percent', 0) < 30:
                    gpu_pressure = "low"
                
                _publish_sse_event(sse_manager.publish_supervisor_decision(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    models=decision.models,
                    decision_type="model_selection",
                    downgraded=decision.downgraded,
                    downgrade_reason=decision.reason if decision.downgraded else None,
                    fallback_models=decision.fallback_models,
                    gpu_pressure=gpu_pressure,
                    estimated_vram_mb=decision.estimated_vram,
                    wait_for_capacity=decision.wait_for_capacity,
                    phase_importance=decision.phase_importance,
                    temperature=decision.temperature if hasattr(decision, 'temperature') else 0.7,
                    parallelism=decision.parallelism if hasattr(decision, 'parallelism') else 1,
                ))
            
            # Log additional info if waiting for capacity
            if decision.wait_for_capacity:
                state.log(
                    f"Supervisor suggests waiting up to {decision.max_wait_minutes:.1f}min "
                    f"for {decision.models} (phase importance: {decision.phase_importance:.1f})"
                )
                if decision.fallback_models:
                    state.log(f"Fallback models if timeout: {decision.fallback_models}")
            
            return decision
            
        except Exception as e:
            state.log(f"Supervisor decision failed: {str(e)}")
            return None
    
    def _apply_supervisor_decision(
        self,
        council: Any,
        decision: "SupervisorDecision"
    ) -> None:
        """
        Apply supervisor decision to a council.
        
        Args:
            council: Council to configure
            decision: Supervisor decision to apply
        """
        if hasattr(council, 'model_pool'):
            council.model_pool.update_from_decision(decision)
    
    def _set_council_mission_id(self, council: Any, mission_id: str) -> None:
        """
        Set the mission ID on a council for SSE event publishing.
        
        Args:
            council: Council to configure
            mission_id: Mission ID for SSE events
        """
        if hasattr(council, '_current_mission_id'):
            council._current_mission_id = mission_id
    
    def _prepare_council_for_execution(
        self,
        council: Any,
        state: "MissionState",
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """
        Prepare a council for execution by setting mission ID and applying supervisor decision.
        
        Args:
            council: Council to configure
            state: Current mission state
            decision: Optional supervisor decision to apply
        """
        self._set_council_mission_id(council, state.mission_id)
        if decision:
            self._apply_supervisor_decision(council, decision)
    
    def create_mission(
        self,
        objective: str,
        constraints: MissionConstraints,
    ) -> MissionState:
        """
        Create a new mission with phases planned by the planner council.
        
        Initialization is parallelized for speed:
        - Phase planning (LLM call)
        - Memory setup (embedding + RAG)
        - Shadow latent retrieval (similarity search)
        
        The deadline is set AFTER initialization completes, so init time
        doesn't eat into the mission's time budget.
        
        Args:
            objective: The mission objective/goal
            constraints: Execution constraints
            
        Returns:
            Initialized MissionState
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        
        mission_id = str(uuid.uuid4())
        init_start = time.time()
        
        # === Run slow init operations in parallel ===
        phases = None
        memory_result = None
        shadow_result = None
        
        _orchestrator_logger.debug(f"[init] Starting parallel initialization for mission {mission_id[:8]}...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            phase_future = executor.submit(self._plan_phases, objective, constraints)
            memory_future = executor.submit(
                self._setup_memory_deferred, mission_id, objective, constraints
            )
            shadow_future = executor.submit(
                self._run_shadow_retrieval_deferred, mission_id, objective
            )
            
            # Get phases with 90s timeout and fallback to defaults
            try:
                phases = phase_future.result(timeout=90)
            except FuturesTimeoutError:
                _orchestrator_logger.warning(
                    f"[init] Phase planning timed out after 90s, using default phases"
                )
                phases = self._get_default_phases_with_steps(objective, constraints)
            except Exception as e:
                _orchestrator_logger.warning(
                    f"[init] Phase planning failed: {e}, using default phases"
                )
                phases = self._get_default_phases_with_steps(objective, constraints)
            
            # Get memory result (60s timeout, non-critical)
            try:
                memory_result = memory_future.result(timeout=60)
            except Exception as e:
                _orchestrator_logger.debug(f"[init] Memory setup failed (non-critical): {e}")
                memory_result = None
            
            # Get shadow result (30s timeout, non-critical)
            try:
                shadow_result = shadow_future.result(timeout=30)
            except Exception as e:
                _orchestrator_logger.debug(f"[init] Shadow retrieval failed (non-critical): {e}")
                shadow_result = None
        
        init_elapsed = time.time() - init_start
        _orchestrator_logger.info(f"[init] Parallel initialization completed in {init_elapsed:.1f}s")
        
        # === Set deadline AFTER init completes ===
        # This ensures init time doesn't eat into mission time budget
        created_at = self._now()
        deadline_at = created_at + timedelta(minutes=constraints.time_budget_minutes)
        
        state = MissionState(
            mission_id=mission_id,
            objective=objective,
            constraints=constraints,
            created_at=created_at,
            deadline_at=deadline_at,
            phases=phases,
            status="pending",
        )
        
        state.log(f"Mission created with {len(phases)} phases (init took {init_elapsed:.1f}s)")
        state.log(f"Time budget: {constraints.time_budget_minutes} minutes")
        state.log(f"Deadline: {deadline_at.isoformat()}")
        
        # Log supervision status
        if self.supervisor is not None:
            state.log("Supervisor enabled for dynamic model selection")
        if self.gpu_manager is not None:
            state.log("GPU resource management enabled")
        
        # === Allocate time budgets to phases ===
        self._allocate_phase_time_budgets(state, phases, constraints)
        
        # === Apply memory result if available ===
        if memory_result:
            self._apply_memory_result(state, memory_result)
        else:
            # Initialize empty knowledge context
            state.knowledge_context = {
                "formatted": "",
                "prior_knowledge": [],
                "known_gaps": [],
                "sources": [],
                "items_count": 0,
            }
        
        # === Log shadow retrieval result ===
        if shadow_result:
            _orchestrator_logger.info(
                f"[latent-shadow] Retrieved {len(shadow_result.get('retrieved_missions', []))} "
                f"similar missions for mission_id={state.mission_id[:8]}..."
            )
        
        self.store.save(state)
        return state
    
    def _allocate_phase_time_budgets(
        self,
        state: MissionState,
        phases: List[MissionPhase],
        constraints: MissionConstraints
    ) -> None:
        """
        Allocate time budgets to phases using PhaseTimeAllocator.
        
        Uses historical data from PolicyMemory when available to predict
        phase durations, then scales allocations to meet target utilization
        (default 80% of total budget).
        
        Args:
            state: Mission state to store allocation info
            phases: List of mission phases
            constraints: Mission constraints with time budget
        """
        if not PHASE_TIME_ALLOCATOR_AVAILABLE or self._phase_time_allocator is None:
            state.log("Phase time allocation skipped (allocator not available)")
            return
        
        if not phases:
            return
        
        # Get phase names for allocation
        phase_names = [p.name for p in phases]
        total_budget_seconds = constraints.time_budget_minutes * 60.0
        
        # Allocate time budgets
        self._time_allocation = self._phase_time_allocator.allocate(
            phase_names=phase_names,
            total_budget_seconds=total_budget_seconds,
            reserved_synthesis_seconds=120.0,  # 2 minutes for synthesis
            target_utilization=self._target_utilization
        )
        
        # Apply budgets to phases
        for phase in phases:
            budget = self._time_allocation.get_budget(phase.name)
            if budget:
                phase.time_budget_seconds = budget.max_seconds
                _orchestrator_logger.debug(
                    f"Phase '{phase.name}' budget: {budget.allocated_seconds:.0f}s allocated, "
                    f"{budget.max_seconds:.0f}s max"
                )
        
        # Create MissionTimeManager with allocation
        self._mission_time_manager = create_time_manager_from_constraints(
            time_budget_minutes=constraints.time_budget_minutes,
            reserved_synthesis_minutes=2.0,
            grace_period_seconds=60.0
        )
        self._mission_time_manager.target_utilization = self._target_utilization
        self._mission_time_manager.set_time_allocation(self._time_allocation)
        
        # Log allocation summary
        total_allocated = sum(
            (self._time_allocation.get_budget(p.name).allocated_seconds 
             if self._time_allocation.get_budget(p.name) else 0)
            for p in phases
        )
        state.log(
            f"Time allocation: {total_allocated:.0f}s allocated across {len(phases)} phases "
            f"(target {self._target_utilization:.0%} of {total_budget_seconds:.0f}s budget, "
            f"scale={self._time_allocation.scale_factor:.2f})"
        )
    
    def _plan_phases(
        self,
        objective: str,
        constraints: MissionConstraints
    ) -> List[MissionPhase]:
        """
        Ask the planner council to propose mission phases with steps.
        
        Uses the Step Engine format when enabled, falling back to legacy
        phase-only planning if step execution is disabled.
        
        Args:
            objective: Mission objective
            constraints: Mission constraints
            
        Returns:
            List of MissionPhase objects (with steps if step execution enabled)
        """
        if self.enable_step_execution:
            return self._plan_phases_with_steps(objective, constraints)
        else:
            return self._plan_phases_legacy(objective, constraints)
    
    def _plan_phases_with_steps(
        self,
        objective: str,
        constraints: MissionConstraints
    ) -> List[MissionPhase]:
        """
        Plan phases with embedded step definitions using PlannerCouncil.
        
        This is the Step Engine approach: each phase contains discrete steps
        that will be executed by single specialized models.
        """
        # Use the new structured prompt for phases+steps
        prompt = self.planner.build_mission_phases_prompt(
            objective=objective,
            time_budget_minutes=constraints.time_budget_minutes,
            allow_internet=constraints.allow_internet,
            allow_code_execution=constraints.allow_code_execution,
            notes=constraints.notes,
        )
        
        planner_context = PlannerContext(
            objective=prompt,
            max_iterations=1,
            quality_threshold=5.0
        )
        
        try:
            result = self.planner.execute(planner_context)
            
            if result.success and result.output:
                # Parse the structured output
                raw_output = str(result.output)
                phases_data = self.planner.parse_phases_with_steps(raw_output)
                
                if phases_data:
                    phases = []
                    for phase_name, phase_desc, steps in phases_data:
                        phase = MissionPhase(
                            name=phase_name,
                            description=phase_desc,
                            steps=steps,
                        )
                        phases.append(phase)
                    return phases
                    
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error planning phases with planner council: {e}. Using default phases.")
        
        # Fallback: use default phases with auto-generated steps
        return self._get_default_phases_with_steps(objective, constraints)
    
    def _plan_phases_legacy(
        self,
        objective: str,
        constraints: MissionConstraints
    ) -> List[MissionPhase]:
        """
        Legacy phase planning without steps.
        
        Used when step execution is disabled.
        """
        # Build a specialized prompt for mission phase planning
        constraint_info = []
        if not constraints.allow_internet:
            constraint_info.append("- Internet/web search is NOT allowed")
        if not constraints.allow_code_execution:
            constraint_info.append("- Code execution is NOT allowed (design only)")
        constraint_info.append(f"- Time budget: {constraints.time_budget_minutes} minutes")
        constraint_info.append(f"- Max iterations per phase: {constraints.max_iterations}")
        
        constraint_str = "\n".join(constraint_info) if constraint_info else "No special constraints"
        
        planning_prompt = f"""Plan the phases for this long-running mission.

OBJECTIVE:
{objective}

CONSTRAINTS:
{constraint_str}

Create a list of phases that will accomplish this objective. Each phase should be:
1. Focused on a specific aspect (research, design, implementation, testing, synthesis)
2. Named with a clear identifier
3. Described with what it will accomplish

OUTPUT FORMAT:
For each phase, output exactly in this format:

PHASE: [phase_name]
DESCRIPTION: [what this phase will accomplish]

List 3-7 phases depending on complexity. Common phase types:
- Reconnaissance/Research: Gather context and information
- Design/Architecture: Plan the approach
- Implementation: Build/code the solution
- Testing/Validation: Verify and test
- Synthesis/Report: Consolidate findings and produce final output

Start your response with the phases:"""

        # Create a custom context for phase planning
        planner_context = PlannerContext(
            objective=planning_prompt,
            max_iterations=1,
            quality_threshold=5.0
        )
        
        try:
            result = self.planner.execute(planner_context)
            
            if result.success and result.output:
                phases = self._parse_phases_from_output(str(result.output))
                if phases:
                    return phases
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Legacy phase planning failed: {e}. Using default phases.")
        
        # Fallback to default phases based on constraints
        return self._get_default_phases(objective, constraints)
    
    def _parse_phases_from_output(self, output: str) -> List[MissionPhase]:
        """Parse phase definitions from planner output."""
        phases = []
        
        # Try to parse PHASE: / DESCRIPTION: format
        phase_pattern = r'PHASE:\s*(.+?)(?:\n|$)'
        desc_pattern = r'DESCRIPTION:\s*(.+?)(?=PHASE:|$)'
        
        phase_matches = re.findall(phase_pattern, output, re.IGNORECASE)
        desc_matches = re.findall(desc_pattern, output, re.IGNORECASE | re.DOTALL)
        
        for i, phase_name in enumerate(phase_matches):
            description = desc_matches[i].strip() if i < len(desc_matches) else ""
            phases.append(MissionPhase(
                name=phase_name.strip(),
                description=description
            ))
        
        # If that didn't work, try numbered list format
        if not phases:
            lines = output.split('\n')
            current_name = None
            current_desc = []
            
            for line in lines:
                line = line.strip()
                # Check for numbered items like "1. Phase Name" or "- Phase Name"
                match = re.match(r'^[\d\-\*\.]+\s*(.+)$', line)
                if match and len(match.group(1)) < 100:
                    if current_name:
                        phases.append(MissionPhase(
                            name=current_name,
                            description=" ".join(current_desc)
                        ))
                    current_name = match.group(1)
                    current_desc = []
                elif current_name and line:
                    current_desc.append(line)
            
            if current_name:
                phases.append(MissionPhase(
                    name=current_name,
                    description=" ".join(current_desc)
                ))
        
        return phases[:7]  # Limit to 7 phases max
    
    def _get_default_phases(
        self,
        objective: str,
        constraints: MissionConstraints
    ) -> List[MissionPhase]:
        """Generate default phases based on objective and constraints."""
        phases = []
        
        # Always start with research/reconnaissance
        if constraints.allow_internet:
            phases.append(MissionPhase(
                name="Reconnaissance",
                description="Gather context, background information, and relevant resources"
            ))
        
        phases.append(MissionPhase(
            name="Analysis & Planning",
            description="Analyze the objective and plan the approach"
        ))
        
        # Check if this looks like a coding task
        coding_keywords = ["implement", "build", "create", "develop", "code", "application", "app", "system"]
        is_coding_task = any(kw in objective.lower() for kw in coding_keywords)
        
        if is_coding_task and constraints.allow_code_execution:
            phases.append(MissionPhase(
                name="Design & Architecture",
                description="Design the solution architecture and components"
            ))
            phases.append(MissionPhase(
                name="Implementation",
                description="Implement the solution code"
            ))
            phases.append(MissionPhase(
                name="Testing & Validation",
                description="Test the implementation and validate correctness"
            ))
        else:
            phases.append(MissionPhase(
                name="Deep Analysis",
                description="Perform detailed analysis and evaluation"
            ))
        
        phases.append(MissionPhase(
            name="Synthesis & Report",
            description="Consolidate findings and produce final deliverables"
        ))
        
        return phases
    
    def _get_default_phases_with_steps(
        self,
        objective: str,
        constraints: MissionConstraints
    ) -> List[MissionPhase]:
        """Generate default phases with embedded step definitions."""
        phases = []
        
        # Reconnaissance phase
        if constraints.allow_internet:
            phases.append(MissionPhase(
                name="Reconnaissance",
                description="Gather context, background information, and relevant resources",
                steps=[
                    StepDefinition(
                        name="Collect key context",
                        description="Search for and gather relevant background information on the topic",
                        step_type="research",
                        tools=["web"],
                    ),
                    StepDefinition(
                        name="Summarize findings",
                        description="Synthesize gathered information into key insights",
                        step_type="analysis",
                    ),
                ]
            ))
        
        # Analysis & Planning phase
        phases.append(MissionPhase(
            name="Analysis & Planning",
            description="Analyze the objective and plan the approach",
            steps=[
                StepDefinition(
                    name="Analyze objective",
                    description="Break down the objective into key components and requirements",
                    step_type="analysis",
                ),
                StepDefinition(
                    name="Create execution plan",
                    description="Design a detailed plan to accomplish the objective",
                    step_type="design",
                ),
            ]
        ))
        
        # Check if this looks like a coding task
        coding_keywords = ["implement", "build", "create", "develop", "code", "application", "app", "system"]
        is_coding_task = any(kw in objective.lower() for kw in coding_keywords)
        
        if is_coding_task and constraints.allow_code_execution:
            # Design phase
            phases.append(MissionPhase(
                name="Design & Architecture",
                description="Design the solution architecture and components",
                steps=[
                    StepDefinition(
                        name="Design architecture",
                        description="Create the high-level architecture and component design",
                        step_type="design",
                    ),
                    StepDefinition(
                        name="Define interfaces",
                        description="Define APIs, interfaces, and data structures",
                        step_type="design",
                    ),
                ]
            ))
            
            # Implementation phase
            phases.append(MissionPhase(
                name="Implementation",
                description="Implement the solution code",
                steps=[
                    StepDefinition(
                        name="Implement core logic",
                        description="Write the main implementation code",
                        step_type="coding",
                    ),
                    StepDefinition(
                        name="Add error handling",
                        description="Implement error handling and edge cases",
                        step_type="coding",
                    ),
                ]
            ))
            
            # Testing phase
            phases.append(MissionPhase(
                name="Testing & Validation",
                description="Test the implementation and validate correctness",
                steps=[
                    StepDefinition(
                        name="Design test cases",
                        description="Create comprehensive test scenarios",
                        step_type="testing",
                    ),
                    StepDefinition(
                        name="Validate implementation",
                        description="Verify the code meets requirements",
                        step_type="testing",
                    ),
                ]
            ))
        else:
            # Deep analysis phase for non-coding tasks
            phases.append(MissionPhase(
                name="Deep Analysis",
                description="Perform detailed analysis and evaluation",
                steps=[
                    StepDefinition(
                        name="Detailed analysis",
                        description="Perform in-depth analysis of the subject matter",
                        step_type="analysis",
                    ),
                    StepDefinition(
                        name="Evaluate findings",
                        description="Assess and evaluate the analysis results",
                        step_type="analysis",
                    ),
                ]
            ))
        
        # Synthesis phase
        phases.append(MissionPhase(
            name="Synthesis & Report",
            description="Consolidate findings and produce final deliverables",
            steps=[
                StepDefinition(
                    name="Consolidate results",
                    description="Bring together all findings and outputs",
                    step_type="synthesis",
                ),
                StepDefinition(
                    name="Create deliverables",
                    description="Produce the final deliverables and report",
                    step_type="synthesis",
                ),
            ]
        ))
        
        return phases
    
    def run_until_complete_or_timeout(
        self,
        mission_id: str,
        heartbeat_callback: Optional[callable] = None,
    ) -> MissionState:
        """
        Main loop: runs the mission until finished, aborted, or time expired.
        
        Enhanced in 2.0 with:
        - Mission-level iterative execution
        - ReasoningSupervisor-driven convergence detection
        - Deepening loops between iterations
        - Context summarization across iterations
        
        Args:
            mission_id: ID of the mission to run
            heartbeat_callback: Optional callback called between phases
            
        Returns:
            Final MissionState (with status="failed" if execution encountered errors)
        """
        state = None
        try:
            state = self.store.load(mission_id)
            state.status = "running"
            state.log("Mission execution started")
            
            # Initialize iteration tracking
            if not hasattr(state, 'iteration_count'):
                state.iteration_count = 0
            
            # === Start time manager if available ===
            if self._mission_time_manager is not None:
                self._mission_time_manager.start_mission()
                state.log(
                    f"Time manager started: budget={self._mission_time_manager.total_budget_seconds:.0f}s, "
                    f"target_util={self._mission_time_manager.target_utilization:.0%}"
                )
            
            # Log initial GPU status if available
            if self.gpu_manager is not None:
                gpu_stats = self.gpu_manager.get_stats()
                state.log(f"Initial GPU status: {gpu_stats.free_mem}MB free, {gpu_stats.utilization}% utilization")
            
            # Log iterative execution mode
            if self.enable_iterative_execution:
                state.log(f"Iterative execution enabled (max {self.max_mission_iterations} iterations)")
            
            self.store.save(state)
            
            # Initialize convergence state
            self._convergence_state = ConvergenceState(
                min_iterations=2,
                min_time_guard_seconds=60.0
            )
            
            # Mission-level iteration loop with enhanced convergence logic
            while True:
                # Increment iteration count
                state.iteration_count += 1
                state.log(f"=== Mission Iteration {state.iteration_count} ===")
                
                # Check time budget
                if state.is_expired():
                    state.status = "expired"
                    state.log("Mission expired due to time budget")
                    self.store.save(state)
                    return state
                
                # Check iteration limit
                if state.iteration_count > self.max_mission_iterations:
                    state.log(f"Reached maximum iterations ({self.max_mission_iterations})")
                    break
                
                # Log current subgoals if any
                if self._current_subgoals:
                    state.log(f"Iteration {state.iteration_count} addressing subgoals: {', '.join(self._current_subgoals[:3])}")
                
                # Run all phases in this iteration
                self._run_all_phases_once(state, heartbeat_callback)
                
                # Resource cleanup after iteration to prevent socket/memory leaks
                try:
                    from deepthinker.models.model_caller import cleanup_resources, count_open_sockets
                    cleanup_resources()
                    socket_count = count_open_sockets()
                    if socket_count >= 0:
                        state.log(f"[Resource] Open sockets after iteration: {socket_count}")
                except ImportError:
                    pass  # model_caller not available
                
                # Check for terminal state
                if state.is_terminal():
                    return state
                
                # Get updated plan subgoals for next iteration
                self._extract_and_update_subgoals(state)
                
                # Check convergence using the new comprehensive logic
                time_remaining_seconds = state.remaining_time().total_seconds()
                total_time_seconds = state.constraints.time_budget_minutes * 60
                
                # Determine if we should continue iterating
                if not self.enable_iterative_execution:
                    # Single-pass mode
                    state.log("Single-pass mode - stopping")
                    break
                
                # Check minimum iterations requirement
                if state.iteration_count < self._convergence_state.min_iterations:
                    state.log(f"Minimum iterations not reached ({state.iteration_count} < {self._convergence_state.min_iterations})")
                    self._reset_phases_for_iteration(state)
                    self.store.save(state)
                    continue
                
                # Check time guard
                if time_remaining_seconds < self._convergence_state.min_time_guard_seconds:
                    state.log(f"Time guard reached ({time_remaining_seconds:.0f}s < {self._convergence_state.min_time_guard_seconds:.0f}s)")
                    break
                
                # Use comprehensive convergence check
                convergence_reached = False
                
                if self.reasoning_supervisor:
                    try:
                        # Analyze mission state
                        metrics = self.reasoning_supervisor.analyze_mission_state(state)
                        
                        # DeepThinker 2.0: Use STRICT convergence check (prioritizes unresolved questions)
                        if self._use_strict_convergence:
                            convergence_result = self.reasoning_supervisor.check_convergence_strict(
                                state=state,
                                evaluator_output=self._last_evaluator_output,
                                current_metrics=metrics if hasattr(metrics, 'novelty_score') else None,
                                multiview_disagreement=self._last_multiview_disagreement if self._last_multiview_disagreement else 0.0
                            )
                            convergence_reached = convergence_result.can_stop
                            
                            # Log detailed convergence status
                            if not convergence_reached:
                                state.log(f"[CONVERGENCE] Blocked: {convergence_result.blocking_criteria}")
                                if convergence_result.secondary_warnings:
                                    state.log(f"[CONVERGENCE] Warnings: {convergence_result.secondary_warnings}")
                            else:
                                state.log(f"[CONVERGENCE] Strict criteria met: unresolved={convergence_result.unresolved_count}")
                        else:
                            # Legacy convergence check
                            convergence_reached = self.reasoning_supervisor.check_convergence(
                                state=state,
                                evaluator_output=self._last_evaluator_output,
                                multiview_disagreement=self._last_multiview_disagreement,
                                updated_plan=state.updated_plan if hasattr(state, 'updated_plan') else None
                            )
                        
                        # Check if we should force another iteration (time+confidence heuristic)
                        if convergence_reached:
                            if self.reasoning_supervisor.should_force_iteration(state, self._last_evaluator_output):
                                convergence_reached = False
                                state.log("Forcing additional iteration due to time+confidence heuristic")
                        
                        # Log verbose metrics
                        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                            if hasattr(verbose_logger, 'log_mission_iteration'):
                                verbose_logger.log_mission_iteration(state.iteration_count, metrics)
                        
                        if not convergence_reached:
                            # Plan and execute deepening
                            deepening_plan = self.reasoning_supervisor.plan_deepening(state, metrics)
                            if deepening_plan and deepening_plan.has_work:
                                state.log(f"Deepening: {deepening_plan.reason}")
                                
                                # =========================================================
                                # Bio Prior Integration (anchor: after plan_deepening)
                                # =========================================================
                                if self._enable_bio_priors and self._bio_prior_engine:
                                    try:
                                        bio_output = self._evaluate_bio_priors(state, metrics, deepening_plan)
                                        if bio_output is not None:
                                            self._log_bio_prior_output(state, bio_output, deepening_plan)
                                    except Exception as bio_err:
                                        _orchestrator_logger.debug(f"BioPrior evaluation error: {bio_err}")
                                
                                self._execute_deepening(state, deepening_plan)
                                
                                # Log deepening
                                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                                    if hasattr(verbose_logger, 'log_deepening_plan'):
                                        verbose_logger.log_deepening_plan(deepening_plan)
                            
                            # Reset phase indices for next iteration
                            self._reset_phases_for_iteration(state)
                        
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"ReasoningSupervisor error: {e}")
                        # Fall back to basic checks
                        convergence_reached = self._basic_convergence_check(state)
                else:
                    # No supervisor - use basic convergence check
                    convergence_reached = self._basic_convergence_check(state)
                    
                    if not convergence_reached:
                        self._reset_phases_for_iteration(state)
                
                # Update convergence state
                self._convergence_state.convergence_reached = convergence_reached
                
                if convergence_reached:
                    state.log("Convergence reached - stopping iterations")
                    break
                
                # Checkpoint after iteration
                self.store.save(state)
            
            # All iterations complete - run final synthesis
            self._run_final_synthesis(state)
            
            # Verbose logging: mission artifact summary
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_mission_artifact_summary(state.phases)
                verbose_logger.log_flow_summary()
            
            # Generate final deliverables
            self._generate_final_deliverables(state)
            
            # Build work summary
            state.work_summary.update({
                "phase_rounds": dict(state.phase_rounds),
                "council_rounds": dict(state.council_rounds),
                "effort_level": state.constraints.effort.value if hasattr(state.constraints, 'effort') and hasattr(state.constraints.effort, 'value') else "unknown",
                "time_used_minutes": state.constraints.time_budget_minutes - state.remaining_minutes(),
                "phases_completed": len([p for p in state.phases if p.status == "completed"]),
                "total_phases": len(state.phases),
                "iterations_completed": state.iteration_count,
            })
            
            # Finalize memory system
            self._finalize_memory(state)
            
            # === Log time utilization ===
            if self._mission_time_manager is not None:
                utilization = self._mission_time_manager.get_utilization()
                util_status = self._mission_time_manager.get_utilization_status()
                state.log(
                    f"Time utilization: {utilization:.0%} "
                    f"({util_status['time_used_seconds']:.0f}s used of {self._mission_time_manager.total_budget_seconds:.0f}s budget)"
                )
                if utilization < self._target_utilization:
                    state.log(
                        f"[NOTE] Utilization below target ({self._target_utilization:.0%}). "
                        f"Mission may benefit from deeper analysis."
                    )
            
            # === Log ML Governance Summary ===
            ml_report = None
            try:
                from ..observability.ml_influence import MLInfluenceReporter
                reporter = MLInfluenceReporter()
                ml_report = reporter.generate_system_report()
                
                if ml_report.get("status") != "no_data":
                    if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                        verbose_logger.log_ml_governance_panel(
                            system_health=ml_report.get("system_health_score"),
                            predictor_status=ml_report.get("predictor_reports"),
                            recent_alerts=ml_report.get("recent_alerts"),
                            advisory_readiness=ml_report.get("advisory_mode_readiness"),
                        )
                    
                    # === SSE: Publish ML governance update for frontend ===
                    if SSE_AVAILABLE and sse_manager:
                        advisory = ml_report.get("advisory_mode_readiness", {})
                        _publish_sse_event(sse_manager.publish_ml_governance(
                            mission_id=state.mission_id,
                            system_health=ml_report.get("system_health_score", 0.0),
                            predictor_status=ml_report.get("predictor_reports", {}),
                            advisory_readiness=advisory.get("overall_score", 0.0) if isinstance(advisory, dict) else 0.0,
                            advisory_ready=advisory.get("ready", False) if isinstance(advisory, dict) else False,
                            alerts=ml_report.get("recent_alerts", []),
                        ))
            except Exception as e:
                _orchestrator_logger.debug(f"ML governance logging failed: {e}")
            
            # === Add Normative Governance Summary to Final Report ===
            governance_summary = self.get_governance_summary()
            state.final_artifacts["governance_summary"] = governance_summary
            state.log(
                f"[GOVERNANCE] Mission summary: "
                f"epistemic_risk={governance_summary.get('epistemic_risk_score', 0):.2f}, "
                f"violations={governance_summary.get('total_violations', 0)}, "
                f"blocked_phases={governance_summary.get('phases_blocked', 0)}"
            )
            
            # Reset governance controller for next mission
            if self._normative_controller is not None:
                self._normative_controller.reset()
            
            # Finalize alignment tracking
            self._finalize_alignment(state)
            
            # Extract claims from final synthesis (if HF instruments enabled)
            self._extract_mission_claims(state)
            
            state.status = "completed"
            state.log(f"Mission completed successfully after {state.iteration_count} iteration(s)")
            self.store.save(state)
            return state
            
        except Exception as e:
            # Mission-level exception boundary - ensures graceful failure
            import traceback
            error_trace = traceback.format_exc()
            error_msg = str(e)
            
            _orchestrator_logger.error(
                f"Mission {mission_id} failed with unhandled exception: {error_msg}\n{error_trace}"
            )
            
            # Ensure we have a state to update
            if state is None:
                try:
                    state = self.store.load(mission_id)
                except Exception:
                    # Cannot load state - create minimal failed state
                    _orchestrator_logger.error(f"Cannot load state for failed mission {mission_id}")
                    raise  # Re-raise if we can't even load state
            
            # Set failure information
            state.set_failed(
                reason=MissionFailureReason.UNKNOWN_ERROR.value,
                details={
                    "error_type": type(e).__name__,
                    "traceback": error_trace[:2000],  # Limit traceback size
                    "phase_index": state.current_phase_index,
                    "iteration": getattr(state, 'iteration_count', 0),
                },
                error_message=f"Unhandled error: {error_msg}"
            )
            
            # Persist failed state
            self.store.save(state)
            
            # Return failed state instead of raising (graceful degradation)
            return state
    
    def _run_all_phases_once(
        self,
        state: MissionState,
        heartbeat_callback: Optional[callable] = None
    ) -> None:
        """
        Run all phases once in sequence.
        
        Phase 3.1: Includes model prefetching for next phase.
        
        Args:
            state: Current mission state
            heartbeat_callback: Optional heartbeat callback
        """
        while True:
            # Check time budget
            if state.is_expired():
                state.status = "expired"
                state.log("Mission expired during phase execution")
                self.store.save(state)
                return
            
            # Get current phase
            phase = state.current_phase()
            if phase is None:
                # All phases completed for this iteration
                return
            
            # If phase is already in a terminal state, advance past it immediately
            # This prevents infinite loops when phases are already failed/completed/skipped
            # from previous iterations or resets
            if phase.status in ("completed", "failed", "skipped"):
                state.advance_phase()
                self.store.save(state)
                if heartbeat_callback:
                    heartbeat_callback(state)
                # Check for terminal state after advancing
                if state.is_terminal():
                    return
                continue  # Skip to next phase
            
            # Check if we have enough time for this phase
            if state.remaining_minutes() < MIN_PHASE_TIME_MINUTES:
                phase.mark_skipped(f"Insufficient time remaining ({state.remaining_minutes():.1f} min)")
                state.log(f"Skipping phase '{phase.name}' - insufficient time")
                state.advance_phase()
                self.store.save(state)
                continue
            
            # Phase 3.1: Wait for any pending prefetch before starting this phase
            self._wait_for_prefetch(timeout=5.0)
            
            # Run the phase
            if phase.status in ("pending", "running"):
                state.log(f"Starting phase: {phase.name}")
                
                # Phase 3.1: Start prefetching models for next phase in background
                self._prefetch_thread = self._prefetch_next_phase_models(
                    state, state.current_phase_index
                )
                
                self._run_phase(state, phase)
                
                # Advance to next phase if completed
                if phase.status in ("completed", "failed", "skipped"):
                    # Track consecutive failures to prevent infinite loops
                    if phase.status == "failed":
                        failure_count = self._phase_failure_counts.get(phase.name, 0) + 1
                        self._phase_failure_counts[phase.name] = failure_count
                        
                        if failure_count >= self._max_consecutive_failures:
                            state.log(
                                f"[SAFEGUARD] Phase '{phase.name}' failed {failure_count} times consecutively, "
                                f"skipping to prevent infinite loop"
                            )
                            # Mark as skipped instead of failed to allow mission to continue
                            phase.mark_skipped(f"Too many consecutive failures ({failure_count})")
                    else:
                        # Reset failure count on success or skip
                        self._phase_failure_counts.pop(phase.name, None)
                    
                    state.advance_phase()
            
            # Checkpoint
            self.store.save(state)
            
            # Call heartbeat if provided
            if heartbeat_callback:
                heartbeat_callback(state)
            
            # Check for terminal state
            if state.is_terminal():
                return
    
    def _reset_phases_for_iteration(self, state: MissionState) -> None:
        """
        Reset phases to allow re-execution in next iteration.
        
        Only resets phases that can benefit from additional passes.
        Keeps artifacts from previous iterations.
        
        Args:
            state: Current mission state
        """
        state.current_phase_index = 0
        
        for phase in state.phases:
            if phase.status == "completed":
                # Keep artifacts but allow re-running
                phase.status = "pending"
                phase.started_at = None
                phase.ended_at = None
                # Note: artifacts are preserved for context
        
        # Reset failure counts for fresh start in new iteration
        self._phase_failure_counts.clear()
        
        state.log("Phases reset for next iteration")
    
    def _basic_convergence_check(self, state: MissionState) -> bool:
        """
        Basic convergence check when ReasoningSupervisor is unavailable.
        
        Args:
            state: Current mission state
            
        Returns:
            True if should stop iterating
        """
        # Check evaluator output
        if self._last_evaluator_output:
            # Has unresolved issues?
            if hasattr(self._last_evaluator_output, 'iteration_should_continue'):
                if self._last_evaluator_output.iteration_should_continue():
                    return False
            
            # Low quality score?
            if hasattr(self._last_evaluator_output, 'quality_score'):
                if self._last_evaluator_output.quality_score < 6.0:
                    return False
        
        # Check multiview disagreement
        if self._last_multiview_disagreement > 0.25:
            return False
        
        # Check pending subgoals
        if self._current_subgoals:
            return False
        
        return True
    
    # =========================================================================
    # Bio Prior Integration Methods
    # =========================================================================
    
    def _evaluate_bio_priors(
        self,
        state: MissionState,
        metrics: Any,
        deepening_plan: Any,
    ) -> Optional["BioPriorOutput"]:
        """
        Evaluate bio priors against current context.
        
        Pure evaluation - no side effects on state or plan.
        
        Args:
            state: Current mission state
            metrics: Mission metrics from supervisor
            deepening_plan: Current deepening plan
            
        Returns:
            BioPriorOutput if evaluation succeeds, None otherwise
        """
        if not self._enable_bio_priors or not self._bio_prior_engine:
            return None
        
        if not BIO_PRIORS_AVAILABLE or build_bio_context is None:
            return None
        
        try:
            # Build context from state
            bio_ctx = build_bio_context(state, metrics)
            
            # Evaluate (pure, deterministic)
            bio_output = self._bio_prior_engine.evaluate(bio_ctx)
            
            return bio_output
            
        except Exception as e:
            _orchestrator_logger.debug(f"BioPrior context build error: {e}")
            return None
    
    def _log_bio_prior_output(
        self,
        state: MissionState,
        bio_output: "BioPriorOutput",
        deepening_plan: Any,
    ) -> None:
        """
        Log bio prior output and optionally apply signals.
        
        Mode behavior:
        - advisory: Log only, no application
        - shadow: Log + compute "would_apply" diff, no application
        - soft: Log + apply bounded modifications (v1: depth_budget_delta only)
        
        Args:
            state: Current mission state
            bio_output: Output from bio prior evaluation
            deepening_plan: Deepening plan to potentially modify
        """
        if bio_output is None:
            return
        
        # Log the bio prior evaluation event
        state.log_event("bio_prior_evaluation", {
            "mode": bio_output.mode,
            "selected_patterns": bio_output.selected_patterns,
            "signals": bio_output.signals.to_dict(),
            "advisory_text": bio_output.advisory_text,
            "applied": bio_output.applied,
            "applied_fields": bio_output.applied_fields,
            "trace": bio_output.trace,
        })
        
        # Shadow mode: compute and log "would_apply" diff
        if bio_output.mode == "shadow":
            if compute_would_apply_diff is not None and deepening_plan:
                try:
                    would_apply = compute_would_apply_diff(deepening_plan, bio_output.signals)
                    state.log_event("bio_prior_shadow_diff", would_apply)
                except Exception as e:
                    _orchestrator_logger.debug(f"BioPrior shadow diff error: {e}")
        
        # Soft mode: apply bounded modifications
        elif bio_output.mode == "soft":
            if apply_bio_pressures_to_deepening_plan is not None and deepening_plan:
                try:
                    applied_fields = apply_bio_pressures_to_deepening_plan(
                        deepening_plan,
                        bio_output.signals,
                    )
                    
                    # Log constitution event (tagged non-evidence)
                    self._log_prior_influence_event(state, bio_output, applied_fields)
                    
                    if applied_fields:
                        state.log(f"[BIO_PRIORS] Applied: {applied_fields}")
                        
                except Exception as e:
                    _orchestrator_logger.debug(f"BioPrior soft apply error: {e}")
        
        # Log advisory text in debug
        if bio_output.selected_patterns:
            _orchestrator_logger.debug(
                f"BioPrior patterns: {[p['id'] for p in bio_output.selected_patterns]}"
            )
    
    def _log_prior_influence_event(
        self,
        state: MissionState,
        bio_output: "BioPriorOutput",
        applied_fields: List[str],
    ) -> None:
        """
        Log a PRIOR_INFLUENCE constitution event.
        
        This event is explicitly tagged as NON-EVIDENCE so that
        constitution invariants ignore it for confidence checks.
        
        Args:
            state: Current mission state
            bio_output: Bio prior output
            applied_fields: Fields that were actually applied
        """
        if not BIO_PRIORS_AVAILABLE or PriorInfluenceEvent is None:
            return
        
        try:
            # Create the prior influence event
            event = PriorInfluenceEvent(
                mission_id=state.mission_id,
                phase_id=state.current_phase().id if state.current_phase() else "",
                source="bio_priors",
                mode=bio_output.mode,
                selected_patterns=[p["id"] for p in bio_output.selected_patterns],
                signals_applied=bio_output.signals.to_dict(),
                applied_fields=applied_fields,
                context_snapshot=bio_output.trace.get("context_snapshot", {}),
            )
            
            # Log to constitution ledger if available
            if self._constitution_engine is not None:
                if hasattr(self._constitution_engine, 'log_event'):
                    self._constitution_engine.log_event(event)
            
            # Also log via state's event system
            state.log_event("constitution_prior_influence", event.to_dict())
            
        except Exception as e:
            _orchestrator_logger.debug(f"PriorInfluenceEvent logging error: {e}")
    
    def _extract_and_update_subgoals(self, state: MissionState) -> None:
        """
        Extract subgoals from planner output for next iteration.
        
        Looks in phase artifacts and updated_plan for new_subgoals.
        
        Args:
            state: Current mission state
        """
        subgoals = []
        
        # Check for subgoals in planner phase artifacts
        for phase in state.phases:
            phase_type = self._classify_phase(phase)
            if phase_type == "design" and phase.status == "completed":
                # Check for new_subgoals in artifacts
                if "new_subgoals" in phase.artifacts:
                    try:
                        import json
                        goals = json.loads(phase.artifacts["new_subgoals"])
                        if isinstance(goals, list):
                            subgoals.extend(goals)
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Check state.updated_plan if available
        if hasattr(state, 'updated_plan') and state.updated_plan:
            if isinstance(state.updated_plan, dict):
                plan_subgoals = state.updated_plan.get('new_subgoals', [])
                if plan_subgoals:
                    subgoals.extend(plan_subgoals)
        
        # Also check evaluator data_needs as implicit subgoals
        if self._last_evaluator_output:
            if hasattr(self._last_evaluator_output, 'data_needs'):
                for need in self._last_evaluator_output.data_needs[:2]:
                    if need not in subgoals:
                        subgoals.append(f"Address data need: {need}")
        
        # Update current subgoals (top 3 priority)
        self._current_subgoals = subgoals[:3]
        
        # Update convergence state
        if self._convergence_state:
            self._convergence_state.pending_subgoals = len(subgoals)
        
        if subgoals:
            state.log(f"Identified {len(subgoals)} subgoals for next iteration")
    
    def _extract_planner_context(
        self,
        state: MissionState
    ) -> Tuple[List[str], Optional[str]]:
        """
        Extract focus areas and requirements from planner phase artifacts.
        
        This ensures research context receives proper guidance from planning phases.
        Fixes Issue #7: Empty research context due to missing planner mapping.
        
        Args:
            state: Current mission state
            
        Returns:
            Tuple of (focus_areas list, planner_requirements string or None)
        """
        focus_areas = []
        planner_requirements = None
        
        # Extract from completed design/planning phases
        for phase in state.phases:
            if phase.status != "completed":
                continue
            
            phase_type = self._classify_phase(phase)
            if phase_type not in ["design", "planning"]:
                continue
            
            artifacts = phase.artifacts
            
            # Extract focus areas
            if "focus_areas" in artifacts:
                try:
                    import json
                    areas = json.loads(artifacts["focus_areas"]) if isinstance(artifacts["focus_areas"], str) else artifacts["focus_areas"]
                    if isinstance(areas, list):
                        focus_areas.extend(areas)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Also check for priorities or key_areas
            for key in ["priorities", "key_areas", "research_areas"]:
                if key in artifacts:
                    try:
                        import json
                        items = json.loads(artifacts[key]) if isinstance(artifacts[key], str) else artifacts[key]
                        if isinstance(items, list):
                            focus_areas.extend(items)
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            # Extract requirements text
            for key in ["requirements", "planner_output", "plan_summary"]:
                if key in artifacts and isinstance(artifacts[key], str):
                    if len(artifacts[key]) > 50:  # Non-trivial content
                        planner_requirements = artifacts[key][:2000]  # Truncate
                        break
        
        # Fallback: extract from current subgoals if no focus areas found
        if not focus_areas and self._current_subgoals:
            focus_areas = self._current_subgoals.copy()
        
        # Fallback: extract from iteration context manager
        if not focus_areas and self._iteration_context_manager:
            ctx_updates = self._iteration_context_manager.get_research_context_updates()
            if ctx_updates.get("focus_areas"):
                focus_areas = ctx_updates["focus_areas"]
        
        # Deduplicate and limit
        seen = set()
        unique_areas = []
        for area in focus_areas:
            if area and area not in seen:
                seen.add(area)
                unique_areas.append(area)
        
        return unique_areas[:5], planner_requirements
    
    def _compute_multiview_disagreement(
        self,
        optimist_output: Any,
        skeptic_output: Any
    ) -> float:
        """
        Compute semantic disagreement between optimist and skeptic perspectives.
        
        Uses embedding-based cosine similarity to measure disagreement.
        Disagreement = 1 - similarity
        
        Args:
            optimist_output: Output from OptimistCouncil
            skeptic_output: Output from SkepticCouncil
            
        Returns:
            Disagreement score (0 = agreement, 1 = complete disagreement)
        """
        if optimist_output is None or skeptic_output is None:
            return 0.0
        
        # Extract text from outputs
        opt_text = ""
        skep_text = ""
        
        if hasattr(optimist_output, 'raw_output'):
            opt_text = optimist_output.raw_output
        elif hasattr(optimist_output, 'reasoning'):
            opt_text = str(optimist_output.reasoning)
        else:
            opt_text = str(optimist_output)
        
        if hasattr(skeptic_output, 'raw_output'):
            skep_text = skeptic_output.raw_output
        elif hasattr(skeptic_output, 'reasoning'):
            skep_text = str(skeptic_output.reasoning)
        else:
            skep_text = str(skeptic_output)
        
        # Use semantic distance consensus if available
        if self._has_semantic_consensus and self._semantic_consensus:
            try:
                # Get embeddings
                opt_emb = self._semantic_consensus._get_embedding(opt_text[:2000])
                skep_emb = self._semantic_consensus._get_embedding(skep_text[:2000])
                
                if opt_emb and skep_emb:
                    similarity = self._semantic_consensus._cosine_similarity(opt_emb, skep_emb)
                    disagreement = 1.0 - similarity
                    return max(0.0, min(1.0, disagreement))
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Semantic disagreement calculation failed: {e}")
        
        # Fallback: simple confidence-based disagreement
        opt_conf = getattr(optimist_output, 'confidence', 0.5)
        skep_conf = getattr(skeptic_output, 'confidence', 0.5)
        
        # High disagreement if confidences are both high but opposing
        # (optimist very confident, skeptic very confident)
        if opt_conf > 0.7 and skep_conf > 0.7:
            return 0.6  # Likely disagreement
        
        # Moderate disagreement if one is confident and other uncertain
        if abs(opt_conf - skep_conf) > 0.4:
            return 0.3
        
        return 0.1  # Low disagreement by default
    
    def _handle_multiview_disagreement(
        self,
        state: MissionState,
        phase: MissionPhase,
        disagreement: float,
        optimist_output: Any,
        skeptic_output: Any
    ) -> None:
        """
        Handle high multiview disagreement by triggering deeper investigation.
        
        When disagreement > 0.25:
        - Adds planner subgoal to resolve disagreement
        - Logs the disagreement for tracking
        - Prevents convergence in this iteration
        
        Args:
            state: Current mission state
            phase: Current phase
            disagreement: Disagreement score (0-1)
            optimist_output: Optimist perspective
            skeptic_output: Skeptic perspective
        """
        if disagreement <= 0.25:
            return
        
        # Log the disagreement
        state.log(f"High multiview disagreement detected ({disagreement:.2f} > 0.25)")
        
        # Extract key points of disagreement
        opt_points = []
        skep_points = []
        
        if hasattr(optimist_output, 'opportunities'):
            opt_points = optimist_output.opportunities[:2]
        if hasattr(skeptic_output, 'concerns'):
            skep_points = skeptic_output.concerns[:2]
        
        # Add subgoal to resolve disagreement
        if opt_points or skep_points:
            disagreement_topic = opt_points[0] if opt_points else (skep_points[0] if skep_points else "the analysis")
            subgoal = f"Resolve disagreement on: {disagreement_topic[:100]} by gathering additional evidence"
            
            if subgoal not in self._current_subgoals:
                self._current_subgoals.append(subgoal)
        
        # Store disagreement details
        disagreement_record = {
            "phase": phase.name,
            "disagreement_score": disagreement,
            "optimist_summary": str(opt_points)[:200] if opt_points else "",
            "skeptic_summary": str(skep_points)[:200] if skep_points else "",
            "agreement_score": 1 - disagreement
        }
        self._multiview_disagreements.append(disagreement_record)
        
        # Store in phase artifacts
        phase.artifacts["multiview_disagreement"] = f"{disagreement:.2f}"
        
        # Update convergence state
        if self._convergence_state:
            self._convergence_state.multiview_disagreement = disagreement
    
    def _update_research_iteration_context(
        self,
        state: MissionState,
        phase: MissionPhase,
        findings: Any
    ) -> None:
        """
        Update iteration context manager with research findings.
        
        This ensures the next iteration receives evolved context with:
        - Accumulated prior_knowledge
        - Updated focus_areas from gaps
        - Data_needs from evidence_requests
        
        Args:
            state: Current mission state
            phase: Current phase
            findings: ResearchFindings from the council
        """
        if not ITERATION_CONTEXT_AVAILABLE or self._iteration_context_manager is None:
            return
        
        try:
            # Initialize if needed
            if self._iteration_context_manager.get_state() is None:
                self._iteration_context_manager.initialize()
            
            # Update context with findings
            delta = self._iteration_context_manager.update_from_research(
                findings=findings,
                evaluation=None  # Will be updated separately if evaluation runs
            )
            
            # Log context evolution
            state.log(f"Context evolution: {delta.summary()}")
            
            # Store delta info in phase artifacts
            phase.artifacts["context_delta"] = delta.summary()
            
            # Get current state for detailed logging
            ctx_state = self._iteration_context_manager.get_state()
            if ctx_state is not None:
                # Use the new structured logging method
                state.log_context_evolution(
                    iteration=ctx_state.current_iteration,
                    focus_areas=ctx_state.focus_areas,
                    unresolved_questions=ctx_state.unresolved_questions,
                    data_needs=ctx_state.data_needs,
                    web_searches=getattr(findings, 'web_search_count', 0),
                    prior_knowledge_size=len(ctx_state.prior_knowledge),
                    delta_summary=delta.summary()
                )
                
                # Log web search if performed
                if hasattr(findings, 'web_search_count') and findings.web_search_count > 0:
                    queries = getattr(findings, 'queries_executed', [])
                    state.log_web_search(
                        phase_name=phase.name,
                        queries=queries,
                        results_count=findings.web_search_count,
                        triggered_by="data_needs" if ctx_state.data_needs else "iteration"
                    )
                
                # Phase 6.1: Update total_web_searches from SearchTriggerManager (single source of truth)
                if self._search_trigger_manager:
                    stats = self._search_trigger_manager.get_search_stats()
                    state.total_web_searches = stats.get("total_searches", 0)
            
            # Update mission state with new context values
            ctx_updates = self._iteration_context_manager.get_research_context_updates()
            if ctx_updates.get("focus_areas"):
                self._current_subgoals = ctx_updates.get("subgoals", [])
                state.log(f"Updated subgoals: {len(self._current_subgoals)}")
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Context evolution update failed: {e}")
    
    def _evaluate_and_update_research_context(
        self,
        state: MissionState,
        phase: MissionPhase,
        findings: Any
    ) -> None:
        """
        Evaluate research findings and update context for next iteration.
        
        Uses EvaluatorCouncil's evaluate_research method to produce
        actionable deltas that drive the next research iteration.
        
        Args:
            state: Current mission state
            phase: Current phase
            findings: ResearchFindings from researcher council
        """
        if not RESEARCH_EVALUATION_AVAILABLE:
            return
        
        # Only evaluate if we have time
        if state.remaining_minutes() < 0.5:
            return
        
        try:
            # Get prior gaps for comparison
            prior_gaps = []
            if hasattr(findings, 'gaps') and findings.gaps:
                prior_gaps = findings.gaps
            
            # Run research evaluation
            eval_result = self.evaluator.evaluate_research(
                objective=state.objective,
                research_findings=findings,
                iteration=phase.iterations,
                prior_gaps=prior_gaps
            )
            
            if eval_result.success and eval_result.output:
                evaluation = eval_result.output
                
                # Store evaluation artifacts
                phase.artifacts["research_evaluation"] = str(evaluation.raw_output)[:1000] if hasattr(evaluation, 'raw_output') else ""
                phase.artifacts["research_completeness"] = f"{getattr(evaluation, 'completeness_score', 0.5):.2f}"
                phase.artifacts["should_continue_research"] = str(getattr(evaluation, 'should_continue', True))
                
                # Log evaluation results
                completeness = getattr(evaluation, 'completeness_score', 0.5)
                gaps = getattr(evaluation, 'gaps', [])
                should_continue = getattr(evaluation, 'should_continue', True)
                
                state.log(
                    f"Research evaluation: completeness={completeness:.2f}, "
                    f"gaps={len(gaps)}, "
                    f"continue={should_continue}"
                )
                
                # === SSE: Publish research progress for frontend ===
                if SSE_AVAILABLE and sse_manager:
                    key_points = getattr(evaluation, 'key_points', [])
                    confidence = getattr(evaluation, 'confidence', 0.5)
                    context_delta = phase.artifacts.get("context_delta", "")
                    
                    _publish_sse_event(sse_manager.publish_research_progress(
                        mission_id=state.mission_id,
                        phase_name=phase.name,
                        iteration=phase.iterations,
                        completeness=completeness,
                        should_continue=should_continue,
                        context_delta=context_delta,
                        web_searches_performed=state.web_searches_performed if hasattr(state, 'web_searches_performed') else 0,
                        key_points_count=len(key_points) if key_points else 0,
                        gaps_count=len(gaps),
                        confidence_score=confidence,
                    ))
                
                # Update iteration context with evaluation
                if ITERATION_CONTEXT_AVAILABLE and self._iteration_context_manager is not None:
                    delta = self._iteration_context_manager.update_from_research(
                        findings=findings,
                        evaluation=evaluation
                    )
                    phase.artifacts["context_delta_with_eval"] = delta.summary()
                
                # Store for convergence tracking
                self._last_evaluator_output = evaluation
                if self._convergence_state:
                    self._convergence_state.update_from_evaluator(evaluation)
                
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Research evaluation failed: {e}")
    
    def _extract_and_apply_multiview_disagreements(
        self,
        state: MissionState,
        phase: MissionPhase,
        optimist_output: Any,
        skeptic_output: Any
    ) -> None:
        """
        Extract disagreements between Optimist and Skeptic and apply to context.
        
        Disagreements become:
        - unresolved_questions for next iteration
        - evidence_requests for web search
        - next_focus_areas for deeper investigation
        
        Args:
            state: Current mission state
            phase: Current phase
            optimist_output: OptimistPerspective
            skeptic_output: SkepticPerspective
        """
        if not MULTIVIEW_COUNCILS_AVAILABLE or extract_disagreements is None:
            return
        
        try:
            # Extract structured disagreements
            disagreement = extract_disagreements(optimist_output, skeptic_output)
            
            # Phase 7.3: Require contested claims when disagreement > 0.25
            if disagreement.agreement_score < 0.5:  # High disagreement (agreement < 0.5 means disagreement > 0.5)
                if (len(disagreement.contested_risks) == 0 and 
                    len(disagreement.contested_opportunities) == 0):
                    # Extract from raw outputs if needed
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"[MULTIVIEW] High disagreement ({disagreement.agreement_score:.2f}) but no contested claims. "
                        "Extracting from raw outputs..."
                    )
                    
                    # Extract contested claims from raw outputs using keyword matching
                    contested_risks, contested_opps = self._extract_contested_claims_from_raw(
                        optimist_output, skeptic_output
                    )
                    
                    if contested_risks or contested_opps:
                        disagreement.contested_risks = contested_risks
                        disagreement.contested_opportunities = contested_opps
                        logger.info(
                            f"[MULTIVIEW] Extracted {len(contested_risks)} risks and "
                            f"{len(contested_opps)} opportunities from raw outputs"
                        )
                    else:
                        logger.warning(
                            f"[MULTIVIEW] Failed to extract contested claims from raw outputs. "
                            "Disagreement analysis may be incomplete."
                        )
            
            # Log disagreement details
            state.log(f"Multi-view disagreement analysis: {disagreement.summary()}")
            
            # Store in phase artifacts
            phase.artifacts["multiview_agreement_score"] = f"{disagreement.agreement_score:.2f}"
            if disagreement.contested_risks:
                phase.artifacts["contested_risks"] = "\n".join(f"- {r}" for r in disagreement.contested_risks[:3])
            if disagreement.contested_opportunities:
                phase.artifacts["contested_opportunities"] = "\n".join(f"- {o}" for o in disagreement.contested_opportunities[:3])
            
            # Apply to iteration context
            if ITERATION_CONTEXT_AVAILABLE and self._iteration_context_manager is not None:
                ctx_state = self._iteration_context_manager.get_state()
                if ctx_state is not None:
                    # Add disagreement-derived items to context
                    ctx_state.unresolved_questions.extend(disagreement.unresolved_questions)
                    ctx_state.evidence_requests.extend(disagreement.evidence_requests)
                    ctx_state.focus_areas.extend(disagreement.next_focus_areas)
                    
                    # Deduplicate
                    ctx_state.unresolved_questions = list(set(ctx_state.unresolved_questions))[:10]
                    ctx_state.evidence_requests = list(set(ctx_state.evidence_requests))[:10]
                    ctx_state.focus_areas = list(set(ctx_state.focus_areas))[:10]
            
            # Add to current subgoals
            for question in disagreement.unresolved_questions[:2]:
                if question not in self._current_subgoals:
                    self._current_subgoals.append(question)
            
            # Log multiview disagreement with structured logging
            state.log_multiview_disagreement(
                phase_name=phase.name,
                agreement_score=disagreement.agreement_score,
                contested_risks=disagreement.contested_risks,
                contested_opportunities=disagreement.contested_opportunities,
                unresolved_from_disagreement=disagreement.unresolved_questions
            )
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to extract and apply multiview disagreements: {e}")
    
    def _extract_contested_claims_from_raw(
        self,
        optimist_output: Any,
        skeptic_output: Any
    ) -> Tuple[List[str], List[str]]:
        """
        Extract contested claims from raw outputs when structured extraction fails.
        
        Phase 7.3: Keyword-based extraction for high disagreement scenarios.
        
        Args:
            optimist_output: OptimistPerspective
            skeptic_output: SkepticPerspective
            
        Returns:
            Tuple of (contested_risks, contested_opportunities)
        """
        contested_risks = []
        contested_opportunities = []
        
        # Get raw output strings
        optimist_raw = getattr(optimist_output, 'raw_output', '') or str(optimist_output)
        skeptic_raw = getattr(skeptic_output, 'raw_output', '') or str(skeptic_output)
        
        # Look for sentences with opposing confidence markers
        import re
        
        # Find risk mentions in skeptic output
        risk_patterns = [
            r'(?:high|significant|major|critical|serious)\s+risk[^.]*\.',
            r'risk[^.]*(?:high|significant|major|critical|serious)[^.]*\.',
            r'could fail[^.]*\.',
            r'failure mode[^.]*\.'
        ]
        
        skeptic_risks = []
        for pattern in risk_patterns:
            matches = re.findall(pattern, skeptic_raw, re.IGNORECASE)
            skeptic_risks.extend([m.strip() for m in matches[:5]])  # Limit to 5 per pattern
        
        # Find opportunity mentions in optimist output
        opp_patterns = [
            r'(?:significant|major|high|strong)\s+opportunity[^.]*\.',
            r'opportunity[^.]*(?:significant|major|high|strong)[^.]*\.',
            r'could succeed[^.]*\.',
            r'success factor[^.]*\.'
        ]
        
        optimist_opps = []
        for pattern in opp_patterns:
            matches = re.findall(pattern, optimist_raw, re.IGNORECASE)
            optimist_opps.extend([m.strip() for m in matches[:5]])  # Limit to 5 per pattern
        
        # Check if optimist dismisses skeptic's risks
        for risk in skeptic_risks[:3]:  # Check top 3 risks
            risk_keywords = set(re.findall(r'\b\w{4,}\b', risk.lower()))  # Extract meaningful words
            # Check if optimist output contains opposing language
            if any(
                keyword in optimist_raw.lower() and 
                any(opp_word in optimist_raw.lower() for opp_word in ['low risk', 'minimal', 'unlikely', 'overstated'])
                for keyword in risk_keywords
                if len(keyword) > 4
            ):
                contested_risks.append(risk[:200])  # Limit length
        
        # Check if skeptic dismisses optimist's opportunities
        for opp in optimist_opps[:3]:  # Check top 3 opportunities
            opp_keywords = set(re.findall(r'\b\w{4,}\b', opp.lower()))  # Extract meaningful words
            # Check if skeptic output contains opposing language
            if any(
                keyword in skeptic_raw.lower() and 
                any(skept_word in skeptic_raw.lower() for skept_word in ['unlikely', 'overstated', 'risky', 'doubtful'])
                for keyword in opp_keywords
                if len(keyword) > 4
            ):
                contested_opportunities.append(opp[:200])  # Limit length
        
        return contested_risks[:3], contested_opportunities[:3]  # Return max 3 of each
    
    # =========================================================================
    # Depth Control Methods
    # =========================================================================
    
    def _gather_phase_output_for_depth(self, phase: MissionPhase) -> str:
        """
        Gather phase output text for depth evaluation.
        
        Combines relevant artifacts into a single text for depth scoring.
        
        Args:
            phase: The phase to gather output from
            
        Returns:
            Combined output text string
        """
        if not phase.artifacts:
            return ""
        
        # Prioritize certain artifact types for depth evaluation
        priority_keys = [
            "research_findings",
            "analysis",
            "deep_analysis",
            "synthesis",
            "findings",
            "recommendations",
            "plan",
            "design",
            "code",
            "evaluation",
            "enrichment_output",
        ]
        
        parts = []
        
        # Add priority artifacts first
        for key in priority_keys:
            if key in phase.artifacts:
                value = phase.artifacts[key]
                if value and isinstance(value, str) and len(value) > 50:
                    parts.append(value)
        
        # Add other artifacts
        for key, value in phase.artifacts.items():
            if key.startswith("_"):  # Skip internal fields
                continue
            if key in priority_keys:  # Already added
                continue
            if key in ("governance_verdict", "governance_violations", "depth_achieved", 
                      "depth_target", "depth_gap", "enrichment_passes"):
                continue  # Skip metadata
            if value and isinstance(value, str) and len(value) > 50:
                parts.append(value)
        
        # Combine with reasonable truncation
        combined = "\n\n".join(parts)
        
        # Truncate if too long (depth eval works on representative sample)
        if len(combined) > 10000:
            combined = combined[:10000]
        
        return combined
    
    def _run_enrichment_pass(
        self,
        state: MissionState,
        phase: MissionPhase,
        current_output: str,
        phase_type: str,
        decision: Optional["SupervisorDecision"] = None
    ) -> bool:
        """
        Run a single depth enrichment pass.
        
        Selects the appropriate enrichment type based on the weakest depth
        indicator and runs an enrichment prompt to improve depth.
        
        Args:
            state: Current mission state
            phase: The phase being enriched
            current_output: Current phase output text
            phase_type: Type of phase
            decision: Optional supervisor decision for model selection
            
        Returns:
            True if enrichment was successful and added content
        """
        if not DEPTH_EVALUATOR_AVAILABLE:
            return False
        
        try:
            # Get depth indicators to select enrichment type
            indicators = extract_depth_indicators(current_output)
            enrichment_type = select_enrichment_type(indicators)
            enrichment_prompt = get_enrichment_prompt(enrichment_type)
            
            state.log(
                f"[ENRICHMENT] Starting '{enrichment_type}' pass "
                f"({phase.enrichment_passes + 1}/{self.max_enrichment_passes})"
            )
            
            # Build enrichment context
            # Truncate current output for prompt
            truncated_output = current_output[:4000] if len(current_output) > 4000 else current_output
            
            enrichment_context = f"""Previous analysis:
{truncated_output}

---

{enrichment_prompt}"""
            
            # Prepare and run the appropriate council for enrichment
            self._prepare_council_for_execution(self.researcher, state, decision)
            
            # Use researcher council for enrichment (it's the most general)
            research_ctx = ResearchContext(
                objective=f"Enrich analysis for phase '{phase.name}': {enrichment_prompt}",
                prior_knowledge=truncated_output[:2000],
                constraints=None,
            )
            
            result = self.researcher.execute(research_ctx)
            
            if result.success and result.output:
                # Extract enrichment text
                enrichment_text = ""
                if hasattr(result.output, 'raw_output'):
                    enrichment_text = result.output.raw_output
                elif hasattr(result.output, 'findings'):
                    enrichment_text = result.output.findings
                else:
                    enrichment_text = str(result.output)
                
                if enrichment_text and len(enrichment_text) > 100:
                    # Store enrichment output
                    existing_enrichment = phase.artifacts.get("enrichment_output", "")
                    phase.artifacts["enrichment_output"] = (
                        existing_enrichment + 
                        f"\n\n--- Enrichment Pass ({enrichment_type}) ---\n" +
                        enrichment_text
                    )
                    
                    state.log(
                        f"[ENRICHMENT] Added {len(enrichment_text)} chars "
                        f"via '{enrichment_type}'"
                    )
                    return True
            
            return False
            
        except Exception as e:
            _orchestrator_logger.warning(f"Enrichment pass failed: {e}")
            state.log(f"[ENRICHMENT] Pass failed: {e}")
            return False
    
    def _execute_deepening(
        self,
        state: MissionState,
        plan: "DeepeningPlan"
    ) -> None:
        """
        Execute a deepening plan by running specified councils.
        
        Args:
            state: Current mission state
            plan: Deepening plan from ReasoningSupervisor
        """
        if not plan.has_work:
            return
        
        state.log(f"Executing deepening plan: {plan.reason}")
        
        # Get accumulated context
        context = self._get_accumulated_context(state, include_full_history=True)
        
        # Get supervisor decision for deepening
        decision = None
        if self.enable_supervision and self.supervisor and self.gpu_manager:
            # Create a dummy phase for supervisor
            from .mission_types import MissionPhase
            dummy_phase = MissionPhase(name="Deepening", description="Deepening pass")
            decision = self._get_supervisor_decision(state, dummy_phase)
        
        for round_idx in range(plan.max_deepening_rounds):
            # Check time
            if state.remaining_minutes() < 1.0:
                state.log("Deepening stopped - insufficient time")
                break
            
            # Run researcher if needed
            if plan.run_researcher and state.remaining_minutes() > 0.5:
                try:
                    research_ctx = ResearchContext(
                        objective=f"Deepen analysis: {state.objective}",
                        prior_knowledge=context[:3000],
                        constraints=None
                    )
                    self._prepare_council_for_execution(self.researcher, state, decision)
                    result = self.researcher.execute(research_ctx)
                    if result.success and result.output:
                        # Store in work summary
                        if "deepening" not in state.work_summary:
                            state.work_summary["deepening"] = {}
                        state.work_summary["deepening"]["research"] = str(result.output)[:1000]
                except Exception as e:
                    state.log(f"Deepening research error: {e}")
            
            # Run planner if needed
            if plan.run_planner and state.remaining_minutes() > 0.5:
                try:
                    planner_ctx = PlannerContext(
                        objective=f"Refine plan: {state.objective}",
                        context={"prior_work": context[:2000]},
                        max_iterations=1
                    )
                    self._prepare_council_for_execution(self.planner, state, decision)
                    result = self.planner.execute(planner_ctx)
                    if result.success and result.output:
                        if "deepening" not in state.work_summary:
                            state.work_summary["deepening"] = {}
                        state.work_summary["deepening"]["planner"] = str(result.output)[:1000]
                except Exception as e:
                    state.log(f"Deepening planner error: {e}")
            
            # Run evaluator if needed
            if plan.run_evaluator and state.remaining_minutes() > 0.5:
                try:
                    eval_ctx = EvaluatorContext(
                        objective=f"Evaluate progress: {state.objective}",
                        content_to_evaluate=context[:3000],
                        quality_threshold=6.0
                    )
                    self._prepare_council_for_execution(self.evaluator, state, decision)
                    result = self.evaluator.execute(eval_ctx)
                    if result.success and result.output:
                        if "deepening" not in state.work_summary:
                            state.work_summary["deepening"] = {}
                        state.work_summary["deepening"]["evaluator"] = str(result.output)[:1000]
                except Exception as e:
                    state.log(f"Deepening evaluator error: {e}")
        
        state.log("Deepening pass completed")
    
    def _classify_phase(self, phase: MissionPhase) -> str:
        """Classify a phase based on its name to determine which councils to use."""
        name_lower = phase.name.lower()
        
        for phase_type, keywords in PHASE_KEYWORDS.items():
            if any(kw in name_lower for kw in keywords):
                return phase_type
        
        # Default to research if unclear
        return "research"
    
    def _track_council_output(
        self,
        council_name: str,
        result: Any,
        state: MissionState,
        duration_s: float = 0.0
    ) -> None:
        """
        Track council output through the CognitiveSpine.
        
        This provides visibility into:
        - Output size tracking
        - Budget monitoring
        - Contraction mode triggering
        
        Args:
            council_name: Name of the council that produced output
            result: Council result object
            state: Current mission state
            duration_s: Execution duration in seconds
        """
        if self.cognitive_spine is None:
            return
        
        # Track output through spine
        if hasattr(result, 'output') and result.output:
            self.cognitive_spine.track_output(result.output, council_name, log_status=True)
        elif hasattr(result, 'raw_output'):
            self.cognitive_spine.track_output(result.raw_output, council_name, log_status=True)
        
        # Check time and potentially enter contraction mode
        time_remaining = state.remaining_time().total_seconds()
        total_time = state.constraints.time_budget_minutes * 60
        
        if self.cognitive_spine.should_enter_contraction(time_remaining, total_time):
            self.cognitive_spine.enter_contraction_mode()
            if self.reasoning_supervisor is not None:
                self.reasoning_supervisor.enter_contraction_mode(state)
            state.log(f"[SPINE] Contraction mode triggered after {council_name}")
    
    def _time_exhausted(self, state: MissionState, safety_margin_seconds: float = 20.0) -> bool:
        """
        Check if mission time is exhausted (with safety margin).
        
        Args:
            state: Current mission state
            safety_margin_seconds: Buffer time to reserve before deadline
            
        Returns:
            True if remaining time is less than safety margin
        """
        remaining = state.remaining_time().total_seconds()
        return remaining <= safety_margin_seconds
    
    def _should_continue_phase(
        self,
        results: List[Any],
        constraints: MissionConstraints
    ) -> bool:
        """
        Determine if phase should continue based on quality.
        
        Args:
            results: List of results from previous rounds
            constraints: Mission constraints with quality thresholds
            
        Returns:
            True if phase should continue, False if quality is sufficient
        """
        if not results:
            return True
        
        last = results[-1]
        
        # Check for quality_score attribute in output
        if hasattr(last, 'output') and last.output is not None:
            output = last.output
            if hasattr(output, 'quality_score') and output.quality_score is not None:
                return output.quality_score < constraints.min_quality_to_stop
        
        # Continue if no quality signal available
        return True
    
    def _phase_good_enough(self, evaluator_output: Any, threshold: float = 7.0) -> bool:
        """
        Check if phase output is good enough to stop rounds.
        
        Good enough means:
        - Quality score >= threshold
        - No missing_info (or empty)
        - Confidence >= 0.6
        
        Args:
            evaluator_output: Output from EvaluatorCouncil
            threshold: Quality threshold
            
        Returns:
            True if phase is good enough to stop
        """
        if evaluator_output is None:
            return False
        
        # Check quality score
        if hasattr(evaluator_output, 'quality_score'):
            if evaluator_output.quality_score < threshold:
                return False
        else:
            return False
        
        # Check for missing info
        if hasattr(evaluator_output, 'missing_info'):
            if evaluator_output.missing_info:
                return False
        
        # Check confidence
        if hasattr(evaluator_output, 'confidence_score'):
            if evaluator_output.confidence_score < 0.6:
                return False
        
        # Check for critical issues
        if hasattr(evaluator_output, 'has_critical_issues'):
            if evaluator_output.has_critical_issues():
                return False
        
        return True
    
    def _get_max_rounds_for_phase(self, phase_type: str, constraints: MissionConstraints) -> int:
        """
        Get max rounds based on phase type and constraints.
        
        Args:
            phase_type: Type of phase (research, design, implementation, etc.)
            constraints: Mission constraints with round limits
            
        Returns:
            Maximum number of rounds for this phase type
        """
        mapping = {
            "research": constraints.max_recon_rounds,
            "design": constraints.max_analysis_rounds,
            "implementation": constraints.max_deep_rounds,
            "testing": constraints.max_analysis_rounds,
            "synthesis": constraints.max_analysis_rounds,
        }
        return mapping.get(phase_type, constraints.max_analysis_rounds)
    
    def _run_phase(self, state: MissionState, phase: MissionPhase) -> None:
        """
        Execute a phase with multi-round support.
        
        If step execution is enabled and the phase has steps, uses the
        StepExecutor to run each step with a single specialized model.
        Otherwise falls back to council-based execution with multi-round support.
        
        Uses DepthContract from ReasoningSupervisor when available for:
        - max_rounds: Based on phase difficulty and uncertainty
        - model_tier: For supervisor model selection
        - exploration_depth: For adjusting thoroughness
        
        CognitiveSpine integration:
        - Pre-phase validation of context
        - Contraction mode check and skip decision
        - Memory compression at phase boundaries
        
        Args:
            state: Current mission state
            phase: Phase to execute
        """
        # Check if this is a Deep Analysis phase - use special pipeline
        phase_lower = phase.name.lower()
        if "deep" in phase_lower and "analysis" in phase_lower:
            return self._run_deep_analysis_pipeline(state, phase)
        
        # === CognitiveSpine: Check for contraction mode skip ===
        if self.reasoning_supervisor is not None and self.reasoning_supervisor.is_contraction_mode():
            should_skip, skip_reason = self.reasoning_supervisor.should_skip_phase(
                phase.name, state
            )
            if should_skip:
                phase.mark_skipped(skip_reason)
                state.log(f"[SPINE] Skipped phase '{phase.name}': {skip_reason}")
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_spine_decision(
                        "phase_skipped",
                        {"phase": phase.name, "reason": skip_reason}
                    )
                return
        
        phase.mark_running()
        phase_type = self._classify_phase(phase)
        start_time = time.time()
        
        # === Sprint 1-2: Metrics Hook - Phase Start ===
        _metrics_ctx: Optional["PhaseMetricsContext"] = None
        if self._metrics_hook is not None:
            try:
                _metrics_ctx = self._metrics_hook.on_phase_start(state, phase)
                
                # Get routing advice if enabled
                if self._metrics_config and self._metrics_config.learning_router_enabled:
                    routing_advice = self._metrics_hook.get_routing_advice(
                        state, phase, self._recent_scores
                    )
                    if routing_advice and routing_advice.confidence > 0.6:
                        state.log(
                            f"[METRICS] Router advice: tier={routing_advice.model_tier}, "
                            f"rounds={routing_advice.num_rounds}, "
                            f"confidence={routing_advice.confidence:.2f}"
                        )
                        # Store in phase artifacts for potential use
                        phase.artifacts["_routing_advice"] = routing_advice.to_dict()
            except Exception as e:
                _orchestrator_logger.debug(f"Metrics phase start hook failed: {e}")
        
        # === Cognitive Constitution: Snapshot Baseline ===
        if self._constitution_engine is not None:
            try:
                # Get current scorecard if available
                current_scorecard = None
                if _metrics_ctx and _metrics_ctx.score_before:
                    current_scorecard = _metrics_ctx.score_before.scorecard
                
                self._constitution_ctx = self._constitution_engine.snapshot_baseline(
                    state=state,
                    phase=phase,
                    scorecard=current_scorecard,
                )
            except Exception as e:
                _orchestrator_logger.debug(f"Constitution baseline snapshot failed: {e}")
        
        # Verbose logging: phase start
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_phase_start(phase)
        
        # === CognitiveSpine: Log phase boundary ===
        if self.cognitive_spine is not None:
            self.cognitive_spine.log_phase_boundary(phase.name, "entering", {
                "type": phase_type,
                "remaining_minutes": state.remaining_minutes(),
            })
        
        # === CognitiveSpine: Check time and trigger contraction if needed ===
        if self.cognitive_spine is not None:
            time_remaining = state.remaining_time().total_seconds()
            total_time = state.constraints.time_budget_minutes * 60
            
            # Check contraction more aggressively - at 25% remaining
            if self.cognitive_spine.should_enter_contraction(time_remaining, total_time):
                self.cognitive_spine.enter_contraction_mode()
                if self.reasoning_supervisor is not None:
                    self.reasoning_supervisor.enter_contraction_mode(state)
                state.log("[SPINE] Entered contraction mode - time exhausted")
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_contraction_mode(
                        time_remaining / 60.0,
                        "time_threshold"
                    )
        
        # === DeepThinker 2.0: Get PhaseSpec for phase contract enforcement ===
        phase_spec = self._get_phase_spec(phase.name)
        if phase_spec is not None:
            _orchestrator_logger.debug(
                f"Phase '{phase.name}' spec: allowed_councils={phase_spec.allowed_councils}, "
                f"memory_policy={phase_spec.memory_write_policy.value}"
            )
            # Store phase spec for use in council validation
            phase._phase_spec = phase_spec
        
        # Get depth contract from ReasoningSupervisor if available
        depth_contract = None
        if self.reasoning_supervisor and REASONING_SUPERVISOR_AVAILABLE:
            try:
                # Analyze phase for metrics if we have previous output
                phase_output = phase.artifacts if phase.artifacts else {}
                phase_metrics = self.reasoning_supervisor.analyze_phase_output(
                    phase.name, phase_output, state
                )
                
                # Create depth contract based on metrics
                time_remaining = state.remaining_minutes()
                depth_contract = self.reasoning_supervisor.create_depth_contract(
                    phase_metrics, phase.name, time_remaining
                )
                
                state.log(f"Depth contract: max_rounds={depth_contract.max_rounds}, "
                         f"tier={depth_contract.model_tier}, depth={depth_contract.exploration_depth:.2f}")
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Depth contract creation failed: {e}")
        
        # Determine max rounds - use depth contract if available, else constraints
        if depth_contract:
            max_rounds = depth_contract.max_rounds
        else:
            max_rounds = self._get_max_rounds_for_phase(phase_type, state.constraints)
        
        # === Dynamic Council Configuration ===
        # Apply dynamic council configurations based on phase/mission context
        difficulty = None
        uncertainty = None
        if depth_contract:
            # Use metrics from depth contract if available
            difficulty = getattr(depth_contract, 'exploration_depth', None)
        
        self._apply_dynamic_council_config(
            state=state,
            phase=phase,
            difficulty=difficulty,
            uncertainty=uncertainty,
        )
        
        results = []
        decision = None
        
        # === SHADOW MODE: Cost/Time Prediction ===
        cost_prediction: Optional["CostTimePrediction"] = None
        
        # === SHADOW MODE: Phase Risk Prediction ===
        risk_prediction: Optional["PhaseRiskPrediction"] = None
        
        # === SHADOW MODE: Web Search Prediction ===
        web_search_prediction: Optional["WebSearchPrediction"] = None
        
        for round_idx in range(max_rounds):
            # Check if time is exhausted
            if self._time_exhausted(state):
                state.log(f"Phase '{phase.name}' stopping - time exhausted")
                break
            
            # Get supervisor decision for this phase
            if self.enable_supervision:
                decision = self._get_supervisor_decision(state, phase)
            
            # === SHADOW MODE: Make prediction on first round ===
            if round_idx == 0 and self._enable_cost_prediction and self._cost_predictor is not None:
                try:
                    cost_prediction = self._make_cost_prediction(
                        state, phase, phase_type, decision
                    )
                except Exception as e:
                    _orchestrator_logger.debug(f"Cost prediction failed: {e}")
            
            # === SHADOW MODE: Make risk prediction on first round ===
            if round_idx == 0 and self._enable_risk_prediction and self._risk_predictor is not None:
                try:
                    risk_prediction = self._make_risk_prediction(
                        state, phase, phase_type, decision
                    )
                except Exception as e:
                    _orchestrator_logger.debug(f"Risk prediction failed: {e}")
            
            # === SHADOW MODE: Make web search prediction on first round ===
            if round_idx == 0 and self._enable_web_search_prediction and self._web_search_predictor is not None:
                try:
                    web_search_prediction = self._make_web_search_prediction(
                        state, phase, phase_type, decision
                    )
                except Exception as e:
                    _orchestrator_logger.debug(f"Web search prediction failed: {e}")
            
            # === LOG ML PREDICTIONS TO VERBOSE LOGGER ===
            if round_idx == 0 and VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                if cost_prediction or risk_prediction or web_search_prediction:
                    try:
                        verbose_logger.log_ml_predictions_panel(
                            phase_name=phase.name,
                            cost_prediction=cost_prediction.to_dict() if cost_prediction else None,
                            risk_prediction=risk_prediction.to_dict() if risk_prediction else None,
                            web_search_prediction=web_search_prediction.to_dict() if web_search_prediction else None,
                        )
                    except Exception as e:
                        _orchestrator_logger.debug(f"ML predictions logging failed: {e}")
            
            # === PHASE 4.1: LIVE COST PREDICTION - USE FOR MODEL SELECTION ===
            if round_idx == 0 and cost_prediction and decision:
                try:
                    decision = self._apply_cost_aware_decision(
                        state, phase, decision, cost_prediction
                    )
                except Exception as e:
                    _orchestrator_logger.debug(f"Cost-aware decision failed: {e}")
            
            try:
                # Use step-based execution if enabled and phase has steps
                if self.enable_step_execution and phase.steps and self.step_executor:
                    self._run_phase_with_steps(state, phase)
                    # Step execution doesn't return CouncilResult, break after one pass
                    state.phase_rounds[phase.name] = 1
                    break
                else:
                    # Legacy council-based execution (single round)
                    self._run_phase_with_councils_single(state, phase, decision, phase_type)
                    results.append({"phase": phase.name, "round": round_idx + 1})
                
                # Track round
                state.phase_rounds[phase.name] = round_idx + 1
                
                # Check quality-based stopping
                if not self._should_continue_phase(results, state.constraints):
                    state.log(f"Phase '{phase.name}' stopping - quality threshold met")
                    break
                
                # Check resource budget via CognitiveSpine
                if self.cognitive_spine is not None:
                    council_name = f"{phase_type}_council"
                    if self.cognitive_spine.is_budget_exceeded(council_name):
                        state.log(f"Phase '{phase.name}' stopping - resource budget exceeded")
                        phase.mark_failed("Resource budget exceeded")
                        phase.artifacts["failure_reason"] = MissionFailureReason.BUDGET_EXCEEDED.value
                        phase.artifacts["budget_status"] = self.cognitive_spine.get_budget(council_name).to_dict()
                        return
                
            except Exception as e:
                state.log(f"Phase '{phase.name}' round {round_idx + 1} error: {e}")
                if round_idx == 0:
                    # First round failed, mark phase as failed with reason code
                    error_msg = str(e)
                    phase.mark_failed(error_msg)
                    
                    # Determine failure reason based on error type/message
                    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        failure_reason = MissionFailureReason.TIMEOUT.value
                    elif "refused" in error_msg.lower() or "refusal" in error_msg.lower():
                        failure_reason = MissionFailureReason.MODEL_REFUSAL.value
                    elif "budget" in error_msg.lower() or "exceeded" in error_msg.lower():
                        failure_reason = MissionFailureReason.BUDGET_EXCEEDED.value
                    else:
                        failure_reason = MissionFailureReason.COUNCIL_FAILURE.value
                    
                    phase.artifacts["failure_reason"] = failure_reason
                    
                    duration = time.time() - start_time
                    state.log_council_execution(
                        council_name=f"{phase_type}_council",
                        phase_name=phase.name,
                        models_used=decision.models if decision else [],
                        success=False,
                        duration_s=duration,
                        error=error_msg
                    )
                    return
                break
        
        # === NORMATIVE GOVERNANCE CHECK ===
        # Evaluate phase output against governance rules before allowing completion
        governance_verdict = None
        if GOVERNANCE_AVAILABLE and self._normative_controller is not None:
            try:
                governance_verdict = self._normative_controller.evaluate(
                    phase_name=phase.name,
                    phase_output=phase.artifacts or {},
                    mission_state=state,
                )
                
                # Store verdict in artifacts
                phase.artifacts["governance_verdict"] = governance_verdict.status.value
                phase.artifacts["governance_violations"] = len(governance_verdict.violations)
                
                # === SSE: Publish governance update for frontend ===
                if SSE_AVAILABLE and sse_manager:
                    retry_count = self._normative_controller.get_retry_count(phase.name) if self._normative_controller else 0
                    violation_details = [
                        {"type": v.rule_id, "severity": v.severity}
                        for v in governance_verdict.violations
                    ] if governance_verdict.violations else []
                    
                    _publish_sse_event(sse_manager.publish_governance_update(
                        mission_id=state.mission_id,
                        phase_name=phase.name,
                        verdict=governance_verdict.status.value,
                        violations=len(governance_verdict.violations),
                        violation_details=violation_details,
                        retry_count=retry_count,
                        max_retries=governance_verdict.max_retries if hasattr(governance_verdict, 'max_retries') else 2,
                        retry_reason=governance_verdict.retry_reason if hasattr(governance_verdict, 'retry_reason') else None,
                        force_web_search=governance_verdict.force_web_search if hasattr(governance_verdict, 'force_web_search') else False,
                        epistemic_risk_score=governance_verdict.epistemic_risk if hasattr(governance_verdict, 'epistemic_risk') else 0.0,
                    ))
                
                # Handle verdict
                if governance_verdict.status == VerdictStatus.BLOCK:
                    blocked = self._handle_governance_block(
                        state, phase, governance_verdict, phase_type, decision, start_time
                    )
                    if blocked:
                        return  # Phase not completed, retry scheduled
                
                elif governance_verdict.status == VerdictStatus.WARN:
                    self._apply_governance_penalties(state, phase, governance_verdict)
                
                # Apply confidence clamping (always applied)
                if governance_verdict.confidence_penalty > 0:
                    state.log(
                        f"[GOVERNANCE] Confidence clamped by {governance_verdict.confidence_penalty:.2f} "
                        f"due to epistemic risk {governance_verdict.epistemic_risk:.2f}"
                    )
                
                # Update epistemic telemetry
                state.update_epistemic_telemetry(
                    epistemic_risk=governance_verdict.epistemic_risk
                )
                
            except Exception as e:
                _orchestrator_logger.warning(f"Governance evaluation failed: {e}")
                state.log(f"[GOVERNANCE] Evaluation error: {e}")
        
        # === DEPTH CONTROL: Evaluate depth and run enrichment if needed ===
        termination_reason = "convergence"  # Default reason
        if self.enable_depth_control and DEPTH_EVALUATOR_AVAILABLE:
            try:
                # Gather phase output for depth evaluation
                phase_output = self._gather_phase_output_for_depth(phase)
                
                if phase_output:
                    # Compute depth score
                    depth_achieved = compute_depth_score(phase_output, phase_type)
                    depth_target = get_depth_target(phase_type)
                    depth_gap = compute_depth_gap(depth_achieved, depth_target)
                    
                    # Store initial depth metrics
                    phase.depth_achieved = depth_achieved
                    phase.artifacts["depth_achieved"] = f"{depth_achieved:.2f}"
                    phase.artifacts["depth_target"] = f"{depth_target:.2f}"
                    phase.artifacts["depth_gap"] = f"{depth_gap:.2f}"
                    
                    _orchestrator_logger.info(
                        f"[DEPTH] Computed depth={depth_achieved:.2f} for phase '{phase.name}' "
                        f"(target={depth_target:.2f}, gap={depth_gap:.2f})"
                    )
                    state.log(
                        f"[DEPTH] Phase '{phase.name}': depth={depth_achieved:.2f}/{depth_target:.2f}"
                    )
                    
                    # Check if enrichment is needed and possible
                    time_remaining = state.remaining_time().total_seconds()
                    enrichment_passes = 0
                    
                    while (
                        depth_gap > self.depth_gap_threshold and
                        enrichment_passes < self.max_enrichment_passes and
                        time_remaining > self.min_enrichment_time_seconds
                    ):
                        # Log depth pressure
                        state.log(
                            f"[DEPTH_PRESSURE] Phase '{phase.name}' depth_gap={depth_gap:.2f}, "
                            f"time_surplus={time_remaining:.0f}s, enrichment_encouraged=True"
                        )
                        
                        # Run enrichment pass
                        enrichment_result = self._run_enrichment_pass(
                            state, phase, phase_output, phase_type, decision
                        )
                        enrichment_passes += 1
                        phase.enrichment_passes = enrichment_passes
                        
                        if enrichment_result:
                            # Recompute depth with enriched content
                            phase_output = self._gather_phase_output_for_depth(phase)
                            new_depth = compute_depth_score(phase_output, phase_type)
                            
                            state.log(
                                f"[ENRICHMENT] Pass complete, depth improved "
                                f"{depth_achieved:.2f} → {new_depth:.2f}"
                            )
                            
                            depth_achieved = new_depth
                            depth_gap = compute_depth_gap(depth_achieved, depth_target)
                            phase.depth_achieved = depth_achieved
                            phase.artifacts["depth_achieved"] = f"{depth_achieved:.2f}"
                            phase.artifacts["depth_gap"] = f"{depth_gap:.2f}"
                        
                        # Update remaining time
                        time_remaining = state.remaining_time().total_seconds()
                    
                    # Set termination reason
                    if depth_gap <= self.depth_gap_threshold:
                        termination_reason = "depth_target_reached"
                    elif enrichment_passes >= self.max_enrichment_passes:
                        termination_reason = "enrichment_limit_reached"
                    elif time_remaining <= self.min_enrichment_time_seconds:
                        termination_reason = "time_exhausted"
                    
                    phase.artifacts["enrichment_passes"] = str(enrichment_passes)
                    
            except Exception as e:
                _orchestrator_logger.warning(f"Depth evaluation failed: {e}")
                state.log(f"[DEPTH] Evaluation error: {e}")
        
        # Set termination reason
        phase.termination_reason = termination_reason
        
        # Log phase end with attribution
        time_unused = state.remaining_time().total_seconds()
        state.log(
            f"[PHASE_END] Phase '{phase.name}' ended: reason={termination_reason}, "
            f"depth={phase.depth_achieved:.2f}/{get_depth_target(phase_type) if DEPTH_EVALUATOR_AVAILABLE else 0:.2f}, "
            f"enrichment_passes={phase.enrichment_passes}, time_unused={time_unused:.0f}s"
        )
        
        # Mark completed
        phase.mark_completed()
        duration = time.time() - start_time
        rounds_completed = state.phase_rounds.get(phase.name, 1)
        state.log(f"Phase '{phase.name}' completed in {duration:.1f}s ({rounds_completed} round(s))")
        
        # === Proof-Carrying Reasoning: Build and store proof packet ===
        if self._enable_proof_packets and self._proof_builder is not None and self._proof_store is not None:
            try:
                # Get phase output text from artifacts
                phase_output = "\n".join(
                    f"{k}: {v}" for k, v in phase.artifacts.items()
                    if not k.startswith("_") and isinstance(v, str)
                )
                
                # Get previous packet for this phase (if any)
                prev_packet = self._proof_store.get_latest(state.mission_id, phase.name)
                
                # Update builder with evidence store if available
                if self.memory is not None and hasattr(self.memory, 'mission_rag'):
                    self._proof_builder.set_evidence_store(self.memory.mission_rag)
                
                # Get model name from artifacts if available
                model_name = phase.artifacts.get("_model_tier", "")
                
                # Build proof packet
                proof_packet = self._proof_builder.build_from_phase_output(
                    output_text=phase_output,
                    phase_name=phase.name,
                    mission_id=state.mission_id,
                    model_name=model_name,
                    prev_packet=prev_packet,
                    depth_increased=enrichment_passes > 0 if 'enrichment_passes' in dir() else False,
                )
                
                # Store packet
                self._proof_store.write(proof_packet)
                
                # Store packet ID in phase artifacts
                phase.artifacts["_proof_packet_id"] = proof_packet.packet_id
                
                # Log summary
                summary = proof_packet.get_summary()
                state.log(
                    f"[PROOF] Packet {proof_packet.packet_id[:12]}: "
                    f"{summary['claim_count']} claims, "
                    f"{summary['evidence_coverage_ratio']:.0%} evidence coverage"
                )
                
                # Log integrity violations if any
                if proof_packet.integrity_flags.has_violations:
                    state.log(
                        f"[PROOF] Integrity violations: "
                        f"{proof_packet.integrity_flags.violation_count}"
                    )
                    
            except Exception as e:
                _orchestrator_logger.debug(f"Proof packet generation failed: {e}")
        
        # === Sprint 1-2: Metrics Hook - Phase End ===
        if self._metrics_hook is not None and _metrics_ctx is not None:
            try:
                scorecard, policy_decision = self._metrics_hook.on_phase_end(
                    ctx=_metrics_ctx,
                    phase=phase,
                    decision_emitter=self._decision_emitter,
                )
                
                # Track score for router features
                if scorecard is not None:
                    self._recent_scores.append(scorecard.overall)
                    # Keep only last 5 scores
                    if len(self._recent_scores) > 5:
                        self._recent_scores.pop(0)
                    
                    # Update bandit if enabled (check constitution learning block)
                    if (
                        self._metrics_config and 
                        self._metrics_config.bandit_enabled and
                        phase.artifacts.get("_model_tier") and
                        not getattr(self, '_learning_blocked', False)
                    ):
                        tier = phase.artifacts["_model_tier"]
                        score_delta = scorecard.score_delta or 0.0
                        cost_delta = 0.01  # Placeholder, would compute from tokens
                        self._metrics_hook.update_bandit(tier, score_delta, cost_delta)
                    elif getattr(self, '_learning_blocked', False):
                        _orchestrator_logger.debug(
                            "[CONSTITUTION] Bandit update skipped - learning blocked"
                        )
                
                # Log policy decision
                if policy_decision is not None:
                    state.log(
                        f"[METRICS] Policy: {policy_decision.action.value} - "
                        f"{policy_decision.rationale}"
                    )
            except Exception as e:
                _orchestrator_logger.debug(f"Metrics phase end hook failed: {e}")
        
        # === Cognitive Constitution: Evaluate Phase ===
        if self._constitution_engine is not None and self._constitution_ctx is not None:
            try:
                # Get scorecard from metrics if available
                phase_scorecard = scorecard if 'scorecard' in dir() else None
                if _metrics_ctx and _metrics_ctx.score_after:
                    phase_scorecard = _metrics_ctx.score_after.scorecard
                
                # Count evidence added (approximate from tool usage)
                evidence_added = len([t for t in phase.artifacts.get("_tools_used", [])
                                      if "search" in t.lower() or "web" in t.lower()])
                
                constitution_flags = self._constitution_engine.evaluate_phase(
                    ctx=self._constitution_ctx,
                    scorecard=phase_scorecard,
                    evidence_added=evidence_added,
                    rounds_used=phase.deepening_rounds + 1,
                    tools_used=phase.artifacts.get("_tools_used", []),
                )
                
                # Apply enforcement in enforce mode
                from ..constitution import get_constitution_config
                constitution_config = get_constitution_config()
                
                if constitution_config.is_enforcing:
                    if constitution_flags.stop_deepening:
                        phase.termination_reason = "constitution:no_free_lunch"
                        state.log("[CONSTITUTION] Stopping deepening - no measurable gain")
                    if constitution_flags.block_learning:
                        self._learning_blocked = True
                        state.log("[CONSTITUTION] Learning updates blocked - invariant violation")
                
                # Log violations (in any mode)
                if constitution_flags.violations:
                    for v in constitution_flags.violations:
                        state.log(f"[CONSTITUTION] Violation: {v}")
                        
            except Exception as e:
                _orchestrator_logger.debug(f"Constitution evaluation failed: {e}")
        
        # Log resource management panel at phase boundary
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            gpu_stats_dict = None
            if self.gpu_manager is not None:
                try:
                    gpu_stats = self.gpu_manager.get_stats()
                    gpu_stats_dict = {
                        'available_gpus': getattr(gpu_stats, 'available_gpus', 0),
                        'utilization_percent': getattr(gpu_stats, 'utilization', 0),
                        'vram_used_mb': getattr(gpu_stats, 'used_mem', 0),
                        'vram_total_mb': getattr(gpu_stats, 'total_mem', 0),
                    }
                except Exception:
                    pass
            
            # Collect resource budgets from cognitive spine
            resource_budgets = {}
            if self.cognitive_spine is not None:
                try:
                    # Get budgets for all councils
                    for council_name in ['ResearcherCouncil', 'PlannerCouncil', 'CoderCouncil', 'EvaluatorCouncil', 'SimulationCouncil']:
                        budget = self.cognitive_spine.get_budget(council_name)
                        if budget:
                            resource_budgets[council_name] = {
                                'tokens_used': getattr(budget, 'tokens_used', 0),
                                'max_tokens': getattr(budget, 'max_tokens', 0),
                                'is_exceeded': getattr(budget, 'is_exceeded', False) if hasattr(budget, 'is_exceeded') else False
                            }
                except Exception:
                    pass
            
            contraction_mode = False
            if self.reasoning_supervisor is not None:
                try:
                    contraction_mode = self.reasoning_supervisor.is_contraction_mode()
                except Exception:
                    pass
            
            verbose_logger.log_resource_management_panel(
                gpu_stats=gpu_stats_dict,
                token_usage=None,  # Will be collected from council results
                time_remaining=state.remaining_minutes(),
                time_total=state.constraints.time_budget_minutes,
                resource_budgets=resource_budgets if resource_budgets else None,
                contraction_mode=contraction_mode
            )
        
        # === CognitiveSpine: Compress memory at phase boundary ===
        if self.cognitive_spine is not None:
            # Add phase artifacts to memory for compression
            if phase.artifacts:
                memory_slot = self.cognitive_spine.get_memory_slot(phase.name)
                artifact_summary = "\n".join(
                    f"- {k}: {str(v)[:500]}" 
                    for k, v in phase.artifacts.items() 
                    if not k.startswith("_")
                )
                memory_slot.add_to_delta(artifact_summary)
            
            # Compress the phase memory
            self.cognitive_spine.compress_phase_memory(phase.name)
            self.cognitive_spine.log_phase_boundary(phase.name, "completed", {
                "duration_s": duration,
                "rounds": rounds_completed,
            })
        
        # Verbose logging: phase artifacts
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_phase_artifacts(phase)
            
            # Log phase layer summary
            layers = {
                "Model Selection": decision is not None,
                "Memory Operations": self.memory is not None,
                "Internet Usage": state.constraints.allow_internet,
                "Step Execution": self.enable_step_execution and phase.steps,
                "Model Execution": True,  # Always happens in councils
                "Consensus Mechanism": True,  # Always happens in councils
                "Multi-View": self.enable_multiview,
                "Resource Management": True,  # Always tracked
                "Security & Execution": state.constraints.allow_code_execution,
                "Arbiter": False  # Only at synthesis
            }
            verbose_logger.log_phase_layer_summary(
                phase_name=phase.name,
                layers=layers,
                duration_s=duration,
                quality_score=None  # Will be added if available
            )
        
        # Log council execution
        state.log_council_execution(
            council_name=f"{phase_type}_council",
            phase_name=phase.name,
            models_used=decision.models if decision else [],
            success=True,
            duration_s=duration
        )
        
        # === Orchestration Outcome Logging ===
        if self.enable_orchestration_logging and self.orchestration_store:
            try:
                self._log_phase_outcome(state, phase, decision, phase_type, start_time, duration)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to log phase outcome: {e}")
        
        # === PHASE 4.2: Log prediction vs actual for online learning ===
        if cost_prediction is not None:
            self._log_prediction_accuracy(phase, cost_prediction, duration)
        
        # === SHADOW MODE: Log prediction vs actual ===
        if cost_prediction is not None and self._cost_eval_logger is not None:
            try:
                # Get actual GPU stats
                actual_gpu_seconds = duration
                actual_vram_peak = 0
                if self.gpu_manager:
                    try:
                        stats = self.gpu_manager.get_stats()
                        if stats.gpu_count > 0:
                            actual_gpu_seconds = duration * (stats.utilization / 100.0)
                            actual_vram_peak = stats.used_mem
                    except Exception:
                        pass
                
                self._cost_eval_logger.log_evaluation(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    phase_type=phase_type,
                    prediction=cost_prediction,
                    actual_wall_time=duration,
                    actual_gpu_seconds=actual_gpu_seconds,
                    actual_vram_peak=actual_vram_peak,
                )
            except Exception as e:
                _orchestrator_logger.debug(f"Cost prediction eval logging failed: {e}")
        
        # === SHADOW MODE: Log risk prediction vs actual ===
        if risk_prediction is not None and self._risk_eval_logger is not None:
            try:
                # Calculate actual retry count from phase rounds
                actual_retry_count = max(0, state.phase_rounds.get(phase.name, 1) - 1)
                
                # Infer failure mode from phase status
                actual_failure_mode = self._infer_failure_mode(phase, state)
                
                self._risk_eval_logger.log_evaluation(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    phase_type=phase_type,
                    prediction=risk_prediction,
                    actual_retry_count=actual_retry_count,
                    actual_failure_mode=actual_failure_mode,
                )
            except Exception as e:
                _orchestrator_logger.debug(f"Risk prediction eval logging failed: {e}")
        
        # === SHADOW MODE: Log web search prediction vs actual ===
        if web_search_prediction is not None and self._web_search_eval_logger is not None:
            try:
                # Determine actual web search usage from state
                actual_search_used, actual_num_queries = self._get_actual_web_search_usage(
                    state, phase
                )
                
                # Infer hallucination from phase artifacts/status
                actual_hallucination_detected = self._infer_hallucination(phase, state)
                
                self._web_search_eval_logger.log_evaluation(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    phase_type=phase_type,
                    prediction=web_search_prediction,
                    actual_search_used=actual_search_used,
                    actual_num_queries=actual_num_queries,
                    actual_hallucination_detected=actual_hallucination_detected,
                )
            except Exception as e:
                _orchestrator_logger.debug(f"Web search prediction eval logging failed: {e}")
        
        # Run multi-view evaluation on phase artifacts
        if phase.artifacts and self.enable_multiview:
            content = "\n".join(str(v)[:1000] for k, v in phase.artifacts.items() if not k.startswith("_"))
            if content.strip():
                multiview_result = self._run_multiview_evaluation(state, phase, content, decision)
                if multiview_result.get("disagreement_summary"):
                    phase.artifacts["multiview_summary"] = multiview_result["disagreement_summary"]
        
        # === EPISTEMIC VALIDATION ===
        # Run claim validation and epistemic risk assessment
        epistemic_result = self._run_epistemic_validation(state, phase)
        if epistemic_result:
            phase.artifacts["epistemic_validation"] = str(epistemic_result)
            
            # Log epistemic metrics
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                try:
                    verbose_logger.log_info(
                        f"Epistemic validation: risk={epistemic_result.get('epistemic_risk', 0):.2f}, "
                        f"grounded={epistemic_result.get('grounded_ratio', 0):.2%}, "
                        f"contamination={epistemic_result.get('contamination_score', 0):.2f}"
                    )
                except Exception:
                    pass
        
        # Run alignment check if enabled
        self._run_alignment_check(state, phase)
        
        # Run phase deepening if enabled
        self._run_phase_deepening(state, phase, decision)
        
        # Run meta-cognition processing
        self._run_meta_cognition(state, phase)
        
        # === Decision Accountability: Emit PHASE_TERMINATION decision ===
        self._emit_phase_termination_decision(
            state, phase, phase_type, governance_verdict, duration
        )
    
    def _compute_phase_outcome_cause(
        self,
        phase: MissionPhase,
        governance_verdict: Optional["NormativeVerdict"],
        time_remaining: float,
        retry_count: int,
        max_retries: int,
    ) -> str:
        """
        Compute the primary cause of phase outcome.
        
        Decision Accountability Layer: Determines why a phase ended
        for attribution and learning.
        
        Args:
            phase: The completed phase
            governance_verdict: Final governance verdict (if any)
            time_remaining: Minutes remaining at phase end
            retry_count: Number of retries attempted
            max_retries: Maximum retries allowed
            
        Returns:
            OutcomeCause value as string
        """
        if not DECISION_ACCOUNTABILITY_AVAILABLE:
            return "unknown"
        
        if phase.status == "completed":
            return OutcomeCause.SUCCESSFUL_CONVERGENCE.value
        
        if phase.status == "skipped" and time_remaining < MIN_PHASE_TIME_MINUTES:
            return OutcomeCause.TIME_EXHAUSTION.value
        
        if governance_verdict:
            if GOVERNANCE_AVAILABLE and governance_verdict.status == VerdictStatus.BLOCK:
                if retry_count >= max_retries:
                    return OutcomeCause.RETRY_EXHAUSTION.value
                return OutcomeCause.GOVERNANCE_VETO.value
        
        if phase.status == "completed_degraded":
            return OutcomeCause.MODEL_UNDERPOWERED.value
        
        if phase.artifacts.get("error"):
            return OutcomeCause.EXECUTION_ERROR.value
        
        return OutcomeCause.MODEL_UNDERPOWERED.value
    
    def _emit_phase_termination_decision(
        self,
        state: MissionState,
        phase: MissionPhase,
        phase_type: str,
        governance_verdict: Optional["NormativeVerdict"],
        duration: float,
    ) -> Optional[str]:
        """
        Emit a PHASE_TERMINATION decision record.
        
        Decision Accountability Layer: Records phase completion/failure
        as a first-class decision with outcome attribution.
        
        Args:
            state: Current mission state
            phase: The completed phase
            phase_type: Type of phase
            governance_verdict: Final governance verdict
            duration: Phase duration in seconds
            
        Returns:
            decision_id if emitted, None otherwise
        """
        if not self._decision_emitter or not self._enable_decision_accountability:
            return None
        
        try:
            # Compute outcome cause
            retry_count = self._normative_controller.get_retry_count(phase.name) if self._normative_controller else 0
            max_retries = governance_verdict.max_retries if governance_verdict else 2
            
            outcome_cause = self._compute_phase_outcome_cause(
                phase=phase,
                governance_verdict=governance_verdict,
                time_remaining=state.remaining_minutes(),
                retry_count=retry_count,
                max_retries=max_retries,
            )
            
            # Get quality score from artifacts if available
            quality_score = None
            if phase.artifacts:
                quality_score = phase.artifacts.get("quality_score")
                if quality_score is None:
                    quality_score = phase.artifacts.get("evaluator_score")
            
            decision_id = self._decision_emitter.emit_phase_termination(
                mission_id=state.mission_id,
                phase_id=phase.name,
                phase_type=phase_type,
                status=phase.status,
                outcome_cause=outcome_cause,
                time_remaining=state.remaining_minutes(),
                quality_score=quality_score,
                retry_count=retry_count,
                governance_severity=governance_verdict.aggregate_severity if governance_verdict else 0.0,
                triggered_by=state.last_model_decision_id,
            )
            
            if decision_id:
                state.track_decision(decision_id)
            
            return decision_id
            
        except Exception as e:
            _orchestrator_logger.debug(f"[DECISION] Failed to emit phase termination: {e}")
            return None
    
    def _log_phase_outcome(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"],
        phase_type: str,
        start_time: float,
        duration: float
    ) -> None:
        """
        Log phase outcome to orchestration store.
        
        Args:
            state: Mission state
            phase: Completed phase
            decision: Supervisor decision (if available)
            phase_type: Type of phase
            start_time: Phase start timestamp
            duration: Phase duration in seconds
        """
        if not self.orchestration_store:
            return
        
        try:
            # Collect council information
            councils_invoked = []
            models_used = []
            
            # Extract from phase artifacts or state
            if hasattr(phase, 'artifacts') and phase.artifacts:
                # Try to extract council info from artifacts
                for key in phase.artifacts.keys():
                    if 'council' in key.lower():
                        councils_invoked.append(key)
            
            # Get models from supervisor decision
            if decision:
                models_used = [
                    (model, self._get_model_tier(model))
                    for model in decision.models
                ]
                councils_invoked.append(f"{phase_type}_council")
            
            # Get token usage from LiteLLM monitor if available
            tokens_consumed = 0
            try:
                from ..models.litellm_monitor import LiteLLMMonitor
                if LiteLLMMonitor.is_enabled():
                    stats = LiteLLMMonitor.get_stats()
                    tokens_consumed = stats.get("total_tokens", 0)
            except Exception:
                pass
            
            # Get GPU stats if available
            gpu_seconds = duration  # Default: assume 100% utilization
            vram_peak_mb = 0
            gpu_pressure = "unknown"
            if self.gpu_manager:
                try:
                    stats = self.gpu_manager.get_stats()
                    if stats.gpu_count > 0:
                        utilization = stats.utilization / 100.0
                        gpu_seconds = duration * utilization
                        vram_peak_mb = stats.used_mem
                        gpu_pressure = self.gpu_manager.get_resource_pressure()
                except Exception:
                    pass
            
            # Phase 3.3: Add RAM stats to phase artifacts
            ram_available_mb = None
            ram_used_pct = None
            if self.ram_monitor and self.ram_monitor.is_available():
                try:
                    ram_stats = self.ram_monitor.get_stats()
                    ram_available_mb = ram_stats.available_ram_mb
                    ram_used_pct = ram_stats.percent_used
                    phase.artifacts["ram_available_mb"] = str(ram_available_mb)
                    phase.artifacts["ram_used_pct"] = f"{ram_used_pct:.1f}"
                except Exception:
                    pass
            
            # Add GPU pressure to phase artifacts
            phase.artifacts["gpu_pressure_at_execution"] = gpu_pressure
            
            # Get memory stats (approximate)
            memory_chars_written = 0
            memory_stable_chars = 0
            memory_ephemeral_chars = 0
            if self.memory:
                try:
                    summary = self.memory.get_state_summary()
                    # Approximate from RAG docs
                    rag_docs = summary.get("mission_rag_docs", 0)
                    memory_chars_written = rag_docs * 500  # Rough estimate
                except Exception:
                    pass
            
            # Get quality/confidence from arbiter or evaluator
            quality_score = None
            confidence_score = None
            arbiter_raw_output = None
            
            # Try to extract from phase artifacts
            if phase.artifacts:
                quality_score = phase.artifacts.get("quality_score")
                confidence_score = phase.artifacts.get("confidence_score")
                arbiter_raw_output = phase.artifacts.get("arbiter_output")
            
            # Determine consensus execution
            consensus_executed = False
            consensus_skipped_reason = None
            # Check if consensus was used (from phase artifacts or state)
            if phase.artifacts:
                consensus_executed = phase.artifacts.get("consensus_executed", False)
                consensus_skipped_reason = phase.artifacts.get("consensus_skipped_reason")
            
            # Infer phase type if not provided
            if not phase_type:
                phase_type = self._infer_phase_type(phase.name)
            
            # Create outcome
            outcome = PhaseOutcome(
                mission_id=state.mission_id,
                phase_name=phase.name,
                phase_type=phase_type,
                timestamp_start=datetime.fromtimestamp(start_time),
                timestamp_end=datetime.fromtimestamp(start_time + duration),
                councils_invoked=councils_invoked,
                models_used=models_used,
                consensus_executed=consensus_executed,
                consensus_skipped_reason=consensus_skipped_reason,
                tokens_consumed=tokens_consumed,
                wall_time_seconds=duration,
                gpu_seconds=gpu_seconds,
                vram_peak_mb=vram_peak_mb,
                memory_chars_written=memory_chars_written,
                memory_stable_chars=memory_stable_chars,
                memory_ephemeral_chars=memory_ephemeral_chars,
                quality_score=quality_score,
                confidence_score=confidence_score,
                arbiter_raw_output=arbiter_raw_output,
                time_remaining_at_start=state.remaining_minutes() * 60.0 + duration,
                effort_level=state.constraints.effort.value if hasattr(state.constraints.effort, 'value') else str(state.constraints.effort),
                constraints=state.constraints.as_dict(),
            )
            
            self.orchestration_store.write_outcome(outcome)
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create phase outcome: {e}")
    
    def _get_model_tier(self, model_name: str) -> str:
        """Get model tier from model name."""
        name_lower = model_name.lower()
        if any(x in name_lower for x in ["70b", "72b"]):
            return "xlarge"
        elif any(x in name_lower for x in ["27b", "33b", "34b"]):
            return "large"
        elif any(x in name_lower for x in ["12b", "13b", "14b"]):
            return "large"
        elif any(x in name_lower for x in ["7b", "8b", "9b"]):
            return "medium"
        elif any(x in name_lower for x in ["3b", "4b"]):
            return "small"
        elif any(x in name_lower for x in ["1b", "2b"]):
            return "small"
        return "medium"
    
    def _infer_phase_type(self, phase_name: str) -> str:
        """Infer phase type from phase name."""
        name_lower = phase_name.lower()
        if any(kw in name_lower for kw in ["recon", "gather", "context", "initial"]):
            return "reconnaissance"
        elif any(kw in name_lower for kw in ["deep", "thorough"]):
            return "deep_analysis"
        elif any(kw in name_lower for kw in ["analy", "plan", "design"]):
            return "analysis"
        elif any(kw in name_lower for kw in ["synth", "report", "final", "conclusion"]):
            return "synthesis"
        elif any(kw in name_lower for kw in ["impl", "code", "build"]):
            return "implementation"
        elif any(kw in name_lower for kw in ["simul", "scenario"]):
            return "simulation"
        return "default"
    
    def _make_cost_prediction(
        self,
        state: MissionState,
        phase: MissionPhase,
        phase_type: str,
        decision: Optional["SupervisorDecision"]
    ) -> Optional["CostTimePrediction"]:
        """
        Make a cost/time prediction for shadow mode logging.
        
        Args:
            state: Current mission state
            phase: Phase about to execute
            phase_type: Inferred phase type
            decision: Supervisor decision (may be None)
            
        Returns:
            CostTimePrediction or None if prediction fails
        """
        if not COST_PREDICTOR_AVAILABLE or self._cost_predictor is None:
            return None
        
        try:
            # Build PhaseContext
            effort_level = "standard"
            if hasattr(state.constraints, 'effort'):
                effort_val = state.constraints.effort
                effort_level = effort_val.value if hasattr(effort_val, 'value') else str(effort_val)
            
            ctx = PredictorPhaseContext(
                phase_name=phase.name,
                phase_type=phase_type,
                effort_level=effort_level,
                mission_time_budget_seconds=state.constraints.time_budget_minutes * 60,
                time_remaining_seconds=state.remaining_minutes() * 60,
                iteration_index=state.current_phase_index,
            )
            
            # Build ExecutionPlan from supervisor decision
            model_names = []
            model_tier = "medium"
            councils_invoked = [f"{phase_type}_council"]
            consensus_enabled = False
            per_call_timeout = 90.0
            
            if decision is not None:
                model_names = decision.models if decision.models else []
                # Infer tier from first model
                if model_names:
                    model_tier = self._get_model_tier(model_names[0])
                consensus_enabled = decision.parallelism > 1
                councils_invoked = [decision.council_type] if hasattr(decision, 'council_type') else councils_invoked
            
            plan = PredictorExecutionPlan(
                model_tier=model_tier,
                model_names=model_names,
                councils_invoked=councils_invoked,
                consensus_enabled=consensus_enabled,
                max_iterations=state.constraints.max_iterations,
                per_call_timeout_seconds=per_call_timeout,
                search_enabled=state.constraints.allow_internet,
            )
            
            # Build SystemState from GPU manager
            available_vram = 30000
            gpu_load = 0.5
            memory_pressure = 0.3
            
            if self.gpu_manager:
                try:
                    stats = self.gpu_manager.get_stats()
                    if stats.gpu_count > 0:
                        available_vram = stats.free_mem
                        gpu_load = stats.utilization / 100.0
                        # Memory pressure from usage ratio
                        if stats.total_mem > 0:
                            memory_pressure = stats.used_mem / stats.total_mem
                except Exception:
                    pass
            
            sys_state = PredictorSystemState(
                available_vram_mb=available_vram,
                gpu_load_ratio=gpu_load,
                memory_pressure_ratio=memory_pressure,
            )
            
            # Make prediction
            prediction = self._cost_predictor.predict(ctx, plan, sys_state)
            
            _orchestrator_logger.debug(
                f"[SHADOW] Cost prediction for '{phase.name}': "
                f"wall_time={prediction.wall_time_seconds:.1f}s, "
                f"gpu_seconds={prediction.gpu_seconds:.1f}s, "
                f"vram={prediction.vram_peak_mb}MB, "
                f"confidence={prediction.confidence:.2f}, "
                f"fallback={prediction.used_fallback}"
            )
            
            return prediction
            
        except Exception as e:
            _orchestrator_logger.debug(f"Cost prediction failed: {e}")
            return None
    
    def _apply_cost_aware_decision(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: "SupervisorDecision",
        prediction: "CostTimePrediction"
    ) -> "SupervisorDecision":
        """
        Apply cost/time prediction to adjust model selection.
        
        Phase 4.1: Uses prediction to downgrade models if budget is threatened.
        
        Args:
            state: Current mission state
            phase: Phase about to execute
            decision: Original supervisor decision
            prediction: Cost/time prediction
            
        Returns:
            Adjusted SupervisorDecision (may be unchanged)
        """
        if not SUPERVISOR_AVAILABLE:
            return decision
        
        remaining_seconds = state.remaining_time().total_seconds()
        predicted_time = prediction.wall_time_seconds
        
        # Check if prediction threatens budget
        # Use 80% of remaining time as threshold to leave room for synthesis
        time_threshold = remaining_seconds * 0.8
        
        if predicted_time <= time_threshold:
            # Within budget - no change needed
            return decision
        
        # Prediction exceeds threshold - downgrade model selection
        state.log(
            f"[COST-AWARE] Phase '{phase.name}' predicted {predicted_time:.0f}s "
            f"exceeds budget threshold ({time_threshold:.0f}s). Downgrading models."
        )
        
        # Get downgraded decision
        downgraded = self._get_downgraded_decision(decision, prediction)
        
        if downgraded.models != decision.models:
            state.log(
                f"[COST-AWARE] Model selection changed: {decision.models} -> {downgraded.models}"
            )
        
        return downgraded
    
    def _get_downgraded_decision(
        self,
        original: "SupervisorDecision",
        prediction: "CostTimePrediction"
    ) -> "SupervisorDecision":
        """
        Create a downgraded supervisor decision with lighter models.
        
        Phase 4.1: Selects smaller/faster models to meet time budget.
        
        Args:
            original: Original supervisor decision
            prediction: Cost/time prediction
            
        Returns:
            New SupervisorDecision with lighter models
        """
        if not SUPERVISOR_AVAILABLE:
            return original
        
        # Model tier downgrade mapping
        tier_downgrade = {
            "gemma3:27b": "gemma3:12b",
            "llama3.3:70b": "llama3.2:3b",
            "qwen3:32b": "qwen3:14b",
            "qwen3:14b": "qwen3:8b",
            "gemma3:12b": "llama3.2:3b",
            "deepseek-r1:32b": "deepseek-r1:14b",
            "deepseek-r1:14b": "gemma3:12b",
        }
        
        downgraded_models = []
        for model in original.models:
            downgraded = tier_downgrade.get(model, model)
            if downgraded not in downgraded_models:
                downgraded_models.append(downgraded)
        
        # Reduce council size if prediction is severely over budget
        if prediction.wall_time_seconds > prediction.wall_time_seconds * 1.5:
            downgraded_models = downgraded_models[:1]  # Single model only
        
        # Create new decision
        return SupervisorDecision(
            models=downgraded_models,
            temperature=original.temperature,
            parallelism=min(original.parallelism, len(downgraded_models)),
            downgraded=True,
            reason=f"Cost-aware downgrade (predicted {prediction.wall_time_seconds:.0f}s)",
            council_type=original.council_type,
            estimated_vram=original.estimated_vram // 2 if original.estimated_vram else 0,
            wait_for_capacity=False,
            phase_importance=original.phase_importance,
        )
    
    def _log_prediction_accuracy(
        self,
        phase: MissionPhase,
        prediction: "CostTimePrediction",
        actual_duration: float
    ) -> None:
        """
        Log prediction accuracy for online learning.
        
        Phase 4.2: Compares predicted vs actual to improve future predictions.
        
        Args:
            phase: Completed phase
            prediction: Pre-execution prediction
            actual_duration: Actual execution time in seconds
        """
        if not COST_PREDICTOR_AVAILABLE or self._cost_eval_logger is None:
            return
        
        try:
            error = actual_duration - prediction.wall_time_seconds
            error_pct = (error / prediction.wall_time_seconds * 100) if prediction.wall_time_seconds > 0 else 0
            
            _orchestrator_logger.info(
                f"[PREDICTION] Phase '{phase.name}': predicted={prediction.wall_time_seconds:.1f}s, "
                f"actual={actual_duration:.1f}s, error={error:+.1f}s ({error_pct:+.1f}%)"
            )
            
            # Log to evaluation logger for training
            self._cost_eval_logger.log(
                phase_name=phase.name,
                predicted_wall_time=prediction.wall_time_seconds,
                predicted_gpu_seconds=prediction.gpu_seconds,
                predicted_vram_peak=prediction.vram_peak_mb,
                actual_wall_time=actual_duration,
                actual_gpu_seconds=None,  # Not tracked yet
                actual_vram_peak=None,  # Not tracked yet
                model_version=prediction.model_version,
            )
        except Exception as e:
            _orchestrator_logger.debug(f"Prediction accuracy logging failed: {e}")
    
    def _make_risk_prediction(
        self,
        state: MissionState,
        phase: MissionPhase,
        phase_type: str,
        decision: Optional["SupervisorDecision"]
    ) -> Optional["PhaseRiskPrediction"]:
        """
        Make a phase risk prediction for shadow mode logging.
        
        Args:
            state: Current mission state
            phase: Phase about to execute
            phase_type: Inferred phase type
            decision: Supervisor decision (may be None)
            
        Returns:
            PhaseRiskPrediction or None if prediction fails
        """
        if not RISK_PREDICTOR_AVAILABLE or self._risk_predictor is None:
            return None
        
        try:
            # Build PhaseRiskContext
            effort_level = "standard"
            if hasattr(state.constraints, 'effort'):
                effort_val = state.constraints.effort
                effort_level = effort_val.value if hasattr(effort_val, 'value') else str(effort_val)
            
            # Get retry count so far for this phase
            retry_count_so_far = max(0, state.phase_rounds.get(phase.name, 0))
            
            ctx = PhaseRiskContext(
                phase_name=phase.name,
                phase_type=phase_type,
                effort_level=effort_level,
                iteration_index=state.current_phase_index,
                retry_count_so_far=retry_count_so_far,
                mission_time_remaining_seconds=state.remaining_minutes() * 60,
            )
            
            # Build PhaseRiskExecutionPlan from supervisor decision
            model_names = []
            model_tier = "medium"
            councils_invoked = [f"{phase_type}_council"]
            consensus_enabled = False
            
            if decision is not None:
                model_names = decision.models if decision.models else []
                # Infer tier from first model
                if model_names:
                    model_tier = self._get_model_tier(model_names[0])
                consensus_enabled = decision.parallelism > 1
                councils_invoked = [decision.council_type] if hasattr(decision, 'council_type') else councils_invoked
            
            plan = PhaseRiskExecutionPlan(
                model_tier=model_tier,
                model_names=model_names,
                councils_invoked=councils_invoked,
                consensus_enabled=consensus_enabled,
                search_enabled=state.constraints.allow_internet,
                max_iterations=state.constraints.max_iterations,
            )
            
            # Build PhaseRiskSystemState from GPU manager
            available_vram = 30000
            gpu_load = 0.5
            memory_pressure = 0.3
            
            if self.gpu_manager:
                try:
                    stats = self.gpu_manager.get_stats()
                    if stats.gpu_count > 0:
                        available_vram = stats.free_mem
                        gpu_load = stats.utilization / 100.0
                        # Memory pressure from usage ratio
                        if stats.total_mem > 0:
                            memory_pressure = stats.used_mem / stats.total_mem
                except Exception:
                    pass
            
            sys_state = PhaseRiskSystemState(
                available_vram_mb=available_vram,
                gpu_load_ratio=gpu_load,
                memory_pressure_ratio=memory_pressure,
            )
            
            # Make prediction
            prediction = self._risk_predictor.predict(ctx, plan, sys_state)
            
            _orchestrator_logger.debug(
                f"[SHADOW] Risk prediction for '{phase.name}': "
                f"retry_prob={prediction.retry_probability:.2f}, "
                f"expected_retries={prediction.expected_retries:.2f}, "
                f"failure_mode={prediction.dominant_failure_mode}, "
                f"confidence={prediction.confidence:.2f}, "
                f"fallback={prediction.used_fallback}"
            )
            
            return prediction
            
        except Exception as e:
            _orchestrator_logger.debug(f"Risk prediction failed: {e}")
            return None
    
    def _infer_failure_mode(
        self,
        phase: MissionPhase,
        state: MissionState
    ) -> str:
        """
        Infer the failure mode from phase execution results.
        
        Args:
            phase: Completed phase
            state: Current mission state
            
        Returns:
            Failure mode string (timeout, hallucination, low_quality, incoherent, unknown)
        """
        # Check if phase failed
        if phase.status == "failed":
            error_msg = phase.error.lower() if phase.error else ""
            if "timeout" in error_msg or "time" in error_msg:
                return "timeout"
            if "hallucin" in error_msg:
                return "hallucination"
            if "incoherent" in error_msg or "inconsistent" in error_msg:
                return "incoherent"
            return "unknown"
        
        # Check quality score if available in artifacts
        if phase.artifacts:
            quality_score = phase.artifacts.get("quality_score")
            if quality_score is not None and float(quality_score) < 0.3:
                return "low_quality"
            
            # Check arbiter output for hints
            arbiter_output = phase.artifacts.get("arbiter_raw_output", "")
            if arbiter_output:
                arbiter_lower = str(arbiter_output).lower()
                if "hallucin" in arbiter_lower:
                    return "hallucination"
                if "incoherent" in arbiter_lower or "inconsistent" in arbiter_lower:
                    return "incoherent"
                if "low quality" in arbiter_lower or "poor" in arbiter_lower:
                    return "low_quality"
        
        # Phase completed successfully
        return "unknown"
    
    def _make_web_search_prediction(
        self,
        state: MissionState,
        phase: MissionPhase,
        phase_type: str,
        decision: Optional["SupervisorDecision"]
    ) -> Optional["WebSearchPrediction"]:
        """
        Make a web search prediction for shadow mode logging.
        
        Predicts whether the phase requires web search to avoid hallucinations.
        
        Args:
            state: Current mission state
            phase: Phase to execute
            phase_type: Classified phase type
            decision: Optional supervisor decision
            
        Returns:
            WebSearchPrediction or None if prediction fails
        """
        if not WEB_SEARCH_PREDICTOR_AVAILABLE or self._web_search_predictor is None:
            return None
        
        try:
            # Build WebSearchContext
            effort_level = "standard"
            if hasattr(state.constraints, 'effort'):
                effort_val = state.constraints.effort
                effort_level = effort_val.value if hasattr(effort_val, 'value') else str(effort_val)
            
            # Analyze phase content for search-relevant patterns
            content_to_analyze = f"{phase.name} {phase.description}"
            if state.objective:
                content_to_analyze += f" {state.objective}"
            
            content_analysis = analyze_content(content_to_analyze)
            
            # Estimate prompt token count (rough heuristic)
            prompt_token_count = len(content_to_analyze.split()) * 2
            
            ctx = WebSearchContext(
                phase_name=phase.name,
                phase_type=phase_type,
                effort_level=effort_level,
                iteration_index=state.current_phase_index,
                prompt_token_count=prompt_token_count,
                contains_dates=content_analysis["contains_dates"],
                contains_named_entities=content_analysis["contains_named_entities"],
                contains_factual_claims=content_analysis["contains_factual_claims"],
            )
            
            # Build WebSearchExecutionPlan from supervisor decision
            model_names = []
            model_tier = "medium"
            councils_invoked = [f"{phase_type}_council"]
            consensus_enabled = False
            
            if decision is not None:
                model_names = decision.models if decision.models else []
                if model_names:
                    model_tier = self._get_model_tier(model_names[0])
                consensus_enabled = decision.parallelism > 1
                councils_invoked = [decision.council_type] if hasattr(decision, 'council_type') else councils_invoked
            
            plan = WebSearchExecutionPlan(
                model_tier=model_tier,
                model_names=model_names,
                councils_invoked=councils_invoked,
                consensus_enabled=consensus_enabled,
                search_enabled_by_planner=state.constraints.allow_internet,
                max_iterations=state.constraints.max_iterations,
            )
            
            # Build WebSearchSystemState
            sys_state = WebSearchSystemState(
                available_time_seconds=state.remaining_minutes() * 60,
            )
            
            # Make prediction
            prediction = self._web_search_predictor.predict(ctx, plan, sys_state)
            
            _orchestrator_logger.debug(
                f"[SHADOW] Web search prediction for '{phase.name}': "
                f"search_required={prediction.search_required}, "
                f"expected_queries={prediction.expected_queries}, "
                f"hallucination_risk={prediction.hallucination_risk_without_search:.2f}, "
                f"confidence={prediction.confidence:.2f}, "
                f"fallback={prediction.used_fallback}"
            )
            
            return prediction
            
        except Exception as e:
            _orchestrator_logger.debug(f"Web search prediction failed: {e}")
            return None
    
    def _get_actual_web_search_usage(
        self,
        state: MissionState,
        phase: MissionPhase
    ) -> tuple:
        """
        Get actual web search usage for a completed phase.
        
        Args:
            state: Mission state
            phase: Completed phase
            
        Returns:
            Tuple of (search_used: bool, num_queries: int)
        """
        # Check web_search_history for this phase
        phase_searches = [
            entry for entry in state.web_search_history
            if entry.get("phase") == phase.name
        ]
        
        if phase_searches:
            total_queries = sum(len(entry.get("queries", [])) for entry in phase_searches)
            return True, total_queries
        
        # Check phase artifacts for web search indicators
        if phase.artifacts:
            web_searches = phase.artifacts.get("web_searches_performed")
            if web_searches:
                try:
                    num = int(web_searches)
                    return num > 0, num
                except (ValueError, TypeError):
                    pass
            
            # Check for web_queries artifact
            web_queries = phase.artifacts.get("web_queries")
            if web_queries:
                queries = web_queries.strip().split("\n")
                return True, len(queries)
        
        return False, 0
    
    def _infer_hallucination(
        self,
        phase: MissionPhase,
        state: MissionState
    ) -> bool:
        """
        Infer whether hallucination occurred during phase execution.
        
        Args:
            phase: Completed phase
            state: Mission state
            
        Returns:
            True if hallucination indicators are present
        """
        # Check phase artifacts for arbiter feedback
        if phase.artifacts:
            arbiter_output = phase.artifacts.get("arbiter_raw_output", "")
            if arbiter_output:
                arbiter_lower = str(arbiter_output).lower()
                if "hallucin" in arbiter_lower:
                    return True
                if "factual error" in arbiter_lower or "incorrect fact" in arbiter_lower:
                    return True
                if "unverified claim" in arbiter_lower or "unsupported" in arbiter_lower:
                    return True
            
            # Check quality score
            quality_score = phase.artifacts.get("quality_score")
            if quality_score is not None:
                try:
                    if float(quality_score) < 0.3:
                        return True
                except (ValueError, TypeError):
                    pass
        
        # Check phase error message
        if phase.error:
            error_lower = phase.error.lower()
            if "hallucin" in error_lower or "factual" in error_lower:
                return True
        
        # Check event logs for retry due to hallucination
        for event in state.event_logs:
            if event.get("phase_name") == phase.name:
                if event.get("event_type") == "phase_retry":
                    reason = str(event.get("data", {}).get("reason", "")).lower()
                    if "hallucin" in reason or "factual" in reason:
                        return True
        
        return False
    
    def _run_deep_analysis_pipeline(self, state: MissionState, phase: MissionPhase) -> None:
        """
        Execute Deep Analysis phase with multi-council pipeline.
        
        Unlike standard phases that use a single council, Deep Analysis uses:
        1. PlannerCouncil: Generate scenarios, failure modes, and hypotheses
        2. SimulationCouncil: Stress-test scenarios and edge cases
        3. EvaluatorCouncil: Assess trade-offs and confidence
        4. ResearcherCouncil: Fill gaps identified by evaluation
        
        This produces structured output with:
        - scenarios: Possible outcomes and their likelihoods
        - failure_modes: What could go wrong
        - tradeoffs: Key decision trade-offs
        - long_horizon_impacts: Long-term implications
        
        Args:
            state: Current mission state
            phase: Deep Analysis phase to execute
        """
        import logging
        logger = logging.getLogger(__name__)
        
        phase.mark_running()
        start_time = time.time()
        
        # Verbose logging: phase start
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_phase_start(phase)
        
        state.log(f"Deep Analysis pipeline started for phase '{phase.name}'")
        
        # Initialize structured output
        deep_analysis_result = {
            "scenarios": [],
            "failure_modes": [],
            "tradeoffs": [],
            "long_horizon_impacts": [],
            "research_gaps": [],
            "confidence_assessment": {},
        }
        
        # Gather prior context
        prior_context = self._get_accumulated_context(state)
        
        # Check if time allows full pipeline
        if self._time_exhausted(state):
            state.log("Deep Analysis: Insufficient time - skipping to lightweight synthesis")
            phase.artifacts["deep_analysis"] = "Time exhausted - used prior findings"
            phase.mark_completed()
            return
        
        # Get supervisor decision if enabled
        decision = None
        if self.enable_supervision:
            decision = self._get_supervisor_decision(state, phase)
        
        try:
            # === Stage 1: PlannerCouncil - Scenario Generation ===
            if state.remaining_minutes() > 1.0:
                state.log("Deep Analysis Stage 1: Scenario generation (PlannerCouncil)")
                
                planner_prompt = f"""Analyze the following objective and prior research to generate:

1. SCENARIOS: 3-5 possible outcomes or paths forward, each with likelihood assessment
2. FAILURE_MODES: Key risks and what could go wrong
3. HYPOTHESES: Testable assumptions that need validation

## Objective
{state.objective}

## Prior Research
{prior_context[:4000]}

## Focus Areas
Provide your analysis in a structured format with clear sections for each category.
Focus on actionable insights and testable predictions."""

                planner_ctx = PlannerContext(
                    objective=state.objective,
                    context={
                        "analysis_prompt": planner_prompt,
                        "constraints": state.constraints.as_dict(),
                    },
                )
                
                self._prepare_council_for_execution(self.planner, state, decision)
                
                council_start = time.time()
                result = self.planner.execute(planner_ctx)
                council_duration = time.time() - council_start
                
                if result.success and result.output:
                    planner_output = str(result.output)
                    
                    # === STRUCTURED SCENARIO MODELING ===
                    # Use ScenarioFactory to parse into structured Scenario objects
                    structured_scenario_set = None
                    if SCENARIO_MODEL_AVAILABLE and self._scenario_factory is not None:
                        try:
                            structured_scenario_set = self._scenario_factory.parse_from_output(
                                planner_output, state.objective
                            )
                            
                            # Validate distinctness
                            structured_scenario_set.validate()
                            
                            # Store structured scenarios
                            deep_analysis_result["structured_scenarios"] = structured_scenario_set.to_dict()
                            
                            # Update epistemic telemetry with scenario distinctness
                            state.update_epistemic_telemetry(
                                scenario_distinctness=structured_scenario_set.distinctness_score
                            )
                            
                            # Extract scenario names for backward compatibility
                            deep_analysis_result["scenarios"] = [
                                f"{s.name}: {s.description[:100]}" 
                                for s in structured_scenario_set.scenarios
                            ]
                            
                            # Log scenario quality
                            state.log(
                                f"[SCENARIO] Parsed {len(structured_scenario_set.scenarios)} structured scenarios, "
                                f"distinctness={structured_scenario_set.distinctness_score:.2f}"
                            )
                            
                            if structured_scenario_set.validation_errors:
                                state.log(f"[SCENARIO] Validation issues: {structured_scenario_set.validation_errors[:3]}")
                            
                        except Exception as e:
                            logger.warning(f"Structured scenario parsing failed: {e}")
                            # Fallback to text extraction
                            deep_analysis_result["scenarios"] = self._extract_list_items(
                                planner_output, ["scenario", "outcome", "path"]
                            )
                    else:
                        # Fallback: extract scenarios from text
                        deep_analysis_result["scenarios"] = self._extract_list_items(
                            planner_output, ["scenario", "outcome", "path"]
                        )
                    
                    deep_analysis_result["failure_modes"] = self._extract_list_items(
                        planner_output, ["risk", "failure", "wrong"]
                    )
                    phase.artifacts["scenario_analysis"] = planner_output[:2000]
                    
                    # Store synthesis-ready scenario text if available
                    if structured_scenario_set is not None:
                        phase.artifacts["structured_scenarios"] = self._scenario_factory.serialize_for_synthesis(
                            structured_scenario_set
                        )
                    
                    state.log(f"Stage 1 complete: {len(deep_analysis_result['scenarios'])} scenarios identified")
                    
                    # Track council time
                    if hasattr(state, 'phase_council_times'):
                        state.phase_council_times[f"{phase.name}_planner"] = council_duration
            
            # === Stage 2: SimulationCouncil - Stress Testing ===
            # Run stress testing if we have scenarios OR if we have sufficient time and context
            if state.remaining_minutes() > 0.5 and (deep_analysis_result["scenarios"] or len(prior_context) > 500):
                state.log("Deep Analysis Stage 2: Stress testing (SimulationCouncil)")
                
                scenarios_str = "\n".join(f"- {s}" for s in deep_analysis_result["scenarios"][:5]) if deep_analysis_result["scenarios"] else "No specific scenarios identified yet"
                
                sim_ctx = SimulationContext(
                    code="",  # No code for analysis mode
                    objective=f"Stress-test potential scenarios and edge cases for {state.objective}:\n{scenarios_str}",
                    focus_scenarios=[s[:200] for s in deep_analysis_result["scenarios"][:3]] if deep_analysis_result["scenarios"] else [],
                )
                
                self._prepare_council_for_execution(self.simulator, state, decision)
                
                council_start = time.time()
                result = self.simulator.execute(sim_ctx)
                council_duration = time.time() - council_start
                
                if result.success and result.output:
                    sim_output = str(result.output)
                    # Extract edge cases and stress test results
                    deep_analysis_result["stress_test_results"] = sim_output[:1500]
                    additional_risks = self._extract_list_items(sim_output, ["edge case", "stress", "vulnerability"])
                    deep_analysis_result["failure_modes"].extend(additional_risks[:5])
                    phase.artifacts["stress_tests"] = sim_output[:1500]
                    state.log(f"Stage 2 complete: Stress testing revealed {len(additional_risks)} additional risks")
                    
                    if hasattr(state, 'phase_council_times'):
                        state.phase_council_times[f"{phase.name}_simulation"] = council_duration
            
            # === Stage 3: EvaluatorCouncil - Trade-off Assessment ===
            if state.remaining_minutes() > 0.5:
                state.log("Deep Analysis Stage 3: Trade-off assessment (EvaluatorCouncil)")
                
                eval_ctx = EvaluatorContext(
                    content_to_evaluate="",  # No content for analysis mode
                    objective=state.objective,
                    iteration=1,
                    prior_analysis=f"Scenarios: {deep_analysis_result['scenarios']}\nRisks: {deep_analysis_result['failure_modes']}",
                )
                
                self._prepare_council_for_execution(self.evaluator, state, decision)
                
                council_start = time.time()
                result = self.evaluator.execute(eval_ctx)
                council_duration = time.time() - council_start
                
                if result.success and result.output:
                    eval_output = str(result.output)
                    # Extract trade-offs and confidence
                    deep_analysis_result["tradeoffs"] = self._extract_list_items(
                        eval_output, ["trade-off", "tradeoff", "balance", "vs", "versus"]
                    )
                    deep_analysis_result["research_gaps"] = self._extract_list_items(
                        eval_output, ["gap", "missing", "unclear", "unknown", "question"]
                    )
                    
                    # Try to extract confidence
                    if hasattr(result.output, 'quality_score'):
                        deep_analysis_result["confidence_assessment"]["quality"] = result.output.quality_score
                    
                    phase.artifacts["tradeoff_analysis"] = eval_output[:1500]
                    state.log(f"Stage 3 complete: {len(deep_analysis_result['tradeoffs'])} trade-offs, {len(deep_analysis_result['research_gaps'])} gaps")
                    
                    # === SSE: Publish deep analysis update for frontend ===
                    if SSE_AVAILABLE and sse_manager:
                        _publish_sse_event(sse_manager.publish_deep_analysis_update(
                            mission_id=state.mission_id,
                            scenarios=deep_analysis_result.get("scenarios", [])[:5],
                            stress_tests=self._extract_list_items(
                                deep_analysis_result.get("stress_test_results", ""),
                                ["test", "stress", "edge"]
                            )[:5] if deep_analysis_result.get("stress_test_results") else [],
                            tradeoffs=deep_analysis_result.get("tradeoffs", [])[:5],
                            robustness_score=deep_analysis_result.get("confidence_assessment", {}).get("quality", 0.0),
                            failure_modes=deep_analysis_result.get("failure_modes", [])[:5],
                            recommendations=deep_analysis_result.get("research_gaps", [])[:5],
                        ))
                    
                    if hasattr(state, 'phase_council_times'):
                        state.phase_council_times[f"{phase.name}_evaluator"] = council_duration
            
            # === Stage 4: ResearcherCouncil - Gap Filling ===
            if state.remaining_minutes() > 0.5 and deep_analysis_result["research_gaps"]:
                state.log("Deep Analysis Stage 4: Gap filling (ResearcherCouncil)")
                
                gaps_str = "\n".join(f"- {g}" for g in deep_analysis_result["research_gaps"][:5])
                
                research_ctx = ResearchContext(
                    objective=f"Address these knowledge gaps for '{state.objective}':\n{gaps_str}",
                    prior_knowledge=prior_context[:2000],
                )
                
                self._prepare_council_for_execution(self.researcher, state, decision)
                
                council_start = time.time()
                result = self.researcher.execute(research_ctx)
                council_duration = time.time() - council_start
                
                if result.success and result.output:
                    research_output = str(result.output)
                    # Extract long-horizon impacts
                    deep_analysis_result["long_horizon_impacts"] = self._extract_list_items(
                        research_output, ["long-term", "future", "impact", "implication", "consequence"]
                    )
                    phase.artifacts["gap_research"] = research_output[:1500]
                    state.log(f"Stage 4 complete: {len(deep_analysis_result['long_horizon_impacts'])} long-term impacts identified")
                    
                    if hasattr(state, 'phase_council_times'):
                        state.phase_council_times[f"{phase.name}_researcher"] = council_duration
            
            # Consolidate results
            phase.artifacts["deep_analysis_structured"] = str(deep_analysis_result)
            
            # Create summary
            summary_parts = []
            if deep_analysis_result["scenarios"]:
                summary_parts.append(f"**Scenarios**: {len(deep_analysis_result['scenarios'])} identified")
            if deep_analysis_result["failure_modes"]:
                summary_parts.append(f"**Risks**: {len(deep_analysis_result['failure_modes'])} failure modes")
            if deep_analysis_result["tradeoffs"]:
                summary_parts.append(f"**Trade-offs**: {len(deep_analysis_result['tradeoffs'])} key decisions")
            if deep_analysis_result["long_horizon_impacts"]:
                summary_parts.append(f"**Long-term**: {len(deep_analysis_result['long_horizon_impacts'])} impacts")
            
            phase.artifacts["deep_analysis_summary"] = "\n".join(summary_parts) if summary_parts else "Analysis complete"
            
        except Exception as e:
            logger.error(f"Deep Analysis pipeline error: {e}")
            state.log(f"Deep Analysis error: {e}")
            phase.artifacts["deep_analysis_error"] = str(e)
        
        # Mark completed
        phase.mark_completed()
        duration = time.time() - start_time
        state.log(f"Deep Analysis phase completed in {duration:.1f}s")
        
        # Track wall time
        if hasattr(state, 'phase_wall_times'):
            state.phase_wall_times[phase.name] = duration
        
        # Verbose logging: phase artifacts
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_phase_artifacts(phase)
        
        # Run multi-view if enabled
        if self.enable_multiview and phase.artifacts:
            content = phase.artifacts.get("deep_analysis_structured", "")
            if content:
                multiview_result = self._run_multiview_evaluation(state, phase, content[:3000], decision)
                if multiview_result.get("disagreement_summary"):
                    phase.artifacts["multiview_summary"] = multiview_result["disagreement_summary"]
        
        # === Orchestration Outcome Logging ===
        if self.enable_orchestration_logging and self.orchestration_store:
            try:
                phase_type = "deep_analysis"
                self._log_phase_outcome(state, phase, decision, phase_type, start_time, duration)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to log Deep Analysis phase outcome: {e}")
    
    def _extract_list_items(self, text: str, keywords: List[str], max_items: int = 10) -> List[str]:
        """
        Extract list items from text that relate to given keywords.
        
        Enhanced with fallback: if no keyword matches found, extracts any list items.
        
        Args:
            text: Text to search
            keywords: Keywords to look for
            max_items: Maximum items to return
            
        Returns:
            List of extracted items
        """
        items = []
        fallback_items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Check if line is a list item (starts with -, *, number, or bullet)
            if line and (
                line.startswith('-') or 
                line.startswith('*') or 
                line.startswith('•') or
                (len(line) > 2 and line[0].isdigit() and line[1] in '.)')
            ):
                # Clean up the item
                cleaned = line.lstrip('-*•0123456789.) ').strip()
                if cleaned and len(cleaned) > 10:
                    # Check if any keyword is in this line (case-insensitive)
                    line_lower = line.lower()
                    if any(kw.lower() in line_lower for kw in keywords):
                        items.append(cleaned[:300])
                    else:
                        # Store as fallback
                        fallback_items.append(cleaned[:300])
        
        # If no keyword matches found, use fallback items
        if not items and fallback_items:
            items = fallback_items[:max_items]
        
        return items[:max_items]
    
    def _run_phase_with_steps(self, state: MissionState, phase: MissionPhase) -> None:
        """
        Execute a phase by running each step through StepExecutor.
        
        This is the Step Engine execution path:
        - Each step is executed by a single specialized model
        - Steps can optionally be evaluated for quality
        - Pivot suggestions can trigger plan modifications
        
        Args:
            state: Current mission state
            phase: Phase containing steps to execute
        """
        state.log(f"Running phase '{phase.name}' with {len(phase.steps)} steps")
        
        # Build shared artifacts from previous phases
        shared_artifacts = self._get_shared_artifacts(state)
        
        # Track previous steps summary as we execute
        previous_steps_summary = ""
        
        for step_index, step in enumerate(phase.steps):
            # Check if we should execute this step
            if not step.needs_execution():
                continue
            
            # Check time budget
            if state.is_expired():
                step.mark_skipped("Mission time expired")
                state.log(f"Step '{step.name}' skipped - time expired")
                continue
            
            # Check remaining time - skip if very low
            remaining_minutes = state.remaining_minutes()
            if remaining_minutes < 0.5:
                step.mark_skipped(f"Insufficient time ({remaining_minutes:.1f} min remaining)")
                state.log(f"Step '{step.name}' skipped - insufficient time")
                continue
            
            # Build execution context
            constraints_notes = self._build_constraints_notes(state.constraints)
            ctx = StepExecutionContext(
                mission_id=state.mission_id,
                mission_objective=state.objective,
                phase_name=phase.name,
                phase_description=phase.description,
                step_index=step_index,
                previous_steps_summary=previous_steps_summary,
                shared_artifacts=shared_artifacts,
                remaining_time_minutes=remaining_minutes,
                constraints_notes=constraints_notes,
            )
            
            state.log(f"Executing step: {step.name} (type: {step.step_type})")
            
            # Execute the step
            result = self.step_executor.execute_step(step, ctx)
            
            # Log the step execution
            state.log_step_execution(
                phase_name=phase.name,
                step_name=step.name,
                step_type=step.step_type,
                chosen_model=result.model_used,
                status=result.status,
                attempts=result.attempts,
                duration_s=result.duration_seconds(),
                pivot_suggestion=result.pivot_suggestion,
                error=result.error,
            )
            
            # Log step execution panel
            output_preview = result.output[:500] + "..." if len(result.output) > 500 else result.output if result.output else None
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_step_execution_panel(
                    step_name=step.name,
                    step_type=step.step_type,
                    model_used=result.model_used,
                    duration_s=result.duration_seconds(),
                    attempts=result.attempts,
                    status=result.status,
                    output_preview=output_preview,
                    pivot_suggestion=result.pivot_suggestion,
                    error=result.error,
                    artifacts=result.artifacts if hasattr(result, 'artifacts') else None
                )
            
            # Publish SSE event for frontend
            if SSE_AVAILABLE and sse_manager:
                _publish_sse_event(sse_manager.publish_step_execution(
                    mission_id=state.mission_id,
                    step_name=step.name,
                    step_type=step.step_type,
                    model_used=result.model_used,
                    duration_s=result.duration_seconds(),
                    status=result.status,
                    attempts=result.attempts,
                    output_preview=output_preview,
                    pivot_suggestion=result.pivot_suggestion,
                    error=result.error,
                    artifacts=result.artifacts if hasattr(result, 'artifacts') else None
                ))
            
            # Handle result
            if result.is_success():
                state.log(f"Step '{step.name}' completed successfully")
                
                # Update previous steps summary
                output_preview = result.output[:500] + "..." if len(result.output) > 500 else result.output
                previous_steps_summary += f"\n\n### {step.name}\n{output_preview}"
                
                # Merge artifacts into shared artifacts
                for name, content in result.artifacts.items():
                    shared_artifacts[f"{step.name}_{name}"] = content
                
                # Store step output in phase artifacts
                phase.artifacts[f"step_{step_index}_{step.name}"] = result.output
                
            else:
                state.log(f"Step '{step.name}' failed: {result.error or 'Unknown error'}")
            
            # Handle pivot suggestions
            if result.pivot_suggestion:
                state.log(f"Pivot suggested: {result.pivot_suggestion}")
                self._handle_pivot_suggestion(state, phase, step_index, result.pivot_suggestion)
            
            # Checkpoint after each step
            self.store.save(state)
        
        # After all steps, consolidate step artifacts into phase
        step_artifacts = phase.get_step_artifacts()
        phase.artifacts.update(step_artifacts)
    
    def _run_phase_with_councils_single(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"],
        phase_type: str
    ) -> None:
        """
        Single-round council-based phase execution.
        
        This path is used when step execution is disabled or when
        a phase has no steps defined. Called from _run_phase for each round.
        
        DeepThinker 2.0:
        - Validates councils against phase spec before execution
        - Validates output artifacts after execution
        """
        # === DeepThinker 2.0: Get phase spec for validation ===
        phase_spec = getattr(phase, '_phase_spec', None)
        
        # Map phase_type to primary council name
        council_mapping = {
            "research": "ResearcherCouncil",
            "design": "PlannerCouncil",
            "implementation": "CoderCouncil",
            "testing": "SimulationCouncil",
            "synthesis": "SynthesisCouncil",
        }
        council_name = council_mapping.get(phase_type, "ResearcherCouncil")
        
        # Validate council usage if strict enforcement is enabled
        if self._strict_phase_enforcement and phase_spec is not None:
            if not self._validate_council_for_phase(phase_spec, council_name):
                state.log(f"[2.0] Council '{council_name}' blocked in phase '{phase.name}'")
                phase.artifacts["council_blocked"] = council_name
                return
        
        if phase_type == "research":
            self._run_research_phase(state, phase, decision)
        elif phase_type == "design":
            self._run_design_phase(state, phase, decision)
        elif phase_type == "implementation":
            self._run_implementation_phase(state, phase, decision)
        elif phase_type == "testing":
            self._run_testing_phase(state, phase, decision)
        elif phase_type == "synthesis":
            self._run_synthesis_phase(state, phase, decision)
        else:
            self._run_research_phase(state, phase, decision)  # Default
        
        # === DeepThinker 2.0: Validate phase output ===
        if phase_spec is not None and phase.artifacts:
            phase.artifacts = self._validate_phase_output(phase_spec, phase.artifacts)
    
    def _build_constraints_notes(self, constraints: MissionConstraints) -> str:
        """Build a string describing mission constraints for step context."""
        notes = []
        if not constraints.allow_internet:
            notes.append("Internet/web search is NOT allowed.")
        if not constraints.allow_code_execution:
            notes.append("Code execution is NOT allowed (design only).")
        if constraints.notes:
            notes.append(constraints.notes)
        return " ".join(notes) if notes else ""
    
    def _get_shared_artifacts(self, state: MissionState) -> Dict[str, str]:
        """Gather artifacts from all completed phases."""
        artifacts = {}
        for phase in state.phases:
            if phase.status == "completed" and phase.artifacts:
                for key, value in phase.artifacts.items():
                    # Truncate large artifacts using OutputLimits (Phase 5.5)
                    from ..core.cognitive_spine import OutputLimits
                    if self.cognitive_spine and len(value) > OutputLimits.MAX_ARTIFACT_CHARS:
                        truncated = self.cognitive_spine.compress_text(
                            value, max_chars=OutputLimits.MAX_ARTIFACT_CHARS
                        )
                    elif len(value) > OutputLimits.MAX_ARTIFACT_CHARS:
                        truncated = value[:OutputLimits.MAX_ARTIFACT_CHARS] + "..."
                    else:
                        truncated = value
                    artifacts[f"{phase.name}_{key}"] = truncated
        return artifacts
    
    def _handle_pivot_suggestion(
        self,
        state: MissionState,
        phase: MissionPhase,
        current_step_index: int,
        suggestion: str,
    ) -> None:
        """
        Handle a pivot suggestion from step execution.
        
        This could:
        - Insert new steps
        - Skip remaining steps
        - Modify future step descriptions
        
        For now, we just log the suggestion. A full implementation would
        parse the suggestion and modify the plan accordingly.
        
        Args:
            state: Current mission state
            phase: Current phase
            current_step_index: Index of the step that made the suggestion
            suggestion: The pivot suggestion text
        """
        # Log the suggestion as an event
        state.log_event(
            event_type="pivot_suggestion",
            data={
                "step_index": current_step_index,
                "suggestion": suggestion,
            },
            phase_name=phase.name
        )
        
        # Future: Parse suggestion and modify plan
        # For now, this is informational only
        # 
        # Example enhancements:
        # - If suggestion mentions "skip", mark remaining steps as skipped
        # - If suggestion mentions "add step", insert a new StepDefinition
        # - If suggestion mentions "focus on X", update subsequent step descriptions
    
    def _get_accumulated_context(self, state: MissionState, include_full_history: bool = False) -> str:
        """
        Gather comprehensive context from previous phases and council outputs.
        
        Enhanced to include:
        - All prior council outputs (not just artifacts)
        - Multiple iteration summaries
        - Evaluation score evolution
        - Multi-view disagreement summaries
        - Memory system insights (from past missions)
        
        Args:
            state: Current mission state
            include_full_history: If True, include full council output history
            
        Returns:
            Accumulated context string
        """
        context_parts = [f"Mission Objective: {state.objective}"]
        
        # Add time context
        remaining = state.remaining_minutes()
        elapsed = state.constraints.time_budget_minutes - remaining
        context_parts.append(f"Time: {elapsed:.1f}min elapsed, {remaining:.1f}min remaining")
        
        # Add memory insights (from past missions)
        memory_context = self._get_memory_context(state)
        if memory_context:
            context_parts.append(memory_context)
        
        # Add phase summaries
        for phase in state.phases:
            if phase.status == "completed" and phase.artifacts:
                context_parts.append(f"\n--- {phase.name} Results ---")
                context_parts.append(f"Iterations: {phase.iterations}")
                
                for key, value in phase.artifacts.items():
                    # Skip internal artifacts
                    if key.startswith("_"):
                        continue
                    # Truncate long artifacts
                    value_str = str(value)
                    truncated = value_str[:2000] + "..." if len(value_str) > 2000 else value_str
                    context_parts.append(f"{key}: {truncated}")
        
        # Add evaluation score evolution
        evaluation_scores = []
        for phase in state.phases:
            if phase.artifacts.get("quality_score"):
                try:
                    score = float(phase.artifacts["quality_score"])
                    evaluation_scores.append(f"{phase.name}: {score:.1f}")
                except (ValueError, TypeError):
                    pass
        
        if evaluation_scores:
            context_parts.append("\n--- Evaluation Evolution ---")
            context_parts.append(", ".join(evaluation_scores))
        
        # Add multi-view disagreement summaries
        if self._multiview_disagreements:
            context_parts.append("\n--- Multi-View Analysis ---")
            for disagreement in self._multiview_disagreements[-5:]:  # Last 5
                phase_name = disagreement.get("phase", "Unknown")
                agreement = disagreement.get("agreement_score", 0)
                context_parts.append(f"{phase_name}: Agreement {agreement:.1%}")
                
                # Add brief summaries if available
                opt_summary = disagreement.get("optimist_summary", "")
                skep_summary = disagreement.get("skeptic_summary", "")
                if opt_summary:
                    context_parts.append(f"  Optimist: {opt_summary[:200]}...")
                if skep_summary:
                    context_parts.append(f"  Skeptic: {skep_summary[:200]}...")
        
        # Add council output history if requested
        if include_full_history:
            context_parts.append("\n--- Council Output History ---")
            for council_name, outputs in self._council_output_history.items():
                if outputs:
                    context_parts.append(f"\n{council_name.title()} Council ({len(outputs)} outputs):")
                    for i, output in enumerate(outputs[-3:]):  # Last 3 outputs
                        output_str = str(output)[:500]
                        context_parts.append(f"  [{i+1}] {output_str}...")
        
        return "\n".join(context_parts)
    
    def _get_memory_context(self, state: MissionState) -> str:
        """
        Get context from memory system for council enrichment.
        
        This provides councils with read-only access to past mission insights.
        Uses the reason_over method for structured memory analysis.
        
        Args:
            state: Current mission state
            
        Returns:
            Memory context string, or empty string if unavailable
        """
        if not self.memory:
            return ""
        
        try:
            # Use reason_over for structured memory analysis
            memory_summary = self.memory.reason_over(
                objective=state.objective,
                limit=5
            )
            
            # Track memory usage in state
            if hasattr(state, 'memory_items_used'):
                state.memory_items_used = memory_summary.get("memory_used_count", 0)
            
            # Log memory operations panel
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                memory_items = memory_summary.get("memory_items", [])
                known_gaps = memory_summary.get("known_gaps", [])
                used_in_prompt = bool(memory_summary.get("memory_used_count", 0) > 0)
                
                # Extract memory item details
                memory_items_list = []
                if isinstance(memory_items, list):
                    for item in memory_items:
                        if isinstance(item, tuple):
                            doc, score = item
                            memory_items_list.append({
                                'title': getattr(doc, 'title', str(doc)[:60]),
                                'summary': str(doc)[:100] if hasattr(doc, '__str__') else '',
                                'score': score
                            })
                        elif isinstance(item, dict):
                            memory_items_list.append(item)
                
                verbose_logger.log_memory_operations_panel(
                    memory_used_count=memory_summary.get("memory_used_count", 0),
                    memory_items=memory_items_list,
                    used_in_prompt=used_in_prompt,
                    known_gaps=known_gaps if isinstance(known_gaps, list) else [],
                    compression_operations=None  # Will be added when compression happens
                )
            
            # Format for prompt injection
            formatted = self.memory.format_for_prompt(
                memory_summary,
                include_gaps=True
            )
            
            # Also populate state.knowledge_context if we have formatted knowledge
            if formatted and hasattr(state, 'knowledge_context'):
                if not state.knowledge_context.get("formatted"):
                    state.knowledge_context["formatted"] = formatted
                    state.knowledge_context["prior_knowledge"] = memory_summary.get("prior_knowledge", [])
                    state.knowledge_context["sources"] = memory_summary.get("memory_sources", [])
                    state.knowledge_context["items_count"] = memory_summary.get("memory_used_count", 0)
            elif formatted and not hasattr(state, 'knowledge_context'):
                state.knowledge_context = {
                    "formatted": formatted,
                    "prior_knowledge": memory_summary.get("prior_knowledge", []),
                    "known_gaps": memory_summary.get("known_gaps", []),
                    "sources": memory_summary.get("memory_sources", []),
                    "items_count": memory_summary.get("memory_used_count", 0),
                }
            
            if formatted:
                return formatted
            
            # Fallback to legacy format if format_for_prompt returns empty
            parts = []
            
            # Get relevant past insights
            past_insights = self.memory.retrieve_relevant_past_insights(
                state.objective, limit=2
            )
            if past_insights:
                parts.append("\n--- Insights from Similar Past Missions ---")
                for summary, score in past_insights:
                    if score >= 0.4:  # Only include relevant matches
                        parts.append(f"Mission Type: {summary.mission_type or 'General'}")
                        for insight in summary.key_insights[:2]:
                            parts.append(f"  - {insight[:200]}")
                        if summary.resolved_hypotheses:
                            parts.append(f"  Confirmed: {summary.resolved_hypotheses[0][:150]}")
            
            # Get relevant evidence from global store
            evidence = self.memory.retrieve_relevant_evidence(state.objective, limit=3)
            if evidence:
                parts.append("\n--- Relevant Knowledge from Past Missions ---")
                for doc, score in evidence:
                    if score >= 0.4:
                        text = doc.get("text", "")[:200]
                        source = doc.get("mission_id", "unknown")[:8]
                        parts.append(f"  [from memory: {source}] {text}...")
            
            return "\n".join(parts) if parts else ""
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Memory context retrieval skipped: {e}")
            return ""
    
    def _get_memory_summary(self, state: MissionState) -> Dict[str, Any]:
        """
        Get structured memory summary for logging and CLI display.
        
        Args:
            state: Current mission state
            
        Returns:
            Memory summary dictionary
        """
        if not self.memory:
            return {
                "memory_used_count": 0,
                "memory_items_titles": [],
                "used_in_prompt": False,
            }
        
        try:
            return self.memory.reason_over(
                objective=state.objective,
                limit=5
            )
        except Exception:
            return {
                "memory_used_count": 0,
                "memory_items_titles": [],
                "used_in_prompt": False,
            }
    
    def _get_multiview_summary(self) -> str:
        """
        Get a summary of all multi-view disagreements.
        
        Returns:
            Summary string for arbiter consumption
        """
        if not self._multiview_disagreements:
            return "No multi-view analysis performed."
        
        parts = ["Multi-View Perspective Summary:"]
        
        # Calculate overall statistics
        agreements = [d.get("agreement_score", 0) for d in self._multiview_disagreements]
        avg_agreement = sum(agreements) / len(agreements) if agreements else 0
        
        parts.append(f"Average Agreement: {avg_agreement:.1%}")
        parts.append(f"Phases Analyzed: {len(self._multiview_disagreements)}")
        
        # Identify key disagreements (low agreement)
        key_disagreements = [d for d in self._multiview_disagreements if d.get("agreement_score", 1) < 0.5]
        if key_disagreements:
            parts.append("\nKey Disagreements:")
            for d in key_disagreements:
                parts.append(f"- {d.get('phase', 'Unknown')}: {d.get('agreement_score', 0):.1%} agreement")
        
        return "\n".join(parts)
    
    def _run_research_phase(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """Execute a research/reconnaissance phase."""
        # Prepare council for execution (sets mission_id and applies supervisor decision)
        self._prepare_council_for_execution(self.researcher, state, decision)
        
        # Check internet constraint
        if not state.constraints.allow_internet:
            phase.artifacts["note"] = "Internet research disabled by constraints"
            state.log("Skipping web research (disabled by constraints)")
        
        context = self._get_accumulated_context(state)
        
        # Get data needs and questions from last evaluator output
        data_needs = []
        unresolved_questions = []
        if self._last_evaluator_output:
            if hasattr(self._last_evaluator_output, 'data_needs'):
                data_needs = self._last_evaluator_output.data_needs
            if hasattr(self._last_evaluator_output, 'questions'):
                unresolved_questions = self._last_evaluator_output.questions
        
        # Get current subgoals
        subgoals = self._current_subgoals.copy() if self._current_subgoals else []
        
        # Determine if evidence is required (for depth contracts)
        requires_evidence = False
        if hasattr(state, 'constraints') and state.constraints.effort.value in ["deep", "marathon"]:
            requires_evidence = True
        
        # Get knowledge context from state if available
        knowledge_ctx = None
        if hasattr(state, 'knowledge_context') and state.knowledge_context:
            knowledge_ctx = state.knowledge_context.get("formatted", "")
        
        # Extract planner artifacts from previous phases for context propagation
        focus_areas, planner_requirements = self._extract_planner_context(state)
        
        research_context = ResearchContext(
            objective=f"{state.objective}\n\nPhase: {phase.name}\n{phase.description}",
            focus_areas=focus_areas,  # Pass planner focus areas
            prior_knowledge=context if context else None,
            constraints="No internet access" if not state.constraints.allow_internet else None,
            planner_requirements=planner_requirements,  # Pass planner requirements
            allow_internet=state.constraints.allow_internet,
            data_needs=data_needs,
            unresolved_questions=unresolved_questions,
            requires_evidence=requires_evidence,
            subgoals=subgoals,
            knowledge_context=knowledge_ctx,
            current_phase=phase.name,  # Inject phase name to prevent confusion
        )
        
        # Log iteration context
        if state.iteration_count > 1 and (data_needs or unresolved_questions or subgoals):
            state.log(f"Research phase using: {len(data_needs)} data needs, "
                     f"{len(unresolved_questions)} questions, {len(subgoals)} subgoals")
        
        # Check search trigger with objective-aware logic
        trigger_search = research_context.allow_internet
        search_rationale = ""
        if self._has_search_triggers and self._search_trigger_manager:
            # Get uncertainty from convergence state
            uncertainty = 0.5
            if self._convergence_state:
                uncertainty = 1.0 - self._convergence_state.confidence_score
            
            search_decision = self._search_trigger_manager.should_trigger_search(
                objective=state.objective,
                phase_name=phase.name,
                uncertainty=uncertainty,
                data_needs=data_needs,
                unresolved_questions=unresolved_questions,
            )
            trigger_search = search_decision.should_search and research_context.allow_internet
            search_rationale = search_decision.reason
            
            # Log search decision
            if trigger_search:
                state.log(f"Internet search triggered: {search_rationale}")
                if search_decision.queries:
                    state.log(f"Search queries: {search_decision.queries[:2]}")
            else:
                state.log(f"Internet search skipped: {search_rationale}")
        
        # Override allow_internet if trigger says no (preserve other fields from first context)
        research_context = ResearchContext(
            objective=research_context.objective,
            focus_areas=research_context.focus_areas,  # Preserve planner focus areas
            prior_knowledge=research_context.prior_knowledge,
            constraints=research_context.constraints,
            planner_requirements=research_context.planner_requirements,  # Preserve planner requirements
            allow_internet=trigger_search,
            data_needs=data_needs,
            unresolved_questions=unresolved_questions,
            requires_evidence=requires_evidence,
            subgoals=subgoals,
            knowledge_context=knowledge_ctx,
            current_phase=research_context.current_phase,  # Preserve phase context
        )
        
        # Verbose logging: council activation
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.researcher.__class__)
            verbose_logger.log_council_activated(
                "ResearcherCouncil",
                research_context,
                models=decision.models if decision else None
            )
        
        # Use research method which includes auto web search
        result = self.researcher.research(
            objective=research_context.objective,
            focus_areas=research_context.focus_areas,
            prior_knowledge=research_context.prior_knowledge,
            planner_requirements=research_context.planner_requirements,
            allow_internet=research_context.allow_internet,
            data_needs=research_context.data_needs,
            unresolved_questions=research_context.unresolved_questions,
            requires_evidence=research_context.requires_evidence,
            subgoals=research_context.subgoals
        )
        
        # Verbose logging: council output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete(
                "ResearcherCouncil",
                success=result.success,
                model_used=decision.models[0] if decision and decision.models else None
            )
            if result.output:
                verbose_logger.log_context_output("Research Output", result.output)
        phase.iterations += 1
        
        # Track output through spine for budget monitoring
        self._track_council_output("ResearcherCouncil", result, state)
        
        if result.success and result.output:
            findings = result.output
            phase.artifacts["research_findings"] = findings.raw_output if hasattr(findings, 'raw_output') else str(findings)
            if hasattr(findings, 'key_points') and findings.key_points:
                phase.artifacts["key_points"] = "\n".join(f"- {p}" for p in findings.key_points)
            if hasattr(findings, 'recommendations') and findings.recommendations:
                phase.artifacts["recommendations"] = "\n".join(f"- {r}" for r in findings.recommendations)
            # Store sources for governance evidence budget checks
            if hasattr(findings, 'sources_suggested') and findings.sources_suggested:
                phase.artifacts["sources"] = findings.sources_suggested
            # Capture confidence score from findings
            if hasattr(findings, 'confidence_score') and findings.confidence_score:
                phase.artifacts["confidence_score"] = f"{findings.confidence_score:.2f}"
            
            # Track web search activity
            if hasattr(findings, 'web_search_count'):
                phase.artifacts["web_searches_performed"] = str(findings.web_search_count)
                if hasattr(findings, 'queries_executed') and findings.queries_executed:
                    phase.artifacts["web_queries"] = "\n".join(findings.queries_executed)
                
                # Log web search activity
                if findings.web_search_count > 0:
                    state.log(f"Performed {findings.web_search_count} web searches")
            
            # Track external knowledge artifact (anti-hallucination)
            if hasattr(findings, 'external_knowledge') and findings.external_knowledge:
                import json
                try:
                    external_knowledge_data = {
                        "queries": findings.external_knowledge.queries,
                        "sources": findings.external_knowledge.sources,
                        "evidence_strength": findings.external_knowledge.evidence_strength,
                        "coverage": findings.external_knowledge.coverage,
                        "confidence_delta": findings.external_knowledge.confidence_delta,
                        "search_failed": findings.external_knowledge.search_failed
                    }
                    phase.artifacts["external_knowledge"] = json.dumps(external_knowledge_data)
                    state.log(
                        f"External knowledge: strength={findings.external_knowledge.evidence_strength}, "
                        f"coverage={findings.external_knowledge.coverage}, "
                        f"confidence_delta={findings.external_knowledge.confidence_delta:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to serialize external_knowledge artifact: {e}")
            
            # Track iteration-driving fields (with deduplication)
            if hasattr(findings, 'gaps') and findings.gaps:
                unique_gaps = self._deduplicate_list(findings.gaps)
                phase.artifacts["research_gaps"] = "\n".join(f"- {g}" for g in unique_gaps[:10])
            if hasattr(findings, 'unresolved_questions') and findings.unresolved_questions:
                unique_questions = self._deduplicate_list(findings.unresolved_questions)
                phase.artifacts["unresolved_questions"] = "\n".join(f"- {q}" for q in unique_questions[:10])
            if hasattr(findings, 'evidence_requests') and findings.evidence_requests:
                unique_requests = self._deduplicate_list(findings.evidence_requests)
                phase.artifacts["evidence_requests"] = "\n".join(f"- {e}" for e in unique_requests[:10])
            if hasattr(findings, 'next_focus_areas') and findings.next_focus_areas:
                unique_areas = self._deduplicate_list(findings.next_focus_areas)
                phase.artifacts["next_focus_areas"] = "\n".join(f"- {f}" for f in unique_areas[:10])
            
            # === SSE: Publish epistemic update for frontend ===
            if SSE_AVAILABLE and sse_manager:
                questions = getattr(findings, 'unresolved_questions', []) or []
                evidence = getattr(findings, 'evidence_requests', []) or []
                focus = getattr(findings, 'next_focus_areas', []) or []
                
                # Try to get grounding score from epistemic telemetry
                grounding_score = 0.0
                if hasattr(state, 'epistemic_telemetry'):
                    grounding_score = state.epistemic_telemetry.get('grounding_score', 0.0)
                
                _publish_sse_event(sse_manager.publish_epistemic_update(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    unresolved_questions=questions[:10],
                    evidence_requests=evidence[:10],
                    focus_areas=focus[:10],
                    claims_count=getattr(findings, 'claims_count', 0) if hasattr(findings, 'claims_count') else len(getattr(findings, 'key_points', [])),
                    verified_claims=getattr(findings, 'verified_claims', 0) if hasattr(findings, 'verified_claims') else 0,
                    grounding_score=grounding_score,
                ))
            
            # Update iteration context if manager is available
            self._update_research_iteration_context(state, phase, findings)
            
            # Evaluate research if evaluator is available
            self._evaluate_and_update_research_context(state, phase, findings)
        else:
            phase.artifacts["error"] = result.error or "Research failed"
    
    def _run_design_phase(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """Execute a design/architecture phase."""
        # Prepare council for execution
        self._prepare_council_for_execution(self.planner, state, decision)
        
        context = self._get_accumulated_context(state)
        
        # Get knowledge context from state if available
        knowledge_ctx = None
        if hasattr(state, 'knowledge_context') and state.knowledge_context:
            knowledge_ctx = state.knowledge_context.get("formatted", "")
        
        # Use planner for design
        planner_context = PlannerContext(
            objective=f"{state.objective}\n\nPhase: {phase.name}\n{phase.description}",
            context={"prior_work": context},
            max_iterations=min(3, state.constraints.max_iterations),
            knowledge_context=knowledge_ctx,
        )
        
        # Verbose logging: council activation
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.planner.__class__)
            verbose_logger.log_council_activated(
                "PlannerCouncil",
                planner_context,
                models=decision.models if decision else None
            )
        
        result = self.planner.execute(planner_context)
        
        # Verbose logging: council output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete(
                "PlannerCouncil",
                success=result.success,
                model_used=decision.models[0] if decision and decision.models else None
            )
            if result.output:
                verbose_logger.log_context_output("Design Output", result.output)
        phase.iterations += 1
        
        # Track output through spine for budget monitoring
        self._track_council_output("PlannerCouncil", result, state)
        
        if result.success and result.output:
            plan = result.output
            # Store the design/plan
            if hasattr(plan, 'objective_analysis'):
                phase.artifacts["analysis"] = plan.objective_analysis
            if hasattr(plan, 'workflow_strategy'):
                phase.artifacts["strategy"] = str(plan.workflow_strategy)
            if hasattr(plan, 'agent_requirements'):
                phase.artifacts["requirements"] = str(plan.agent_requirements)
            
            # Get raw output as fallback
            raw_str = str(plan)
            if len(phase.artifacts) == 0:
                phase.artifacts["design"] = raw_str
        else:
            phase.artifacts["error"] = result.error or "Design failed"
        
        # Optionally evaluate the design
        if phase.artifacts.get("design") or phase.artifacts.get("analysis"):
            self._prepare_council_for_execution(self.evaluator, state, decision)
            
            design_content = phase.artifacts.get("design") or phase.artifacts.get("analysis", "")
            
            # Get knowledge context
            eval_knowledge = None
            if hasattr(state, 'knowledge_context') and state.knowledge_context:
                eval_knowledge = state.knowledge_context.get("formatted", "")
            
            eval_context = EvaluatorContext(
                objective=f"Evaluate this design for: {state.objective}",
                content_to_evaluate=design_content,  # Design document to evaluate
                quality_threshold=6.0,
                knowledge_context=eval_knowledge,
            )
            eval_result = self.evaluator.execute(eval_context)
            if eval_result.success and eval_result.output:
                phase.artifacts["design_evaluation"] = str(eval_result.output)
                
                # Track evaluator output
                self._last_evaluator_output = eval_result.output
                if self._convergence_state:
                    self._convergence_state.update_from_evaluator(eval_result.output)
    
    def _run_implementation_phase(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """Execute an implementation/coding phase."""
        # Prepare council for execution
        self._prepare_council_for_execution(self.coder, state, decision)
        
        context = self._get_accumulated_context(state)
        
        # Check code execution constraint
        if not state.constraints.allow_code_execution:
            state.log("Code execution disabled - generating design-only code")
        
        # Gather research findings
        research_findings = ""
        for p in state.phases:
            if "research" in self._classify_phase(p) and p.artifacts.get("research_findings"):
                research_findings += p.artifacts["research_findings"] + "\n"
        
        # Get design requirements
        design_reqs = ""
        for p in state.phases:
            if "design" in self._classify_phase(p):
                if p.artifacts.get("requirements"):
                    design_reqs += p.artifacts["requirements"] + "\n"
                if p.artifacts.get("design"):
                    design_reqs += p.artifacts["design"] + "\n"
        
        coder_context = CoderContext(
            objective=f"{state.objective}\n\nPhase: {phase.name}\n{phase.description}",
            research_findings=research_findings if research_findings else None,
            planner_requirements=design_reqs if design_reqs else None
        )
        
        # Verbose logging: council activation
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.coder.__class__)
            verbose_logger.log_council_activated(
                "CoderCouncil",
                coder_context,
                models=decision.models if decision else None
            )
        
        result = self.coder.execute(coder_context)
        phase.iterations += 1
        
        # Verbose logging: council output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete(
                "CoderCouncil",
                success=result.success,
                model_used=decision.models[0] if decision and decision.models else None
            )
            if result.output:
                verbose_logger.log_context_output("Code Output", result.output)
        
        # Track output through spine for budget monitoring
        self._track_council_output("CoderCouncil", result, state)
        
        if result.success and result.output:
            code_output = result.output
            phase.artifacts["code"] = code_output.code if hasattr(code_output, 'code') else str(code_output)
            if hasattr(code_output, 'explanation') and code_output.explanation:
                phase.artifacts["explanation"] = code_output.explanation
            if hasattr(code_output, 'dependencies') and code_output.dependencies:
                phase.artifacts["dependencies"] = ", ".join(code_output.dependencies)
            
            # Evaluate the code
            self._prepare_council_for_execution(self.evaluator, state, decision)
            
            eval_context = EvaluatorContext(
                objective=state.objective,
                content_to_evaluate=phase.artifacts["code"],
                quality_threshold=7.0
            )
            eval_result = self.evaluator.execute(eval_context)
            if eval_result.success and eval_result.output:
                evaluation = eval_result.output
                phase.artifacts["evaluation"] = str(evaluation)
                if hasattr(evaluation, 'quality_score'):
                    phase.artifacts["quality_score"] = str(evaluation.quality_score)
                
                # Track evaluator output for convergence
                self._last_evaluator_output = evaluation
                
                # Update convergence state from evaluator
                if self._convergence_state:
                    self._convergence_state.update_from_evaluator(evaluation)
                
                # Store confidence
                if hasattr(evaluation, 'confidence_score'):
                    phase.artifacts["confidence_score"] = str(evaluation.confidence_score)
                
                # Store missing info for report
                if hasattr(evaluation, 'missing_info') and evaluation.missing_info:
                    phase.artifacts["missing_info"] = "\n".join(f"- {info}" for info in evaluation.missing_info)
                
                # Store questions for report
                if hasattr(evaluation, 'questions') and evaluation.questions:
                    phase.artifacts["outstanding_questions"] = "\n".join(f"- {q}" for q in evaluation.questions)
        else:
            phase.artifacts["error"] = result.error or "Implementation failed"
    
    def _run_testing_phase(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """Execute a testing/simulation phase."""
        # Prepare council for execution
        self._prepare_council_for_execution(self.simulator, state, decision)
        
        # Find code from implementation phase
        code = ""
        for p in state.phases:
            if p.artifacts.get("code"):
                code = p.artifacts["code"]
                break
        
        if not code:
            phase.artifacts["note"] = "No code found to test"
            state.log("Testing phase: No code available to test")
            return
        
        # Check execution constraint
        if not state.constraints.allow_code_execution:
            state.log("Code execution disabled - running simulation analysis only")
        
        sim_context = SimulationContext(
            code=code,
            objective=state.objective
        )
        
        # Verbose logging: council activation
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.simulator.__class__)
            verbose_logger.log_council_activated(
                "SimulationCouncil",
                sim_context,
                models=decision.models if decision else None
            )
        
        result = self.simulator.execute(sim_context)
        phase.iterations += 1
        
        # Verbose logging: council output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete(
                "SimulationCouncil",
                success=result.success,
                model_used=decision.models[0] if decision and decision.models else None
            )
            if result.output:
                verbose_logger.log_context_output("Simulation Output", result.output)
        
        # Track output through spine for budget monitoring
        self._track_council_output("SimulationCouncil", result, state)
        
        if result.success and result.output:
            findings = result.output
            phase.artifacts["simulation_findings"] = findings.raw_output if hasattr(findings, 'raw_output') else str(findings)
            
            if hasattr(findings, 'edge_cases') and findings.edge_cases:
                phase.artifacts["edge_cases"] = "\n".join(f"- {e}" for e in findings.edge_cases)
            if hasattr(findings, 'robustness_score'):
                phase.artifacts["robustness_score"] = str(findings.robustness_score)
            if hasattr(findings, 'recommendations') and findings.recommendations:
                phase.artifacts["test_recommendations"] = "\n".join(f"- {r}" for r in findings.recommendations)
        else:
            phase.artifacts["error"] = result.error or "Testing failed"
        
        # Run evaluator on the simulation findings
        if phase.artifacts.get("simulation_findings"):
            self._prepare_council_for_execution(self.evaluator, state, decision)
            
            eval_context = EvaluatorContext(
                objective=f"Evaluate test coverage for: {state.objective}",
                content_to_evaluate=code,
                quality_threshold=6.0
            )
            eval_result = self.evaluator.execute(eval_context)
            if eval_result.success and eval_result.output:
                phase.artifacts["test_evaluation"] = str(eval_result.output)
    
    def _run_synthesis_phase(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """
        Execute a synthesis/report phase with iterative refinement.
        
        Uses SynthesisContext for stateful iteration when available,
        ensuring each loop produces different output based on gaps.
        """
        # Prepare council for execution
        self._prepare_council_for_execution(self.planner, state, decision)
        
        # Gather all artifacts from previous phases
        all_artifacts = {}
        for p in state.phases:
            if p.artifacts:
                all_artifacts[p.name] = p.artifacts
        
        context = self._get_accumulated_context(state)
        
        # Try to use iterative synthesis with SynthesisContext
        if SYNTHESIS_CONTEXT_AVAILABLE and SynthesisContext is not None:
            self._run_iterative_synthesis(state, phase, context, decision)
            return
        
        # Fallback: Use planner to synthesize (legacy mode)
        synthesis_prompt = f"""Synthesize the results of this mission into a comprehensive final report.

OBJECTIVE:
{state.objective}

PHASE RESULTS:
{context}

Create a structured final report with:
1. Executive Summary
2. Key Findings
3. Deliverables (code, designs, recommendations)
4. Conclusions
5. Next Steps (if applicable)"""

        planner_context = PlannerContext(
            objective=synthesis_prompt,
            max_iterations=1,
            quality_threshold=5.0
        )
        
        # Verbose logging: council activation for synthesis
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_activated(
                "PlannerCouncil (Synthesis)",
                {"objective": "Synthesize mission results"},
                models=decision.models if decision else None
            )
        
        result = self.planner.execute(planner_context)
        phase.iterations += 1
        
        # Verbose logging: synthesis output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete(
                "PlannerCouncil (Synthesis)",
                success=result.success
            )
            if result.output:
                verbose_logger.log_context_output("Synthesis Output", result.output)
        
        # Track output through spine for budget monitoring
        self._track_council_output("PlannerCouncil", result, state)
        
        if result.success and result.output:
            phase.artifacts["synthesis_report"] = str(result.output)
        
        # Store in mission's final artifacts
        state.final_artifacts["synthesis"] = phase.artifacts.get("synthesis_report", "")
    
    def _run_iterative_synthesis(
        self,
        state: MissionState,
        phase: MissionPhase,
        prior_findings: str,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """
        Run iterative synthesis with SynthesisContext.
        
        Iterates until:
        - All gaps are addressed
        - Max iterations reached
        - Time runs out
        
        Args:
            state: Current mission state
            phase: Current phase
            prior_findings: Accumulated findings from prior phases
            decision: Optional supervisor decision
        """
        # Gather unresolved issues from evaluator history
        unresolved_issues = []
        evaluator_feedback = []
        
        if self._last_evaluator_output:
            if hasattr(self._last_evaluator_output, 'issues'):
                unresolved_issues.extend([str(i) for i in self._last_evaluator_output.issues[:5]])
            if hasattr(self._last_evaluator_output, 'questions'):
                unresolved_issues.extend(self._last_evaluator_output.questions[:3])
            if hasattr(self._last_evaluator_output, 'recommendations'):
                evaluator_feedback.extend(self._last_evaluator_output.recommendations[:3])
        
        # Get structural gaps from context manager if available
        structural_gaps = []
        if ITERATION_CONTEXT_AVAILABLE and self._iteration_context_manager is not None:
            ctx_state = self._iteration_context_manager.get_state()
            if ctx_state is not None:
                structural_gaps = ctx_state.gaps[:5]
        
        # Initialize synthesis context
        synthesis_context = SynthesisContext(
            objective=state.objective,
            prior_findings=prior_findings[:4000],
            unresolved_issues=unresolved_issues,
            structural_gaps=structural_gaps,
            evaluator_feedback=evaluator_feedback,
            iteration=1,
            max_iterations=3
        )
        
        max_synthesis_iterations = min(3, max(1, int(state.remaining_minutes() / 2)))
        accumulated_content = []
        
        state.log(f"Starting iterative synthesis (max {max_synthesis_iterations} iterations)")
        
        for iteration in range(1, max_synthesis_iterations + 1):
            # Check time
            if state.remaining_minutes() < 0.5:
                state.log("Synthesis stopped - insufficient time")
                break
            
            synthesis_context.iteration = iteration
            
            # Verbose logging
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_council_activated(
                    f"PlannerCouncil (Synthesis #{iteration})",
                    {"iteration": iteration, "gaps": len(synthesis_context.structural_gaps)},
                    models=decision.models if decision else None
                )
            
            # Run synthesis
            result = self.planner.synthesize(synthesis_context)
            phase.iterations += 1
            
            if result.success and result.output:
                synthesis_result = result.output
                accumulated_content.append(synthesis_result.content)
                
                # Log progress
                state.log(
                    f"Synthesis iteration {iteration}: "
                    f"sections={len(getattr(synthesis_result, 'sections_completed', []))}, "
                    f"remaining_gaps={len(getattr(synthesis_result, 'remaining_gaps', []))}, "
                    f"complete={getattr(synthesis_result, 'is_complete', False)}"
                )
                
                # Store iteration artifacts
                phase.artifacts[f"synthesis_iteration_{iteration}"] = synthesis_result.content[:2000]
                
                # === SSE: Publish synthesis iteration for frontend ===
                if SSE_AVAILABLE and sse_manager:
                    sections_count = len(getattr(synthesis_result, 'sections_completed', []))
                    word_count = len(synthesis_result.content.split()) if synthesis_result.content else 0
                    quality_score = getattr(synthesis_result, 'quality_score', 0.0)
                    is_final = getattr(synthesis_result, 'is_complete', False)
                    
                    _publish_sse_event(sse_manager.publish_synthesis_iteration(
                        mission_id=state.mission_id,
                        iteration=iteration,
                        content_preview=synthesis_result.content[:500] if synthesis_result.content else "",
                        quality_score=quality_score,
                        word_count=word_count,
                        sections_count=sections_count,
                        is_final=is_final,
                    ))
                
                # Check if complete
                if getattr(synthesis_result, 'is_complete', False):
                    state.log("Synthesis complete - no remaining gaps")
                    break
                
                # Update context for next iteration
                synthesis_context = self.planner.update_synthesis_context(
                    synthesis_context,
                    synthesis_result
                )
                
                # Stop if no work remaining
                if not synthesis_context.has_work_remaining():
                    state.log("Synthesis complete - no work remaining")
                    break
            else:
                state.log(f"Synthesis iteration {iteration} failed: {result.error}")
                break
        
        # Combine all synthesis outputs
        final_report = "\n\n---\n\n".join(accumulated_content) if accumulated_content else ""
        
        # DeepThinker 2.0: Validate synthesis output
        validation_passed = self._validate_synthesis_output(final_report)
        if not validation_passed:
            state.log("[SYNTHESIS] Warning: synthesis output failed validation, using fallback")
            # Try to salvage from prior_findings if available
            fallback_report = None
            if prior_findings and len(prior_findings) > 100:
                fallback_report = f"## Synthesis (Fallback)\n\n{prior_findings[:4000]}"
                # Validate the fallback as well
                if self._validate_synthesis_output(fallback_report):
                    final_report = fallback_report
                    state.log("[SYNTHESIS] Using prior findings as fallback synthesis")
                    validation_passed = True
                else:
                    state.log("[SYNTHESIS] Fallback also failed validation")
            
            # If both original and fallback failed validation, mark as terminal failure
            if not validation_passed:
                error_msg = "Synthesis validation failed: no valid output or fallback"
                state.log(f"[SYNTHESIS] {error_msg}")
                phase.artifacts["_validation_failed_terminal"] = True
                phase.artifacts["synthesis_report"] = final_report if final_report else ""
                phase.mark_failed(error_msg)
                # Store empty synthesis to prevent further processing
                state.final_artifacts["synthesis"] = ""
                
                # Verbose logging
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_council_execution_complete(
                        "PlannerCouncil (Synthesis)",
                        success=False
                    )
                return  # Early return to prevent governance check on terminal failure
        
        phase.artifacts["synthesis_report"] = final_report
        
        # Store in mission's final artifacts
        state.final_artifacts["synthesis"] = final_report
        
        # Verbose logging
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete(
                "PlannerCouncil (Synthesis)",
                success=bool(final_report)
            )
    
    def _run_multiview_evaluation(
        self,
        state: MissionState,
        phase: MissionPhase,
        content: str,
        decision: Optional["SupervisorDecision"] = None
    ) -> Dict[str, Any]:
        """
        Run multi-view evaluation with OptimistCouncil and SkepticCouncil.
        
        Args:
            state: Current mission state
            phase: Current phase
            content: Content to evaluate
            decision: Optional supervisor decision
            
        Returns:
            Dictionary with optimist_output, skeptic_output, and disagreement_summary
        """
        result = {
            "optimist_output": None,
            "skeptic_output": None,
            "disagreement_summary": "",
            "agreement_score": 0.0
        }
        
        if not self.enable_multiview:
            return result
        
        # Ensure multi-view councils are set up
        self._setup_multiview_councils()
        
        objective = state.objective
        
        # Run OptimistCouncil
        if self.optimist is not None:
            state.log(f"Running OptimistCouncil for phase '{phase.name}'")
            
            self._prepare_council_for_execution(self.optimist, state, decision)
            
            try:
                from ..councils.multi_view.optimist_council import OptimistContext
                opt_context = OptimistContext(
                    objective=objective,
                    content=content[:4000],  # Truncate for efficiency
                    iteration=phase.iterations
                )
                opt_result = self.optimist.execute(opt_context)
                
                if opt_result.success and opt_result.output:
                    result["optimist_output"] = opt_result.output
                    self._council_output_history["optimist"].append(opt_result.output)
                    phase.artifacts["optimist_perspective"] = opt_result.output.raw_output if hasattr(opt_result.output, 'raw_output') else str(opt_result.output)
            except Exception as e:
                state.log(f"OptimistCouncil error: {e}")
        
        # Run SkepticCouncil
        if self.skeptic is not None:
            state.log(f"Running SkepticCouncil for phase '{phase.name}'")
            
            self._prepare_council_for_execution(self.skeptic, state, decision)
            
            try:
                from ..councils.multi_view.skeptic_council import SkepticContext
                skep_context = SkepticContext(
                    objective=objective,
                    content=content[:4000],
                    iteration=phase.iterations
                )
                skep_result = self.skeptic.execute(skep_context)
                
                if skep_result.success and skep_result.output:
                    result["skeptic_output"] = skep_result.output
                    self._council_output_history["skeptic"].append(skep_result.output)
                    phase.artifacts["skeptic_perspective"] = skep_result.output.raw_output if hasattr(skep_result.output, 'raw_output') else str(skep_result.output)
            except Exception as e:
                state.log(f"SkepticCouncil error: {e}")
        
        # Calculate semantic disagreement and generate summary
        if result["optimist_output"] and result["skeptic_output"]:
            opt_conf = getattr(result["optimist_output"], 'confidence', 0.5)
            skep_conf = getattr(result["skeptic_output"], 'confidence', 0.5)
            
            # Compute semantic disagreement using embeddings
            semantic_disagreement = self._compute_multiview_disagreement(
                result["optimist_output"],
                result["skeptic_output"]
            )
            
            # Update tracking
            self._last_multiview_disagreement = semantic_disagreement
            result["agreement_score"] = 1.0 - semantic_disagreement
            result["semantic_disagreement"] = semantic_disagreement
            
            # Generate disagreement summary
            opt_reasoning = getattr(result["optimist_output"], 'reasoning', '')
            skep_reasoning = getattr(result["skeptic_output"], 'reasoning', '')
            
            disagreement = {
                "phase": phase.name,
                "optimist_confidence": opt_conf,
                "skeptic_confidence": skep_conf,
                "agreement_score": result["agreement_score"],
                "semantic_disagreement": semantic_disagreement,
                "optimist_summary": opt_reasoning[:500] if opt_reasoning else "",
                "skeptic_summary": skep_reasoning[:500] if skep_reasoning else ""
            }
            self._multiview_disagreements.append(disagreement)
            result["disagreement_summary"] = (
                f"Agreement: {result['agreement_score']:.1%} | "
                f"Semantic disagreement: {semantic_disagreement:.2f} | "
                f"Optimist conf: {opt_conf:.2f} | Skeptic conf: {skep_conf:.2f}"
            )
            
            state.log(f"Multi-view: {result['disagreement_summary']}")
            
            # Handle high disagreement - trigger deeper investigation
            self._handle_multiview_disagreement(
                state, phase, semantic_disagreement,
                result["optimist_output"], result["skeptic_output"]
            )
            
            # Extract structured disagreements and apply to iteration context
            self._extract_and_apply_multiview_disagreements(
                state, phase,
                result["optimist_output"], result["skeptic_output"]
            )
            
            # Update convergence state
            if self._convergence_state:
                self._convergence_state.multiview_disagreement = semantic_disagreement
            
            # Verbose logging
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                if hasattr(verbose_logger, 'log_multi_view_comparison'):
                    verbose_logger.log_multi_view_comparison(
                        result["optimist_output"], 
                        result["skeptic_output"],
                        mission_id=state.mission_id
                    )
        
        return result
    
    def _run_alignment_check(
        self,
        state: MissionState,
        phase: MissionPhase,
    ) -> None:
        """
        Run alignment check at phase end if enabled.
        
        This integrates the Hybrid Alignment Control Layer which:
        - Tracks goal alignment as a time series
        - Detects drift using embedding similarity
        - Applies soft corrective actions (never hard stops)
        
        Silent failure: alignment errors never crash the mission.
        
        Args:
            state: Current mission state
            phase: Completed phase
        """
        try:
            from deepthinker.alignment.integration import run_alignment_check
            
            # Get phase artifacts for alignment analysis
            phase_result = phase.artifacts if phase.artifacts else {}
            
            # Run alignment check (returns list of actions applied)
            actions = run_alignment_check(state, phase, phase_result)
            
            if actions:
                state.log(f"[ALIGNMENT] Actions applied: {[a.value for a in actions]}")
                
                # Log for verbose output
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    try:
                        verbose_logger.log_info(
                            f"[ALIGNMENT] Phase '{phase.name}': {len(actions)} corrective action(s) applied"
                        )
                    except Exception:
                        pass
                        
        except ImportError:
            # Alignment module not available - silently skip
            pass
        except Exception as e:
            # Silent failure - alignment errors never crash mission
            _orchestrator_logger.debug(f"[ALIGNMENT] Check failed (non-fatal): {e}")
    
    def _finalize_alignment(self, state: MissionState) -> None:
        """
        Finalize alignment tracking at mission end.
        
        Saves final alignment logs and summary statistics.
        Silent failure: alignment errors never crash the mission.
        
        Args:
            state: Mission state
        """
        try:
            from deepthinker.alignment.integration import finalize_alignment
            finalize_alignment(state)
        except ImportError:
            # Alignment module not available - silently skip
            pass
        except Exception as e:
            # Silent failure
            _orchestrator_logger.debug(f"[ALIGNMENT] Finalize failed (non-fatal): {e}")
    
    def _extract_mission_claims(self, state: MissionState) -> None:
        """
        Extract claims from mission final output and store to kb/claims/.
        
        Called at mission completion if HF instruments are enabled.
        Silent failure: claim extraction errors never crash the mission.
        
        Extracts from:
        - Synthesis report (final_artifacts["synthesis"])
        - Any other significant final artifacts
        
        Args:
            state: Mission state
        """
        try:
            # Check if claim extraction is enabled
            from deepthinker.hf_instruments.config import get_config
            config = get_config()
            
            if not config.is_claim_extractor_active():
                _orchestrator_logger.debug("[CLAIMS] Claim extractor not active, skipping")
                return
            
            from deepthinker.claims import extract_and_store_claims
            
            # Get synthesis report (main source of claims)
            synthesis = state.final_artifacts.get("synthesis", "")
            
            if not synthesis or len(synthesis) < 100:
                _orchestrator_logger.debug("[CLAIMS] No substantial synthesis to extract claims from")
                return
            
            # Extract claims from synthesis
            result = extract_and_store_claims(
                text=synthesis,
                mission_id=state.mission_id,
                source_type="final_answer",
                source_ref="synthesis_report",
                phase="synthesis",
            )
            
            state.log(
                f"[CLAIMS] Extracted {result.claim_count} claims from synthesis "
                f"(mode={result.extractor_mode}, time={result.extraction_time_ms:.1f}ms)"
            )
            
            # Store claim metadata in state for observability
            state.final_artifacts["claim_extraction"] = {
                "claim_count": result.claim_count,
                "extractor_mode": result.extractor_mode,
                "extraction_time_ms": result.extraction_time_ms,
                "error": result.error,
            }
            
        except ImportError:
            # HF instruments or claims module not available - silently skip
            _orchestrator_logger.debug("[CLAIMS] Claims module not available, skipping")
        except Exception as e:
            # Silent failure - claim extraction should never crash the mission
            _orchestrator_logger.debug(f"[CLAIMS] Extraction failed (non-fatal): {e}")
    
    def _run_phase_deepening(
        self,
        state: MissionState,
        phase: MissionPhase,
        decision: Optional["SupervisorDecision"] = None
    ) -> None:
        """
        Run time-aware deepening loop after a phase to enhance analysis.
        
        Enhanced with proactive time budgeting:
        - Uses phase time budgets from PhaseTimeAllocator
        - Tracks convergence/plateau score to stop when quality plateaus
        - Loops while budget allows and quality hasn't converged
        - Aims for 80% time budget utilization
        
        Deepening re-runs key councils to gather more insights:
        - ResearcherCouncil: Additional research
        - PlannerCouncil: Plan refinement
        - EvaluatorCouncil: Quality assessment
        
        Args:
            state: Current mission state
            phase: Completed phase
            decision: Optional supervisor decision
        """
        if not self.enable_phase_deepening:
            return
        
        # Skip synthesis phases - they have their own iteration logic
        if "synthesis" in phase.name.lower():
            return
        
        # Check basic time threshold
        remaining_minutes = state.remaining_minutes()
        if remaining_minutes < self.deepening_time_threshold_minutes:
            state.log(f"Skipping deepening: insufficient mission time ({remaining_minutes:.1f}min)")
            return
        
        # === Time-aware deepening using MissionTimeManager ===
        if self._mission_time_manager is not None:
            # Update phase time used
            phase.update_time_used()
            
            # Check if we should deepen using time manager
            should_deepen, reason = self._mission_time_manager.should_deepen_phase(
                phase_name=phase.name,
                convergence_score=phase.convergence_score,
                deepening_rounds_done=phase.deepening_rounds,
                min_iteration_seconds=self._min_deepening_iteration_seconds,
                max_deepening_rounds=self._max_deepening_rounds,
                convergence_threshold=self._convergence_threshold_for_deepening
            )
            
            if not should_deepen:
                state.log(f"Skipping deepening for '{phase.name}': {reason}")
                return
            
            state.log(f"[TIME-AWARE] {reason}")
        else:
            # Fallback to criteria-based check
            should_deepen = self._check_deepening_criteria(state, phase)
            if not should_deepen:
                state.log(f"Skipping deepening for '{phase.name}': criteria not met (converged)")
                return
        
        # === Deepening Loop ===
        # Continue deepening while budget allows and quality hasn't plateaued
        max_loops = self._max_deepening_rounds - phase.deepening_rounds
        
        for loop_idx in range(max_loops):
            # Update phase time
            phase.update_time_used()
            
            # Check if we should continue deepening
            if self._mission_time_manager is not None:
                should_continue, reason = self._mission_time_manager.should_deepen_phase(
                    phase_name=phase.name,
                    convergence_score=phase.convergence_score,
                    deepening_rounds_done=phase.deepening_rounds,
                    min_iteration_seconds=self._min_deepening_iteration_seconds,
                    max_deepening_rounds=self._max_deepening_rounds,
                    convergence_threshold=self._convergence_threshold_for_deepening
                )
                if not should_continue:
                    state.log(f"[TIME-AWARE] Stopping deepening: {reason}")
                    break
            
            # Check overall mission time
            if state.remaining_minutes() < 1.0:
                state.log(f"Stopping deepening: mission time low ({state.remaining_minutes():.1f}min)")
                break
            
            state.log(f"Running deepening round {phase.deepening_rounds + 1} for '{phase.name}'")
            phase.deepening_rounds += 1
            
            # Verbose logging
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                if hasattr(verbose_logger, 'log_phase_deepening'):
                    verbose_logger.log_phase_deepening(
                        phase.name,
                        ["researcher", "evaluator"]
                    )
            
            # Gather context from phase
            phase_context = self._get_accumulated_context(state)
            
            # Researcher pass for additional insights
            research_success = False
            if state.remaining_minutes() > 1.0:
                try:
                    deepening_prompt = (
                        f"Deepen analysis of phase '{phase.name}' (round {phase.deepening_rounds}). "
                        f"Mission objective: {state.objective}\n\n"
                        f"Seek additional insights, explore edge cases, and strengthen the analysis. "
                        f"Focus on areas that haven't been fully explored."
                    )
                    research_context = ResearchContext(
                        objective=deepening_prompt,
                        prior_knowledge=phase_context[:3000]
                    )
                    result = self.researcher.execute(research_context)
                    if result.success and result.output:
                        artifact_key = f"deepening_research_r{phase.deepening_rounds}"
                        phase.artifacts[artifact_key] = str(result.output)[:2000]
                        self._council_output_history["researcher"].append(result.output)
                        research_success = True
                except Exception as e:
                    state.log(f"Deepening research error: {e}")
            
            # Evaluation pass to assess quality and update convergence
            if state.remaining_minutes() > 0.5 and phase.artifacts:
                try:
                    content = "\n".join(str(v)[:500] for v in phase.artifacts.values())
                    eval_context = EvaluatorContext(
                        objective=f"Evaluate phase '{phase.name}' output quality after deepening round {phase.deepening_rounds}",
                        content_to_evaluate=content[:3000],
                        quality_threshold=6.0
                    )
                    result = self.evaluator.execute(eval_context)
                    if result.success and result.output:
                        artifact_key = f"deepening_evaluation_r{phase.deepening_rounds}"
                        phase.artifacts[artifact_key] = str(result.output)[:1000]
                        self._council_output_history["evaluator"].append(result.output)
                        
                        # Update convergence score from evaluation
                        self._update_convergence_from_evaluation(phase, result.output)
                except Exception as e:
                    state.log(f"Deepening evaluation error: {e}")
            
            # If research didn't succeed, may indicate convergence
            if not research_success:
                phase.convergence_score = min(1.0, phase.convergence_score + 0.2)
                state.log(f"Research didn't yield new insights, convergence={phase.convergence_score:.2f}")
        
        # Update final phase time used
        phase.update_time_used()
        
        # Log summary
        if phase.deepening_rounds > 0:
            state.log(
                f"Phase deepening completed for '{phase.name}': "
                f"{phase.deepening_rounds} round(s), convergence={phase.convergence_score:.2f}, "
                f"time_used={phase.time_used_seconds:.0f}s"
            )
    
    def _update_convergence_from_evaluation(
        self,
        phase: MissionPhase,
        eval_output: Any
    ) -> None:
        """
        Update phase convergence score based on evaluator output.
        
        Args:
            phase: Phase to update
            eval_output: Output from EvaluatorCouncil
        """
        # Extract quality and confidence signals
        quality_score = 5.0
        has_gaps = True
        confidence = 0.5
        
        if hasattr(eval_output, 'quality_score') and eval_output.quality_score:
            quality_score = eval_output.quality_score
        
        if hasattr(eval_output, 'missing_info'):
            has_gaps = bool(eval_output.missing_info)
        elif hasattr(eval_output, 'gaps'):
            has_gaps = bool(eval_output.gaps)
        elif hasattr(eval_output, 'data_needs'):
            has_gaps = bool(eval_output.data_needs)
        
        if hasattr(eval_output, 'confidence') and eval_output.confidence:
            confidence = eval_output.confidence
        
        # Calculate convergence: higher quality + fewer gaps + higher confidence = more converged
        quality_factor = min(1.0, quality_score / 8.0)  # 8.0 is "good enough"
        gaps_factor = 0.0 if has_gaps else 0.3
        confidence_factor = confidence * 0.3
        
        new_convergence = quality_factor * 0.4 + gaps_factor + confidence_factor
        
        # Smooth update (don't jump too fast)
        phase.convergence_score = phase.convergence_score * 0.3 + new_convergence * 0.7
        
        _orchestrator_logger.debug(
            f"Phase '{phase.name}' convergence updated: "
            f"quality={quality_score:.1f}, has_gaps={has_gaps}, "
            f"convergence={phase.convergence_score:.2f}"
        )
    
    def _check_deepening_criteria(self, state: MissionState, phase: MissionPhase) -> bool:
        """
        Check if deepening criteria are met for a phase.
        
        Deepening only occurs if ANY of these criteria are met:
        1. Uncertainty > 0.55
        2. Multi-view agreement < 0.85
        3. Research gaps exist (from evaluator)
        4. Quality score < 6.0
        5. Progress < 0.5 (stalled)
        
        Args:
            state: Current mission state
            phase: Phase to check
            
        Returns:
            True if deepening should occur
        """
        # Get uncertainty from convergence state or reasoning supervisor
        uncertainty = 0.5
        if self._convergence_state:
            uncertainty = 1.0 - self._convergence_state.confidence_score
        
        # Get multi-view agreement
        multiview_agreement = 0.9
        if self._convergence_state:
            multiview_agreement = 1.0 - self._convergence_state.multiview_disagreement
        elif self._multiview_disagreements:
            # Use last disagreement score
            last = self._multiview_disagreements[-1]
            multiview_agreement = last.get("agreement_score", 0.9)
        
        # Get research gaps from last evaluator output
        research_gaps = []
        if self._last_evaluator_output:
            if hasattr(self._last_evaluator_output, 'data_needs'):
                research_gaps.extend(self._last_evaluator_output.data_needs or [])
            if hasattr(self._last_evaluator_output, 'questions'):
                research_gaps.extend(self._last_evaluator_output.questions or [])
        
        # Get quality score
        quality_score = 7.0
        if self._last_evaluator_output and hasattr(self._last_evaluator_output, 'quality_score'):
            quality_score = self._last_evaluator_output.quality_score or 7.0
        
        # Get progress estimate
        progress = 1.0
        if self._convergence_state:
            progress = self._convergence_state.last_quality_score / 10.0 if self._convergence_state.last_quality_score else 0.5
        
        # Check criteria
        # 1. High uncertainty triggers deepening
        if uncertainty > 0.55:
            state.log(f"Deepening criteria met: uncertainty={uncertainty:.2f} > 0.55")
            return True
        
        # 2. Low multi-view agreement triggers deepening
        if multiview_agreement < 0.85:
            state.log(f"Deepening criteria met: agreement={multiview_agreement:.2f} < 0.85")
            return True
        
        # 3. Non-empty research gaps trigger deepening
        if research_gaps:
            state.log(f"Deepening criteria met: {len(research_gaps)} research gaps")
            return True
        
        # 4. Low quality triggers deepening
        if quality_score < 6.0:
            state.log(f"Deepening criteria met: quality={quality_score:.1f} < 6.0")
            return True
        
        # 5. Stalled progress triggers deepening
        if progress < 0.5:
            state.log(f"Deepening criteria met: progress={progress:.2f} < 0.5")
            return True
        
        # No criteria met
        state.log(
            f"Deepening criteria not met: uncertainty={uncertainty:.2f}, "
            f"agreement={multiview_agreement:.2f}, gaps={len(research_gaps)}, "
            f"quality={quality_score:.1f}, progress={progress:.2f}"
        )
        return False
    
    def _run_epistemic_validation(
        self,
        state: MissionState,
        phase: MissionPhase
    ) -> Optional[Dict[str, Any]]:
        """
        Run epistemic validation on phase output.
        
        Performs:
        - Claim extraction and validation
        - Web search gate checking
        - Phase contamination detection
        - Epistemic risk computation
        
        Updates state.epistemic_telemetry with results.
        
        Args:
            state: Current mission state
            phase: Completed phase with artifacts
            
        Returns:
            Dictionary with epistemic validation results, or None if validation skipped
        """
        result: Dict[str, Any] = {
            "phase": phase.name,
            "epistemic_risk": 0.0,
            "grounded_ratio": 1.0,
            "contamination_score": 0.0,
            "web_search_gate": {},
            "claim_validation": {},
            "blocking_issues": [],
        }
        
        # Collect all phase artifacts as text for analysis
        phase_output = ""
        if phase.artifacts:
            phase_output = "\n\n".join(
                f"## {key}\n{str(value)[:2000]}"
                for key, value in phase.artifacts.items()
                if not key.startswith("_")
            )
        
        if not phase_output or len(phase_output) < 100:
            # Not enough content to validate
            return None
        
        sources_found: List[Dict[str, Any]] = []
        
        # === 1. Claim Validation ===
        if CLAIM_VALIDATOR_AVAILABLE and self._claim_validator is not None:
            try:
                # Parse claims from output
                claims = self._claim_validator.parse_claims(phase_output)
                
                # Validate claims
                validation_result = self._claim_validator.validate(claims)
                
                result["claim_validation"] = validation_result.to_dict()
                result["grounded_ratio"] = validation_result.grounded_ratio
                
                # Compute epistemic risk
                risk = self._claim_validator.compute_epistemic_risk(
                    output=phase_output,
                    claims=claims,
                    sources=[],  # Will be populated by web search gate
                    stated_confidence=0.7  # Default confidence
                )
                
                result["epistemic_risk"] = risk.overall_risk
                
                # Check if should block
                should_block, block_reason = self._claim_validator.should_block_phase_advancement(
                    validation_result, risk
                )
                
                if should_block:
                    result["blocking_issues"].append(block_reason)
                    state.log(f"[EPISTEMIC] Claim validation issue: {block_reason}")
                
                _orchestrator_logger.debug(
                    f"Claim validation: {validation_result.grounded_claims}/{validation_result.total_claims} "
                    f"grounded, risk={risk.overall_risk:.2f}"
                )
                
            except Exception as e:
                _orchestrator_logger.warning(f"Claim validation failed: {e}")
        
        # === 2. Web Search Gate ===
        if WEB_SEARCH_GATE_AVAILABLE and self._web_search_gate is not None:
            try:
                # Collect sources from artifacts
                for key, value in (phase.artifacts or {}).items():
                    if "source" in key.lower() or "url" in key.lower():
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and "url" in item:
                                    sources_found.append(item)
                        elif isinstance(value, str) and "http" in value:
                            sources_found.append({"url": value})
                
                # Check web search requirements
                gate_result = self._web_search_gate.check_requirements(
                    objective=state.objective,
                    phase_name=phase.name,
                    sources=sources_found,
                )
                
                result["web_search_gate"] = gate_result.to_dict()
                
                if not gate_result.is_satisfied:
                    result["blocking_issues"].append(gate_result.blocking_reason)
                    state.log(f"[EPISTEMIC] Web search gate: {gate_result.blocking_reason}")
                
                _orchestrator_logger.debug(
                    f"Web search gate: requires={gate_result.requires_search}, "
                    f"sources={gate_result.current_sources}/{gate_result.min_sources_required}, "
                    f"satisfied={gate_result.is_satisfied}"
                )
                
            except Exception as e:
                _orchestrator_logger.warning(f"Web search gate check failed: {e}")
        
        # === 3. Phase Contamination Detection ===
        if PHASE_GUARD_AVAILABLE and self._phase_guard is not None:
            try:
                contamination = self._phase_guard.inspect_output(phase_output, phase.name)
                
                result["contamination_score"] = contamination.contamination_score
                
                if not contamination.is_clean:
                    result["blocking_issues"].append(
                        f"Phase contamination: {len(contamination.violations)} violations"
                    )
                    state.log(
                        f"[EPISTEMIC] Phase contamination: score={contamination.contamination_score:.2f}, "
                        f"violations={len(contamination.violations)}"
                    )
                
                _orchestrator_logger.debug(
                    f"Phase contamination: score={contamination.contamination_score:.2f}, "
                    f"clean={contamination.is_clean}"
                )
                
            except Exception as e:
                _orchestrator_logger.warning(f"Phase contamination check failed: {e}")
        
        # === Update State Telemetry ===
        try:
            state.update_epistemic_telemetry(
                epistemic_risk=result["epistemic_risk"],
                grounded_claim_ratio=result["grounded_ratio"],
                phase_name=phase.name,
                source_count=len(sources_found),
                contamination_score=result["contamination_score"],
            )
        except Exception as e:
            _orchestrator_logger.debug(f"Failed to update epistemic telemetry: {e}")
        
        # Log summary
        if result["blocking_issues"]:
            state.log(
                f"[EPISTEMIC] Phase '{phase.name}' has {len(result['blocking_issues'])} epistemic issues"
            )
        
        return result
    
    # =========================================================================
    # Normative Governance Methods
    # =========================================================================
    
    def _handle_governance_block(
        self,
        state: MissionState,
        phase: MissionPhase,
        verdict: "NormativeVerdict",
        phase_type: str,
        decision: Optional["SupervisorDecision"],
        start_time: float,
    ) -> bool:
        """
        Handle a BLOCK verdict from the governance layer.
        
        Implements corrective actions:
        - RETRY_PHASE: Re-run phase with constraints
        - FORCE_WEB_SEARCH: Trigger web search before retry
        - SCOPE_REDUCTION: Reduce phase scope and retry
        
        Args:
            state: Current mission state
            phase: Phase that was blocked
            verdict: Governance verdict with recommended action
            phase_type: Type of phase (research, synthesis, etc.)
            decision: Supervisor decision (if any)
            start_time: Phase start timestamp
            
        Returns:
            True if phase was blocked and retry is scheduled
        """
        state.log(
            f"[GOVERNANCE] Phase '{phase.name}' BLOCKED: "
            f"{len(verdict.violations)} violations, severity={verdict.aggregate_severity:.2f}"
        )
        
        # Check for terminal validation failure - do not retry these
        if phase.artifacts.get("_validation_failed_terminal", False):
            state.log(
                f"[GOVERNANCE] Phase '{phase.name}' has terminal validation failure, "
                f"not retrying despite governance block"
            )
            # Phase should already be marked as failed, but ensure it is
            if phase.status != "failed":
                phase.mark_failed("Terminal validation failure (governance blocked)")
            phase.artifacts["failure_reason"] = MissionFailureReason.GOVERNANCE_BLOCK.value
            duration = time.time() - start_time
            state.log_council_execution(
                council_name=f"{phase_type}_council",
                phase_name=phase.name,
                models_used=decision.models if decision else [],
                success=False,
                duration_s=duration,
                error="Terminal validation failure - governance blocked"
            )
            return True
        
        # Track the block
        self._normative_controller.record_retry(phase.name)
        retry_count = self._normative_controller.get_retry_count(phase.name)
        
        # Check if retry is allowed
        if not verdict.can_retry or retry_count > verdict.max_retries:
            # Model-Aware Phase Stabilization: Attempt soft failure before hard failure
            partial_output = self._extract_partial_output(phase)
            
            if partial_output and len(str(partial_output)) > 100:
                # We have usable partial output - mark as degraded instead of failed
                reason = (
                    f"Governance block after {retry_count} retries with "
                    f"{len(verdict.violations)} violations (partial output salvaged)"
                )
                phase.mark_completed_degraded(reason)
                
                state.log(
                    f"[SOFT_FAILURE] Phase '{phase.name}' completed with degradation: "
                    f"partial_output_length={len(str(partial_output))}, reason='{reason}'"
                )
                
                duration = time.time() - start_time
                state.log_council_execution(
                    council_name=f"{phase_type}_council",
                    phase_name=phase.name,
                    models_used=decision.models if decision else [],
                    success=True,  # Mark as success since we salvaged output
                    duration_s=duration,
                    error=None
                )
                return True
            
            # No partial output - hard failure
            state.log(
                f"[GOVERNANCE] Max retries exceeded ({retry_count}/{verdict.max_retries}), "
                f"marking phase as failed"
            )
            phase.mark_failed(f"Governance block: {len(verdict.violations)} violations after {retry_count} retries")
            phase.artifacts["failure_reason"] = MissionFailureReason.GOVERNANCE_BLOCK.value
            phase.artifacts["governance_violations_count"] = len(verdict.violations)
            phase.artifacts["governance_retries_attempted"] = retry_count
            
            # === SSE: Publish phase error for frontend ===
            if SSE_AVAILABLE and sse_manager:
                violation_types = [v.rule_id for v in verdict.violations] if verdict.violations else []
                _publish_sse_event(sse_manager.publish_phase_error(
                    mission_id=state.mission_id,
                    phase_name=phase.name,
                    error_type="governance_block",
                    error_message=f"Governance blocked: {len(verdict.violations)} violations after {retry_count} retries",
                    phase_index=state.current_phase_index,
                    retry_available=False,
                    suggestions=[
                        "Add more evidence and citations",
                        "Reduce speculative claims",
                        "Ground assertions in sources"
                    ] if any('epistemic' in str(v).lower() for v in violation_types) else [],
                    context={
                        "violations": len(verdict.violations),
                        "retries_attempted": retry_count,
                        "epistemic_risk": verdict.epistemic_risk if hasattr(verdict, 'epistemic_risk') else 0
                    }
                ))
            
            duration = time.time() - start_time
            state.log_council_execution(
                council_name=f"{phase_type}_council",
                phase_name=phase.name,
                models_used=decision.models if decision else [],
                success=False,
                duration_s=duration,
                error=f"Governance block after {retry_count} retries"
            )
            return True
        
        # Log hard violations
        hard_violations = verdict.get_hard_violations()
        if hard_violations:
            for v in hard_violations[:3]:  # Limit logging
                state.log(f"[GOVERNANCE] Hard violation: {v.description}")
        
        # Apply recommended corrective action
        action = verdict.recommended_action
        
        if action == RecommendedAction.FORCE_WEB_SEARCH:
            state.log(f"[GOVERNANCE] Corrective action: forcing web search before retry")
            # Inject web search requirement into phase artifacts
            phase.artifacts["_governance_force_web_search"] = True
            phase.artifacts["_governance_retry_reason"] = "epistemic_grounding_required"
        
        elif action == RecommendedAction.SCOPE_REDUCTION:
            state.log(f"[GOVERNANCE] Corrective action: reducing scope for retry")
            phase.artifacts["_governance_scope_reduction"] = True
            phase.artifacts["_governance_retry_reason"] = "scope_too_broad"
        
        else:
            state.log(f"[GOVERNANCE] Corrective action: retry phase with constraints")
            phase.artifacts["_governance_retry_reason"] = "violations_detected"
        
        # Store governance context for retry
        phase.artifacts["_governance_retry_count"] = retry_count
        phase.artifacts["_governance_violation_summary"] = [
            {"type": v.type.value, "severity": v.severity} 
            for v in verdict.violations[:5]
        ]
        
        # Model-Aware Phase Stabilization: Store escalation signal for supervisor
        # This will be used by _run_phase to pass to supervisor.decide()
        importance = PHASE_IMPORTANCE.get(phase_type, 0.5)
        phase.artifacts["_escalation_signal"] = {
            "violation_types": verdict.get_violation_types(),
            "failed_models": decision.models if decision else [],
            "retry_count": retry_count,
            "phase_importance": importance,
            "aggregate_severity": verdict.aggregate_severity,
        }
        
        state.log(
            f"[GOVERNANCE_ESCALATION] Phase '{phase.name}' blocked (retry {retry_count}): "
            f"triggering model escalation signal with violations={[v.type.value for v in verdict.violations[:3]]}"
        )
        
        # Reset phase for retry
        phase.status = "pending"
        phase.iterations = 0
        
        state.log(
            f"[GOVERNANCE] Scheduling retry {retry_count}/{verdict.max_retries} "
            f"for phase '{phase.name}'"
        )
        
        # === Decision Accountability: Emit RETRY_ESCALATION decision ===
        self._emit_retry_escalation_decision(
            state=state,
            phase=phase,
            phase_type=phase_type,
            verdict=verdict,
            decision=decision,
            retry_count=retry_count,
        )
        
        return True
    
    def _emit_retry_escalation_decision(
        self,
        state: MissionState,
        phase: MissionPhase,
        phase_type: str,
        verdict: "NormativeVerdict",
        decision: Optional["SupervisorDecision"],
        retry_count: int,
    ) -> Optional[str]:
        """
        Emit a RETRY_ESCALATION decision record.
        
        Decision Accountability Layer: Records when governance triggers
        a retry with model escalation.
        
        Args:
            state: Current mission state
            phase: Phase being retried
            phase_type: Type of phase
            verdict: Governance verdict that triggered retry
            decision: Original supervisor decision
            retry_count: Current retry count
            
        Returns:
            decision_id if emitted, None otherwise
        """
        if not self._decision_emitter or not self._enable_decision_accountability:
            return None
        
        try:
            # Determine escalation targets
            from_models = decision.models if decision else []
            
            # Get escalation hint from verdict
            escalation_hint = verdict.escalation_hint if verdict else {}
            to_tier = escalation_hint.get("recommended_tier", "reasoning")
            
            # Infer target models based on tier
            to_models = []
            if SUPERVISOR_AVAILABLE:
                from ..supervisor.model_supervisor import TIER_MODELS
                to_models = TIER_MODELS.get(to_tier, ["gemma3:27b"])[:2]
            else:
                to_models = ["gemma3:27b"]  # Default escalation target
            
            escalation_reason = (
                f"Governance BLOCK (severity={verdict.aggregate_severity:.2f}, "
                f"violations={len(verdict.violations)}), retry {retry_count}"
            )
            
            decision_id = self._decision_emitter.emit_retry_escalation(
                mission_id=state.mission_id,
                phase_id=phase.name,
                phase_type=phase_type,
                from_models=from_models,
                to_models=to_models,
                retry_count=retry_count,
                escalation_reason=escalation_reason,
                triggered_by=state.last_governance_decision_id,
            )
            
            if decision_id:
                state.track_decision(decision_id)
            
            return decision_id
            
        except Exception as e:
            _orchestrator_logger.debug(f"[DECISION] Failed to emit retry escalation: {e}")
            return None
    
    def _extract_partial_output(self, phase: MissionPhase) -> Optional[str]:
        """
        Extract any partial usable output from a phase.
        
        Model-Aware Phase Stabilization: Used to salvage partial work
        from phases that failed governance checks but produced some output.
        
        Args:
            phase: The phase to extract output from
            
        Returns:
            Partial output string if available, None otherwise
        """
        # Check common artifact keys for output
        output_keys = [
            "analysis", "research", "findings", "report", 
            "synthesis_report", "output", "result", "content"
        ]
        
        for key in output_keys:
            if key in phase.artifacts:
                value = phase.artifacts[key]
                if isinstance(value, str) and len(value) > 50:
                    return value
                elif isinstance(value, dict):
                    # Try to extract from dict
                    for subkey in ["content", "text", "output", "result"]:
                        if subkey in value and isinstance(value[subkey], str):
                            return value[subkey]
        
        # Try to concatenate all string artifacts
        all_text = []
        for key, value in phase.artifacts.items():
            if not key.startswith("_") and isinstance(value, str) and len(value) > 20:
                all_text.append(f"{key}: {value[:500]}")
        
        if all_text:
            return "\n\n".join(all_text)
        
        return None
    
    def _apply_governance_penalties(
        self,
        state: MissionState,
        phase: MissionPhase,
        verdict: "NormativeVerdict",
    ) -> None:
        """
        Apply penalties from a WARN verdict.
        
        Penalties are applied silently and summarized in the report.
        
        Args:
            state: Current mission state
            phase: Phase being processed
            verdict: Governance verdict with penalties
        """
        state.log(
            f"[GOVERNANCE] Phase '{phase.name}' WARNING: "
            f"{len(verdict.violations)} violations (severity={verdict.aggregate_severity:.2f})"
        )
        
        # Apply confidence penalty to any stated confidence
        if verdict.confidence_penalty > 0:
            # Update convergence state if available
            if hasattr(state, '_convergence_state') and state._convergence_state is not None:
                old_confidence = state._convergence_state.confidence_score
                state._convergence_state.confidence_score = max(
                    0.0,
                    old_confidence - verdict.confidence_penalty
                )
                state.log(
                    f"[GOVERNANCE] Confidence adjusted: {old_confidence:.2f} → "
                    f"{state._convergence_state.confidence_score:.2f}"
                )
        
        # Store penalties in phase artifacts for reporting
        phase.artifacts["governance_penalties"] = {
            "confidence_penalty": verdict.confidence_penalty,
            "epistemic_risk": verdict.epistemic_risk,
            "violation_count": len(verdict.violations),
        }
        
        # Update epistemic telemetry with penalty info
        state.update_epistemic_telemetry(
            epistemic_risk=verdict.epistemic_risk
        )
    
    def get_governance_summary(self) -> Dict[str, Any]:
        """
        Get governance summary for final mission report.
        
        Returns summary only (not verbose violation list) as per spec.
        
        Returns:
            Dict with epistemic_risk_score, total_violations, phases_blocked
        """
        if self._normative_controller is not None:
            return self._normative_controller.get_governance_report()
        
        return {
            "epistemic_risk_score": 0.0,
            "total_violations": 0,
            "phases_blocked": 0,
        }
    
    def _run_meta_cognition(self, state: MissionState, phase: MissionPhase) -> None:
        """
        Run meta-cognition processing after a phase completes.
        
        This invokes the meta-cognition engine to:
        - Reflect on phase output
        - Update hypotheses
        - Run internal debate
        - Revise the plan
        - Record to memory system
        
        Args:
            state: Current mission state
            phase: The completed phase
        """
        if self.meta is None:
            return
        
        try:
            # Gather phase output for analysis
            council_output = phase.artifacts if phase.artifacts else {}
            
            # Run meta-cognition
            meta_result = self.meta.process_phase(
                phase_name=phase.name,
                council_output=council_output,
                state=state
            )
            
            # Store in work_summary
            if "meta" not in state.work_summary:
                state.work_summary["meta"] = {}
            state.work_summary["meta"][phase.name] = meta_result
            
            # Record to memory system
            self._record_phase_to_memory(state, phase, meta_result)
            
            # Publish SSE event (fire-and-forget from sync context)
            self._publish_meta_update_async(state.mission_id, phase.name, meta_result)
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Meta-cognition failed for phase '{phase.name}': {e}"
            )
            state.log(f"Meta-cognition skipped for phase '{phase.name}': {e}")
    
    def _record_phase_to_memory(
        self,
        state: MissionState,
        phase: MissionPhase,
        meta_result: Dict[str, Any]
    ) -> None:
        """
        Record phase completion data to memory system.
        
        Args:
            state: Current mission state
            phase: The completed phase
            meta_result: Results from meta-cognition processing
        """
        if not self.memory:
            return
        
        try:
            # Record phase output
            output_summary = None
            if phase.artifacts:
                output_parts = [f"{k}: {str(v)[:200]}" for k, v in list(phase.artifacts.items())[:5]]
                output_summary = "\n".join(output_parts)
            
            self.memory.add_phase_output(
                phase_name=phase.name,
                summary=output_summary,
                final_output=str(phase.artifacts) if phase.artifacts else None,
                artifacts=phase.artifacts or {},
                duration_seconds=phase.duration_seconds(),
                iteration_count=phase.iterations,
            )
            
            # Record reflection if available
            reflection = meta_result.get("reflection", {})
            if reflection:
                self.memory.add_reflection(
                    phase_name=phase.name,
                    assumptions=reflection.get("assumptions", []),
                    risks=reflection.get("risks", []),
                    weaknesses=reflection.get("weaknesses", []),
                    questions=reflection.get("questions", []),
                )
            
            # Record debate if available
            debate = meta_result.get("debate", {})
            if debate and debate.get("ran_debate"):
                self.memory.add_debate(
                    phase_name=phase.name,
                    contradictions_found=debate.get("contradictions_found", []),
                )
            
            # Record supervisor metrics if available
            phase_metrics = meta_result.get("phase_metrics", {})
            if phase_metrics:
                self.memory.record_supervisor_signals(
                    difficulty=phase_metrics.get("difficulty_score", 0.5),
                    uncertainty=phase_metrics.get("uncertainty_score", 0.5),
                    progress=phase_metrics.get("progress_score", 0.5),
                    novelty=phase_metrics.get("novelty_score", 0.5),
                    confidence=phase_metrics.get("confidence_score", 0.5),
                )
            
            # Add significant text to RAG
            if phase.artifacts:
                for key, value in phase.artifacts.items():
                    if isinstance(value, str) and len(value) > 200:
                        self.memory.add_evidence_text(
                            text=value[:3000],
                            phase=phase.name,
                            artifact_type=key,
                            tags=[phase.name, key],
                        )
            
            # Save checkpoint
            self.memory.save_checkpoint()
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to record phase to memory: {e}")
    
    def _finalize_memory(self, state: MissionState) -> None:
        """
        Finalize memory system at mission completion.
        
        Saves mission state, updates global RAG, and stores long-term summary.
        
        Args:
            state: Final mission state
        """
        if not self.memory:
            return
        
        try:
            # Calculate final metrics
            time_used = state.constraints.time_budget_minutes - state.remaining_minutes()
            phases_completed = len([p for p in state.phases if p.status == "completed"])
            
            # Save mission state
            self.memory.save_mission()
            
            # Update global RAG with important evidence
            docs_added = self.memory.update_global_rag(
                min_confidence=0.5,
                max_documents=30,
            )
            if docs_added > 0:
                state.log(f"Added {docs_added} documents to global knowledge base")
            
            # Build and save long-term summary
            summary = self.memory.build_mission_summary(
                time_taken_minutes=time_used,
            )
            
            # Add key insights from final artifacts
            if state.final_artifacts.get("final_report"):
                summary.key_insights.insert(0, state.final_artifacts["final_report"][:300])
            
            if self.memory.save_long_term_summary(summary):
                state.log("Mission summary saved to long-term memory")
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to finalize memory: {e}")
    
    def _publish_meta_update_async(
        self,
        mission_id: str,
        phase_name: str,
        meta_result: Dict[str, Any]
    ) -> None:
        """
        Publish meta update via SSE (fire-and-forget from sync context).
        
        Args:
            mission_id: Mission ID
            phase_name: Phase name
            meta_result: Meta-cognition results
        """
        if not SSE_AVAILABLE:
            return
        
        try:
            # Create a new event loop if needed or use existing
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, schedule the coroutine
                asyncio.create_task(
                    sse_manager.publish_meta_update(mission_id, phase_name, meta_result)
                )
            except RuntimeError:
                # No running loop, create one for this call
                asyncio.run(
                    sse_manager.publish_meta_update(mission_id, phase_name, meta_result)
                )
        except Exception:
            # Silently ignore SSE failures - they shouldn't affect mission execution
            pass
    
    def _run_final_synthesis(self, state: MissionState) -> None:
        """
        Run final synthesis using arbiter to consolidate all outputs.
        
        Enhanced to include:
        - Multi-view perspectives (optimist + skeptic)
        - Council disagreement analysis
        - Evaluation evolution
        """
        # Collect outputs from all completed phases
        council_outputs = []
        
        for phase in state.phases:
            if phase.status == "completed" and phase.artifacts:
                # Create council output for each phase
                output_str = "\n".join(f"{k}: {v[:500]}..." if len(str(v)) > 500 else f"{k}: {v}" 
                                       for k, v in phase.artifacts.items()
                                       if not k.startswith("_"))
                council_outputs.append(CouncilOutput(
                    council_name=phase.name,
                    output=output_str,
                    confidence=0.8
                ))
        
        # Add multi-view summary as a special council output
        multiview_summary = self._get_multiview_summary()
        if multiview_summary and "No multi-view" not in multiview_summary:
            council_outputs.append(CouncilOutput(
                council_name="multi_view_analysis",
                output=multiview_summary,
                confidence=0.9
            ))
        
        # Add council history summary
        history_context = self._get_accumulated_context(state, include_full_history=True)
        
        if council_outputs:
            try:
                # Build enhanced context for arbiter
                context = {
                    "mission_id": state.mission_id,
                    "time_used_minutes": state.constraints.time_budget_minutes - state.remaining_minutes(),
                    "phases_completed": len([p for p in state.phases if p.status == "completed"]),
                    "total_phases": len(state.phases),
                    "multiview_disagreements": len(self._multiview_disagreements),
                    "history_context": self.cognitive_spine.compress_text(
                        history_context, max_chars=OutputLimits.MAX_HISTORY_CHARS
                    ) if self.cognitive_spine and len(history_context) > OutputLimits.MAX_HISTORY_CHARS else history_context[:OutputLimits.MAX_HISTORY_CHARS]
                }
                
                decision = self.arbiter.arbitrate(
                    council_outputs=council_outputs,
                    objective=state.objective,
                    context=context,
                    meta_traces=state.meta_traces if hasattr(state, 'meta_traces') else None
                )
                
                state.final_artifacts["final_report"] = decision.final_output if decision.final_output else ""
                state.final_artifacts["resolution_notes"] = decision.resolution_notes
                
                # Store additional arbiter outputs if available
                if hasattr(decision, 'meta_analysis'):
                    state.final_artifacts["meta_analysis"] = decision.meta_analysis
                if hasattr(decision, 'ranked_insights'):
                    state.final_artifacts["ranked_insights"] = "\n".join(decision.ranked_insights)
                if hasattr(decision, 'optimist_summary'):
                    state.final_artifacts["optimist_synthesis"] = decision.optimist_summary
                if hasattr(decision, 'skeptic_summary'):
                    state.final_artifacts["skeptic_synthesis"] = decision.skeptic_summary
                
                state.log("Arbiter produced final consolidated report with multi-view synthesis")
                
                # Verbose logging
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    if hasattr(verbose_logger, 'log_arbiter_synthesis'):
                        verbose_logger.log_arbiter_synthesis(decision)
                
            except Exception as e:
                state.log(f"Arbiter synthesis failed: {str(e)}")
                # Fallback: concatenate phase artifacts
                state.final_artifacts["final_report"] = self._get_accumulated_context(state)

    def _build_phase_artifacts(self, state: MissionState) -> Dict[str, Dict[str, str]]:
        """Build a dictionary of all phase artifacts."""
        phase_artifacts: Dict[str, Dict[str, str]] = {}
        for phase in state.phases:
            if phase.artifacts:
                phase_artifacts[phase.name] = dict(phase.artifacts)
        
        # Include final_artifacts as a special phase
        if state.final_artifacts:
            phase_artifacts["_final"] = dict(state.final_artifacts)
        
        return phase_artifacts

    def _generate_final_deliverables(self, state: MissionState) -> None:
        """
        Generate final deliverables using OutputManager.
        
        Wraps output generation in try/except for graceful degradation.
        """
        try:
            # Build phase artifacts dictionary
            phase_artifacts = self._build_phase_artifacts(state)
            
            # Get output spec from arbiter
            spec = self.arbiter.decide_output_spec(state, phase_artifacts)
            state.log(f"Arbiter decided output format: {spec.primary_format.value}")
            
            # Generate outputs
            artifacts = self.output_manager.generate_outputs(
                mission_state=state,
                phase_artifacts=phase_artifacts,
                spec=spec
            )
            
            # Store deliverables
            state.output_deliverables = artifacts
            
            if artifacts:
                state.log(f"Generated {len(artifacts)} final deliverable(s)")
                for artifact in artifacts:
                    state.log(f"  - [{artifact.format.value}] {artifact.path}")
            else:
                state.log("[WARN] No final deliverables generated")
                
        except Exception as e:
            state.log(f"[WARN] Failed to generate final deliverables: {str(e)}")
            
            # Fallback: try to generate at least a raw dump
            try:
                phase_artifacts = self._build_phase_artifacts(state)
                raw_artifacts = self.output_manager._generate_raw_dump(
                    state, phase_artifacts, self.output_manager._mission_dir(state.mission_id)
                )
                state.output_deliverables = raw_artifacts
                state.log("[WARN] Raw artifacts stored as fallback")
            except Exception as e2:
                state.log(f"[WARN] Even raw artifact generation failed: {str(e2)}")
    
    def resume_mission(self, mission_id: str) -> MissionState:
        """
        Resume a previously started mission.
        
        Args:
            mission_id: ID of mission to resume
            
        Returns:
            Final MissionState after resumption
        """
        state = self.store.load(mission_id)
        
        if state.is_terminal():
            state.log("Mission already in terminal state, cannot resume")
            return state
        
        if state.is_expired():
            state.status = "expired"
            state.log("Mission expired, cannot resume")
            self.store.save(state)
            return state
        
        state.log(f"Resuming mission from phase {state.current_phase_index + 1}")
        return self.run_until_complete_or_timeout(mission_id)
    
    def abort_mission(self, mission_id: str, reason: str = "User requested abort") -> MissionState:
        """
        Abort a running mission.
        
        Args:
            mission_id: ID of mission to abort
            reason: Reason for abortion
            
        Returns:
            Updated MissionState
        """
        state = self.store.load(mission_id)
        state.status = "aborted"
        state.log(f"Mission aborted: {reason}")
        
        # Mark current phase as failed
        phase = state.current_phase()
        if phase and phase.status == "running":
            phase.mark_failed(reason)
        
        self.store.save(state)
        return state
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status.
        
        Returns:
            Dictionary with resource information
        """
        status = {
            "supervision_enabled": self.enable_supervision,
            "gpu_manager_available": self.gpu_manager is not None,
            "supervisor_available": self.supervisor is not None
        }
        
        if self.gpu_manager is not None:
            stats = self.gpu_manager.get_stats()
            status["gpu"] = stats.to_dict()
            status["resource_pressure"] = self.gpu_manager.get_resource_pressure()
            status["suggested_tier"] = self.gpu_manager.suggest_model_tier()
        
        return status
