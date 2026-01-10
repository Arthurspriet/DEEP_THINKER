"""
Safety Core Registry for DeepThinker.

Centralized registry for optional safety, governance, and infrastructure modules.
Replaces scattered try/except ImportError blocks with a unified system that:
- Explicitly tracks which modules are available
- Logs degradation warnings at startup
- Provides require() for critical paths that should fail loudly
- Provides get() for optional paths with graceful degradation

Usage:
    from deepthinker.core.safety_registry import safety
    
    # Initialize at startup (logs availability)
    safety.initialize()
    
    # For critical paths - fails if missing
    governance = safety.require("governance")
    
    # For optional paths - returns None with warning
    memory_guard = safety.get("memory_guard")
    
    # Check availability without loading
    if safety.is_available("phase_validator"):
        validator = safety.get("phase_validator")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)


class ModuleCategory(str, Enum):
    """Categories of safety/infrastructure modules."""
    
    GOVERNANCE = "governance"
    VALIDATION = "validation"
    MEMORY = "memory"
    CONSENSUS = "consensus"
    ORCHESTRATION = "orchestration"
    EPISTEMICS = "epistemics"
    DECISIONS = "decisions"
    META = "meta"
    RESOURCES = "resources"
    MODELS = "models"
    PREDICTORS = "predictors"


class ImportPriority(str, Enum):
    """Priority levels for module imports."""
    
    CRITICAL = "critical"      # Must be available; system cannot function without
    RECOMMENDED = "recommended"  # Should be available; degraded experience if missing
    OPTIONAL = "optional"       # Nice to have; minimal impact if missing


@dataclass
class ModuleSpec:
    """Specification for a safety/infrastructure module."""
    
    name: str
    import_path: str
    category: ModuleCategory
    priority: ImportPriority = ImportPriority.OPTIONAL
    description: str = ""
    exports: List[str] = field(default_factory=list)
    fallback_value: Any = None
    
    # Loaded state
    _module: Any = field(default=None, repr=False)
    _available: Optional[bool] = field(default=None, repr=False)
    _error: Optional[str] = field(default=None, repr=False)


class SafetyModuleUnavailableError(Exception):
    """Raised when a required safety module is not available."""
    
    def __init__(self, module_name: str, reason: str = ""):
        self.module_name = module_name
        self.reason = reason
        message = f"Required safety module '{module_name}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class SafetyCoreRegistry:
    """
    Central registry for safety and infrastructure modules.
    
    Provides unified access to optional modules with explicit
    availability tracking and graceful degradation.
    """
    
    def __init__(self):
        self._modules: Dict[str, ModuleSpec] = {}
        self._initialized: bool = False
        self._strict_mode: bool = False
        self._degradation_callbacks: List[Callable[[str, str], None]] = []
        
        # Register all known modules
        self._register_default_modules()
    
    def _register_default_modules(self) -> None:
        """Register all known safety/infrastructure modules."""
        
        # === Governance Modules ===
        self.register(ModuleSpec(
            name="governance",
            import_path="deepthinker.governance.normative_layer",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.RECOMMENDED,
            description="Normative control layer for phase governance",
            exports=["NormativeController", "NormativeVerdict", "VerdictStatus", "RecommendedAction"],
        ))
        
        self.register(ModuleSpec(
            name="rule_engine",
            import_path="deepthinker.governance.rule_engine",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.RECOMMENDED,
            description="Deterministic rule engine for governance",
            exports=["RuleEngine", "GovernanceConfig", "load_governance_config"],
        ))
        
        self.register(ModuleSpec(
            name="phase_guard",
            import_path="deepthinker.governance.phase_guard",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.OPTIONAL,
            description="Phase boundary enforcement",
            exports=["PhaseGuard", "get_phase_guard"],
        ))
        
        # === Validation Modules ===
        self.register(ModuleSpec(
            name="phase_validator",
            import_path="deepthinker.core.phase_validator",
            category=ModuleCategory.VALIDATION,
            priority=ImportPriority.RECOMMENDED,
            description="Phase contract validation",
            exports=["PhaseValidator", "get_phase_validator"],
        ))
        
        self.register(ModuleSpec(
            name="claim_validator",
            import_path="deepthinker.epistemics.claim_validator",
            category=ModuleCategory.VALIDATION,
            priority=ImportPriority.OPTIONAL,
            description="Epistemic claim validation",
            exports=["ClaimValidator", "get_claim_validator"],
        ))
        
        self.register(ModuleSpec(
            name="cognitive_spine",
            import_path="deepthinker.core.cognitive_spine",
            category=ModuleCategory.VALIDATION,
            priority=ImportPriority.RECOMMENDED,
            description="Central schema and resource validation",
            exports=["CognitiveSpine"],
        ))
        
        # === Memory Modules ===
        self.register(ModuleSpec(
            name="memory_manager",
            import_path="deepthinker.memory",
            category=ModuleCategory.MEMORY,
            priority=ImportPriority.OPTIONAL,
            description="Memory management system",
            exports=["MemoryManager", "MEMORY_SYSTEM_AVAILABLE"],
        ))
        
        self.register(ModuleSpec(
            name="memory_guard",
            import_path="deepthinker.memory.memory_guard",
            category=ModuleCategory.MEMORY,
            priority=ImportPriority.OPTIONAL,
            description="Memory access control",
            exports=["MemoryGuard", "get_memory_guard"],
        ))
        
        # === Consensus Modules ===
        self.register(ModuleSpec(
            name="consensus_policy",
            import_path="deepthinker.consensus.policy_engine",
            category=ModuleCategory.CONSENSUS,
            priority=ImportPriority.OPTIONAL,
            description="Consensus policy engine",
            exports=["ConsensusPolicyEngine", "get_consensus_policy_engine"],
        ))
        
        self.register(ModuleSpec(
            name="semantic_consensus",
            import_path="deepthinker.consensus.semantic_distance",
            category=ModuleCategory.CONSENSUS,
            priority=ImportPriority.OPTIONAL,
            description="Semantic distance consensus",
            exports=["SemanticDistanceConsensus"],
        ))
        
        # === Orchestration Modules ===
        self.register(ModuleSpec(
            name="orchestration_store",
            import_path="deepthinker.orchestration",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Orchestration state persistence",
            exports=["OrchestrationStore", "PhaseOutcome"],
        ))
        
        self.register(ModuleSpec(
            name="phase_time_allocator",
            import_path="deepthinker.orchestration.phase_time_allocator",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Phase time budget allocation",
            exports=["PhaseTimeAllocator", "TimeAllocation", "create_allocator_from_store"],
        ))
        
        self.register(ModuleSpec(
            name="iteration_context",
            import_path="deepthinker.orchestration.iteration_context_manager",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Iteration context management",
            exports=["IterationContextManager", "IterationState", "ContextDelta"],
        ))
        
        self.register(ModuleSpec(
            name="convergence_tracker",
            import_path="deepthinker.utils.convergence",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Convergence tracking utilities",
            exports=["ConvergenceTracker"],
        ))
        
        # === Epistemics Modules ===
        self.register(ModuleSpec(
            name="proof_packets",
            import_path="deepthinker.proofs.proof_packet",
            category=ModuleCategory.EPISTEMICS,
            priority=ImportPriority.OPTIONAL,
            description="Proof packet builder for evidence chains",
            exports=["ProofPacketBuilder", "ProofStore", "ProofPacket", "EvidenceType"],
        ))
        
        self.register(ModuleSpec(
            name="contradiction_detector",
            import_path="deepthinker.epistemics.contradiction_detector",
            category=ModuleCategory.EPISTEMICS,
            priority=ImportPriority.OPTIONAL,
            description="Contradiction detection in claims",
            exports=["get_contradiction_detector"],
        ))
        
        self.register(ModuleSpec(
            name="research_evaluation",
            import_path="deepthinker.evaluation.research_evaluation",
            category=ModuleCategory.EPISTEMICS,
            priority=ImportPriority.OPTIONAL,
            description="Research quality evaluation",
            exports=["ResearchEvaluation", "ResearchEvaluationContext"],
        ))
        
        self.register(ModuleSpec(
            name="synthesis_context",
            import_path="deepthinker.evaluation.synthesis_context",
            category=ModuleCategory.EPISTEMICS,
            priority=ImportPriority.OPTIONAL,
            description="Synthesis phase context",
            exports=["SynthesisContext", "SynthesisResult"],
        ))
        
        # === Decision Accountability Modules ===
        self.register(ModuleSpec(
            name="decision_emitter",
            import_path="deepthinker.decisions.decision_emitter",
            category=ModuleCategory.DECISIONS,
            priority=ImportPriority.OPTIONAL,
            description="Decision accountability logging",
            exports=["DecisionEmitter", "DecisionStore", "DecisionRecord", "DecisionType", "OutcomeCause"],
        ))
        
        # === Meta Modules ===
        self.register(ModuleSpec(
            name="meta_controller",
            import_path="deepthinker.meta.meta_controller",
            category=ModuleCategory.META,
            priority=ImportPriority.OPTIONAL,
            description="Meta-cognitive controller",
            exports=["MetaController"],
        ))
        
        self.register(ModuleSpec(
            name="reasoning_supervisor",
            import_path="deepthinker.meta.reasoning_supervisor",
            category=ModuleCategory.META,
            priority=ImportPriority.OPTIONAL,
            description="Reasoning process supervision",
            exports=["ReasoningSupervisor", "LoopDetection"],
        ))
        
        self.register(ModuleSpec(
            name="depth_evaluator",
            import_path="deepthinker.meta.depth_evaluator",
            category=ModuleCategory.META,
            priority=ImportPriority.OPTIONAL,
            description="Analysis depth evaluation",
            exports=["compute_depth_score", "get_depth_target", "compute_depth_gap"],
        ))
        
        self.register(ModuleSpec(
            name="convergence_result",
            import_path="deepthinker.meta.supervisor",
            category=ModuleCategory.META,
            priority=ImportPriority.OPTIONAL,
            description="Convergence result types",
            exports=["ConvergenceResult"],
        ))
        
        # === Resource Modules ===
        self.register(ModuleSpec(
            name="gpu_manager",
            import_path="deepthinker.resources.gpu_manager",
            category=ModuleCategory.RESOURCES,
            priority=ImportPriority.OPTIONAL,
            description="GPU resource management",
            exports=["GPUResourceManager"],
        ))
        
        self.register(ModuleSpec(
            name="model_supervisor",
            import_path="deepthinker.supervisor.model_supervisor",
            category=ModuleCategory.RESOURCES,
            priority=ImportPriority.OPTIONAL,
            description="Model selection supervision",
            exports=["ModelSupervisor", "SupervisorDecision", "GovernanceEscalationSignal", "PHASE_IMPORTANCE"],
        ))
        
        self.register(ModuleSpec(
            name="model_selector",
            import_path="deepthinker.models.model_selector",
            category=ModuleCategory.MODELS,
            priority=ImportPriority.OPTIONAL,
            description="Dynamic model selection",
            exports=["ModelSelector", "get_model_selector"],
        ))
        
        # === Predictor Modules ===
        self.register(ModuleSpec(
            name="cost_predictor",
            import_path="deepthinker.learning.cost_time_predictor",
            category=ModuleCategory.PREDICTORS,
            priority=ImportPriority.OPTIONAL,
            description="Cost and time prediction",
            exports=["CostTimePredictor", "PredictorInput", "EvaluationLogger", "PREDICTOR_CONFIG"],
        ))
        
        self.register(ModuleSpec(
            name="risk_predictor",
            import_path="deepthinker.learning.phase_risk_predictor",
            category=ModuleCategory.PREDICTORS,
            priority=ImportPriority.OPTIONAL,
            description="Phase risk prediction",
            exports=["PhaseRiskPredictor", "PhaseRiskEvaluationLogger"],
        ))
        
        self.register(ModuleSpec(
            name="web_search_predictor",
            import_path="deepthinker.learning.web_search_predictor",
            category=ModuleCategory.PREDICTORS,
            priority=ImportPriority.OPTIONAL,
            description="Web search necessity prediction",
            exports=["WebSearchPredictor", "WebSearchEvaluationLogger", "analyze_content"],
        ))
        
        # === Multi-view Councils ===
        self.register(ModuleSpec(
            name="multiview_councils",
            import_path="deepthinker.councils.multi_view.optimist_council",
            category=ModuleCategory.CONSENSUS,
            priority=ImportPriority.OPTIONAL,
            description="Optimist/Skeptic multi-view councils",
            exports=["OptimistCouncil"],
        ))
        
        self.register(ModuleSpec(
            name="skeptic_council",
            import_path="deepthinker.councils.multi_view.skeptic_council",
            category=ModuleCategory.CONSENSUS,
            priority=ImportPriority.OPTIONAL,
            description="Skeptic council for multi-view",
            exports=["SkepticCouncil"],
        ))
        
        self.register(ModuleSpec(
            name="multiview_utils",
            import_path="deepthinker.councils.multi_view.multi_view_utils",
            category=ModuleCategory.CONSENSUS,
            priority=ImportPriority.OPTIONAL,
            description="Multi-view utilities",
            exports=["extract_disagreements", "MultiViewDisagreement"],
        ))
        
        # === Dynamic Councils ===
        self.register(ModuleSpec(
            name="dynamic_council_factory",
            import_path="deepthinker.councils.dynamic_council_factory",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Dynamic council generation",
            exports=["DynamicCouncilFactory", "CouncilDefinition"],
        ))
        
        # === Web Search Gate ===
        self.register(ModuleSpec(
            name="web_search_gate",
            import_path="deepthinker.tooling.web_search_gate",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.OPTIONAL,
            description="Web search access control",
            exports=["WebSearchGate", "get_web_search_gate"],
        ))
        
        # === Scenario Model ===
        self.register(ModuleSpec(
            name="scenario_model",
            import_path="deepthinker.scenarios.scenario_model",
            category=ModuleCategory.EPISTEMICS,
            priority=ImportPriority.OPTIONAL,
            description="Scenario modeling",
            exports=["ScenarioFactory", "get_scenario_factory"],
        ))
        
        # === Phase Spec ===
        self.register(ModuleSpec(
            name="phase_spec",
            import_path="deepthinker.phases.phase_spec",
            category=ModuleCategory.VALIDATION,
            priority=ImportPriority.OPTIONAL,
            description="Phase specifications",
            exports=["PhaseSpec", "get_phase_spec", "RECON_PHASE", "ANALYSIS_PHASE", "DEEP_ANALYSIS_PHASE", "SYNTHESIS_PHASE"],
        ))
        
        # === Metrics Integration ===
        self.register(ModuleSpec(
            name="metrics_integration",
            import_path="deepthinker.metrics.integration",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Metrics collection integration",
            exports=["get_metrics_config", "get_metrics_hook"],
        ))
        
        self.register(ModuleSpec(
            name="policy_action",
            import_path="deepthinker.policy",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.OPTIONAL,
            description="Policy action types",
            exports=["PolicyAction"],
        ))
        
        # === Alignment ===
        self.register(ModuleSpec(
            name="alignment",
            import_path="deepthinker.alignment.integration",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.OPTIONAL,
            description="Alignment control integration",
            exports=["finalize_alignment"],
        ))
        
        # === Constitution ===
        self.register(ModuleSpec(
            name="constitution",
            import_path="deepthinker.constitution.engine",
            category=ModuleCategory.GOVERNANCE,
            priority=ImportPriority.OPTIONAL,
            description="Constitution engine for governance",
            exports=["ConstitutionEngine"],
        ))
        
        # === Verbose Logger ===
        self.register(ModuleSpec(
            name="verbose_logger",
            import_path="deepthinker.cli",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Verbose CLI logging",
            exports=["verbose_logger"],
        ))
        
        # === SSE Manager ===
        self.register(ModuleSpec(
            name="sse_manager",
            import_path="api.sse",
            category=ModuleCategory.ORCHESTRATION,
            priority=ImportPriority.OPTIONAL,
            description="Server-Sent Events for real-time updates",
            exports=["sse_manager"],
        ))
    
    def register(self, spec: ModuleSpec) -> None:
        """
        Register a module specification.
        
        Args:
            spec: Module specification
        """
        self._modules[spec.name] = spec
    
    def initialize(self, strict_mode: bool = False) -> Dict[str, bool]:
        """
        Initialize the registry by checking all module availability.
        
        Args:
            strict_mode: If True, fail on any CRITICAL module unavailability
            
        Returns:
            Dictionary of module name -> availability
        """
        self._strict_mode = strict_mode
        availability: Dict[str, bool] = {}
        
        # Group modules by category for logging
        by_category: Dict[ModuleCategory, List[Tuple[str, bool, str]]] = {}
        
        for name, spec in self._modules.items():
            is_available, error = self._check_availability(spec)
            availability[name] = is_available
            spec._available = is_available
            spec._error = error
            
            if spec.category not in by_category:
                by_category[spec.category] = []
            by_category[spec.category].append((name, is_available, error or ""))
        
        # Log availability by category
        self._log_availability(by_category)
        
        # Check for critical failures in strict mode
        if strict_mode:
            critical_failures = [
                (name, spec._error)
                for name, spec in self._modules.items()
                if spec.priority == ImportPriority.CRITICAL and not spec._available
            ]
            if critical_failures:
                msg = "Critical safety modules unavailable:\n"
                msg += "\n".join(f"  - {name}: {error}" for name, error in critical_failures)
                raise SafetyModuleUnavailableError("multiple", msg)
        
        self._initialized = True
        return availability
    
    def _check_availability(self, spec: ModuleSpec) -> Tuple[bool, Optional[str]]:
        """
        Check if a module is available without fully loading it.
        
        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            import importlib
            importlib.import_module(spec.import_path)
            return True, None
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Load error: {e}"
    
    def _log_availability(
        self, 
        by_category: Dict[ModuleCategory, List[Tuple[str, bool, str]]]
    ) -> None:
        """Log module availability by category."""
        
        logger.info("=" * 60)
        logger.info("SafetyCore Module Availability Report")
        logger.info("=" * 60)
        
        total_available = 0
        total_modules = 0
        
        for category in ModuleCategory:
            if category not in by_category:
                continue
            
            modules = by_category[category]
            available = sum(1 for _, avail, _ in modules if avail)
            total = len(modules)
            total_available += available
            total_modules += total
            
            status_icon = "✅" if available == total else ("⚠️" if available > 0 else "❌")
            logger.info(f"\n{status_icon} {category.value.upper()} ({available}/{total})")
            
            for name, is_available, error in modules:
                spec = self._modules[name]
                priority_marker = ""
                if spec.priority == ImportPriority.CRITICAL:
                    priority_marker = " [CRITICAL]"
                elif spec.priority == ImportPriority.RECOMMENDED:
                    priority_marker = " [RECOMMENDED]"
                
                if is_available:
                    logger.info(f"    ✓ {name}{priority_marker}")
                else:
                    level = logging.WARNING if spec.priority != ImportPriority.OPTIONAL else logging.DEBUG
                    logger.log(level, f"    ✗ {name}{priority_marker}: {error}")
        
        logger.info("")
        logger.info(f"Total: {total_available}/{total_modules} modules available")
        logger.info("=" * 60)
    
    def is_available(self, name: str) -> bool:
        """
        Check if a module is available.
        
        Args:
            name: Module name
            
        Returns:
            True if available
        """
        if name not in self._modules:
            return False
        
        spec = self._modules[name]
        
        if spec._available is None:
            spec._available, spec._error = self._check_availability(spec)
        
        return spec._available
    
    def get(
        self, 
        name: str, 
        export: Optional[str] = None,
        warn_if_missing: bool = True
    ) -> Any:
        """
        Get a module or specific export, returning None if unavailable.
        
        Args:
            name: Module name
            export: Specific export to retrieve (e.g., "NormativeController")
            warn_if_missing: Whether to log a warning if missing
            
        Returns:
            Module, export, or None if unavailable
        """
        if name not in self._modules:
            if warn_if_missing:
                logger.warning(f"Unknown module requested: {name}")
            return None
        
        spec = self._modules[name]
        
        if not self.is_available(name):
            if warn_if_missing and spec.priority != ImportPriority.OPTIONAL:
                logger.warning(
                    f"Optional module '{name}' unavailable: {spec._error}"
                )
            self._notify_degradation(name, spec._error or "Unknown error")
            return spec.fallback_value
        
        # Load the module if not already loaded
        if spec._module is None:
            try:
                import importlib
                spec._module = importlib.import_module(spec.import_path)
            except Exception as e:
                logger.error(f"Failed to load module '{name}': {e}")
                spec._available = False
                spec._error = str(e)
                return spec.fallback_value
        
        # Return specific export or whole module
        if export:
            return getattr(spec._module, export, None)
        return spec._module
    
    def require(self, name: str, export: Optional[str] = None) -> Any:
        """
        Get a module or export, raising if unavailable.
        
        Use this in critical paths where the module MUST be present.
        
        Args:
            name: Module name
            export: Specific export to retrieve
            
        Returns:
            Module or export
            
        Raises:
            SafetyModuleUnavailableError: If module is unavailable
        """
        if name not in self._modules:
            raise SafetyModuleUnavailableError(name, "Module not registered")
        
        spec = self._modules[name]
        
        if not self.is_available(name):
            raise SafetyModuleUnavailableError(name, spec._error or "Import failed")
        
        result = self.get(name, export, warn_if_missing=False)
        
        if result is None:
            if export:
                raise SafetyModuleUnavailableError(
                    name, 
                    f"Export '{export}' not found in module"
                )
            raise SafetyModuleUnavailableError(name, "Module load returned None")
        
        return result
    
    def get_exports(self, name: str) -> Dict[str, Any]:
        """
        Get all exports from a module as a dictionary.
        
        Args:
            name: Module name
            
        Returns:
            Dictionary of export_name -> value, or empty dict if unavailable
        """
        if not self.is_available(name):
            return {}
        
        spec = self._modules[name]
        module = self.get(name)
        
        if module is None:
            return {}
        
        exports = {}
        for export_name in spec.exports:
            value = getattr(module, export_name, None)
            if value is not None:
                exports[export_name] = value
        
        return exports
    
    def add_degradation_callback(
        self, 
        callback: Callable[[str, str], None]
    ) -> None:
        """
        Add a callback for module degradation events.
        
        Args:
            callback: Function(module_name, error) called when degradation occurs
        """
        self._degradation_callbacks.append(callback)
    
    def _notify_degradation(self, module_name: str, error: str) -> None:
        """Notify callbacks of degradation."""
        for callback in self._degradation_callbacks:
            try:
                callback(module_name, error)
            except Exception as e:
                logger.error(f"Degradation callback failed: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive status report.
        
        Returns:
            Dictionary with availability and status information
        """
        by_category: Dict[str, Dict[str, Any]] = {}
        
        for name, spec in self._modules.items():
            cat = spec.category.value
            if cat not in by_category:
                by_category[cat] = {"available": [], "unavailable": []}
            
            if self.is_available(name):
                by_category[cat]["available"].append(name)
            else:
                by_category[cat]["unavailable"].append({
                    "name": name,
                    "error": spec._error,
                    "priority": spec.priority.value
                })
        
        # Calculate totals
        total_available = sum(
            len(cat["available"]) 
            for cat in by_category.values()
        )
        total_unavailable = sum(
            len(cat["unavailable"]) 
            for cat in by_category.values()
        )
        
        return {
            "initialized": self._initialized,
            "strict_mode": self._strict_mode,
            "total_available": total_available,
            "total_unavailable": total_unavailable,
            "by_category": by_category
        }
    
    def list_available(self, category: Optional[ModuleCategory] = None) -> List[str]:
        """
        List available modules, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of available module names
        """
        result = []
        for name, spec in self._modules.items():
            if category and spec.category != category:
                continue
            if self.is_available(name):
                result.append(name)
        return result
    
    def list_unavailable(self, category: Optional[ModuleCategory] = None) -> List[str]:
        """
        List unavailable modules, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of unavailable module names
        """
        result = []
        for name, spec in self._modules.items():
            if category and spec.category != category:
                continue
            if not self.is_available(name):
                result.append(name)
        return result


# Global singleton instance
safety = SafetyCoreRegistry()


# Convenience functions
def initialize_safety(strict_mode: bool = False) -> Dict[str, bool]:
    """Initialize the global safety registry."""
    return safety.initialize(strict_mode)


def is_safety_available(name: str) -> bool:
    """Check if a safety module is available."""
    return safety.is_available(name)


def get_safety_module(name: str, export: Optional[str] = None) -> Any:
    """Get a safety module or export."""
    return safety.get(name, export)


def require_safety_module(name: str, export: Optional[str] = None) -> Any:
    """Require a safety module (raises if unavailable)."""
    return safety.require(name, export)


