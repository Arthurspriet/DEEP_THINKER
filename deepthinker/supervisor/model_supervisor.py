"""
Model Supervisor for DeepThinker 2.0.

A lightweight LLM agent that makes intelligent decisions about:
- Which models to use for each council
- How many models should participate
- When to downgrade due to GPU pressure
- When to wait for GPU capacity vs downgrade immediately
- Temperature tuning per phase
- Fallback strategies for long missions

Updated for autonomous execution:
- Prefers larger models for longer missions (time_budget > 3 minutes)
- Only downgrades on tokenizer overflow or memory issues
- More willing to wait for GPU capacity on long missions

Runs exclusively on CPU to avoid competing for GPU resources.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..resources.gpu_manager import GPUResourceStats
    from ..missions.mission_types import MissionState, MissionPhase
    from ..meta.supervisor import PhaseMetrics, DepthContract
    from ..decisions.decision_emitter import DecisionEmitter

# Step tier policy imports (Epistemic Hardening Phase 7)
try:
    from ..models.step_tier_policy import (
        StepType,
        evaluate_tier_policy,
        infer_step_type,
        is_truth_critical,
        get_minimum_tier,
    )
    from ..models.model_registry import ModelTier
    STEP_TIER_POLICY_AVAILABLE = True
except ImportError:
    STEP_TIER_POLICY_AVAILABLE = False


# Phase importance weights - higher = more critical, justifies heavier models
PHASE_IMPORTANCE = {
    "synthesis": 1.0,      # Critical - needs best models for final output
    "implementation": 0.9, # High - code quality matters significantly
    "design": 0.8,         # Medium-high - architecture decisions are important
    "testing": 0.7,        # Medium - validation needs good reasoning
    "research": 0.6        # Lower - can use diverse smaller models effectively
}

# Hierarchical model tiers by phase type
# Recon/Research -> small/medium, Implementation/Synthesis -> medium/large
PHASE_MODEL_TIERS = {
    "research": "medium",      # Can use diverse smaller models
    "design": "medium",        # Medium models for planning
    "implementation": "large", # Large models for code quality
    "testing": "medium",       # Medium for validation
    "synthesis": "large",      # Large for final synthesis
}

# Tier to model mapping
TIER_MODELS = {
    "small": ["llama3.2:3b", "llama3.2:1b"],
    "medium": ["deepseek-r1:8b", "gemma3:12b", "mistral:instruct"],
    "large": ["cogito:14b", "gemma3:27b"],
    "xlarge": ["gemma3:27b"],  # If 70B+ available
}

# Quality gain estimates when using heavier models (relative improvement)
MODEL_TIER_QUALITY = {
    "small": 0.6,   # llama3.2:1b, llama3.2:3b
    "medium": 0.8,  # deepseek-r1:8b, gemma3:12b
    "large": 0.95,  # cogito:14b, gemma3:27b
    "xlarge": 1.0   # 70B models
}

# =============================================================================
# Model Floor Enforcement (Model-Aware Phase Stabilization)
# =============================================================================
# Maps phase importance thresholds to minimum allowed model tiers.
# Prevents downgrading below these floors regardless of time/resource pressure.

IMPORTANCE_TIER_FLOORS = {
    # importance >= 0.9 → requires reasoning tier (synthesis, critical phases)
    0.9: "large",
    # importance >= 0.7 → requires large tier minimum
    0.7: "medium",
    # importance >= 0.5 → requires medium tier minimum
    0.5: "medium",
    # importance < 0.5 → no floor (can use small)
}

# Tier priority order for escalation (lower index = weaker tier)
TIER_PRIORITY_ORDER = ["small", "medium", "large", "xlarge"]


def get_tier_floor_for_importance(importance: float) -> str:
    """
    Get the minimum allowed tier for a given importance level.
    
    Args:
        importance: Phase importance score (0.0-1.0)
        
    Returns:
        Minimum tier name that should be used
    """
    for threshold in sorted(IMPORTANCE_TIER_FLOORS.keys(), reverse=True):
        if importance >= threshold:
            return IMPORTANCE_TIER_FLOORS[threshold]
    return "small"  # No floor for low-importance phases


def get_tier_priority(tier: str) -> int:
    """Get numeric priority for a tier (higher = stronger)."""
    try:
        return TIER_PRIORITY_ORDER.index(tier)
    except ValueError:
        return 1  # Default to medium priority


@dataclass
class GovernanceEscalationSignal:
    """
    Signal passed from governance to model supervisor on retry.
    
    Carries information about the failure to enable intelligent
    model escalation on subsequent attempts.
    
    Attributes:
        violation_types: List of violation type strings from governance
        failed_models: List of model names that produced the failing output
        retry_count: Current retry count for this phase
        phase_importance: Importance weight of the phase (0.0-1.0)
        aggregate_severity: Severity score from governance verdict
    """
    violation_types: List[str] = field(default_factory=list)
    failed_models: List[str] = field(default_factory=list)
    retry_count: int = 0
    phase_importance: float = 0.5
    aggregate_severity: float = 0.0
    
    def requires_tier_escalation(self) -> bool:
        """Check if this signal warrants tier escalation."""
        # Escalate if severity is high or multiple retries
        return self.aggregate_severity >= 0.5 or self.retry_count >= 2
    
    def get_escalation_tier(self, current_tier: str) -> str:
        """
        Determine the target tier for escalation.
        
        Args:
            current_tier: The tier that failed
            
        Returns:
            Next higher tier to try
        """
        current_priority = get_tier_priority(current_tier)
        
        # On first retry, try same tier with different models
        # On subsequent retries, escalate tier
        if self.retry_count >= 2:
            target_priority = min(current_priority + 1, len(TIER_PRIORITY_ORDER) - 1)
        else:
            target_priority = current_priority
        
        return TIER_PRIORITY_ORDER[target_priority]


@dataclass
class SupervisorDecision:
    """
    Decision made by the model supervisor.
    
    Attributes:
        models: List of model names to use
        temperature: Recommended temperature for generation
        parallelism: Number of models to run in parallel
        downgraded: Whether this is a downgraded configuration
        reason: Human-readable explanation of the decision
        council_type: Type of council this decision is for
        estimated_vram: Estimated total VRAM requirement in MB
        wait_for_capacity: Whether to wait for GPU capacity instead of running now
        max_wait_minutes: Maximum time to wait for capacity before fallback
        fallback_models: Models to use if wait times out
        phase_importance: Importance weight that influenced this decision
    """
    models: List[str]
    temperature: float
    parallelism: int
    downgraded: bool
    reason: str
    council_type: str = "unknown"
    estimated_vram: int = 0
    wait_for_capacity: bool = False
    max_wait_minutes: float = 5.0
    fallback_models: Optional[List[str]] = None
    phase_importance: float = 0.5
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "models": self.models,
            "temperature": self.temperature,
            "parallelism": self.parallelism,
            "downgraded": self.downgraded,
            "reason": self.reason,
            "council_type": self.council_type,
            "estimated_vram": self.estimated_vram,
            "wait_for_capacity": self.wait_for_capacity,
            "max_wait_minutes": self.max_wait_minutes,
            "fallback_models": self.fallback_models,
            "phase_importance": self.phase_importance
        }


# Default model configurations by phase type
DEFAULT_CONFIGS = {
    "research": {
        "models": ["gemma3:12b", "cogito:14b", "deepseek-r1:8b"],
        "temperature": 0.5,
        "parallelism": 3
    },
    "design": {
        "models": ["cogito:14b", "gemma3:27b"],
        "temperature": 0.6,
        "parallelism": 2
    },
    "implementation": {
        "models": ["deepseek-r1:8b", "gemma3:12b"],
        "temperature": 0.3,
        "parallelism": 2
    },
    "testing": {
        "models": ["mistral:instruct", "gemma3:12b"],
        "temperature": 0.7,
        "parallelism": 2
    },
    "synthesis": {
        "models": ["gemma3:27b", "cogito:14b"],
        "temperature": 0.4,
        "parallelism": 1
    }
}

# Fallback configurations for resource pressure
FALLBACK_CONFIGS = {
    "small": {
        "models": ["llama3.2:3b", "llama3.2:1b"],
        "temperature": 0.5,
        "parallelism": 2
    },
    "medium": {
        "models": ["deepseek-r1:8b", "gemma3:12b"],
        "temperature": 0.5,
        "parallelism": 2
    },
    "large": {
        "models": ["gemma3:27b"],
        "temperature": 0.5,
        "parallelism": 1
    }
}


class ModelSupervisor:
    """
    Lightweight LLM supervisor for model selection decisions.
    
    Uses a small model running on CPU only to make intelligent
    decisions about model configuration based on:
    - Current GPU resource availability
    - Mission phase and type
    - Phase importance and quality requirements
    - Time constraints and remaining mission budget
    - Previous phase results
    
    Key features:
    - GPU-aware: considers current VRAM and utilization
    - Time-aware: balances wait time vs quality gain
    - Phase-aware: critical phases get priority for heavy models
    - Adaptive: can suggest waiting for capacity instead of downgrading
    - Long-mission aware: prefers larger models for missions > 3 minutes
    
    This agent NEVER uses GPU - it runs entirely on CPU/RAM.
    """
    
    # Threshold for "long" missions in minutes
    LONG_MISSION_THRESHOLD_MINUTES = 3.0
    
    def __init__(
        self,
        small_model_name: str = "llama3.2:1b",
        ollama_base_url: str = "http://localhost:11434",
        use_llm: bool = True,
        timeout: float = 30.0,
        prefer_large_for_long_missions: bool = True,
        decision_emitter: Optional["DecisionEmitter"] = None,
    ):
        """
        Initialize the model supervisor.
        
        Args:
            small_model_name: Name of small model for CPU inference
            ollama_base_url: URL of Ollama server
            use_llm: Whether to use LLM for decisions (False = rule-based only)
            timeout: Timeout for LLM calls in seconds
            prefer_large_for_long_missions: If True, prefer larger models for long missions
            decision_emitter: Optional DecisionEmitter for accountability logging
        """
        self.model = small_model_name
        self.base_url = ollama_base_url
        self.use_llm = use_llm
        self.timeout = timeout
        self.prefer_large_for_long_missions = prefer_large_for_long_missions
        self._model_costs: Optional[Dict] = None
        self._decision_emitter = decision_emitter
    
    def mission_is_long(self, time_budget_minutes: float) -> bool:
        """
        Check if a mission qualifies as "long".
        
        Long missions get preference for larger models.
        
        Args:
            time_budget_minutes: Total mission time budget in minutes
            
        Returns:
            True if mission is long (> 3 minutes)
        """
        return time_budget_minutes > self.LONG_MISSION_THRESHOLD_MINUTES
    
    def decide(
        self,
        mission_state: Optional["MissionState"],
        phase: Optional["MissionPhase"],
        gpu_stats: "GPUResourceStats",
        council_config: Optional[Dict[str, Any]] = None,
        step_type_hint: Optional[str] = None,
        escalation_signal: Optional["GovernanceEscalationSignal"] = None
    ) -> SupervisorDecision:
        """
        Make a decision about model configuration.
        
        Args:
            mission_state: Current mission state (optional)
            phase: Current phase being executed (optional)
            gpu_stats: Current GPU resource statistics
            council_config: Configuration for the council to run
            step_type_hint: Hint about what type of step is being performed
            escalation_signal: Optional signal from governance indicating retry with escalation
            
        Returns:
            SupervisorDecision with model configuration
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Determine phase type and importance
        phase_type = self._classify_phase(phase) if phase else "research"
        importance = PHASE_IMPORTANCE.get(phase_type, 0.5)
        
        # If escalation signal present, boost importance
        if escalation_signal is not None:
            importance = max(importance, escalation_signal.phase_importance)
        
        # Epistemic Hardening Phase 7: Check step tier policy
        step_tier_enforced = False
        min_tier_required = "medium"
        should_delay_for_tier = False
        
        if STEP_TIER_POLICY_AVAILABLE and mission_state:
            constraints = getattr(mission_state, "constraints", None)
            if constraints and getattr(constraints, "enable_step_tier_policy", False):
                step_tier_enforced = True
                
                # Infer step type from phase and hint
                phase_name = phase.name if phase else "unknown"
                action_desc = step_type_hint or ""
                inferred_step = infer_step_type(phase_name, action_desc)
                
                # Get minimum tier required for this step
                min_tier = get_minimum_tier(inferred_step)
                min_tier_required = min_tier.value
                
                # Check if this is truth-critical
                if is_truth_critical(inferred_step):
                    importance = max(importance, 0.9)  # Boost importance
                    should_delay_for_tier = True
                    
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[STEP TIER POLICY] Truth-critical step detected: {inferred_step.value}. "
                        f"Minimum tier: {min_tier_required}. Will delay rather than downgrade."
                    )
        
        # Get available VRAM
        available_vram = max(0, gpu_stats.free_mem - 2000)  # 2GB safety margin
        
        # Get resource pressure
        pressure = self._get_pressure(gpu_stats)
        
        # Get remaining mission time
        remaining_minutes = 60.0  # Default
        if mission_state:
            remaining_minutes = mission_state.remaining_minutes()
        
        # Try LLM-based decision if enabled
        if self.use_llm:
            import logging
            logger = logging.getLogger(__name__)
            
            try:
                decision = self._llm_decide(
                    mission_state, phase, gpu_stats, council_config, phase_type
                )
                if decision and self._validate_decision(decision, available_vram):
                    decision.phase_importance = importance
                    return decision
            except Exception as e:
                logger.warning(f"LLM-based supervisor decision failed: {e}. Falling back to rule-based.")
        
        # Rule-based fallback with GPU-aware logic
        decision = self._rule_based_decide(
            phase_type=phase_type,
            available_vram=available_vram,
            pressure=pressure,
            mission_state=mission_state,
            phase=phase,
            importance=importance,
            remaining_minutes=remaining_minutes
        )
        
        # Epistemic Hardening Phase 7: Enforce step tier policy
        if step_tier_enforced:
            decision = self._enforce_step_tier_policy(
                decision=decision,
                min_tier_required=min_tier_required,
                should_delay=should_delay_for_tier,
                pressure=pressure,
            )
        
        # Model-Aware Phase Stabilization: Apply escalation if governance signaled retry
        if escalation_signal is not None:
            decision = self._apply_escalation(
                decision=decision,
                escalation_signal=escalation_signal,
                phase_name=phase.name if phase else "unknown",
                available_vram=available_vram,
            )
        
        # Model-Aware Phase Stabilization: Enforce minimum tier floor based on importance
        tier_floor = get_tier_floor_for_importance(importance)
        decision = self._enforce_tier_floor(
            decision=decision,
            tier_floor=tier_floor,
            importance=importance,
            phase_name=phase.name if phase else "unknown",
        )
        
        # Log the decision visibly
        self._log_supervisor_decision(decision, mission_state, phase)
        
        # Decision Accountability: Emit MODEL_SELECTION decision record
        self._emit_model_selection_decision(decision, mission_state, phase, pressure)
        
        return decision
    
    def _enforce_step_tier_policy(
        self,
        decision: SupervisorDecision,
        min_tier_required: str,
        should_delay: bool,
        pressure: str
    ) -> SupervisorDecision:
        """
        Enforce step tier policy on a decision.
        
        Epistemic Hardening Phase 7: Ensures that truth-critical steps
        use sufficiently strong models. VRAM pressure may delay but
        never downgrades truth-critical steps.
        
        Args:
            decision: The current decision
            min_tier_required: Minimum tier required (small, medium, large, reasoning)
            should_delay: Whether to delay instead of downgrade
            pressure: Current VRAM pressure level
            
        Returns:
            Modified SupervisorDecision
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Map tier names to priority (higher = stronger)
        tier_priority = {
            "small": 1,
            "medium": 2,
            "large": 3,
            "xlarge": 4,
            "reasoning": 4,
        }
        
        # Infer current tier from selected models
        current_tier = "small"
        for model in decision.models:
            for tier, models in TIER_MODELS.items():
                if model in models:
                    if tier_priority.get(tier, 0) > tier_priority.get(current_tier, 0):
                        current_tier = tier
                    break
        
        min_priority = tier_priority.get(min_tier_required, 2)
        current_priority = tier_priority.get(current_tier, 1)
        
        # Check if we need to upgrade
        if current_priority < min_priority:
            if should_delay and pressure in ("high", "critical"):
                # Delay instead of downgrade for truth-critical steps
                decision.wait_for_capacity = True
                decision.max_wait_minutes = min(decision.max_wait_minutes + 5.0, 15.0)
                decision.reason += (
                    f" [STEP TIER POLICY] Truth-critical step requires {min_tier_required} tier. "
                    f"Waiting for capacity instead of downgrading."
                )
                logger.info(
                    f"[STEP TIER POLICY] Delaying for capacity: "
                    f"requires {min_tier_required}, current {current_tier}"
                )
            else:
                # Upgrade models to meet requirement
                target_models = TIER_MODELS.get(min_tier_required, TIER_MODELS["medium"])
                if target_models:
                    decision.models = target_models.copy()
                    decision.downgraded = False  # This is an upgrade
                    decision.reason += (
                        f" [STEP TIER POLICY] Upgraded to {min_tier_required} tier "
                        f"for truth-critical step."
                    )
                    logger.info(
                        f"[STEP TIER POLICY] Upgraded models to {target_models} "
                        f"for {min_tier_required} tier requirement"
                    )
        
        return decision
    
    def _apply_escalation(
        self,
        decision: SupervisorDecision,
        escalation_signal: "GovernanceEscalationSignal",
        phase_name: str,
        available_vram: int,
    ) -> SupervisorDecision:
        """
        Apply model escalation based on governance failure signal.
        
        Model-Aware Phase Stabilization: When governance blocks a phase,
        subsequent retries should use different/stronger models to avoid
        repeating the same failing configuration.
        
        Escalation strategy:
        - Retry 1: Same tier, different model (if available)
        - Retry 2: Next higher tier
        - Retry 3+: Highest available tier
        
        Args:
            decision: Current supervisor decision
            escalation_signal: Signal containing failure information
            phase_name: Name of the current phase
            available_vram: Available VRAM in MB
            
        Returns:
            Modified SupervisorDecision with escalated models
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Determine current tier from decision models
        current_tier = self._get_model_tier(decision.models)
        failed_models = set(escalation_signal.failed_models)
        retry_count = escalation_signal.retry_count
        
        # Get target tier based on escalation signal
        target_tier = escalation_signal.get_escalation_tier(current_tier)
        
        # Get candidate models from target tier
        target_models = TIER_MODELS.get(target_tier, TIER_MODELS["medium"]).copy()
        
        # Filter out failed models to ensure we try something different
        if retry_count == 1:
            # First retry: try different models in same tier
            available_models = [m for m in target_models if m not in failed_models]
            if available_models:
                target_models = available_models
        
        # Ensure we can fit in available VRAM
        final_models = []
        for model in target_models:
            estimated_vram = self._estimate_vram([model])
            if estimated_vram <= available_vram:
                final_models.append(model)
                break  # Take first model that fits
        
        if not final_models:
            # Fallback: use smallest available model from target tier
            for tier in TIER_PRIORITY_ORDER:
                tier_models = TIER_MODELS.get(tier, [])
                for model in tier_models:
                    if self._estimate_vram([model]) <= available_vram:
                        final_models = [model]
                        break
                if final_models:
                    break
        
        if not final_models:
            final_models = ["llama3.2:3b"]  # Ultimate fallback
        
        # Only apply escalation if we're changing something
        if set(final_models) != set(decision.models):
            old_models = decision.models
            decision.models = final_models
            decision.downgraded = False  # This is an escalation, not downgrade
            
            # Reduce temperature slightly for more deterministic output
            decision.temperature = max(0.2, decision.temperature - 0.1)
            
            decision.reason += (
                f" [MODEL_ESCALATION] Retry {retry_count}: "
                f"escalated from {old_models} to {final_models}"
            )
            
            logger.info(
                f"[MODEL_ESCALATION] Phase '{phase_name}' retry {retry_count}: "
                f"escalating from {old_models} to {final_models} "
                f"(importance={escalation_signal.phase_importance:.2f}, "
                f"severity={escalation_signal.aggregate_severity:.2f})"
            )
        
        return decision
    
    def _enforce_tier_floor(
        self,
        decision: SupervisorDecision,
        tier_floor: str,
        importance: float,
        phase_name: str,
    ) -> SupervisorDecision:
        """
        Enforce minimum tier floor based on phase importance.
        
        Model-Aware Phase Stabilization: Prevents downgrading below
        the minimum tier required for the phase importance level.
        
        Args:
            decision: Current supervisor decision
            tier_floor: Minimum allowed tier
            importance: Phase importance (0.0-1.0)
            phase_name: Name of the current phase
            
        Returns:
            Modified SupervisorDecision respecting tier floor
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Get current tier from decision
        current_tier = self._get_model_tier(decision.models)
        current_priority = get_tier_priority(current_tier)
        floor_priority = get_tier_priority(tier_floor)
        
        # Check if we're below the floor
        if current_priority < floor_priority:
            # Need to upgrade to meet floor requirement
            floor_models = TIER_MODELS.get(tier_floor, TIER_MODELS["medium"]).copy()
            
            if floor_models:
                old_models = decision.models
                decision.models = floor_models[:1]  # Take first model from floor tier
                decision.downgraded = False  # This is an upgrade
                decision.reason += (
                    f" [MODEL_FLOOR] Enforced minimum tier '{tier_floor}' "
                    f"for importance={importance:.2f}"
                )
                
                logger.info(
                    f"[MODEL_FLOOR] Phase '{phase_name}' enforcing minimum tier '{tier_floor}' "
                    f"(importance={importance:.2f}, upgraded from {old_models} to {decision.models})"
                )
        
        return decision
    
    def _log_supervisor_decision(
        self,
        decision: SupervisorDecision,
        mission_state: Optional["MissionState"],
        phase: Optional["MissionPhase"]
    ) -> None:
        """
        Log supervisor decision with visible output.
        
        Args:
            decision: The decision made
            mission_state: Current mission state
            phase: Current phase
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Calculate time ratio for visibility
        time_ratio = 1.0
        remaining_min = 0.0
        total_min = 0.0
        if mission_state:
            remaining_min = mission_state.remaining_minutes()
            total_min = mission_state.constraints.time_budget_minutes
            if total_min > 0:
                time_ratio = remaining_min / total_min
        
        phase_name = phase.name if phase else "unknown"
        
        # Build status indicator
        if decision.downgraded:
            status = "DOWNGRADED"
        elif decision.wait_for_capacity:
            status = f"WAITING (max {decision.max_wait_minutes:.1f}min)"
        else:
            status = "OPTIMAL"
        
        # Log with visibility
        logger.info(
            f"[SUPERVISOR] Decision for {phase_name}: {status}\n"
            f"  Models: {decision.models}\n"
            f"  Time: {remaining_min:.1f}/{total_min:.1f}min ({time_ratio*100:.0f}% remaining)\n"
            f"  Reason: {decision.reason}"
        )
    
    def _emit_model_selection_decision(
        self,
        decision: SupervisorDecision,
        mission_state: Optional["MissionState"],
        phase: Optional["MissionPhase"],
        gpu_pressure: str = "low",
    ) -> Optional[str]:
        """
        Emit a MODEL_SELECTION decision record for accountability.
        
        Decision Accountability Layer: Records model selection decisions
        as first-class artifacts with context and rationale.
        
        Args:
            decision: The SupervisorDecision made
            mission_state: Current mission state
            phase: Current phase
            gpu_pressure: GPU pressure level at decision time
            
        Returns:
            decision_id if emitted, None otherwise
        """
        if not self._decision_emitter or not mission_state:
            return None
        
        try:
            phase_name = phase.name if phase else "unknown"
            phase_type = decision.council_type.replace("_council", "") if decision.council_type else "unknown"
            
            # Gather options considered (models that could have been selected)
            all_possible_models = []
            for tier in ["small", "medium", "large", "reasoning"]:
                all_possible_models.extend(TIER_MODELS.get(tier, [])[:2])
            
            decision_id = self._decision_emitter.emit_model_selection(
                mission_id=mission_state.mission_id,
                phase_id=phase_name,
                phase_type=phase_type,
                models_considered=all_possible_models[:6],  # Top candidates
                models_selected=decision.models,
                time_remaining=mission_state.remaining_minutes(),
                importance=decision.phase_importance,
                gpu_pressure=gpu_pressure,
                downgraded=decision.downgraded,
                reason=decision.reason,
                triggered_by=mission_state.last_governance_decision_id,
            )
            
            # Track in mission state
            if decision_id:
                mission_state.set_last_model_decision(decision_id)
            
            return decision_id
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[DECISION] Failed to emit model selection: {e}")
            return None
    
    def set_decision_emitter(self, emitter: "DecisionEmitter") -> None:
        """
        Set the decision emitter for accountability logging.
        
        Args:
            emitter: DecisionEmitter instance
        """
        self._decision_emitter = emitter
    
    def get_adaptive_timeout(
        self,
        phase_type: str,
        time_remaining_minutes: float,
        total_time_minutes: float,
        phase_importance: float = 0.5
    ) -> float:
        """
        Get adaptive timeout for a phase based on time remaining.
        
        Dynamically adjusts timeouts based on:
        - Time remaining ratio
        - Phase importance
        - Phase type defaults
        
        Args:
            phase_type: Type of phase
            time_remaining_minutes: Time remaining in mission
            total_time_minutes: Total mission time budget
            phase_importance: Importance of the phase (0-1)
            
        Returns:
            Recommended timeout in seconds
        """
        # Base timeouts by phase type
        base_timeouts = {
            "research": 120.0,
            "design": 90.0,
            "implementation": 90.0,
            "testing": 60.0,
            "synthesis": 120.0,
        }
        
        base = base_timeouts.get(phase_type, 90.0)
        
        # Calculate time ratio
        if total_time_minutes <= 0:
            time_ratio = 1.0
        else:
            time_ratio = time_remaining_minutes / total_time_minutes
        
        # Adjust timeout based on time ratio
        if time_ratio < 0.1:
            # Less than 10% remaining - aggressive reduction
            adjusted = min(30.0, base * 0.25)
        elif time_ratio < 0.2:
            # Less than 20% remaining - significant reduction  
            adjusted = min(45.0, base * 0.4)
        elif time_ratio < 0.3:
            # Less than 30% remaining - moderate reduction
            adjusted = min(60.0, base * 0.6)
        elif time_ratio < 0.5:
            # Less than 50% remaining - slight reduction
            adjusted = base * 0.8
        else:
            # More than 50% remaining - use base
            adjusted = base
        
        # Boost timeout for important phases
        if phase_importance >= 0.9:
            adjusted = min(adjusted * 1.5, 180.0)  # Max 3 min for critical
        elif phase_importance >= 0.7:
            adjusted = min(adjusted * 1.25, 150.0)
        
        return adjusted
    
    def get_model_tier_for_time_ratio(
        self,
        time_ratio: float,
        phase_importance: float,
        available_vram: int
    ) -> str:
        """
        Get appropriate model tier based on time remaining ratio.
        
        Args:
            time_ratio: Remaining time / total time (0-1)
            phase_importance: Phase importance (0-1)
            available_vram: Available VRAM in MB
            
        Returns:
            Model tier: "small", "medium", "large", or "xlarge"
        """
        # Time-based constraints
        if time_ratio < 0.1:
            # Less than 10% time - always use small
            return "small"
        elif time_ratio < 0.2:
            # 10-20% time - medium at best, small if not important
            return "medium" if phase_importance >= 0.7 else "small"
        elif time_ratio < 0.3:
            # 20-30% time - use medium unless critical
            if phase_importance >= 0.9 and available_vram >= 15000:
                return "large"
            return "medium"
        else:
            # More than 30% time - use appropriate tier based on importance
            if phase_importance >= 0.9 and available_vram >= 18000:
                return "xlarge" if available_vram >= 25000 else "large"
            elif phase_importance >= 0.7 and available_vram >= 12000:
                return "large"
            elif available_vram >= 8000:
                return "medium"
            else:
                return "small"
    
    def should_wait_for_capacity(
        self,
        phase_importance: float,
        remaining_time_minutes: float,
        estimated_wait_minutes: float,
        quality_gain: float,
        mission_time_budget_minutes: Optional[float] = None
    ) -> bool:
        """
        Decide whether to wait for GPU capacity or downgrade immediately.
        
        This implements a value-based decision: is the quality gain from
        using heavier models worth the time cost of waiting?
        
        For long missions (>= 10 minutes), we're much more willing to wait.
        
        Args:
            phase_importance: How critical is this phase (0-1)
            remaining_time_minutes: Time left in mission
            estimated_wait_minutes: Estimated wait for GPU capacity
            quality_gain: Relative quality improvement from heavier models (0-1)
            mission_time_budget_minutes: Total mission time budget (for long mission detection)
            
        Returns:
            True if waiting is worth it, False if should downgrade
        """
        # Never wait if we don't have time
        if remaining_time_minutes < estimated_wait_minutes * 1.5:
            return False
        
        # Check if this is a long mission (>= 10 min)
        is_long_mission = False
        is_very_long_mission = False
        if mission_time_budget_minutes is not None:
            is_long_mission = mission_time_budget_minutes >= 10
            is_very_long_mission = mission_time_budget_minutes >= 30
        
        # For very long missions (>= 30 min), almost always wait
        if is_very_long_mission and self.prefer_large_for_long_missions:
            if remaining_time_minutes > 5:
                return True
        
        # For long missions (>= 10 min), be MORE willing to wait
        if is_long_mission and self.prefer_large_for_long_missions:
            # Lower the threshold for waiting significantly
            if remaining_time_minutes > estimated_wait_minutes * 1.5:
                return True
        
        # For critical phases (synthesis, implementation), wait if we have time
        if phase_importance >= 0.9 and remaining_time_minutes > estimated_wait_minutes * 2:
            return True
        
        # Calculate value: is the quality gain worth the time cost?
        # Higher importance = more willing to wait
        # More remaining time = more willing to wait
        # Higher quality gain = more willing to wait
        time_ratio = remaining_time_minutes / max(estimated_wait_minutes, 1)
        value = quality_gain * phase_importance * min(time_ratio, 2.0)
        
        # For long missions, lower the threshold significantly
        if is_very_long_mission:
            threshold = 0.2
        elif is_long_mission:
            threshold = 0.3
        else:
            threshold = 0.5
        
        return value > threshold
    
    def get_models_for_phase_type(self, phase_type: str, available_vram: int) -> List[str]:
        """
        Get appropriate models for a phase type using hierarchical selection.
        
        Uses PHASE_MODEL_TIERS mapping to determine the appropriate tier,
        then returns models that fit within available VRAM.
        
        Args:
            phase_type: Type of phase (research, design, implementation, etc.)
            available_vram: Available VRAM in MB
            
        Returns:
            List of model names appropriate for this phase
        """
        # Get preferred tier for this phase type
        preferred_tier = PHASE_MODEL_TIERS.get(phase_type, "medium")
        
        # Get models for this tier
        preferred_models = TIER_MODELS.get(preferred_tier, TIER_MODELS["medium"])
        
        # Filter by VRAM availability
        result = []
        for model in preferred_models:
            estimated_vram = self._estimate_vram([model])
            if estimated_vram <= available_vram:
                result.append(model)
        
        # If nothing fits, try lower tiers
        if not result:
            tier_order = ["small", "medium", "large", "xlarge"]
            current_idx = tier_order.index(preferred_tier) if preferred_tier in tier_order else 1
            
            for tier in tier_order[:current_idx][::-1]:  # Go down in tiers
                for model in TIER_MODELS.get(tier, []):
                    estimated_vram = self._estimate_vram([model])
                    if estimated_vram <= available_vram:
                        result.append(model)
                        break
                if result:
                    break
        
        # Fallback to smallest
        if not result:
            result = ["llama3.2:1b"]
        
        return result
    
    def ensure_large_model_for_critical_council(
        self,
        phase_type: str,
        current_models: List[str],
        available_vram: int,
        mission_time_budget_minutes: float
    ) -> List[str]:
        """
        Ensure at least one large model is used for critical councils in long missions.
        
        Args:
            phase_type: Type of phase
            current_models: Currently selected models
            available_vram: Available VRAM
            mission_time_budget_minutes: Total mission time budget
            
        Returns:
            Updated model list (may include a large model)
        """
        # Only apply to long missions
        if mission_time_budget_minutes < 10:
            return current_models
        
        # Only apply to critical phases
        critical_phases = ["synthesis", "implementation"]
        if phase_type not in critical_phases:
            return current_models
        
        # Check if any current model is large
        has_large = any(
            any(x in m.lower() for x in ["14b", "27b", "70b"])
            for m in current_models
        )
        
        if has_large:
            return current_models
        
        # Try to add a large model
        large_models = ["cogito:14b", "gemma3:27b"]
        for model in large_models:
            estimated_vram = self._estimate_vram([model])
            if estimated_vram <= available_vram:
                # Replace the first model with the large one
                result = [model] + current_models[1:] if len(current_models) > 1 else [model]
                return result
        
        return current_models
    
    def _classify_phase(self, phase: "MissionPhase") -> str:
        """Classify phase into a known type."""
        if not phase:
            return "research"
        
        name_lower = phase.name.lower()
        
        if any(kw in name_lower for kw in ["research", "recon", "gather", "investigate"]):
            return "research"
        elif any(kw in name_lower for kw in ["design", "plan", "architect", "strategy"]):
            return "design"
        elif any(kw in name_lower for kw in ["implement", "code", "build", "develop"]):
            return "implementation"
        elif any(kw in name_lower for kw in ["test", "simulation", "validate", "verify"]):
            return "testing"
        elif any(kw in name_lower for kw in ["synthesis", "report", "final", "summary"]):
            return "synthesis"
        else:
            return "research"
    
    def _get_pressure(self, gpu_stats: "GPUResourceStats") -> str:
        """Determine resource pressure level."""
        if gpu_stats.gpu_count == 0:
            return "critical"
        
        memory_pct = (gpu_stats.used_mem / gpu_stats.total_mem * 100) if gpu_stats.total_mem > 0 else 100
        
        if memory_pct > 90 or gpu_stats.utilization > 95:
            return "critical"
        elif memory_pct > 75 or gpu_stats.utilization > 80:
            return "high"
        elif memory_pct > 50 or gpu_stats.utilization > 60:
            return "medium"
        else:
            return "low"
    
    def _get_model_tier(self, models: List[str]) -> str:
        """Determine the tier of a model list."""
        for model in models:
            name_lower = model.lower()
            if any(x in name_lower for x in ["70b", "72b"]):
                return "xlarge"
            elif any(x in name_lower for x in ["27b", "33b"]):
                return "large"
            elif any(x in name_lower for x in ["12b", "13b", "14b"]):
                return "large"
            elif any(x in name_lower for x in ["7b", "8b", "9b"]):
                return "medium"
        return "small"
    
    def _rule_based_decide(
        self,
        phase_type: str,
        available_vram: int,
        pressure: str,
        mission_state: Optional["MissionState"],
        phase: Optional["MissionPhase"],
        importance: float,
        remaining_minutes: float
    ) -> SupervisorDecision:
        """
        Make a rule-based decision with GPU-aware and time-aware logic.
        
        Key improvements over simple threshold-based rules:
        - Considers phase importance for model selection
        - Can suggest waiting for capacity instead of immediate downgrade
        - Adjusts thresholds based on mission context
        - Provides fallback models for wait timeouts
        - Prefers larger models for long missions (> 3 minutes)
        
        For long missions:
        - Only downgrade on actual resource constraints (OOM, tokenizer overflow)
        - More willing to wait for GPU capacity
        - Use largest models that fit in available VRAM
        """
        downgraded = False
        wait_for_capacity = False
        max_wait_minutes = 5.0
        fallback_models: List[str] = []
        reason = ""
        
        # Check if this is a long mission
        is_long_mission = False
        time_budget_minutes = 60.0  # Default
        if mission_state is not None:
            time_budget_minutes = mission_state.constraints.time_budget_minutes
            is_long_mission = self.mission_is_long(time_budget_minutes)
        
        # Calculate time ratio for adaptive decisions
        time_ratio = 1.0
        if time_budget_minutes > 0:
            time_ratio = remaining_minutes / time_budget_minutes
        
        # Get default config for phase
        default = DEFAULT_CONFIGS.get(phase_type, DEFAULT_CONFIGS["research"])
        models = list(default["models"])
        temperature = default["temperature"]
        parallelism = default["parallelism"]
        
        # === TIME-CRITICAL OVERRIDE ===
        # If very low on time, force smaller models regardless of other factors
        if time_ratio < 0.1:
            # Less than 10% time remaining - use smallest viable models
            models = ["llama3.2:3b"]
            parallelism = 1
            downgraded = True
            reason = f"Time critical ({time_ratio*100:.0f}% remaining) - using fastest model"
            
            return SupervisorDecision(
                models=models,
                temperature=0.4,  # Lower temp for speed
                parallelism=parallelism,
                downgraded=downgraded,
                reason=reason,
                council_type=phase_type,
                estimated_vram=self._estimate_vram(models),
                wait_for_capacity=False,
                max_wait_minutes=0,
                fallback_models=None,
                phase_importance=importance
            )
        
        # === MODERATE TIME PRESSURE ===
        if time_ratio < 0.25:
            # 10-25% time remaining - prefer faster models
            tier = self.get_model_tier_for_time_ratio(time_ratio, importance, available_vram)
            models = TIER_MODELS.get(tier, TIER_MODELS["medium"])
            parallelism = 1
            downgraded = tier in ("small", "medium") and importance >= 0.7
            reason = f"Time pressure ({time_ratio*100:.0f}% remaining) - using {tier} tier"
            
            # Don't wait for capacity when time is tight
            wait_for_capacity = False
            max_wait_minutes = 0
        
        # === LONG MISSION: PREFER LARGER MODELS ===
        if is_long_mission and self.prefer_large_for_long_missions:
            # For long missions, try to use the largest models available
            # Only downgrade if absolutely necessary (OOM)
            
            # Check if we can fit large models
            if available_vram >= 18000:
                # Can fit 27B models
                models = ["gemma3:27b"]
                parallelism = 1
                reason = f"Long mission ({time_budget_minutes:.0f}min) - using largest model"
            elif available_vram >= 12000:
                # Can fit 14B models
                models = ["cogito:14b", "gemma3:12b"]
                parallelism = 2
                reason = f"Long mission ({time_budget_minutes:.0f}min) - using large models"
            elif available_vram >= 8000:
                # Can fit 8B models
                models = ["deepseek-r1:8b", "gemma3:12b"]
                parallelism = 2
                reason = f"Long mission ({time_budget_minutes:.0f}min) - using medium models"
            else:
                # Must use smaller models due to VRAM
                downgraded = True
                reason = f"Long mission but limited VRAM ({available_vram}MB) - downgraded"
            
            # For long missions, be more willing to wait for capacity
            if pressure in ("high", "medium") and remaining_minutes > 5:
                wait_for_capacity = True
                max_wait_minutes = min(10, remaining_minutes * 0.3)
                fallback_models = ["deepseek-r1:8b"]
            
            # If not downgraded and using large models, return early
            if not downgraded and pressure not in ("critical",):
                return SupervisorDecision(
                    models=models,
                    temperature=temperature,
                    parallelism=parallelism,
                    downgraded=downgraded,
                    reason=reason,
                    council_type=phase_type,
                    estimated_vram=self._estimate_vram(models),
                    wait_for_capacity=wait_for_capacity,
                    max_wait_minutes=max_wait_minutes,
                    fallback_models=fallback_models if fallback_models else None,
                    phase_importance=importance
                )
        
        # === STANDARD LOGIC FOR SHORT MISSIONS ===
        
        # Adjust VRAM thresholds based on phase importance
        # Critical phases get lower thresholds (more willing to use heavy models)
        vram_threshold_multiplier = 1.0 - (importance * 0.2)  # 0.8x - 1.0x
        
        # Calculate adjusted thresholds
        threshold_xlarge = int(40000 * vram_threshold_multiplier)
        threshold_large = int(20000 * vram_threshold_multiplier)
        threshold_medium = int(10000 * vram_threshold_multiplier)
        threshold_small = int(6000 * vram_threshold_multiplier)
        
        # === CRITICAL PRESSURE HANDLING ===
        if pressure == "critical":
            # Even for critical phases, we must use minimal models
            # But we can suggest waiting if we have lots of time
            if importance >= 0.9 and remaining_minutes > 15:
                # Critical phase with time - suggest waiting
                config = FALLBACK_CONFIGS["small"]
                return SupervisorDecision(
                    models=config["models"],
                    temperature=config["temperature"],
                    parallelism=config["parallelism"],
                    downgraded=True,
                    reason="Critical GPU pressure - using minimal models but consider waiting",
                    council_type=phase_type,
                    estimated_vram=self._estimate_vram(config["models"]),
                    wait_for_capacity=True,
                    max_wait_minutes=min(10, remaining_minutes * 0.3),
                    fallback_models=config["models"],
                    phase_importance=importance
                )
            else:
                config = FALLBACK_CONFIGS["small"]
                return SupervisorDecision(
                    models=config["models"],
                    temperature=config["temperature"],
                    parallelism=config["parallelism"],
                    downgraded=True,
                    reason="Critical GPU pressure - using minimal models",
                    council_type=phase_type,
                    estimated_vram=self._estimate_vram(config["models"]),
                    phase_importance=importance
                )
        
        # === HIGH PRESSURE WITH CRITICAL PHASE OVERRIDE ===
        if pressure == "high" and importance >= 0.8 and remaining_minutes > 10:
            # Don't immediately downgrade critical phases
            # Check if we should wait for capacity
            quality_gain = MODEL_TIER_QUALITY["large"] - MODEL_TIER_QUALITY["medium"]
            
            if self.should_wait_for_capacity(
                phase_importance=importance,
                remaining_time_minutes=remaining_minutes,
                estimated_wait_minutes=3.0,
                quality_gain=quality_gain
            ):
                # Suggest waiting for a heavy model
                if available_vram >= 8000:
                    return SupervisorDecision(
                        models=["gemma3:27b"],
                        temperature=temperature,
                        parallelism=1,
                        downgraded=False,
                        reason=f"Critical {phase_type} phase - waiting for large model capacity",
                        council_type=phase_type,
                        estimated_vram=self._estimate_vram(["gemma3:27b"]),
                        wait_for_capacity=True,
                        max_wait_minutes=min(5, remaining_minutes * 0.2),
                        fallback_models=["cogito:14b", "deepseek-r1:8b"],
                        phase_importance=importance
                    )
        
        # === STANDARD VRAM-BASED RULES WITH IMPORTANCE ADJUSTMENT ===
        
        # Check time pressure
        is_urgent = False
        if mission_state:
            total = mission_state.constraints.time_budget_minutes
            if remaining_minutes < total * 0.2:
                is_urgent = True
        
        # Rule: Handle VRAM constraints with wait-or-downgrade logic
        if available_vram < threshold_xlarge:  # Can't run 70B
            models = [m for m in models if "70b" not in m.lower() and "72b" not in m.lower()]
            if not models:
                models = ["gemma3:27b"]
            if "70b" in str(default["models"]).lower():
                downgraded = True
                reason = f"Insufficient VRAM for 70B models (available: {available_vram}MB)"
        
        if available_vram < threshold_large:  # Can't run 27B+
            # Check if we should wait for synthesis/implementation
            if importance >= 0.9 and remaining_minutes > 10 and available_vram >= 10000:
                # Wait for capacity instead of downgrade
                quality_gain = MODEL_TIER_QUALITY["large"] - MODEL_TIER_QUALITY["medium"]
                if self.should_wait_for_capacity(importance, remaining_minutes, 5.0, quality_gain):
                    return SupervisorDecision(
                        models=["gemma3:27b"],
                        temperature=temperature,
                        parallelism=1,
                        downgraded=False,
                        reason=f"Critical {phase_type} - waiting for large model",
                        council_type=phase_type,
                        estimated_vram=self._estimate_vram(["gemma3:27b"]),
                        wait_for_capacity=True,
                        max_wait_minutes=min(5, remaining_minutes * 0.2),
                        fallback_models=["cogito:14b"],
                        phase_importance=importance
                    )
            
            # Standard downgrade
            models = [m for m in models if not any(x in m.lower() for x in ["27b", "33b", "34b"])]
            if not models:
                models = ["deepseek-r1:8b", "gemma3:12b"]
            downgraded = True
            fallback_models = ["deepseek-r1:8b"]
            reason = f"Limited VRAM - using medium models (available: {available_vram}MB)"
        
        if available_vram < threshold_medium:  # Very limited
            # For critical phases, try to use best available
            if importance >= 0.8:
                models = ["deepseek-r1:8b"]
                fallback_models = ["llama3.2:3b"]
            else:
                models = ["deepseek-r1:8b"]
            parallelism = 1
            downgraded = True
            reason = f"Very limited VRAM - single medium model (available: {available_vram}MB)"
        
        if available_vram < threshold_small:  # Minimal
            models = ["llama3.2:3b", "llama3.2:1b"]
            parallelism = 1
            downgraded = True
            reason = f"Minimal VRAM - using small models (available: {available_vram}MB)"
        
        # === PHASE-SPECIFIC OPTIMIZATION ===
        
        # Rule: Urgent missions prefer speed (but not for synthesis)
        if is_urgent and not downgraded:
            if phase_type != "synthesis":
                models = [m for m in models if any(x in m.lower() for x in ["8b", "7b", "9b", "3b", "12b"])]
                if not models:
                    models = ["deepseek-r1:8b"]
                reason = "Urgent mission - prioritizing speed"
        
        # Rule: Research uses multiple diverse models
        if phase_type == "research" and not downgraded:
            if available_vram >= threshold_large:
                models = ["gemma3:12b", "cogito:14b", "deepseek-r1:8b"]
                parallelism = 3
                reason = "Research phase - using diverse medium models"
        
        # Rule: Implementation uses code-specialized models
        if phase_type == "implementation" and not downgraded:
            if available_vram >= 12000:
                models = ["deepseek-r1:8b", "gemma3:12b"]
                parallelism = 2
                temperature = 0.3
                reason = "Coding phase - using code-specialized models"
        
        # Rule: Synthesis gets the best available
        if phase_type == "synthesis" and not downgraded:
            if available_vram >= threshold_large:
                models = ["gemma3:27b"]
                parallelism = 1
                reason = "Final synthesis - using largest available model"
            elif available_vram >= 12000:
                models = ["cogito:14b"]
                parallelism = 1
                reason = "Final synthesis - using large model"
        
        # Rule: High pressure downgrades (unless already handled above)
        if pressure == "high" and not downgraded and importance < 0.8:
            models = ["gemma3:12b"]
            parallelism = 1
            downgraded = True
            fallback_models = ["deepseek-r1:8b"]
            reason = "High GPU pressure - downgrading to efficient model"
        
        if not reason:
            reason = f"Default configuration for {phase_type} phase"
        
        return SupervisorDecision(
            models=models,
            temperature=temperature,
            parallelism=parallelism,
            downgraded=downgraded,
            reason=reason,
            council_type=phase_type,
            estimated_vram=self._estimate_vram(models),
            wait_for_capacity=wait_for_capacity,
            max_wait_minutes=max_wait_minutes,
            fallback_models=fallback_models if fallback_models else None,
            phase_importance=importance
        )
    
    def _llm_decide(
        self,
        mission_state: Optional["MissionState"],
        phase: Optional["MissionPhase"],
        gpu_stats: "GPUResourceStats",
        council_config: Optional[Dict[str, Any]],
        phase_type: str
    ) -> Optional[SupervisorDecision]:
        """
        Use LLM to make a decision.
        
        The LLM runs on CPU only (num_gpu=0).
        Uses centralized model_caller for proper resource management.
        """
        from deepthinker.models.model_caller import call_model, ModelInvocationError
        
        prompt = self._build_prompt(mission_state, phase, gpu_stats, council_config, phase_type)
        
        try:
            result = call_model(
                model=self.model,
                prompt=prompt,
                options={
                    "num_gpu": 0,  # CPU only
                    "temperature": 0.3,
                    "num_predict": 500
                },
                timeout=self.timeout,
                max_retries=2,  # Fewer retries for supervisor decisions
                base_url=self.base_url,
            )
            
            output = result.get("response", "")
            return self._parse_llm_response(output, phase_type)
            
        except ModelInvocationError:
            return None
        except Exception:
            return None
    
    def _build_prompt(
        self,
        mission_state: Optional["MissionState"],
        phase: Optional["MissionPhase"],
        gpu_stats: "GPUResourceStats",
        council_config: Optional[Dict[str, Any]],
        phase_type: str
    ) -> str:
        """Build prompt for LLM decision."""
        
        importance = PHASE_IMPORTANCE.get(phase_type, 0.5)
        
        # Mission context
        mission_info = ""
        if mission_state:
            remaining = mission_state.remaining_minutes()
            total = mission_state.constraints.time_budget_minutes
            mission_info = f"""
Mission: {mission_state.objective[:200]}
Time remaining: {remaining:.1f} / {total} minutes
Completed phases: {mission_state.current_phase_index} / {len(mission_state.phases)}
"""
        
        # Phase info
        phase_info = ""
        if phase:
            phase_info = f"""
Current phase: {phase.name}
Phase type: {phase_type}
Phase importance: {importance} (1.0 = critical)
Description: {phase.description[:200] if phase.description else 'N/A'}
"""
        
        # GPU info
        gpu_info = f"""
GPU Status:
- Total VRAM: {gpu_stats.total_mem} MB
- Used VRAM: {gpu_stats.used_mem} MB
- Free VRAM: {gpu_stats.free_mem} MB
- Utilization: {gpu_stats.utilization}%
- GPU count: {gpu_stats.gpu_count}
"""
        
        # Available models info
        models_info = """
Available models (VRAM requirement):
Small (< 3GB): llama3.2:1b, llama3.2:3b
Medium (5-10GB): deepseek-r1:8b, gemma3:12b, mistral:instruct
Large (10-20GB): cogito:14b, gemma3:27b
"""
        
        prompt = f"""You are a model supervisor deciding which LLMs to use.

{mission_info}
{phase_info}
{gpu_info}
{models_info}

Rules:
1. Never exceed available VRAM (keep 2GB safety margin)
2. Use code-specialized models for implementation
3. Use multiple diverse models for research
4. For critical phases (importance >= 0.8), prefer waiting for heavy models if time allows
5. Use largest possible models for final synthesis
6. If VRAM is tight but phase is critical, suggest wait_for_capacity=true

Respond with JSON only:
{{
    "models": ["model1", "model2"],
    "temperature": 0.5,
    "parallelism": 2,
    "downgraded": false,
    "wait_for_capacity": false,
    "max_wait_minutes": 5,
    "fallback_models": ["smaller_model"],
    "reason": "explanation"
}}
"""
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        phase_type: str
    ) -> Optional[SupervisorDecision]:
        """Parse LLM response into SupervisorDecision."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            models = data.get("models", [])
            if not models or not isinstance(models, list):
                return None
            
            fallback = data.get("fallback_models", [])
            if not isinstance(fallback, list):
                fallback = []
            
            return SupervisorDecision(
                models=models,
                temperature=float(data.get("temperature", 0.5)),
                parallelism=int(data.get("parallelism", 1)),
                downgraded=bool(data.get("downgraded", False)),
                reason=str(data.get("reason", "LLM decision")),
                council_type=phase_type,
                estimated_vram=self._estimate_vram(models),
                wait_for_capacity=bool(data.get("wait_for_capacity", False)),
                max_wait_minutes=float(data.get("max_wait_minutes", 5.0)),
                fallback_models=fallback if fallback else None,
                phase_importance=PHASE_IMPORTANCE.get(phase_type, 0.5)
            )
            
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    
    def _validate_decision(
        self,
        decision: SupervisorDecision,
        available_vram: int
    ) -> bool:
        """Validate that decision fits within resource constraints."""
        if not decision.models:
            return False
        
        # If waiting is suggested, validate fallback models too
        if decision.wait_for_capacity and decision.fallback_models:
            fallback_vram = self._estimate_vram(decision.fallback_models)
            if fallback_vram > available_vram:
                return False
        
        estimated_vram = self._estimate_vram(decision.models)
        
        # If waiting, we can exceed current capacity
        if decision.wait_for_capacity:
            return True
        
        return estimated_vram <= available_vram
    
    def _estimate_vram(self, models: List[str]) -> int:
        """Estimate total VRAM requirement for models."""
        if self._model_costs is None:
            try:
                from ..resources.model_costs import MODEL_COSTS
                self._model_costs = MODEL_COSTS
            except ImportError:
                self._model_costs = {}
        
        total = 0
        for model in models:
            if model in self._model_costs:
                total += self._model_costs[model].get("vram_mb", 8000)
            else:
                # Estimate based on name
                name_lower = model.lower()
                if any(x in name_lower for x in ["70b", "72b"]):
                    total += 42000
                elif any(x in name_lower for x in ["27b", "33b"]):
                    total += 20000
                elif any(x in name_lower for x in ["12b", "13b", "14b"]):
                    total += 10000
                elif any(x in name_lower for x in ["7b", "8b", "9b"]):
                    total += 6000
                elif any(x in name_lower for x in ["3b", "4b"]):
                    total += 3000
                elif any(x in name_lower for x in ["1b", "2b"]):
                    total += 1500
                else:
                    total += 8000
        
        return total
    
    def get_fallback_decision(self, phase_type: str = "research") -> SupervisorDecision:
        """
        Get a safe fallback decision when all else fails.
        
        Args:
            phase_type: Type of phase for context
            
        Returns:
            Safe minimal SupervisorDecision
        """
        importance = PHASE_IMPORTANCE.get(phase_type, 0.5)
        
        return SupervisorDecision(
            models=["llama3.2:3b"],
            temperature=0.5,
            parallelism=1,
            downgraded=True,
            reason="Fallback to minimal safe configuration",
            council_type=phase_type,
            estimated_vram=2500,
            phase_importance=importance
        )
    
    def choose_models_from_metrics(
        self,
        phase_metrics: "PhaseMetrics",
        depth_contract: "DepthContract",
        mission_type: str,
        time_remaining: float,
        gpu_stats: Optional["GPUResourceStats"] = None
    ) -> SupervisorDecision:
        """
        Choose models based on ReasoningSupervisor metrics and depth contract.
        
        This is the integration point between ReasoningSupervisor analysis
        and model selection. Uses metrics to determine:
        - Model tier (small/medium/large)
        - Number of models (parallelism)
        - Temperature
        - Wait vs downgrade strategy
        
        Args:
            phase_metrics: Metrics from ReasoningSupervisor.analyze_phase_output()
            depth_contract: Depth contract for this phase
            mission_type: Type of mission ("research", "coding", "strategic", etc.)
            time_remaining: Time remaining in minutes
            gpu_stats: Optional GPU statistics
            
        Returns:
            SupervisorDecision based on metrics
        """
        # Determine base configuration from depth contract model tier
        model_tier = depth_contract.model_tier
        
        # Map tier to models
        tier_models = {
            "small": ["llama3.2:3b", "llama3.2:1b"],
            "medium": ["deepseek-r1:8b", "gemma3:12b"],
            "large": ["cogito:14b", "gemma3:27b"],
            "xlarge": ["gemma3:27b"],
        }
        
        models = tier_models.get(model_tier, tier_models["medium"])
        
        # Adjust based on difficulty and uncertainty
        difficulty = phase_metrics.difficulty_score
        uncertainty = phase_metrics.uncertainty_score
        
        # High difficulty + high uncertainty -> use larger models
        if difficulty >= 0.7 and uncertainty >= 0.7:
            if model_tier != "xlarge":
                # Upgrade tier
                if model_tier == "small":
                    models = tier_models["medium"]
                    model_tier = "medium"
                elif model_tier == "medium":
                    models = tier_models["large"]
                    model_tier = "large"
        
        # Low difficulty + low uncertainty -> can use smaller models
        elif difficulty < 0.4 and uncertainty < 0.4:
            if model_tier in ("large", "xlarge"):
                models = tier_models["medium"]
                model_tier = "medium"
        
        # Determine parallelism based on exploration needs
        parallelism = 1
        if depth_contract.allow_alternatives and len(models) > 1:
            parallelism = min(2, len(models))
        if uncertainty >= 0.7:
            # High uncertainty benefits from multiple perspectives
            parallelism = min(3, len(models))
        
        # Determine temperature
        # Higher uncertainty -> higher temperature for exploration
        temperature = 0.4 + (uncertainty * 0.3)  # Range: 0.4 - 0.7
        
        # Time-aware adjustments
        if time_remaining < 2.0:
            # Very limited time - use fastest models
            models = tier_models["small"][:1]
            parallelism = 1
            model_tier = "small"
        elif time_remaining < 5.0:
            # Limited time - reduce parallelism
            parallelism = 1
        
        # Check GPU constraints if available
        downgraded = False
        wait_for_capacity = False
        max_wait_minutes = 5.0
        fallback_models = []
        
        if gpu_stats:
            available_vram = max(0, gpu_stats.free_mem - 2000)  # Safety margin
            estimated_vram = self._estimate_vram(models)
            
            # Calculate reserved headroom: 30GB minimum or 30% of total VRAM
            reserved_headroom_mb = max(30000, int(gpu_stats.total_mem * 0.3))
            
            # Only consider downgrade if we're below reserved headroom
            # If available_vram >= reserved_headroom, we have abundant VRAM - do not downgrade
            if available_vram < reserved_headroom_mb and estimated_vram > available_vram:
                # We're below reserved headroom - need to downgrade or wait
                if time_remaining > 5.0 and difficulty >= 0.6:
                    # Worth waiting for better models
                    wait_for_capacity = True
                    max_wait_minutes = min(3.0, time_remaining * 0.2)
                    fallback_models = tier_models.get("small", ["llama3.2:3b"])
                else:
                    # Downgrade immediately (only when below reserved headroom)
                    downgraded = True
                    if available_vram >= 10000:
                        models = tier_models["medium"]
                    elif available_vram >= 6000:
                        models = ["deepseek-r1:8b"]
                    else:
                        models = tier_models["small"]
                    parallelism = min(parallelism, len(models))
        
        # Build reason
        reason_parts = [
            f"Metrics-based selection: difficulty={difficulty:.2f}, uncertainty={uncertainty:.2f}",
            f"Contract tier={depth_contract.model_tier}, exploration={depth_contract.exploration_depth:.2f}",
        ]
        if downgraded:
            reason_parts.append("(downgraded due to GPU constraints)")
        if wait_for_capacity:
            reason_parts.append(f"(waiting up to {max_wait_minutes:.1f}min for capacity)")
        
        reason = " | ".join(reason_parts)
        
        
        # Phase 8.2: Log resource decision (downgrade/wait)
        if downgraded or wait_for_capacity:
            try:
                from ..cli import verbose_logger
                if verbose_logger and verbose_logger.enabled:
                    gpu_pressure = self.gpu_manager.get_resource_pressure() if self.gpu_manager else "unknown"
                    vram_info = f"VRAM: {available_vram}MB free (estimated need: {estimated_vram}MB)" if gpu_stats else "N/A"
                    verbose_logger.log_resource_decision(
                        original_models=tier_models.get(model_tier, []),
                        selected_models=models,
                        downgraded=downgraded,
                        wait_for_capacity=wait_for_capacity,
                        vram_pressure=gpu_pressure,
                        vram_info=vram_info,
                        time_remaining=time_remaining,
                        max_wait_minutes=max_wait_minutes if wait_for_capacity else None
                    )
            except (ImportError, AttributeError):
                pass  # Verbose logger not available
        
        # Determine phase type from mission type
        phase_type = "research"  # Default
        if "code" in mission_type.lower() or "implement" in mission_type.lower():
            phase_type = "implementation"
        elif "design" in mission_type.lower() or "plan" in mission_type.lower():
            phase_type = "design"
        elif "test" in mission_type.lower() or "eval" in mission_type.lower():
            phase_type = "testing"
        elif "synth" in mission_type.lower() or "report" in mission_type.lower():
            phase_type = "synthesis"
        
        importance = PHASE_IMPORTANCE.get(phase_type, 0.5)
        
        decision = SupervisorDecision(
            models=models,
            temperature=round(temperature, 2),
            parallelism=parallelism,
            downgraded=downgraded,
            reason=reason,
            council_type=phase_type,
            estimated_vram=self._estimate_vram(models),
            wait_for_capacity=wait_for_capacity,
            max_wait_minutes=max_wait_minutes,
            fallback_models=fallback_models if fallback_models else None,
            phase_importance=importance
        )
        
        # Phase 8.1: Log model selection decision
        try:
            from ..cli import verbose_logger
            if verbose_logger and verbose_logger.enabled:
                verbose_logger.log_model_selection(
                    decision=decision,
                    confidence=1.0 - uncertainty  # Higher uncertainty = lower confidence
                )
        except (ImportError, AttributeError):
            pass  # Verbose logger not available
        
        # Phase 8.2: Log resource decision (downgrade/wait) - already done above, but now with decision object
        # The log_resource_decision was already called above, so we're good
        
        return decision
