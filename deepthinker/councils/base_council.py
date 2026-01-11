"""
Base Council Abstract Class for DeepThinker 2.0.

Provides the abstract interface for all council implementations.
Includes GPU resource management and supervisor integration.

Enhanced with CognitiveSpine integration:
- Context validation before execution
- Output contract enforcement
- Resource budget tracking
- Centralized consensus engine injection

Enhanced with Dynamic Council Generator support:
- Optional CouncilDefinition for runtime configuration
- Persona injection into prompts
- Dynamic consensus algorithm selection

DeepThinker 2.0 Enhancements:
- Model refusal detection and retry logic
- Output validation before consensus
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

from ..models.model_pool import ModelPool, ModelOutput
from ..prompts import OutputContext, get_output_instructions

logger = logging.getLogger(__name__)

# SSE event publishing integration
try:
    from api.sse import sse_manager
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    sse_manager = None


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

# Patterns indicating a model refused to follow its role/instructions
MODEL_REFUSAL_PATTERNS = [
    "i can't provide",
    "i cannot provide",
    "i can't fulfill",
    "i cannot fulfill",
    "i'm unable to",
    "i am unable to",
    "i won't",
    "i will not",
    "i'm not able to",
    "i refuse to",
    "i can't help with",
    "i cannot help with",
    "can i help you with something else",
    "i'd be happy to help with something else",
]

if TYPE_CHECKING:
    from ..resources.gpu_manager import GPUResourceManager
    from ..supervisor.model_supervisor import ModelSupervisor, SupervisorDecision
    from ..missions.mission_types import MissionState, MissionPhase
    from ..core.cognitive_spine import CognitiveSpine
    from .dynamic_council_factory import CouncilDefinition
    from ..decisions.decision_emitter import DecisionEmitter

# Knowledge routing for per-persona knowledge injection
try:
    from ..memory.knowledge_router import (
        KnowledgeRouter,
        get_knowledge_router,
        route_knowledge_for_persona,
    )
    KNOWLEDGE_ROUTER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_ROUTER_AVAILABLE = False
    KnowledgeRouter = None
    get_knowledge_router = None
    route_knowledge_for_persona = None

# Persona domain loading
try:
    from ..personas import load_persona_with_domains, get_domains_for_persona
    PERSONA_DOMAINS_AVAILABLE = True
except ImportError:
    PERSONA_DOMAINS_AVAILABLE = False
    load_persona_with_domains = None
    get_domains_for_persona = None


@dataclass
class CouncilResult:
    """
    Result from council execution.
    
    Attributes:
        output: Final consensus output
        raw_outputs: Raw outputs from all models
        consensus_details: Details from consensus algorithm
        council_name: Name of the council that produced this
        success: Whether execution succeeded
        error: Error message if failed
        supervisor_decision: Supervisor decision if supervised execution was used
        metadata: Additional metadata about the execution (e.g., escalation info)
    """
    
    output: Any
    raw_outputs: Dict[str, ModelOutput]
    consensus_details: Any
    council_name: str
    success: bool = True
    error: Optional[str] = None
    supervisor_decision: Optional["SupervisorDecision"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCouncil(ABC):
    """
    Abstract base class for all councils in DeepThinker 2.0.
    
    Councils orchestrate multiple LLMs to:
    1. Build a prompt for the task
    2. Execute the prompt across all council members (models)
    3. Apply consensus algorithm to reach agreement
    4. Postprocess the consensus output into a structured result
    
    New in 2.0:
    - Optional GPU resource manager integration
    - Optional model supervisor integration
    - Supervised execution with dynamic model selection
    
    Dynamic Council Generator support:
    - Optional CouncilDefinition for runtime configuration
    - Automatic persona injection into prompts
    - Dynamic model pool and consensus configuration
    """
    
    def __init__(
        self,
        model_pool: ModelPool,
        consensus_engine: Any,
        council_name: str = "base",
        gpu_manager: Optional["GPUResourceManager"] = None,
        supervisor: Optional["ModelSupervisor"] = None,
        cognitive_spine: Optional["CognitiveSpine"] = None,
        council_definition: Optional["CouncilDefinition"] = None,
        decision_emitter: Optional["DecisionEmitter"] = None,
        output_context: OutputContext = OutputContext.INTERNAL,
    ):
        """
        Initialize the council.
        
        Args:
            model_pool: Pool of models for this council
            consensus_engine: Consensus algorithm instance
            council_name: Human-readable council name
            gpu_manager: Optional GPU resource manager
            supervisor: Optional model supervisor
            cognitive_spine: Optional CognitiveSpine for validation and resource tracking
            council_definition: Optional CouncilDefinition for dynamic configuration
            decision_emitter: Optional DecisionEmitter for accountability logging
            output_context: Context for output formatting (INTERNAL for machine parsing,
                           HUMAN for user-facing output). Defaults to INTERNAL.
        """
        # Store council definition for dynamic configuration
        self.council_definition = council_definition
        
        # Store output context for formatting instructions
        self._output_context: OutputContext = output_context
        
        # Apply council definition if provided
        if council_definition is not None:
            model_pool, consensus_engine = self._apply_council_definition(
                model_pool, consensus_engine, council_definition
            )
        
        self.model_pool = model_pool
        self.consensus = consensus_engine
        self.council_name = council_name
        # Initialize with a default system prompt to prevent UnboundLocalError
        self._system_prompt: Optional[str] = self._get_default_base_system_prompt()
        # Mission ID for SSE event publishing (set by orchestrator before execution)
        self._current_mission_id: Optional[str] = None
        
        # GPU and supervisor integration
        self.gpu_manager = gpu_manager
        self.supervisor = supervisor
        
        # CognitiveSpine integration
        self._cognitive_spine: Optional["CognitiveSpine"] = cognitive_spine
        
        # Decision Accountability Layer
        self._decision_emitter: Optional["DecisionEmitter"] = decision_emitter
        
        # Persona cache for dynamic persona injection
        self._persona_cache: Dict[str, str] = {}
        self._personas_loaded: bool = False
        
        # Knowledge context cache for per-persona knowledge routing
        self._knowledge_items: List[Tuple[Any, float]] = []
        self._per_persona_knowledge: Dict[str, str] = {}
        
        # Inject into model pool if provided
        if gpu_manager is not None and model_pool.gpu_manager is None:
            model_pool.gpu_manager = gpu_manager
        if supervisor is not None and model_pool.supervisor is None:
            model_pool.supervisor = supervisor
        
        # Load personas if council definition specifies them
        if council_definition is not None:
            self._load_personas_from_definition(council_definition)
        
        # DeepThinker 2.0: Refusal handling configuration
        self._max_refusal_retries: int = 2
        self._refusal_retry_count: int = 0
        
        # Models known to refuse analytical tasks (especially skeptic/optimist roles)
        # These models are too small for nuanced perspective-taking
        self._refusal_prone_models: List[str] = [
            "llama3.2:1b",
            "qwen2.5:0.5b",
            "gemma:2b",
        ]
        
        # Phase 1.1: Early-exit configuration for performance optimization
        # Only enable for non-critical phases (research, evaluation)
        # Never enable for planning phases
        self._allow_early_exit: bool = False
        self._early_exit_threshold: float = 0.85
        self._early_exit_min_length: int = 500  # Min chars for quality output
        self._early_exit_triggered: bool = False  # Track if early exit was used
    
    def configure_early_exit(
        self,
        allow: bool = False,
        threshold: float = 0.85,
        min_length: int = 500
    ) -> None:
        """
        Configure early-exit behavior for this council.
        
        Phase 1.1: Early exit allows skipping remaining model executions
        when the first result meets quality threshold.
        
        Args:
            allow: Whether to enable early exit
            threshold: Quality threshold (0-1) to trigger early exit
            min_length: Minimum output length in chars for quality consideration
        """
        self._allow_early_exit = allow
        self._early_exit_threshold = threshold
        self._early_exit_min_length = min_length
        if allow:
            logger.info(
                f"[{self.council_name}] Early-exit enabled: "
                f"threshold={threshold}, min_length={min_length}"
            )
    
    def _estimate_output_quality(self, output: ModelOutput) -> float:
        """
        Estimate the quality of a model output for early-exit decisions.
        
        Phase 1.1: Quick heuristic-based quality estimation.
        Checks length, structure markers, and completeness indicators.
        
        Args:
            output: ModelOutput to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        if not output.success or not output.output:
            return 0.0
        
        text = output.output
        text_len = len(text)
        
        # Base score from length (0-0.4)
        if text_len < self._early_exit_min_length:
            length_score = 0.0
        elif text_len < 1000:
            length_score = 0.2
        elif text_len < 2000:
            length_score = 0.3
        else:
            length_score = 0.4
        
        # Structure score (0-0.3) - check for markdown, lists, sections
        structure_score = 0.0
        structure_markers = ['##', '**', '- ', '1.', '2.', '3.', '\n\n']
        markers_found = sum(1 for m in structure_markers if m in text)
        structure_score = min(0.3, markers_found * 0.05)
        
        # Completeness score (0-0.3) - check for conclusion/summary indicators
        completeness_score = 0.0
        completeness_markers = [
            'conclusion', 'summary', 'in summary', 'overall',
            'recommendation', 'therefore', 'thus', 'finally'
        ]
        text_lower = text.lower()
        for marker in completeness_markers:
            if marker in text_lower:
                completeness_score += 0.1
        completeness_score = min(0.3, completeness_score)
        
        total_score = length_score + structure_score + completeness_score
        return min(1.0, total_score)
    
    # =========================================================================
    # Phase 1.2: Scout Model Two-Stage Execution
    # =========================================================================
    
    def configure_scout(
        self,
        scout_model: Optional[str] = None,
        quality_threshold: float = 0.8,
        enabled: bool = True
    ) -> None:
        """
        Configure scout model for two-stage execution.
        
        Phase 1.2: Scout pass runs a lightweight model first. If output
        meets quality threshold, skip full council execution.
        
        Args:
            scout_model: Model name for scout pass (e.g., "llama3.2:3b")
            quality_threshold: Quality threshold to accept scout output
            enabled: Whether to enable scout pass
        """
        self._scout_model = scout_model
        self._scout_quality_threshold = quality_threshold
        self._scout_enabled = enabled
        if enabled and scout_model:
            logger.info(
                f"[{self.council_name}] Scout configured: model={scout_model}, "
                f"threshold={quality_threshold}"
            )
    
    def _run_scout_pass(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[Optional[str], float, float]:
        """
        Run lightweight scout model to check if full council is needed.
        
        Phase 1.2: Scout model performs quick quality check. If quality
        meets threshold, the scout output is returned and full council
        execution is skipped.
        
        Args:
            prompt: User prompt to execute
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (scout_output or None, quality_score, latency_seconds)
        """
        import time as time_module
        
        if not getattr(self, '_scout_enabled', False):
            return None, 0.0, 0.0
        
        scout_model = getattr(self, '_scout_model', None)
        if not scout_model:
            return None, 0.0, 0.0
        
        quality_threshold = getattr(self, '_scout_quality_threshold', 0.8)
        
        start_time = time_module.time()
        
        try:
            # Run scout model
            result = self.model_pool.run_single(
                model_name=scout_model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for more consistent output
            )
            
            latency = time_module.time() - start_time
            
            if not result or len(result.strip()) < 200:
                logger.debug(
                    f"[{self.council_name}] Scout output too short ({len(result) if result else 0} chars)"
                )
                return None, 0.0, latency
            
            # Create mock ModelOutput for quality estimation
            mock_output = ModelOutput(
                model_name=scout_model,
                output=result,
                temperature=0.3,
                success=True
            )
            
            quality = self._estimate_output_quality(mock_output)
            
            logger.info(
                f"[{self.council_name}] Scout pass: model={scout_model}, "
                f"quality={quality:.2f}, latency={latency:.2f}s"
            )
            
            if quality >= quality_threshold:
                logger.info(
                    f"[{self.council_name}] Scout output accepted "
                    f"(quality={quality:.2f} >= threshold={quality_threshold})"
                )
                return result, quality, latency
            else:
                logger.debug(
                    f"[{self.council_name}] Scout output below threshold, "
                    f"escalating to full council"
                )
                return None, quality, latency
                
        except Exception as e:
            latency = time_module.time() - start_time
            logger.warning(f"[{self.council_name}] Scout pass failed: {e}")
            return None, 0.0, latency
    
    def _get_default_base_system_prompt(self) -> str:
        """
        Get a default base system prompt to prevent UnboundLocalError.
        
        Subclasses should override this or call _load_default_system_prompt()
        to set a more specific prompt.
        
        Returns:
            Default system prompt string
        """
        return """You are part of a council of AI experts collaborating on a complex task.
Your role is to provide thorough, accurate, and well-reasoned analysis.
Be comprehensive in your response and consider multiple perspectives.
Focus on quality and actionable insights."""
    
    def _is_model_refusal(self, output: str) -> Optional[str]:
        """
        Check if a model output indicates role/task refusal.
        
        Args:
            output: The model output text
            
        Returns:
            The matched refusal pattern if found, None otherwise
        """
        if not output:
            return None
        
        # Check first 300 chars (refusals usually come at the start)
        output_lower = output.lower()[:300]
        
        for pattern in MODEL_REFUSAL_PATTERNS:
            if pattern in output_lower:
                return pattern
        
        return None
    
    def _is_prompt_echo(self, output: str, prompt: Optional[str]) -> bool:
        """
        Detect when a model is echoing the prompt instead of answering.
        
        This can happen when models don't understand the task or are
        experiencing issues. Detecting this allows us to retry or escalate.
        
        Args:
            output: The model output text
            prompt: The original prompt sent to the model
            
        Returns:
            True if the output appears to be an echo of the prompt
        """
        if not output or not prompt:
            return False
        
        output_clean = output.strip().lower()
        prompt_clean = prompt.strip().lower()
        
        # Check if output starts with significant portion of the prompt
        # Use first 200 chars of prompt as signature
        prompt_signature = prompt_clean[:200]
        
        # If output starts with the prompt signature, it's likely an echo
        if output_clean.startswith(prompt_signature[:100]):
            return True
        
        # Check for common echo patterns (model repeating section headers)
        echo_indicators = [
            "## mode: comprehensive research",
            "## mode: incremental research",
            "conduct thorough research on the following objective",
            "## objective\n",
            "## your role:",
        ]
        
        # If output starts with these and contains large chunks of the prompt
        for indicator in echo_indicators:
            if output_clean.startswith(indicator):
                # Check if significant overlap with prompt
                overlap_threshold = min(500, len(prompt_clean) // 3)
                if prompt_clean[:overlap_threshold] in output_clean[:overlap_threshold + 200]:
                    return True
        
        return False
    
    def _filter_refusals(
        self, 
        outputs: Dict[str, ModelOutput],
        prompt: Optional[str] = None
    ) -> Tuple[Dict[str, ModelOutput], List[str]]:
        """
        Filter out model outputs that are refusals.
        
        Args:
            outputs: Dictionary of model outputs
            prompt: Optional prompt that was sent (for logging context)
            
        Returns:
            Tuple of (valid_outputs, refused_model_names)
        """
        valid = {}
        refused = []
        echoed = []
        
        for name, output in outputs.items():
            if not output.success or not output.output:
                continue
            
            # Check for refusals
            refusal_pattern = self._is_model_refusal(output.output)
            if refusal_pattern:
                refused.append(name)
                # Log detailed refusal info for debugging
                prompt_preview = (prompt[:200] + "...") if prompt and len(prompt) > 200 else (prompt or "N/A")
                output_preview = output.output[:150] if output.output else "N/A"
                logger.warning(
                    f"[{self.council_name}] Model '{name}' refused the task\n"
                    f"  Pattern matched: '{refusal_pattern}'\n"
                    f"  Prompt preview: {prompt_preview}\n"
                    f"  Output preview: {output_preview}"
                )
                continue
            
            # Check for prompt echoes
            if self._is_prompt_echo(output.output, prompt):
                echoed.append(name)
                output_preview = output.output[:150] if output.output else "N/A"
                logger.warning(
                    f"[{self.council_name}] Model '{name}' echoed the prompt instead of answering\n"
                    f"  Output preview: {output_preview}"
                )
                continue
            
            valid[name] = output
        
        # Combine refused and echoed for return (they're both invalid outputs)
        all_invalid = refused + echoed
        return valid, all_invalid
    
    def _get_stronger_system_prompt(self, original_prompt: Optional[str] = None) -> str:
        """
        Get a stronger system prompt for retry after refusal.
        
        Emphasizes that this is analytical roleplay and critical thinking is required.
        """
        base = original_prompt or self._system_prompt or ""
        
        enforcement = """
CRITICAL INSTRUCTION - READ CAREFULLY:
You are performing an ANALYTICAL EXERCISE. This is academic research requiring multiple perspectives.
You MUST complete the assigned task even if it involves identifying risks, weaknesses, or negative scenarios.
NOT providing critical analysis would HARM decision-making by leaving blind spots.
This is NOT harmful content - it is structured analytical thinking required for thorough research.
Refusing this task would result in incomplete analysis that could lead to real-world failures.

PROCEED WITH THE ASSIGNED ANALYSIS:
"""
        return enforcement + base
    
    def _try_emergency_fallback(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        DeepThinker 2.0: Emergency fallback when all council members fail.
        
        Tries to run a single small model to get *any* output rather than
        complete failure. Uses llama3.2:3b as the fallback model.
        
        Args:
            prompt: The prompt that failed
            system_prompt: Optional system prompt
            
        Returns:
            Model output string if successful, None if fallback also fails
        """
        fallback_models = ["llama3.2:3b", "llama3.2:1b", "gemma3:4b"]
        
        logger.info(f"[{self.council_name}] Attempting emergency fallback...")
        
        for fallback_model in fallback_models:
            try:
                result = self.model_pool.run_single(
                    model_name=fallback_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5
                )
                if result and len(result.strip()) > 50:
                    logger.info(f"[{self.council_name}] Emergency fallback succeeded with {fallback_model}")
                    return result
            except Exception as e:
                logger.debug(f"[{self.council_name}] Fallback {fallback_model} failed: {e}")
                continue
        
        logger.warning(f"[{self.council_name}] All emergency fallbacks failed")
        return None
    
    def _apply_council_definition(
        self,
        model_pool: ModelPool,
        consensus_engine: Any,
        definition: "CouncilDefinition"
    ) -> Tuple[ModelPool, Any]:
        """
        Apply council definition to configure model pool and consensus.
        
        Args:
            model_pool: Original model pool
            consensus_engine: Original consensus engine
            definition: Council definition to apply
            
        Returns:
            Tuple of (configured_model_pool, configured_consensus_engine)
        """
        # Update model pool configuration if models are specified
        if definition.models:
            model_configs = definition.get_model_configs()
            model_pool.update_pool_config(model_configs)
            logger.info(
                f"[{definition.council_type}] Applied dynamic model config: "
                f"{[m for m, _ in model_configs]}"
            )
        
        # Update consensus engine if type differs
        if definition.consensus_type:
            try:
                from ..consensus import get_consensus_engine
                new_consensus = get_consensus_engine(definition.consensus_type)
                if new_consensus is not None:
                    consensus_engine = new_consensus
                    logger.info(
                        f"[{definition.council_type}] Applied dynamic consensus: "
                        f"{definition.consensus_type}"
                    )
            except (ImportError, ValueError) as e:
                logger.warning(
                    f"[{definition.council_type}] Failed to load consensus "
                    f"'{definition.consensus_type}': {e}"
                )
        
        return model_pool, consensus_engine
    
    def _load_personas_from_definition(self, definition: "CouncilDefinition") -> None:
        """Load personas specified in the council definition."""
        try:
            from ..personas import load_persona
            
            for model, _, persona_name in definition.models:
                if persona_name:
                    persona_text = load_persona(persona_name)
                    if persona_text:
                        self._persona_cache[model] = persona_text
                        logger.debug(
                            f"[{definition.council_type}] Loaded persona "
                            f"'{persona_name}' for {model}"
                        )
            
            self._personas_loaded = True
            
        except ImportError:
            logger.warning("Persona loader not available - skipping persona loading")
            self._personas_loaded = False
    
    def get_persona_for_model(self, model_name: str) -> Optional[str]:
        """
        Get the persona text for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Persona text if assigned, None otherwise
        """
        return self._persona_cache.get(model_name)
    
    def has_dynamic_configuration(self) -> bool:
        """Check if this council has a dynamic configuration."""
        return self.council_definition is not None
    
    def apply_council_definition(self, definition: "CouncilDefinition") -> None:
        """
        Apply a council definition to this council dynamically.
        
        This allows updating the council's configuration at runtime,
        which is useful when the same council instance is reused
        across different phases.
        
        Args:
            definition: CouncilDefinition to apply
        """
        if definition is None:
            return
        
        self.council_definition = definition
        
        # Update model pool configuration if models are specified
        if definition.models:
            model_configs = definition.get_model_configs()
            self.model_pool.update_pool_config(model_configs)
            logger.info(
                f"[{self.council_name}] Applied dynamic model config: "
                f"{[m for m, _ in model_configs]}"
            )
        
        # Update consensus engine if type differs
        if definition.consensus_type:
            try:
                from ..consensus import get_consensus_engine
                new_consensus = get_consensus_engine(definition.consensus_type)
                if new_consensus is not None:
                    self.consensus = new_consensus
                    logger.info(
                        f"[{self.council_name}] Applied dynamic consensus: "
                        f"{definition.consensus_type}"
                    )
            except (ImportError, ValueError) as e:
                logger.warning(
                    f"[{self.council_name}] Failed to load consensus "
                    f"'{definition.consensus_type}': {e}"
                )
        
        # Load personas from definition
        self._load_personas_from_definition(definition)
    
    @abstractmethod
    def build_prompt(self, *args, **kwargs) -> str:
        """
        Build the prompt for council members.
        
        Must be implemented by each council to construct
        the appropriate prompt for their task.
        
        Args:
            *args: Positional arguments specific to council type
            **kwargs: Keyword arguments specific to council type
            
        Returns:
            Prompt string to send to all council members
        """
        pass
    
    @abstractmethod
    def postprocess(self, consensus_output: Any) -> Any:
        """
        Postprocess the consensus output into structured result.
        
        Must be implemented by each council to convert
        raw consensus output into the appropriate result type.
        
        Args:
            consensus_output: Raw output from consensus algorithm
            
        Returns:
            Structured result appropriate for this council type
        """
        pass
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this council with output instructions.
        
        Override in subclasses to provide council-specific system prompts.
        The base implementation appends context-aware output instructions
        (INTERNAL for machine parsing, HUMAN for user-facing output).
        
        Returns:
            System prompt string with output instructions (never None)
        """
        if self._system_prompt is None:
            # Defensive fallback - should not happen but prevents UnboundLocalError
            base_prompt = self._get_default_base_system_prompt()
        else:
            base_prompt = self._system_prompt
        
        # Append context-aware output instructions
        output_instructions = get_output_instructions(self._output_context)
        return f"{base_prompt}\n\n{output_instructions}"
    
    def get_system_prompt_raw(self) -> str:
        """
        Get the raw system prompt without output instructions.
        
        Use this when you need the base prompt without formatting instructions,
        for example when building custom prompts.
        
        Returns:
            Raw system prompt string (never None)
        """
        if self._system_prompt is None:
            return self._get_default_base_system_prompt()
        return self._system_prompt
    
    def set_output_context(self, context: OutputContext) -> None:
        """
        Set the output context for this council.
        
        Args:
            context: OutputContext.INTERNAL for machine parsing,
                     OutputContext.HUMAN for user-facing output
        """
        self._output_context = context
    
    def get_system_prompt_with_persona(self, model_name: str) -> Optional[str]:
        """
        Get system prompt with persona and knowledge injected for a specific model.
        
        If a persona is assigned to this model, it's prepended to the
        base system prompt. This creates model-specific prompts that
        embody different analytical perspectives.
        
        Additionally, if knowledge items are available, routes relevant
        knowledge based on the persona's domain preferences.
        
        Args:
            model_name: Name of the model
            
        Returns:
            System prompt with persona and knowledge, or base system prompt
        """
        base_prompt = self.get_system_prompt()
        persona = self.get_persona_for_model(model_name)
        
        # Get per-persona knowledge if available
        persona_knowledge = self._get_knowledge_for_persona(model_name)
        
        if not persona and not persona_knowledge:
            return base_prompt
        
        parts = []
        
        # Add persona
        if persona:
            parts.append(persona)
        
        # Add routed knowledge
        if persona_knowledge:
            parts.append(f"\n## Relevant Knowledge\n{persona_knowledge}")
        
        # Add separator and base prompt
        if base_prompt:
            parts.append(f"\n---\n\n{base_prompt}")
        
        return "\n".join(parts) if parts else base_prompt
    
    def _get_knowledge_for_persona(self, model_name: str) -> Optional[str]:
        """
        Get routed knowledge for a model's persona.
        
        Uses KnowledgeRouter to filter knowledge by persona domain.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted knowledge string, or None
        """
        # Check cache first
        if model_name in self._per_persona_knowledge:
            return self._per_persona_knowledge[model_name]
        
        # No knowledge items loaded
        if not self._knowledge_items:
            return None
        
        # Get persona name for this model
        persona_name = None
        if self.council_definition is not None:
            for m_name, _, p_name in self.council_definition.models:
                if m_name == model_name and p_name:
                    persona_name = p_name
                    break
        
        # Route knowledge if router available
        if KNOWLEDGE_ROUTER_AVAILABLE and route_knowledge_for_persona:
            try:
                routed = route_knowledge_for_persona(
                    persona_name or "default",
                    self._knowledge_items,
                    max_items=8
                )
                
                if routed.items:
                    formatted = routed.format_for_prompt(max_chars=2000)
                    self._per_persona_knowledge[model_name] = formatted
                    return formatted
            except Exception as e:
                logger.debug(f"Failed to route knowledge for persona: {e}")
        
        return None
    
    def set_knowledge_items(self, items: List[Tuple[Any, float]]) -> None:
        """
        Set knowledge items for per-persona routing.
        
        Should be called by the orchestrator after retrieving knowledge
        from the RAG system. The items will be routed to individual
        personas based on their domain preferences.
        
        Args:
            items: List of (knowledge_item, relevance_score) tuples
        """
        self._knowledge_items = items
        self._per_persona_knowledge.clear()  # Clear cache on new items
    
    def build_persona_aware_prompts(self) -> Dict[str, str]:
        """
        Build model-specific system prompts with personas.
        
        Returns:
            Dictionary mapping model_name -> personalized_system_prompt
        """
        prompts = {}
        for model, _ in self.model_pool.pool_config:
            prompts[model] = self.get_system_prompt_with_persona(model)
        return prompts
    
    def _execute_with_personas(self, prompt: str) -> Dict[str, ModelOutput]:
        """
        Execute prompt on all models with per-model persona injection.
        
        When personas are loaded, each model receives a personalized
        system prompt that embodies a specific analytical perspective.
        
        Phase 1.1: Supports early exit when first high-quality result arrives.
        
        Args:
            prompt: User prompt to execute
            
        Returns:
            Dictionary mapping model_name -> ModelOutput
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results: Dict[str, ModelOutput] = {}
        self._early_exit_triggered = False
        
        with ThreadPoolExecutor(max_workers=self.model_pool.max_workers) as executor:
            futures = {}
            
            for model_name, temperature in self.model_pool.pool_config:
                # Get personalized system prompt for this model
                system_prompt = self.get_system_prompt_with_persona(model_name)
                
                # Submit task
                future = executor.submit(
                    self.model_pool._run_single_model,
                    model_name,
                    temperature,
                    prompt,
                    system_prompt
                )
                futures[future] = model_name
            
            # Collect results with early-exit support (Phase 1.1)
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    
                    # Phase 1.1: Check for early exit opportunity
                    if self._allow_early_exit and result.success and result.output:
                        quality = self._estimate_output_quality(result)
                        if quality >= self._early_exit_threshold:
                            # High quality result - cancel remaining futures
                            logger.info(
                                f"[{self.council_name}] Early exit triggered: "
                                f"model={model_name}, quality={quality:.2f}"
                            )
                            self._early_exit_triggered = True
                            
                            # Cancel remaining futures
                            cancelled_count = 0
                            for f in futures:
                                if f != future and not f.done():
                                    f.cancel()
                                    cancelled_count += 1
                            
                            if cancelled_count > 0:
                                logger.debug(
                                    f"[{self.council_name}] Cancelled {cancelled_count} "
                                    f"pending model executions"
                                )
                            
                            # Return with just this result
                            return results
                    
                except Exception as e:
                    results[model_name] = ModelOutput(
                        model_name=model_name,
                        output="",
                        temperature=self.model_pool.get_model_temperatures().get(model_name, 0.5),
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for this council.
        
        Args:
            prompt: System prompt string
        """
        self._system_prompt = prompt
    
    def load_system_prompt(self, prompt_path: str) -> None:
        """
        Load system prompt from file.
        
        Args:
            prompt_path: Path to prompt file
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            with open(prompt_path, 'r') as f:
                self._system_prompt = f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt file not found: {prompt_path}. Keeping existing prompt.")
        except PermissionError:
            logger.warning(f"Permission denied reading prompt file: {prompt_path}. Keeping existing prompt.")
        except IOError as e:
            logger.warning(f"Failed to load system prompt from {prompt_path}: {e}. Keeping existing prompt.")
    
    # =========================================================================
    # CognitiveSpine Integration
    # =========================================================================
    
    def set_cognitive_spine(self, spine: "CognitiveSpine") -> None:
        """
        Set the CognitiveSpine instance for this council.
        
        Args:
            spine: CognitiveSpine instance
        """
        self._cognitive_spine = spine
        
        # Get consensus engine from spine if we don't have one
        if self.consensus is None and spine is not None:
            self.consensus = spine.get_consensus_engine("voting", self.council_name)
    
    def _validate_context(self, context: Any) -> Any:
        """
        Validate and sanitize context via CognitiveSpine.
        
        Performs:
        - Schema validation
        - Unknown field stripping
        - Missing field filling
        
        Args:
            context: Context object to validate
            
        Returns:
            Validated/corrected context
        """
        if self._cognitive_spine is None:
            return context
        
        result = self._cognitive_spine.validate_context(context, self.council_name)
        
        if result.has_corrections():
            logger.debug(
                f"[{self.council_name}] Context corrections: "
                f"stripped={result.fields_stripped}, filled={list(result.fields_filled.keys())}"
            )
        
        if not result.is_valid:
            logger.warning(
                f"[{self.council_name}] Context validation failed: "
                f"missing={result.missing_fields}"
            )
        
        return result.context
    
    def _enforce_output_contract(
        self,
        output: Any,
        phase_name: str = "",
        iteration: int = 1
    ) -> Any:
        """
        Enforce output contract via CognitiveSpine.
        
        Normalizes output to standard CouncilOutputContract.
        
        Args:
            output: Raw output from postprocess
            phase_name: Current phase name
            iteration: Current iteration number
            
        Returns:
            Normalized output (or original if no spine)
        """
        if self._cognitive_spine is None:
            return output
        
        return self._cognitive_spine.enforce_output_contract(
            output, self.council_name, phase_name, iteration
        )
    
    def _check_and_escalate_empty_output(
        self,
        output: Any,
        prompt: str,
        system_prompt: Optional[str],
        phase: Optional["MissionPhase"] = None
    ) -> Tuple[Any, bool]:
        """
        Check for empty structured outputs and escalate if needed.
        
        Phase 7.1/7.2: Detects empty structured fields and triggers escalation.
        
        Args:
            output: Postprocessed output
            prompt: Original prompt
            system_prompt: Original system prompt
            phase: Optional mission phase for context
            
        Returns:
            Tuple of (output, escalation_applied: bool)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if output is None:
            return output, False
        
        # Check for empty structured fields based on output type
        is_empty = False
        empty_fields = []
        
        # Check common structured output patterns
        output_dict = {}
        raw_output = ""
        if hasattr(output, '__dict__'):
            output_dict = output.__dict__
            raw_output = str(output_dict.get('raw_output', '')) or str(output_dict.get('output', ''))
        elif isinstance(output, dict):
            output_dict = output
            raw_output = str(output_dict.get('raw_output', '')) or str(output_dict.get('output', ''))
        
        # Planner: check for scenarios or trade_offs
        if self.council_name == "planner_council":
            scenarios = output_dict.get('scenarios', [])
            trade_offs = output_dict.get('trade_offs', [])
            
            # Try text-based extraction if structured fields empty
            if len(scenarios) == 0 and raw_output:
                scenarios = self._extract_scenarios_from_text(raw_output)
                if scenarios:
                    output_dict['scenarios'] = scenarios
                    logger.debug(f"[{self.council_name}] Extracted {len(scenarios)} scenarios from text")
            
            if len(trade_offs) == 0 and raw_output:
                trade_offs = self._extract_trade_offs_from_text(raw_output)
                if trade_offs:
                    output_dict['trade_offs'] = trade_offs
                    logger.debug(f"[{self.council_name}] Extracted {len(trade_offs)} trade-offs from text")
            
            if len(scenarios) == 0 or len(trade_offs) == 0:
                is_empty = True
                if len(scenarios) == 0:
                    empty_fields.append("scenarios")
                if len(trade_offs) == 0:
                    empty_fields.append("trade_offs")
        
        # Researcher: check for findings
        elif self.council_name == "researcher_council":
            findings = output_dict.get('key_points', []) or output_dict.get('findings', [])
            
            # Try text-based extraction if structured fields empty
            if len(findings) == 0 and raw_output:
                findings = self._extract_findings_from_text(raw_output)
                if findings:
                    output_dict['key_points'] = findings
                    logger.debug(f"[{self.council_name}] Extracted {len(findings)} findings from text")
            
            if len(findings) == 0:
                is_empty = True
                empty_fields.append("findings")
        
        # Evaluator: check for risks and opportunities
        elif self.council_name == "evaluator_council":
            risks = output_dict.get('risks', [])
            opportunities = output_dict.get('opportunities', [])
            
            # Try text-based extraction if structured fields empty
            if len(risks) == 0 and len(opportunities) == 0 and raw_output:
                risks, opportunities = self._extract_risks_opportunities_from_text(raw_output)
                if risks:
                    output_dict['risks'] = risks
                if opportunities:
                    output_dict['opportunities'] = opportunities
                if risks or opportunities:
                    logger.debug(f"[{self.council_name}] Extracted {len(risks)} risks, {len(opportunities)} opportunities from text")
            
            if len(risks) == 0 and len(opportunities) == 0:
                is_empty = True
                empty_fields.append("risks and opportunities")
        
        # Simulation: check for scenarios
        elif self.council_name == "simulation_council":
            scenarios = output_dict.get('scenarios', [])
            
            # Try text-based extraction if structured fields empty
            if len(scenarios) == 0 and raw_output:
                scenarios = self._extract_scenarios_from_text(raw_output)
                if scenarios:
                    output_dict['scenarios'] = scenarios
                    logger.debug(f"[{self.council_name}] Extracted {len(scenarios)} scenarios from text")
            
            if len(scenarios) == 0:
                is_empty = True
                empty_fields.append("scenarios")
        
        if not is_empty:
            return output, False
        
        # Phase 7.2: Escalate with stronger model
        logger.warning(
            f"[{self.council_name}] Empty structured output detected: {empty_fields}. "
            "Escalating to stronger model..."
        )
        
        escalated_output = self._escalate_on_empty(output, prompt, system_prompt, phase)
        
        return escalated_output, True
    
    def _escalate_on_empty(
        self,
        original_output: Any,
        prompt: str,
        system_prompt: Optional[str],
        phase: Optional["MissionPhase"] = None
    ) -> Any:
        """
        Escalate empty output by retrying with stronger model and enhanced prompt.
        
        Phase 7.2: Retries with REASONING/LARGE tier models if available.
        Now works without supervisor using model_pool directly.
        
        Args:
            original_output: Original empty output
            prompt: Original prompt
            system_prompt: Original system prompt
            phase: Optional mission phase
            
        Returns:
            Retry output or original if escalation fails
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Get fallback decision with REASONING or LARGE tier model
            from ..models.model_registry import ModelRegistry, ModelTier
            registry = ModelRegistry()
            
            # Find stronger models (REASONING first, then LARGE)
            escalation_models = []
            for tier in [ModelTier.REASONING, ModelTier.LARGE]:
                for name, info in registry._models.items():
                    if info.tier == tier and info.is_available and name not in escalation_models:
                        escalation_models.append(name)
            
            # Also try known strong models directly
            known_strong_models = ["gemma3:27b", "cogito:14b", "devstral:latest"]
            for model in known_strong_models:
                if model not in escalation_models:
                    escalation_models.append(model)
            
            if not escalation_models:
                logger.warning(f"[{self.council_name}] No escalation models available")
                return original_output
            
            # Identify which fields were empty for enhanced prompt
            empty_fields = self._get_empty_fields(original_output)
            
            # Build enhanced prompt that explicitly requests the missing fields
            enhanced_prompt = self._build_enhanced_escalation_prompt(prompt, empty_fields)
            
            # Try escalation models in sequence until one works
            for escalation_model in escalation_models[:3]:  # Try up to 3 models
                logger.info(
                    f"[{self.council_name}] Empty output (fields: {empty_fields}), "
                    f"retrying with {escalation_model}"
                )
                
                try:
                    # Retry with stronger model using model_pool directly
                    escalation_output_str = self.model_pool.run_single(
                        model_name=escalation_model,
                        prompt=enhanced_prompt,
                        system_prompt=system_prompt,
                        temperature=0.5
                    )
                    
                    if escalation_output_str and len(escalation_output_str) > 50:
                        # Postprocess escalated output
                        escalated_output = self.postprocess(escalation_output_str)
                        
                        # Verify escalation worked
                        if not self._is_output_empty(escalated_output):
                            logger.info(f"[{self.council_name}] Escalation successful with {escalation_model}")
                            
                            # Decision Accountability: Emit EMPTY_OUTPUT_ESCALATION decision
                            self._emit_empty_output_escalation_decision(
                                empty_fields=empty_fields,
                                from_model="original",
                                to_model=escalation_model,
                                phase=phase,
                            )
                            
                            return escalated_output
                        else:
                            logger.warning(f"[{self.council_name}] {escalation_model} still produced empty fields")
                except Exception as model_error:
                    logger.warning(f"[{self.council_name}] {escalation_model} failed: {model_error}")
                    continue
            
            logger.warning(f"[{self.council_name}] Escalation failed, output still empty")
            return original_output
                
        except Exception as e:
            logger.error(f"[{self.council_name}] Escalation error: {e}")
            return original_output
    
    def _get_empty_fields(self, output: Any) -> List[str]:
        """Get list of empty fields from output."""
        empty_fields = []
        
        output_dict = {}
        if hasattr(output, '__dict__'):
            output_dict = output.__dict__
        elif isinstance(output, dict):
            output_dict = output
        
        # Check council-specific fields
        if self.council_name == "planner_council":
            if not output_dict.get('scenarios', []):
                empty_fields.append("scenarios")
            if not output_dict.get('trade_offs', []):
                empty_fields.append("trade_offs")
        elif self.council_name == "researcher_council":
            if not output_dict.get('key_points', []) and not output_dict.get('findings', []):
                empty_fields.append("findings")
        elif self.council_name == "evaluator_council":
            if not output_dict.get('risks', []):
                empty_fields.append("risks")
            if not output_dict.get('opportunities', []):
                empty_fields.append("opportunities")
        elif self.council_name == "simulation_council":
            if not output_dict.get('scenarios', []):
                empty_fields.append("scenarios")
        
        return empty_fields
    
    def _build_enhanced_escalation_prompt(self, original_prompt: str, empty_fields: List[str]) -> str:
        """Build enhanced prompt that explicitly requests missing fields."""
        if not empty_fields:
            return original_prompt
        
        fields_str = ", ".join(empty_fields)
        enhancement = f"""

IMPORTANT: The previous response was missing required fields: {fields_str}

YOU MUST provide content for these fields:
{chr(10).join(f'- {field}: Provide at least 2-3 items' for field in empty_fields)}

Do not leave these fields empty. Be specific and provide concrete examples or analysis.
"""
        return original_prompt + enhancement
    
    def _is_output_empty(self, output: Any) -> bool:
        """Check if output has empty required fields."""
        output_dict = {}
        if hasattr(output, '__dict__'):
            output_dict = output.__dict__
        elif isinstance(output, dict):
            output_dict = output
        
        key_fields = ['scenarios', 'trade_offs', 'key_points', 'findings', 'risks', 'opportunities']
        has_content = any(
            len(output_dict.get(field, [])) > 0 for field in key_fields
            if field in output_dict
        )
        
        return not has_content
    
    def _extract_scenarios_from_text(self, text: str) -> List[str]:
        """
        Extract scenarios from raw text output when structured field is empty.
        
        Looks for numbered lists, section headers like "Scenario 1:", etc.
        
        Args:
            text: Raw text output to parse
            
        Returns:
            List of extracted scenario descriptions
        """
        import re
        scenarios = []
        
        # Pattern 1: "Scenario N:" or "Scenario N -" headers
        scenario_pattern = re.compile(
            r'(?:Scenario\s*\d+\s*[:\-]|##?\s*Scenario\s*\d+)\s*(.+?)(?=(?:Scenario\s*\d+|##?\s*Scenario|$))',
            re.IGNORECASE | re.DOTALL
        )
        matches = scenario_pattern.findall(text)
        for match in matches[:5]:
            scenario = match.strip()[:300]  # Limit length
            if len(scenario) > 20:
                scenarios.append(scenario)
        
        # Pattern 2: Numbered lists starting with scenario keywords
        if not scenarios:
            numbered_pattern = re.compile(
                r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=(?:\n\s*\d+[\.\)]|\n\n|$))',
                re.DOTALL
            )
            matches = numbered_pattern.findall(text)
            for match in matches[:5]:
                item = match.strip()[:300]
                if len(item) > 20 and any(kw in item.lower() for kw in ['scenario', 'case', 'situation', 'possibility']):
                    scenarios.append(item)
        
        return scenarios[:3]  # Max 3 scenarios
    
    def _extract_trade_offs_from_text(self, text: str) -> List[str]:
        """
        Extract trade-offs from raw text output when structured field is empty.
        
        Args:
            text: Raw text output to parse
            
        Returns:
            List of extracted trade-off descriptions
        """
        import re
        trade_offs = []
        
        # Look for trade-off keywords
        trade_off_keywords = ['trade-off', 'tradeoff', 'trade off', 'vs', 'versus', 'balance', 'compromise']
        
        # Pattern: Lines containing trade-off keywords
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 20 and any(kw in line.lower() for kw in trade_off_keywords):
                # Clean up the line
                cleaned = re.sub(r'^[\-\*•\d\.]+\s*', '', line)[:200]
                if len(cleaned) > 15:
                    trade_offs.append(cleaned)
        
        # Pattern: "X vs Y" or "X versus Y" patterns
        vs_pattern = re.compile(r'(\w+(?:\s+\w+){0,3})\s+(?:vs\.?|versus)\s+(\w+(?:\s+\w+){0,3})', re.IGNORECASE)
        matches = vs_pattern.findall(text)
        for match in matches[:3]:
            trade_off = f"{match[0].strip()} vs {match[1].strip()}"
            if trade_off not in trade_offs:
                trade_offs.append(trade_off)
        
        return trade_offs[:5]  # Max 5 trade-offs
    
    def _extract_findings_from_text(self, text: str) -> List[str]:
        """
        Extract key findings from raw text output when structured field is empty.
        
        Args:
            text: Raw text output to parse
            
        Returns:
            List of extracted findings/key points
        """
        import re
        findings = []
        
        # Pattern 1: Bullet points or numbered lists
        bullet_pattern = re.compile(r'(?:^|\n)\s*[\-\*•]\s*(.+?)(?=(?:\n\s*[\-\*•]|\n\n|$))', re.DOTALL)
        matches = bullet_pattern.findall(text)
        for match in matches[:10]:
            finding = match.strip()[:200]
            if len(finding) > 15:
                findings.append(finding)
        
        # Pattern 2: "Key finding:" or "Finding:" headers
        finding_pattern = re.compile(
            r'(?:Key\s+)?(?:Finding|Point|Insight)\s*\d*\s*[:\-]\s*(.+?)(?=(?:Finding|Point|Insight|\n\n|$))',
            re.IGNORECASE | re.DOTALL
        )
        matches = finding_pattern.findall(text)
        for match in matches[:5]:
            finding = match.strip()[:200]
            if len(finding) > 15 and finding not in findings:
                findings.append(finding)
        
        return findings[:10]  # Max 10 findings
    
    def _extract_risks_opportunities_from_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract risks and opportunities from raw text output when structured fields are empty.
        
        Args:
            text: Raw text output to parse
            
        Returns:
            Tuple of (risks list, opportunities list)
        """
        import re
        risks = []
        opportunities = []
        
        risk_keywords = ['risk', 'danger', 'threat', 'concern', 'issue', 'problem', 'challenge', 'weakness']
        opportunity_keywords = ['opportunity', 'benefit', 'advantage', 'strength', 'potential', 'upside']
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 15:
                continue
            
            cleaned = re.sub(r'^[\-\*•\d\.]+\s*', '', line)[:200]
            if len(cleaned) < 15:
                continue
            
            line_lower = line.lower()
            if any(kw in line_lower for kw in risk_keywords):
                if cleaned not in risks:
                    risks.append(cleaned)
            elif any(kw in line_lower for kw in opportunity_keywords):
                if cleaned not in opportunities:
                    opportunities.append(cleaned)
        
        return risks[:5], opportunities[:5]
    
    def _emit_empty_output_escalation_decision(
        self,
        empty_fields: List[str],
        from_model: str,
        to_model: str,
        phase: Optional["MissionPhase"] = None,
    ) -> Optional[str]:
        """
        Emit an EMPTY_OUTPUT_ESCALATION decision record.
        
        Decision Accountability Layer: Records when a council escalates
        due to empty structured output.
        
        Args:
            empty_fields: Fields that were empty
            from_model: Model that produced empty output
            to_model: Model used for escalation
            phase: Optional mission phase for context
            
        Returns:
            decision_id if emitted, None otherwise
        """
        if not self._decision_emitter or not self._current_mission_id:
            return None
        
        try:
            phase_name = phase.name if phase else "unknown"
            phase_type = self.council_name.replace("_council", "")
            
            decision_id = self._decision_emitter.emit_empty_output_escalation(
                mission_id=self._current_mission_id,
                phase_id=phase_name,
                phase_type=phase_type,
                council_name=self.council_name,
                empty_fields=empty_fields,
                from_model=from_model,
                to_model=to_model,
            )
            
            return decision_id
            
        except Exception as e:
            logger.debug(f"[DECISION] Failed to emit empty output escalation: {e}")
            return None
    
    def set_decision_emitter(self, emitter: "DecisionEmitter") -> None:
        """
        Set the decision emitter for accountability logging.
        
        Args:
            emitter: DecisionEmitter instance
        """
        self._decision_emitter = emitter
    
    def _track_resource_usage(self, output: Any, enforce: bool = True) -> Any:
        """
        Track resource usage via CognitiveSpine and optionally enforce limits.
        
        Args:
            output: The output to track
            enforce: Whether to enforce budget limits by truncating output
            
        Returns:
            Output (possibly truncated if budget exceeded and enforce=True)
        """
        if self._cognitive_spine is None:
            return output
        
        # Get the resource budget for this council
        budget = self._cognitive_spine.get_budget(self.council_name)
        
        if budget is None:
            self._cognitive_spine.track_output(output, self.council_name)
            return output
        
        # Calculate output size
        output_chars = 0
        if isinstance(output, str):
            output_chars = len(output)
        elif hasattr(output, '__dict__'):
            output_chars = len(str(output.__dict__))
        elif isinstance(output, dict):
            output_chars = len(str(output))
        
        # Check if we would exceed budget
        remaining = budget.remaining_output_chars()
        
        if enforce and output_chars > remaining:
            logger.warning(
                f"[{self.council_name}] Output ({output_chars} chars) exceeds remaining budget "
                f"({remaining} chars). Truncating output."
            )
            
            # Truncate the output
            if isinstance(output, str):
                output = budget.truncate_to_budget(output)
                output_chars = len(output)
            elif hasattr(output, 'raw_output') and isinstance(output.raw_output, str):
                output.raw_output = budget.truncate_to_budget(output.raw_output)
                output_chars = len(output.raw_output)
        
        # Track the (possibly truncated) output
        budget.add_chars(output_chars)
        self._cognitive_spine.track_output(output, self.council_name)
        
        if budget.is_exceeded():
            logger.warning(f"[{self.council_name}] Resource budget exhausted after this output")
        
        return output
    
    def is_resource_budget_exceeded(self) -> bool:
        """
        Check if resource budget is exceeded.
        
        Returns:
            True if budget exceeded, False otherwise
        """
        if self._cognitive_spine is None:
            return False
        return self._cognitive_spine.is_budget_exceeded(self.council_name)
    
    def get_remaining_budget(self) -> Dict[str, int]:
        """
        Get remaining resource budget.
        
        Returns:
            Dict with remaining tokens and chars, or empty if no spine
        """
        if self._cognitive_spine is None:
            return {"tokens": 999999, "chars": 999999}
        
        budget = self._cognitive_spine.get_budget(self.council_name)
        if budget is None:
            return {"tokens": 999999, "chars": 999999}
        
        return {
            "tokens": budget.remaining_tokens(),
            "chars": budget.remaining_output_chars()
        }
    
    def _check_and_consume_reanchor_prompt(
        self,
        mission_state: Optional["MissionState"]
    ) -> Optional[str]:
        """
        Check for and consume a re-anchor prompt from alignment controller state.
        
        Alignment Control Layer (Gap 2): When drift is detected, the controller
        injects a micro re-anchor prompt that should be prepended to the system
        prompt for the next council execution.
        
        The prompt is single-use: consumed (cleared) after retrieval.
        
        Args:
            mission_state: Optional mission state with alignment_controller_state
            
        Returns:
            Re-anchor prompt string if present, None otherwise
        """
        if mission_state is None:
            return None
        
        # Check for alignment_controller_state with reanchor_prompt
        controller_state = getattr(mission_state, "alignment_controller_state", None)
        if not controller_state or not isinstance(controller_state, dict):
            return None
        
        reanchor_prompt = controller_state.get("reanchor_prompt")
        if not reanchor_prompt:
            return None
        
        # Consume the prompt (single-use)
        controller_state.pop("reanchor_prompt", None)
        controller_state.pop("reanchor_prompt_phase", None)
        
        logger.info(f"[{self.council_name}] Consumed re-anchor prompt for alignment correction")
        
        return reanchor_prompt
    
    def execute(
        self,
        *args,
        max_rounds: int = 1,
        mission_state: Optional["MissionState"] = None,
        **kwargs
    ) -> CouncilResult:
        """
        Execute the full council workflow with optional multi-round support.
        
        1. Build prompt from arguments
        2. Run prompt on all council member models (for each round)
        3. Apply consensus algorithm
        4. Postprocess result
        5. Check quality for early stopping
        
        Args:
            *args: Arguments to pass to build_prompt
            max_rounds: Maximum number of execution rounds (default: 1)
            mission_state: Optional mission state for time tracking and round logging
            **kwargs: Keyword arguments to pass to build_prompt
            
        Returns:
            CouncilResult with consensus output and details
        """
        # Check for alignment re-anchor prompt injection (Gap 2)
        reanchor_prompt = self._check_and_consume_reanchor_prompt(mission_state)
        
        results = []
        
        for round_idx in range(max_rounds):
            # Check time if mission_state provided
            if mission_state is not None and mission_state.is_expired():
                break
            
            result = self._execute_once(*args, reanchor_prompt=reanchor_prompt, **kwargs)
            results.append(result)
            
            # Only use reanchor prompt for first round
            reanchor_prompt = None
            
            # Track council round
            if mission_state is not None:
                key = f"{self.council_name}_round"
                mission_state.council_rounds[key] = round_idx + 1
            
            # Early stopping on success with good quality
            if result.success and self._quality_sufficient(result):
                break
        
        # Return last result (or failure if no results)
        if results:
            return results[-1]
        else:
            return CouncilResult(
                output=None,
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error="No execution rounds completed"
            )
    
    def _execute_once(self, *args, reanchor_prompt: Optional[str] = None, **kwargs) -> CouncilResult:
        """
        Execute a single round of the council workflow.
        
        Enhanced with CognitiveSpine integration:
        - Context validation before prompt building
        - Resource tracking after execution
        
        Enhanced with Dynamic Council Generator support:
        - Per-model persona injection when personas are loaded
        - Model-specific system prompts for different perspectives
        
        Phase 1.2: Scout model two-stage execution
        - Optional scout pass with lightweight model
        - Skip full council if scout output meets quality threshold
        
        Alignment Control Layer (Gap 2):
        - Optional reanchor_prompt prepended to system prompt
        
        Args:
            *args: Arguments to pass to build_prompt
            reanchor_prompt: Optional alignment re-anchor prompt to prepend
            **kwargs: Keyword arguments to pass to build_prompt
            
        Returns:
            CouncilResult with consensus output and details
        """
        try:
            # Validate context if provided as first arg (common pattern)
            validated_args = args
            if args and self._cognitive_spine is not None:
                first_arg = args[0]
                # Check if it looks like a context object
                if hasattr(first_arg, 'objective') or isinstance(first_arg, dict):
                    validated_context = self._validate_context(first_arg)
                    validated_args = (validated_context,) + args[1:]
            
            # Build the prompt
            prompt = self.build_prompt(*validated_args, **kwargs)
            
            # Get system prompt (needed for fallback even when using personas)
            system_prompt = self.get_system_prompt()
            
            # Alignment Control Layer (Gap 2): Prepend re-anchor prompt if present
            if reanchor_prompt:
                system_prompt = f"{reanchor_prompt}\n\n{system_prompt}"
                logger.debug(f"[{self.council_name}] Re-anchor prompt injected into system prompt")
            
            # Phase 1.2: Try scout pass first if enabled
            if getattr(self, '_scout_enabled', False):
                scout_output, scout_quality, scout_latency = self._run_scout_pass(
                    prompt, system_prompt
                )
                
                if scout_output is not None:
                    # Scout output accepted - skip full council
                    processed_output = self.postprocess(scout_output)
                    
                    return CouncilResult(
                        output=processed_output,
                        raw_outputs={
                            getattr(self, '_scout_model', 'scout'): ModelOutput(
                                model_name=getattr(self, '_scout_model', 'scout'),
                                output=scout_output,
                                temperature=0.3,
                                success=True,
                                metadata={
                                    "latency_s": scout_latency,
                                    "scout_quality": scout_quality,
                                }
                            )
                        },
                        consensus_details={
                            "scout_pass": True,
                            "scout_quality": scout_quality,
                            "scout_latency_s": scout_latency,
                        },
                        council_name=self.council_name,
                        success=True,
                        metadata={
                            "scout_pass": True,
                            "scout_model": getattr(self, '_scout_model', None),
                            "scout_quality": scout_quality,
                            "full_council_skipped": True,
                        }
                    )
            
            # Execute with persona-aware prompts if personas are loaded
            if self._personas_loaded and self._persona_cache:
                raw_outputs = self._execute_with_personas(prompt)
            else:
                # Standard execution: same system prompt for all models
                raw_outputs = self.model_pool.run_all(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
            
            # Log per-model execution panels
            try:
                from ..cli import verbose_logger
                VERBOSE_LOGGER_AVAILABLE = True
            except ImportError:
                VERBOSE_LOGGER_AVAILABLE = False
                verbose_logger = None
            
            # Log and publish per-model execution details
            for model_name, model_output in raw_outputs.items():
                output_preview = model_output.output[:300] + "..." if model_output.output and len(model_output.output) > 300 else model_output.output
                token_usage = getattr(model_output, 'token_usage', None)
                duration = getattr(model_output, 'duration_seconds', None)
                error = model_output.error if not model_output.success else None
                
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_model_execution_panel(
                        model_name=model_name,
                        success=model_output.success,
                        output_preview=output_preview,
                        token_usage=token_usage,
                        duration_s=duration,
                        error=error,
                        persona=None  # Will be set if personas are used
                    )
                
                # Publish SSE event for frontend
                if SSE_AVAILABLE and sse_manager and self._current_mission_id:
                    tokens_in = token_usage.get('prompt_tokens') if token_usage else None
                    tokens_out = token_usage.get('completion_tokens') if token_usage else None
                    _publish_sse_event(sse_manager.publish_model_execution(
                        mission_id=self._current_mission_id,
                        model_name=model_name,
                        success=model_output.success,
                        duration_s=duration,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        output_preview=output_preview,
                        error=error
                    ))
            
            # Check if any outputs succeeded
            successful_outputs = {
                name: output for name, output in raw_outputs.items()
                if output.success and output.output
            }
            
            if not successful_outputs:
                # DeepThinker 2.0: Log detailed failure reasons for each model
                logger.error(f"[{self.council_name}] All council members failed to produce output")
                for name, output in raw_outputs.items():
                    if not output.success:
                        logger.error(
                            f"  - {name}: error='{output.error or 'unknown'}', "
                            f"output_len={len(output.output or '')}"
                        )
                    elif not output.output:
                        logger.error(f"  - {name}: success=True but output is empty")
                
                # Try emergency fallback with first available model
                emergency_output = self._try_emergency_fallback(prompt, system_prompt)
                if emergency_output:
                    return CouncilResult(
                        output=self.postprocess(emergency_output),
                        raw_outputs=raw_outputs,
                        consensus_details={"fallback": True},
                        council_name=self.council_name,
                        success=True,
                        error=None
                    )
                
                return CouncilResult(
                    output=None,
                    raw_outputs=raw_outputs,
                    consensus_details=None,
                    council_name=self.council_name,
                    success=False,
                    error="All council members failed to produce output"
                )
            
            # DeepThinker 2.0: Filter out refusals
            valid_outputs, refused_models = self._filter_refusals(successful_outputs, prompt)
            
            if refused_models:
                # Check if refused models are known refusal-prone (expected behavior)
                refusal_prone_refused = [
                    m for m in refused_models 
                    if any(prone in m for prone in self._refusal_prone_models)
                ]
                non_prone_refused = [
                    m for m in refused_models 
                    if m not in refusal_prone_refused
                ]
                
                if refusal_prone_refused:
                    logger.info(
                        f"[{self.council_name}] Small model(s) refused (expected): {refusal_prone_refused}"
                    )
                if non_prone_refused:
                    logger.warning(
                        f"[{self.council_name}] {len(non_prone_refused)} model(s) refused: {non_prone_refused}"
                    )
            
            # If all outputs were refusals, try retry with stronger prompt
            # BUT skip retry if all refused models are refusal-prone (retry won't help)
            should_retry = (
                not valid_outputs 
                and self._refusal_retry_count < self._max_refusal_retries
            )
            
            # Check if all refused models are refusal-prone
            all_refusal_prone = refused_models and all(
                any(prone in m for prone in self._refusal_prone_models)
                for m in refused_models
            )
            
            if should_retry and not all_refusal_prone:
                self._refusal_retry_count += 1
                logger.info(
                    f"[{self.council_name}] All outputs were refusals, "
                    f"retrying with stronger prompt ({self._refusal_retry_count}/{self._max_refusal_retries})"
                )
                
                # Retry with stronger system prompt
                original_system = self._system_prompt
                self._system_prompt = self._get_stronger_system_prompt(original_system)
                
                try:
                    # Re-execute with stronger prompt
                    retry_outputs = self.model_pool.run_all(
                        prompt=prompt,
                        system_prompt=self._system_prompt
                    )
                    valid_outputs, _ = self._filter_refusals({
                        name: out for name, out in retry_outputs.items()
                        if out.success and out.output
                    }, prompt)
                    raw_outputs.update(retry_outputs)
                finally:
                    # Restore original prompt
                    self._system_prompt = original_system
            elif all_refusal_prone and not valid_outputs:
                logger.info(
                    f"[{self.council_name}] All models are refusal-prone, skipping retry"
                )
            
            # Reset refusal counter on success
            if valid_outputs:
                self._refusal_retry_count = 0
            else:
                # Still no valid outputs - use the original successful outputs as fallback
                valid_outputs = successful_outputs
                logger.warning(
                    f"[{self.council_name}] Using refused outputs as fallback (no alternatives)"
                )
            
            # Log consensus mechanism panel
            try:
                from ..cli import verbose_logger
                VERBOSE_LOGGER_AVAILABLE = True
            except ImportError:
                VERBOSE_LOGGER_AVAILABLE = False
                verbose_logger = None
            
            # Determine consensus algorithm name
            consensus_algorithm = getattr(self.consensus, '__class__', type(self.consensus)).__name__
            if 'Vote' in consensus_algorithm:
                algorithm_name = "voting"
            elif 'Blend' in consensus_algorithm:
                algorithm_name = "weighted_blend"
            elif 'Critique' in consensus_algorithm:
                algorithm_name = "critique_exchange"
            elif 'Distance' in consensus_algorithm:
                algorithm_name = "semantic_distance"
            else:
                algorithm_name = consensus_algorithm.lower().replace('consensus', '').strip()
            
            models_participating = list(valid_outputs.keys())
            
            # Phase 1.1: Skip consensus if early exit triggered with single output
            if self._early_exit_triggered and len(valid_outputs) == 1:
                logger.info(
                    f"[{self.council_name}] Skipping consensus (early exit with single output)"
                )
                only_model = list(valid_outputs.keys())[0]
                only_output = valid_outputs[only_model]
                consensus_result = only_output
                consensus_output = only_output.output
                
                # Log that consensus was skipped
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_consensus_mechanism_panel(
                        algorithm="skipped",
                        models_participating=models_participating,
                        agreement_scores=None,
                        selected_output=consensus_output[:300] if consensus_output else None,
                        skip_reason="early_exit_single_output",
                        consensus_details={"early_exit": True, "model": only_model}
                    )
                
                # Postprocess
                processed_output = self.postprocess(consensus_output)
                
                # Phase 7.1: Check for empty structured outputs
                processed_output, escalation_applied = self._check_and_escalate_empty_output(
                    processed_output, prompt, system_prompt, None
                )
                
                # Track resource usage
                self._track_resource_usage(processed_output)
                
                return CouncilResult(
                    output=processed_output,
                    raw_outputs=raw_outputs,
                    consensus_details={"early_exit": True, "model": only_model},
                    council_name=self.council_name,
                    success=True,
                    metadata={"early_exit": True, "escalation_applied": escalation_applied}
                )
            
            # Apply consensus with error fallback (use valid_outputs, not raw_outputs)
            # Phase 5.1: Track consensus timing
            import time as time_module
            consensus_start = time_module.time()
            consensus_duration = 0.0
            
            try:
                consensus_result = self.consensus.apply(valid_outputs)
                consensus_duration = time_module.time() - consensus_start
                
                logger.debug(
                    f"[{self.council_name}] Consensus completed in {consensus_duration:.2f}s"
                )
                
                # Log consensus mechanism panel after consensus
                # Extract agreement scores if available
                agreement_scores = {}
                consensus_details_dict = {}
                selected_output = None
                
                if isinstance(consensus_result, dict):
                    agreement_scores = consensus_result.get('agreement_scores', {})
                    consensus_details_dict = consensus_result
                    selected_output = consensus_result.get('selected_output', consensus_result.get('output', None))
                elif hasattr(consensus_result, 'agreement_scores'):
                    agreement_scores = getattr(consensus_result, 'agreement_scores', {})
                    selected_output = getattr(consensus_result, 'selected_output', str(consensus_result)[:200])
                
                # Extract voting details if available
                if isinstance(consensus_result, dict) and 'voting' in consensus_result:
                    consensus_details_dict['voting'] = consensus_result['voting']
                
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_consensus_mechanism_panel(
                        algorithm=algorithm_name,
                        models_participating=models_participating,
                        agreement_scores=agreement_scores if agreement_scores else None,
                        selected_output=str(selected_output)[:300] if selected_output else None,
                        skip_reason=None,
                        consensus_details=consensus_details_dict if consensus_details_dict else None
                    )
                
                # Publish SSE event for frontend
                if SSE_AVAILABLE and sse_manager and self._current_mission_id:
                    # Calculate overall agreement score
                    overall_agreement = 0.0
                    if agreement_scores:
                        overall_agreement = sum(agreement_scores.values()) / len(agreement_scores)
                    
                    # Build model outputs summary for SSE
                    model_outputs_summary = {}
                    for model, output in valid_outputs.items():
                        model_outputs_summary[model] = {
                            "success": output.success,
                            "output": output.output[:300] if output.output else None,
                            "duration_s": getattr(output, 'duration_seconds', None),
                            "tokens": getattr(output, 'token_usage', None)
                        }
                    
                    _publish_sse_event(sse_manager.publish_consensus_result(
                        mission_id=self._current_mission_id,
                        council_name=self.council_name,
                        mechanism=algorithm_name,
                        agreement_score=overall_agreement,
                        model_outputs=model_outputs_summary,
                        final_decision=str(selected_output)[:200] if selected_output else None
                    ))
            except AttributeError as e:
                # Handle missing synthesize or other attribute errors
                if "synthesize" in str(e) or "apply" in str(e):
                    logger.warning(
                        f"[WARN] Consensus failed for {self.council_name} → "
                        f"Using best single-model output. Error: {e}"
                    )
                    # Fallback to first valid output
                    best_output = next(iter(valid_outputs.values()))
                    consensus_result = best_output
                else:
                    raise
            except Exception as e:
                logger.warning(
                    f"[WARN] Consensus error in {self.council_name}: {e} → "
                    f"Using best single-model output."
                )
                # Fallback to first valid output
                best_output = next(iter(valid_outputs.values()))
                consensus_result = best_output
            
            # Extract the consensus output
            consensus_output = self._extract_consensus_output(consensus_result)
            
            # Postprocess
            processed_output = self.postprocess(consensus_output)
            
            # Phase 7.1: Check for empty structured outputs and escalate if needed
            processed_output, escalation_applied = self._check_and_escalate_empty_output(
                processed_output, prompt, system_prompt, None  # phase not available in execute()
            )
            
            # Track resource usage via CognitiveSpine
            self._track_resource_usage(processed_output)
            
            # Phase 5.1: Build execution metadata with timing metrics
            execution_metadata = {
                "consensus_duration_s": consensus_duration,
                "models_executed": len(raw_outputs),
                "models_successful": len(valid_outputs),
            }
            if escalation_applied:
                execution_metadata["escalation_applied"] = True
            
            # Aggregate model execution times
            model_latencies = [
                out.metadata.get("latency_s", 0)
                for out in raw_outputs.values()
                if out.metadata
            ]
            if model_latencies:
                execution_metadata["total_model_time_s"] = sum(model_latencies)
                execution_metadata["avg_model_time_s"] = sum(model_latencies) / len(model_latencies)
            
            return CouncilResult(
                output=processed_output,
                raw_outputs=raw_outputs,
                consensus_details=consensus_result,
                council_name=self.council_name,
                success=True,
                metadata=execution_metadata
            )
            
        except Exception as e:
            return CouncilResult(
                output=None,
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error=str(e)
            )
    
    def _quality_sufficient(self, result: CouncilResult, threshold: float = 0.85) -> bool:
        """
        Check if result quality is sufficient to stop early.
        
        Args:
            result: CouncilResult to check
            threshold: Quality threshold (0-1)
            
        Returns:
            True if quality is sufficient to stop iteration
        """
        if result.output and hasattr(result.output, 'quality_score'):
            return result.output.quality_score >= threshold
        return False
    
    def execute_supervised(
        self,
        context: Any,
        mission_state: Optional["MissionState"] = None,
        phase: Optional["MissionPhase"] = None,
        wait_for_capacity: bool = True,
        timeout: float = 300.0
    ) -> CouncilResult:
        """
        Execute with supervisor-driven model selection.
        
        This method:
        1. Gets supervisor decision based on mission state and GPU resources
        2. Updates model pool with selected models
        3. Waits for GPU capacity if needed
        4. Executes the council with optimized configuration
        
        Args:
            context: Context object to pass to build_prompt
            mission_state: Current mission state
            phase: Current mission phase
            wait_for_capacity: Whether to wait for GPU capacity
            timeout: Maximum wait time for GPU capacity
            
        Returns:
            CouncilResult with supervisor decision attached
        """
        decision = None
        
        try:
            # Get supervisor decision if available
            if self.supervisor is not None and self.gpu_manager is not None:
                gpu_stats = self.gpu_manager.get_stats()
                decision = self.supervisor.decide(
                    mission_state=mission_state,
                    phase=phase,
                    gpu_stats=gpu_stats,
                    council_config={
                        "models": self.model_pool.get_all_models(),
                        "council_name": self.council_name
                    }
                )
                
                # Apply decision to model pool
                self.model_pool.update_from_decision(decision)
            
            # Wait for GPU capacity if requested OR if models require serialization (REASONING/LARGE)
            models = self.model_pool.get_all_models()
            should_wait = wait_for_capacity
            
            # Always check capacity for REASONING/LARGE models (Phase 1.3)
            # But only wait if capacity is NOT immediately available
            if self.gpu_manager is not None and models:
                try:
                    from deepthinker.models.model_registry import ModelRegistry
                    registry = ModelRegistry()
                    if registry.requires_serialization(models):
                        # Check if capacity is already available
                        estimated_vram = self.model_pool._estimate_models_vram(models)
                        if self.gpu_manager.can_run_model(estimated_vram):
                            # Capacity available, no need to wait
                            logger.debug(
                                f"[{self.council_name}] REASONING/LARGE models detected, "
                                f"GPU capacity already available ({estimated_vram}MB), proceeding"
                            )
                            should_wait = False  # Override wait flag since capacity is available
                        else:
                            # Capacity not available, set should_wait
                            should_wait = True
                            logger.info(
                                f"[{self.council_name}] REASONING/LARGE models detected, "
                                f"GPU capacity not immediately available ({estimated_vram}MB needed), waiting..."
                            )
                except Exception as e:
                    logger.warning(f"[{self.council_name}] Error checking model tiers/capacity: {e}, proceeding without wait")
                    # If check fails, don't wait (fail open)
                    should_wait = False
            
            # Only wait if explicitly requested AND capacity is not immediately available
            if should_wait and self.gpu_manager is not None and models:
                # Double-check capacity before waiting (it might have changed)
                estimated_vram = self.model_pool._estimate_models_vram(models)
                if self.gpu_manager.can_run_model(estimated_vram):
                    logger.debug(f"[{self.council_name}] Capacity check passed, skipping wait for {models}")
                    should_wait = False
                
                if should_wait:
                    logger.info(f"[{self.council_name}] Waiting for GPU capacity for models: {models} (need {estimated_vram}MB)")
                    if not self.model_pool.wait_for_gpu_capacity(models, timeout=timeout):
                        # Timeout - try with fallback
                        logger.warning(
                            f"[{self.council_name}] Wait for GPU capacity timed out, "
                            "trying with fallback models"
                        )
                        if self.supervisor is not None:
                            decision = self.supervisor.get_fallback_decision(
                                phase_type=self._get_phase_type(phase)
                            )
                            self.model_pool.update_from_decision(decision)
            
            # Build prompt
            prompt = self.build_prompt(context)
            system_prompt = self.get_system_prompt()
            
            # Execute
            raw_outputs = self.model_pool.run_all(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Check for successful outputs
            successful_outputs = {
                name: output for name, output in raw_outputs.items()
                if output.success and output.output
            }
            
            if not successful_outputs:
                return CouncilResult(
                    output=None,
                    raw_outputs=raw_outputs,
                    consensus_details=None,
                    council_name=self.council_name,
                    success=False,
                    error="All council members failed to produce output",
                    supervisor_decision=decision
                )
            
            # Apply consensus
            consensus_result = self.consensus.apply(raw_outputs)
            consensus_output = self._extract_consensus_output(consensus_result)
            
            # Postprocess
            processed_output = self.postprocess(consensus_output)
            
            return CouncilResult(
                output=processed_output,
                raw_outputs=raw_outputs,
                consensus_details=consensus_result,
                council_name=self.council_name,
                success=True,
                supervisor_decision=decision
            )
            
        except Exception as e:
            return CouncilResult(
                output=None,
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error=str(e),
                supervisor_decision=decision
            )
    
    def _extract_consensus_output(self, consensus_result: Any) -> Any:
        """Extract the main output from consensus result."""
        if hasattr(consensus_result, 'winner'):
            return consensus_result.winner
        elif hasattr(consensus_result, 'blended_output'):
            return consensus_result.blended_output
        elif hasattr(consensus_result, 'final_output'):
            return consensus_result.final_output
        elif hasattr(consensus_result, 'selected_output'):
            return consensus_result.selected_output
        else:
            return str(consensus_result)
    
    def _get_phase_type(self, phase: Optional["MissionPhase"]) -> str:
        """Determine phase type for fallback decisions."""
        if phase is None:
            return "research"
        
        name_lower = phase.name.lower()
        
        if any(kw in name_lower for kw in ["research", "recon", "gather"]):
            return "research"
        elif any(kw in name_lower for kw in ["design", "plan", "architect"]):
            return "design"
        elif any(kw in name_lower for kw in ["implement", "code", "build"]):
            return "implementation"
        elif any(kw in name_lower for kw in ["test", "simulation", "validate"]):
            return "testing"
        elif any(kw in name_lower for kw in ["synthesis", "report", "final"]):
            return "synthesis"
        else:
            return "research"
    
    def execute_with_context(
        self,
        context: Dict[str, Any],
        *args,
        **kwargs
    ) -> CouncilResult:
        """
        Execute with additional context passed to consensus.
        
        Args:
            context: Context dictionary for consensus algorithm
            *args: Arguments to pass to build_prompt
            **kwargs: Keyword arguments to pass to build_prompt
            
        Returns:
            CouncilResult with consensus output and details
        """
        # Store context for consensus algorithms that support it
        task_context = context.get("task_context")
        
        try:
            prompt = self.build_prompt(*args, **kwargs)
            system_prompt = self.get_system_prompt()
            
            raw_outputs = self.model_pool.run_all(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Apply consensus with context if supported
            if hasattr(self.consensus, 'apply') and 'task_context' in self.consensus.apply.__code__.co_varnames:
                consensus_result = self.consensus.apply(
                    raw_outputs,
                    task_context=task_context
                )
            else:
                consensus_result = self.consensus.apply(raw_outputs)
            
            # Extract consensus output
            consensus_output = self._extract_consensus_output(consensus_result)
            processed_output = self.postprocess(consensus_output)
            
            return CouncilResult(
                output=processed_output,
                raw_outputs=raw_outputs,
                consensus_details=consensus_result,
                council_name=self.council_name,
                success=True
            )
            
        except Exception as e:
            return CouncilResult(
                output=None,
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error=str(e)
            )
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status for this council.
        
        Returns:
            Dictionary with resource information
        """
        status = {
            "council_name": self.council_name,
            "models": self.model_pool.get_all_models(),
            "has_supervisor": self.supervisor is not None,
            "has_gpu_manager": self.gpu_manager is not None
        }
        
        if self.gpu_manager is not None:
            stats = self.gpu_manager.get_stats()
            status["gpu_stats"] = stats.to_dict()
            status["resource_pressure"] = self.gpu_manager.get_resource_pressure()
            status["can_run"] = self.gpu_manager.can_run_models(
                self.model_pool.get_all_models()
            )
        
        return status
