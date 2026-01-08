"""
Model Pool for DeepThinker 2.0 Council System.

Manages concurrent execution across multiple LLMs for council-based decision making.
Includes GPU resource management and request queuing.
"""

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field

try:
    from langchain_ollama import ChatOllama
    USE_CHAT_OLLAMA = True
except ImportError:
    from langchain_community.llms import Ollama
    USE_CHAT_OLLAMA = False

if TYPE_CHECKING:
    from ..resources.gpu_manager import GPUResourceManager
    from ..supervisor.model_supervisor import ModelSupervisor, SupervisorDecision
    from ..missions.mission_types import MissionState, MissionPhase, EffortLevel


@dataclass
class ModelOutput:
    """
    Output from a single model execution.
    
    Attributes:
        model_name: Name of the model that produced this output
        output: The generated text output
        temperature: Temperature used for generation
        success: Whether the generation succeeded
        error: Error message if generation failed
        metadata: Additional metadata (tokens, latency, etc.)
    """
    
    model_name: str
    output: str
    temperature: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueuedRequest:
    """
    A queued model execution request.
    
    Attributes:
        request_id: Unique identifier for this request
        models: List of model names to execute
        prompt: User prompt text
        system_prompt: Optional system prompt
        temperature: Temperature override (if any)
        priority: Request priority (higher = more urgent)
        created_at: Timestamp when request was created
        estimated_vram: Estimated VRAM requirement in MB
    """
    request_id: str
    models: List[str]
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    estimated_vram: int = 0


class ModelPool:
    """
    Manages a pool of LLM models for council-based execution.
    
    Supports running prompts across multiple models concurrently
    and collecting their outputs for consensus algorithms.
    
    New in 2.0:
    - GPU resource management integration
    - Request queuing for capacity management
    - Supervisor-based model selection
    """
    
    def __init__(
        self,
        pool_config: List[Tuple[str, float]],
        base_url: str = "http://localhost:11434",
        max_workers: int = 3,
        gpu_manager: Optional["GPUResourceManager"] = None,
        supervisor: Optional["ModelSupervisor"] = None,
        enable_queue: bool = True
    ):
        """
        Initialize the model pool.
        
        Args:
            pool_config: List of (model_name, temperature) tuples
            base_url: URL of the Ollama server
            max_workers: Maximum concurrent model executions
            gpu_manager: Optional GPU resource manager for capacity checking
            supervisor: Optional model supervisor for dynamic selection
            enable_queue: Whether to enable request queuing
        """
        self.pool_config = pool_config
        self.base_url = base_url
        self.max_workers = max_workers
        self._model_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # GPU and supervisor integration
        self.gpu_manager = gpu_manager
        self.supervisor = supervisor
        
        # Request queue
        self.enable_queue = enable_queue
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queue_lock = threading.Lock()
        self._active_requests: Dict[str, QueuedRequest] = {}
        self._request_counter = 0
        
        # Model registry for tier-aware parallelism (lazy-loaded)
        self._registry: Optional[Any] = None
    
    def get_all_models(self) -> List[str]:
        """
        Return list of all model names in the pool.
        
        Returns:
            List of model name strings
        """
        return [model for model, _ in self.pool_config]
    
    def _get_registry(self):
        """Lazy-load model registry."""
        if self._registry is None:
            from .model_registry import ModelRegistry
            self._registry = ModelRegistry()
        return self._registry
    
    def _compute_tier_aware_max_workers(self, models: Optional[List[str]] = None) -> int:
        """
        Compute tier-aware max_workers for ThreadPoolExecutor.
        
        Phase 2.1: Enforces serialization for REASONING/LARGE models,
        allows limited parallelism for MEDIUM/SMALL models.
        
        Args:
            models: List of model names (defaults to pool_config models)
            
        Returns:
            Maximum workers (1 for REASONING/LARGE, 2-4 for others based on VRAM)
        """
        if models is None:
            models = self.get_all_models()
        
        if not models:
            return 1
        
        registry = self._get_registry()
        
        # Phase 2.2: Hard serialization rule - REASONING/LARGE must serialize
        if registry.requires_serialization(models):
            return 1
        
        # Determine if all models are same tier
        from .model_registry import ModelTier
        tiers = set()
        for name in models:
            info = registry._models.get(name)
            if info:
                tiers.add(info.tier)
            else:
                # Unknown model - assume MEDIUM tier
                tiers.add(ModelTier.MEDIUM)
        
        # Get available VRAM for VRAM-based caps
        available_vram = 8000  # Default fallback
        if self.gpu_manager is not None:
            available_vram = self.gpu_manager.get_available_vram()
        
        # All MEDIUM tier
        if len(tiers) == 1 and list(tiers)[0] == ModelTier.MEDIUM:
            # Cap at 2 max, or VRAM-based (each MEDIUM model needs ~12GB)
            return min(2, max(1, available_vram // 12000))
        
        # All SMALL tier
        if len(tiers) == 1 and list(tiers)[0] == ModelTier.SMALL:
            # Cap at 4 max, or VRAM-based (each SMALL model needs ~8GB)
            return min(4, max(1, available_vram // 8000))
        
        # Mixed tiers: use most restrictive rule (serialize)
        return 1
    
    def get_model_temperatures(self) -> Dict[str, float]:
        """
        Return mapping of model names to their temperatures.
        
        Returns:
            Dictionary mapping model_name -> temperature
        """
        return {model: temp for model, temp in self.pool_config}
    
    def get_models_for_effort(self, effort: "EffortLevel") -> List[str]:
        """
        Get appropriate models based on effort level.
        
        Uses model name heuristics to categorize by size, then returns
        an appropriate subset based on the effort level.
        
        Args:
            effort: EffortLevel enum value
            
        Returns:
            List of model names appropriate for this effort level
        """
        # Import here to avoid circular imports
        from deepthinker.missions.mission_types import EffortLevel
        
        all_models = self.get_all_models()
        
        if not all_models:
            return []
        
        # Model size heuristics based on common naming conventions
        small_keywords = ["3b", "7b", "8b", "small", "mini", "tiny", "nano"]
        medium_keywords = ["12b", "13b", "14b", "medium"]
        large_keywords = ["27b", "32b", "70b", "large", "huge", "xxl"]
        
        small_models = [
            m for m in all_models 
            if any(s in m.lower() for s in small_keywords)
        ]
        medium_models = [
            m for m in all_models 
            if any(s in m.lower() for s in medium_keywords)
        ]
        large_models = [
            m for m in all_models 
            if any(s in m.lower() for s in large_keywords)
        ]
        
        # Fallback to all models if categorization fails
        if not small_models:
            small_models = all_models[:2] if len(all_models) >= 2 else all_models
        if not medium_models:
            medium_models = all_models
        if not large_models:
            large_models = all_models
        
        if effort == EffortLevel.QUICK:
            return small_models[:2]
        elif effort == EffortLevel.STANDARD:
            return medium_models[:3]
        elif effort == EffortLevel.DEEP:
            return large_models[:3]
        else:  # MARATHON
            return all_models  # Use all available models
    
    def update_pool_config(
        self,
        new_config: List[Tuple[str, float]]
    ) -> None:
        """
        Update the pool configuration dynamically.
        
        Args:
            new_config: New list of (model_name, temperature) tuples
        """
        with self._lock:
            self.pool_config = new_config
    
    def update_from_decision(
        self,
        decision: "SupervisorDecision"
    ) -> None:
        """
        Update pool configuration from a supervisor decision.
        
        Args:
            decision: SupervisorDecision with model list and temperature
        """
        new_config = [
            (model, decision.temperature)
            for model in decision.models
        ]
        self.update_pool_config(new_config)
        self.max_workers = min(self.max_workers, decision.parallelism)
    
    def _create_llm(self, model_name: str, temperature: float) -> Any:
        """
        Create or retrieve cached LLM instance.
        
        Args:
            model_name: Name of the Ollama model
            temperature: Sampling temperature
            
        Returns:
            Configured LLM instance
        """
        cache_key = f"{model_name}_{temperature}"
        
        with self._lock:
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]
            
            if USE_CHAT_OLLAMA:
                # ChatOllama expects just the model name (e.g., "gemma3:27b")
                llm = ChatOllama(
                    model=model_name,
                    base_url=self.base_url,
                    temperature=temperature,
                )
            else:
                # Legacy Ollama class may need prefix
                prefixed_model = f"ollama/{model_name}" if not model_name.startswith("ollama") else model_name
                llm = Ollama(
                    model=prefixed_model,
                    base_url=self.base_url,
                    temperature=temperature,
                )
            
            self._model_cache[cache_key] = llm
            return llm
    
    def _run_single_model(
        self,
        model_name: str,
        temperature: float,
        prompt: str,
        system_prompt: Optional[str] = None,
        prefix_embeds: Optional[Any] = None
    ) -> ModelOutput:
        """
        Execute prompt on a single model.
        
        Phase 5.1: Enhanced with structured timing metrics.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            prompt: User prompt text
            system_prompt: Optional system prompt
            prefix_embeds: Optional prefix embeddings tensor (for future HuggingFace integration)
                          Currently not used with Ollama/LangChain, but logged if provided
            
        Returns:
            ModelOutput with results and detailed timing metadata
        """
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        
        # Log prefix embeddings if provided (placeholder for future HuggingFace integration)
        if prefix_embeds is not None:
            logger.debug(
                f"[LATENT_MEMORY] Prefix embeddings provided (shape: {prefix_embeds.shape}), "
                "but current Ollama/LangChain backend doesn't support inputs_embeds. "
                "This will be used when HuggingFace models are integrated directly."
            )
        
        try:
            # Track model creation/cache lookup time
            create_start = time.time()
            llm = self._create_llm(model_name, temperature)
            model_init_time = time.time() - create_start
            
            # Track inference time
            inference_start = time.time()
            
            if system_prompt and USE_CHAT_OLLAMA:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = llm.invoke(full_prompt)
                output = response.content if hasattr(response, 'content') else str(response)
            
            inference_time = time.time() - inference_start
            total_latency = time.time() - start_time
            
            # Phase 5.2: Extract token counts and compute tokens/sec
            # Estimate tokens from output (rough: ~4 chars per token)
            output_chars = len(output) if output else 0
            estimated_tokens = output_chars // 4
            tokens_per_second = estimated_tokens / inference_time if inference_time > 0 else 0.0
            
            # Try to get actual token counts from response metadata
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            
            if hasattr(response, 'response_metadata'):
                meta = response.response_metadata
                prompt_tokens = meta.get('prompt_eval_count') or meta.get('prompt_tokens')
                completion_tokens = meta.get('eval_count') or meta.get('completion_tokens')
                if completion_tokens and inference_time > 0:
                    tokens_per_second = completion_tokens / inference_time
                if prompt_tokens and completion_tokens:
                    total_tokens = prompt_tokens + completion_tokens
            
            # Build detailed metadata
            metadata = {
                "latency_s": total_latency,
                "inference_time_s": inference_time,
                "model_init_time_s": model_init_time,
                "tokens_per_second": round(tokens_per_second, 2),
                "estimated_output_tokens": estimated_tokens,
            }
            
            # Add actual token counts if available
            if prompt_tokens is not None:
                metadata["prompt_tokens"] = prompt_tokens
            if completion_tokens is not None:
                metadata["completion_tokens"] = completion_tokens
            if total_tokens is not None:
                metadata["total_tokens"] = total_tokens
            
            logger.debug(
                f"[MODEL] {model_name}: {total_latency:.2f}s total, "
                f"{inference_time:.2f}s inference, {tokens_per_second:.1f} tok/s"
            )
            
            return ModelOutput(
                model_name=model_name,
                output=output,
                temperature=temperature,
                success=True,
                metadata=metadata
            )
            
        except Exception as e:
            total_latency = time.time() - start_time
            return ModelOutput(
                model_name=model_name,
                output="",
                temperature=temperature,
                success=False,
                error=str(e),
                metadata={"latency_s": total_latency, "error_type": type(e).__name__}
            )
    
    def run_single(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Execute prompt on a single model and return just the output string.
        
        This is a convenience method for step execution where only a single
        model is needed, not a full council.
        
        Args:
            model_name: Name of the model to use
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Optional temperature override (uses pool default if not specified)
            
        Returns:
            Output string from the model, or empty string on failure
        """
        # Get temperature from pool config or use provided/default
        temp_map = self.get_model_temperatures()
        temp = temperature if temperature is not None else temp_map.get(model_name, 0.5)
        
        result = self._run_single_model(model_name, temp, prompt, system_prompt)
        
        if result.success:
            return result.output
        else:
            return ""
    
    def run_all(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, ModelOutput]:
        """
        Execute prompt on all models in the pool concurrently.
        
        For REASONING/LARGE tier models, waits for GPU capacity before execution
        to prevent GPU thrashing and ensure stable execution.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            timeout: Optional timeout in seconds for each model (None = no timeout)
            
        Returns:
            Dictionary mapping model_name -> ModelOutput
        """
        import logging
        logger = logging.getLogger(__name__)
        
        models = self.get_all_models()
        
        # Check if we need to wait for GPU capacity (Phase 1.3)
        if self.gpu_manager is not None and models:
            # Check if models require serialization (REASONING/LARGE tier)
            try:
                from .model_registry import ModelRegistry
                registry = ModelRegistry()
                requires_serial = registry.requires_serialization(models)
                
                # If models require serialization or cannot run, wait for capacity
                if not self.gpu_manager.can_run_models(models):
                    if requires_serial:
                        # REASONING/LARGE models: wait for capacity
                        estimated_vram = sum(
                            self.gpu_manager.get_loading_cost(m).get("vram_mb", 8000)
                            for m in models
                        )
                        logger.info(
                            f"[POOL] REASONING/LARGE models detected, checking GPU capacity "
                            f"(estimated VRAM: {estimated_vram}MB)"
                        )
                        # wait_for_capacity now checks capacity first, so won't wait unnecessarily
                        if not self.gpu_manager.wait_for_capacity(estimated_vram, timeout=300.0):
                            logger.warning(
                                f"[POOL] Wait for capacity timed out after 300s, proceeding anyway"
                            )
                    else:
                        # Non-serialization models: log warning but proceed
                        logger.debug(f"[POOL] Cannot run models {models} immediately, but proceeding")
            except Exception as e:
                # If registry check fails, still check capacity but don't enforce wait
                logger.debug(f"[POOL] Could not check model tiers: {e}")
                if not self.gpu_manager.can_run_models(models):
                    logger.warning(f"[POOL] Cannot run models {models}, but proceeding")
        
        results: Dict[str, ModelOutput] = {}
        
        models = self.get_all_models()
        registry = self._get_registry()
        
        # Phase 2.2: Hard serialization rule - if REASONING/LARGE, execute sequentially
        if registry.requires_serialization(models):
            logger.info(
                f"[POOL] REASONING/LARGE models detected - executing sequentially "
                f"(models: {models})"
            )
            # Sequential execution for large models
            for model_name, temperature in self.pool_config:
                try:
                    result = self._run_single_model(
                        model_name, temperature, prompt, system_prompt
                    )
                    results[model_name] = result
                except Exception as e:
                    results[model_name] = ModelOutput(
                        model_name=model_name,
                        output="",
                        temperature=self.get_model_temperatures().get(model_name, 0.5),
                        success=False,
                        error=str(e)
                    )
        else:
            # Parallel execution for MEDIUM/SMALL models with tier-aware max_workers
            tier_aware_workers = self._compute_tier_aware_max_workers(models)
            logger.debug(
                f"[POOL] Using tier-aware parallelism: {tier_aware_workers} workers "
                f"(models: {models})"
            )
            
            with ThreadPoolExecutor(max_workers=tier_aware_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_single_model,
                        model_name,
                        temperature,
                        prompt,
                        system_prompt
                    ): model_name
                    for model_name, temperature in self.pool_config
                }
                
                for future in as_completed(futures, timeout=timeout):
                    model_name = futures[future]
                    try:
                        result = future.result(timeout=timeout)
                        results[model_name] = result
                    except TimeoutError:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=self.get_model_temperatures().get(model_name, 0.5),
                            success=False,
                            error=f"Timeout after {timeout}s",
                            metadata={"timeout": True, "timeout_seconds": timeout}
                        )
                    except Exception as e:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=self.get_model_temperatures().get(model_name, 0.5),
                            success=False,
                            error=str(e)
                        )
        
        return results
    
    def run_all_with_timeout(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
        on_timeout: Optional[str] = "partial"
    ) -> Tuple[Dict[str, ModelOutput], bool]:
        """
        Execute prompt on all models with strict timeout enforcement.
        
        This method ensures that execution completes within the timeout,
        returning partial results if some models are still running.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            timeout: Total timeout in seconds for all models
            on_timeout: Behavior on timeout - "partial" returns completed results,
                       "fail" treats entire call as failed
            
        Returns:
            Tuple of (results dict, timed_out bool)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        results: Dict[str, ModelOutput] = {}
        timed_out = False
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_model,
                    model_name,
                    temperature,
                    prompt,
                    system_prompt
                ): model_name
                for model_name, temperature in self.pool_config
            }
            
            try:
                # Wait for all futures with timeout
                for future in as_completed(futures, timeout=timeout):
                    elapsed = time.time() - start_time
                    remaining = max(0, timeout - elapsed)
                    
                    model_name = futures[future]
                    try:
                        result = future.result(timeout=remaining)
                        results[model_name] = result
                    except TimeoutError:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=self.get_model_temperatures().get(model_name, 0.5),
                            success=False,
                            error=f"Model timeout after {elapsed:.1f}s",
                            metadata={"timeout": True}
                        )
                    except Exception as e:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=self.get_model_temperatures().get(model_name, 0.5),
                            success=False,
                            error=str(e)
                        )
                        
            except TimeoutError:
                # Overall timeout reached
                timed_out = True
                elapsed = time.time() - start_time
                logger.warning(
                    f"[POOL] Overall timeout ({timeout}s) reached after {elapsed:.1f}s. "
                    f"Completed: {len(results)}/{len(futures)} models"
                )
                
                # Mark remaining futures as timed out
                for future, model_name in futures.items():
                    if model_name not in results:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=self.get_model_temperatures().get(model_name, 0.5),
                            success=False,
                            error=f"Overall timeout after {elapsed:.1f}s",
                            metadata={"timeout": True, "overall_timeout": True}
                        )
                        # Cancel the future
                        future.cancel()
        
        return results, timed_out
    
    def run_with_temperature(
        self,
        model: str,
        prompt: str,
        temp: float,
        system_prompt: Optional[str] = None
    ) -> ModelOutput:
        """
        Run a specific model with a custom temperature.
        
        Args:
            model: Model name to use
            prompt: User prompt text
            temp: Custom temperature override
            system_prompt: Optional system prompt
            
        Returns:
            ModelOutput with results
        """
        return self._run_single_model(model, temp, prompt, system_prompt)
    
    def run_queued(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, ModelOutput]:
        """
        Execute prompt using queue infrastructure with backpressure.
        
        Phase 4.1: Checks capacity first, enqueues if unavailable, processes via queue.
        Uses priority: REASONING/LARGE=high, MEDIUM=normal, SMALL=low.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            timeout: Optional timeout in seconds (for queue wait, not execution)
            
        Returns:
            Dictionary mapping model_name -> ModelOutput
        """
        import logging
        logger = logging.getLogger(__name__)
        
        models = self.get_all_models()
        if not models:
            return {}
        
        # Calculate priority based on model tiers
        registry = self._get_registry()
        priority = 0  # Default normal priority
        if registry.requires_serialization(models):
            priority = 10  # High priority for REASONING/LARGE
        else:
            # Check if all MEDIUM
            tiers = set()
            for name in models:
                info = registry._models.get(name)
                if info:
                    tiers.add(info.tier)
            if len(tiers) == 1:
                from .model_registry import ModelTier
                tier = list(tiers)[0]
                if tier == ModelTier.MEDIUM:
                    priority = 5  # Normal priority
                elif tier == ModelTier.SMALL:
                    priority = 1  # Low priority
        
        # Estimate VRAM
        estimated_vram = 0
        if self.gpu_manager is not None:
            estimated_vram = sum(
                self.gpu_manager.get_loading_cost(m).get("vram_mb", 8000)
                for m in models
            )
        
        # Check if we can run immediately
        if self.gpu_manager is None or self.gpu_manager.can_run_models(models):
            # Execute immediately via run_all()
            logger.debug(f"[QUEUE] Capacity available, executing {models} immediately")
            return self.run_all(prompt=prompt, system_prompt=system_prompt, timeout=timeout)
        
        # Capacity unavailable - enqueue request
        logger.info(
            f"[QUEUE] Enqueuing request, models={models}, priority={priority}, "
            f"estimated_vram={estimated_vram}MB, queue_size={self.get_queue_size()}"
        )
        
        request_id = self.enqueue_request(
            models=models,
            prompt=prompt,
            system_prompt=system_prompt,
            priority=priority,
            estimated_vram=estimated_vram
        )
        
        # Process queue until our request is done
        wait_start = time.time()
        max_wait = timeout if timeout else 300.0
        
        while time.time() - wait_start < max_wait:
            result = self.process_next_request()
            if result and result[0] == request_id:
                return result[1]
            time.sleep(0.1)  # Small delay before checking again
        
        # Timeout - remove from queue and return empty results
        logger.warning(f"[QUEUE] Timeout waiting for request {request_id}")
        with self._queue_lock:
            self._active_requests.pop(request_id, None)
        
        return {}
    
    def run_queued(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, ModelOutput]:
        """
        Execute prompt using queue infrastructure with backpressure.
        
        Phase 4.1: Checks capacity first, enqueues if unavailable, processes via queue.
        Uses priority: REASONING/LARGE=high, MEDIUM=normal, SMALL=low.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            timeout: Optional timeout in seconds (for queue wait, not execution)
            
        Returns:
            Dictionary mapping model_name -> ModelOutput
        """
        import logging
        logger = logging.getLogger(__name__)
        
        models = self.get_all_models()
        if not models:
            return {}
        
        # Calculate priority based on model tiers
        registry = self._get_registry()
        priority = 0  # Default normal priority
        if registry.requires_serialization(models):
            priority = 10  # High priority for REASONING/LARGE
        else:
            # Check if all MEDIUM
            tiers = set()
            for name in models:
                info = registry._models.get(name)
                if info:
                    tiers.add(info.tier)
            if len(tiers) == 1:
                from .model_registry import ModelTier
                tier = list(tiers)[0]
                if tier == ModelTier.MEDIUM:
                    priority = 5  # Normal priority
                elif tier == ModelTier.SMALL:
                    priority = 1  # Low priority
        
        # Estimate VRAM
        estimated_vram = 0
        if self.gpu_manager is not None:
            estimated_vram = sum(
                self.gpu_manager.get_loading_cost(m).get("vram_mb", 8000)
                for m in models
            )
        
        # Check if we can run immediately
        if self.gpu_manager is None or self.gpu_manager.can_run_models(models):
            # Execute immediately via run_all()
            logger.debug(f"[QUEUE] Capacity available, executing {models} immediately")
            return self.run_all(prompt=prompt, system_prompt=system_prompt, timeout=timeout)
        
        # Capacity unavailable - enqueue request
        logger.info(
            f"[QUEUE] Enqueuing request, models={models}, priority={priority}, "
            f"estimated_vram={estimated_vram}MB, queue_size={self.get_queue_size()}"
        )
        
        request_id = self.enqueue_request(
            models=models,
            prompt=prompt,
            system_prompt=system_prompt,
            priority=priority,
            estimated_vram=estimated_vram
        )
        
        # Process queue until our request is done
        wait_start = time.time()
        max_wait = timeout if timeout else 300.0
        
        while time.time() - wait_start < max_wait:
            result = self.process_next_request()
            if result and result[0] == request_id:
                return result[1]
            time.sleep(0.1)  # Small delay before checking again
        
        # Timeout - remove from queue and return empty results
        logger.warning(f"[QUEUE] Timeout waiting for request {request_id}")
        with self._queue_lock:
            self._active_requests.pop(request_id, None)
        
        return {}
    
    def run_subset(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, ModelOutput]:
        """
        Execute prompt on a subset of models from the pool.
        
        Args:
            models: List of model names to use
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Optional temperature override for all models
            timeout: Optional timeout in seconds (None = no timeout)
            
        Returns:
            Dictionary mapping model_name -> ModelOutput
        """
        temp_map = self.get_model_temperatures()
        results: Dict[str, ModelOutput] = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_model,
                    model_name,
                    temperature if temperature is not None else temp_map.get(model_name, 0.5),
                    prompt,
                    system_prompt
                ): model_name
                for model_name in models
            }
            
            try:
                for future in as_completed(futures, timeout=timeout):
                    elapsed = time.time() - start_time
                    remaining = (timeout - elapsed) if timeout else None
                    
                    model_name = futures[future]
                    try:
                        result = future.result(timeout=remaining)
                        results[model_name] = result
                    except TimeoutError:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=temperature if temperature is not None else temp_map.get(model_name, 0.5),
                            success=False,
                            error=f"Timeout after {elapsed:.1f}s",
                            metadata={"timeout": True}
                        )
                    except Exception as e:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=temperature if temperature is not None else temp_map.get(model_name, 0.5),
                            success=False,
                            error=str(e)
                        )
            except TimeoutError:
                # Overall timeout reached - mark remaining as timed out
                import logging
                logger = logging.getLogger(__name__)
                elapsed = time.time() - start_time
                logger.warning(
                    f"[POOL] Subset timeout ({timeout}s) reached after {elapsed:.1f}s. "
                    f"Completed: {len(results)}/{len(futures)} models"
                )
                for future, model_name in futures.items():
                    if model_name not in results:
                        results[model_name] = ModelOutput(
                            model_name=model_name,
                            output="",
                            temperature=temperature if temperature is not None else temp_map.get(model_name, 0.5),
                            success=False,
                            error=f"Overall timeout after {elapsed:.1f}s",
                            metadata={"timeout": True, "overall_timeout": True}
                        )
                        future.cancel()
        
        return results
    
    # =========================================================================
    # GPU-Aware Execution Methods
    # =========================================================================
    
    def check_gpu_capacity(
        self,
        models: Optional[List[str]] = None
    ) -> bool:
        """
        Check if GPU has capacity to run the specified models.
        
        Args:
            models: List of model names to check (defaults to pool config)
            
        Returns:
            True if GPU can run the models
        """
        if self.gpu_manager is None:
            return True  # No GPU manager = assume capacity available
        
        if models is None:
            models = self.get_all_models()
        
        return self.gpu_manager.can_run_models(models)
    
    def wait_for_gpu_capacity(
        self,
        models: Optional[List[str]] = None,
        timeout: float = 300.0
    ) -> bool:
        """
        Wait until GPU has capacity for the specified models.
        
        Args:
            models: List of model names to check
            timeout: Maximum wait time in seconds
            
        Returns:
            True if capacity became available, False on timeout
        """
        if self.gpu_manager is None:
            return True
        
        if models is None:
            models = self.get_all_models()
        
        # Calculate total VRAM needed
        total_vram = sum(
            self.gpu_manager.get_loading_cost(m).get("vram_mb", 8000)
            for m in models
        )
        
        # Check if capacity is already available - return immediately if so
        if self.gpu_manager.can_run_model(total_vram):
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[POOL] GPU capacity already available for {models} (need {total_vram}MB)")
            return True
        
        # Capacity not immediately available, wait for it
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[POOL] Waiting for GPU capacity for {models} (need {total_vram}MB)")
        return self.gpu_manager.wait_for_capacity(total_vram, timeout=timeout)
    
    def run_supervised(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        mission_state: Optional["MissionState"] = None,
        phase: Optional["MissionPhase"] = None,
        wait_for_capacity: bool = True,
        timeout: float = 300.0
    ) -> Tuple[Dict[str, ModelOutput], Optional["SupervisorDecision"]]:
        """
        Execute with supervisor-driven model selection.
        
        This is the main entry point for GPU-aware, supervised execution:
        1. Get supervisor decision based on mission state and GPU resources
        2. Handle wait-for-capacity decisions with fallback support
        3. Execute with the selected models
        
        The supervisor may decide to wait for GPU capacity instead of
        downgrading immediately. In that case, we wait up to max_wait_minutes
        and fall back to fallback_models if timeout occurs.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            mission_state: Current mission state for context
            phase: Current mission phase
            wait_for_capacity: Whether to wait for GPU capacity
            timeout: Maximum wait time for GPU capacity (overridden by decision)
            
        Returns:
            Tuple of (results dict, supervisor decision)
        """
        decision = None
        used_fallback = False
        
        # Get supervisor decision if available
        if self.supervisor is not None and self.gpu_manager is not None:
            gpu_stats = self.gpu_manager.get_stats()
            decision = self.supervisor.decide(
                mission_state=mission_state,
                phase=phase,
                gpu_stats=gpu_stats,
                council_config={"models": self.get_all_models()}
            )
            
            # Handle wait-for-capacity decisions
            if decision.wait_for_capacity:
                # Supervisor says to wait for heavy models
                wait_timeout = decision.max_wait_minutes * 60  # Convert to seconds
                
                if self.gpu_manager.wait_for_capacity(
                    decision.estimated_vram,
                    timeout=wait_timeout
                ):
                    # Got capacity - use preferred models
                    self.update_from_decision(decision)
                else:
                    # Timeout - use fallback models
                    used_fallback = True
                    if decision.fallback_models:
                        # Create a modified decision with fallback models
                        fallback_decision = type(decision)(
                            models=decision.fallback_models,
                            temperature=decision.temperature,
                            parallelism=min(decision.parallelism, len(decision.fallback_models)),
                            downgraded=True,
                            reason=f"Fallback after {decision.max_wait_minutes:.1f}min wait timeout",
                            council_type=decision.council_type,
                            estimated_vram=self._estimate_models_vram(decision.fallback_models),
                            wait_for_capacity=False,
                            phase_importance=decision.phase_importance
                        )
                        self.update_from_decision(fallback_decision)
                        decision = fallback_decision
                    elif self.supervisor is not None:
                        # No fallback specified - get safe fallback
                        decision = self.supervisor.get_fallback_decision(
                            phase_type=decision.council_type
                        )
                        self.update_from_decision(decision)
            else:
                # No waiting needed - apply decision directly
                self.update_from_decision(decision)
        
        # Get models to run
        models = self.get_all_models()
        
        # Standard wait for capacity if requested (and not already handled above)
        if wait_for_capacity and self.gpu_manager is not None and not (decision and decision.wait_for_capacity):
            if not self.wait_for_gpu_capacity(models, timeout=timeout):
                # Timeout - try with fallback models
                if self.supervisor is not None:
                    decision = self.supervisor.get_fallback_decision()
                    self.update_from_decision(decision)
                    models = self.get_all_models()
        
        # Execute
        results = self.run_all(prompt, system_prompt)
        
        # Add metadata about fallback usage
        if decision and used_fallback:
            for model_output in results.values():
                model_output.metadata["used_fallback"] = True
        
        return results, decision
    
    def _estimate_models_vram(self, models: List[str]) -> int:
        """Estimate total VRAM for a list of models."""
        if self.gpu_manager is None:
            return 0
        return sum(
            self.gpu_manager.get_loading_cost(m).get("vram_mb", 8000)
            for m in models
        )
    
    # =========================================================================
    # Queue Management Methods
    # =========================================================================
    
    def enqueue_request(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        priority: int = 0,
        estimated_vram: int = 0
    ) -> str:
        """
        Enqueue a request for later processing.
        
        Alias for queue_request for consistency with Phase 4.1 naming.
        """
        return self.queue_request(
            models=models,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            priority=priority,
            estimated_vram=estimated_vram
        )
    
    def queue_request(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        priority: int = 0,
        estimated_vram: int = 0
    ) -> str:
        """
        Add a request to the queue.
        
        Args:
            models: List of model names to execute
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            priority: Request priority (higher = more urgent)
            estimated_vram: Estimated VRAM requirement
            
        Returns:
            Request ID for tracking
        """
        with self._queue_lock:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}_{int(time.time())}"
        
        request = QueuedRequest(
            request_id=request_id,
            models=models,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            priority=priority,
            estimated_vram=estimated_vram
        )
        
        # Priority queue uses (priority, request) - negate priority for max-heap behavior
        self._request_queue.put((-priority, request_id, request))
        
        with self._queue_lock:
            self._active_requests[request_id] = request
        
        return request_id
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._request_queue.qsize()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get queue status information.
        
        Returns:
            Dictionary with queue statistics
        """
        with self._queue_lock:
            return {
                "queue_size": self._request_queue.qsize(),
                "active_requests": len(self._active_requests),
                "total_estimated_vram": sum(
                    r.estimated_vram for r in self._active_requests.values()
                )
            }
    
    def process_next_request(self) -> Optional[Tuple[str, Dict[str, ModelOutput]]]:
        """
        Process the next request from the queue.
        
        Phase 4.1: Worker thread processes queue, calls wait_for_capacity() before dequeue.
        
        Returns:
            Tuple of (request_id, results) or None if queue is empty
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            _, request_id, request = self._request_queue.get_nowait()
        except queue.Empty:
            return None
        
        queue_size = self._request_queue.qsize()
        logger.info(
            f"[QUEUE] Processing request {request_id}, models={request.models}, "
            f"priority={request.priority}, queue_size={queue_size}"
        )
        
        wait_start = time.time()
        
        # Wait for GPU capacity if GPU manager is available (Phase 4.1)
        if self.gpu_manager is not None and request.estimated_vram > 0:
            if not self.gpu_manager.wait_for_capacity(request.estimated_vram, timeout=300.0):
                # Put request back in queue
                logger.warning(
                    f"[QUEUE] Wait for capacity timed out for {request_id}, re-queuing"
                )
                self._request_queue.put((-request.priority, request_id, request))
                return None
            
            wait_time = time.time() - wait_start
            vram_available = self.gpu_manager.get_available_vram()
            logger.info(
                f"[QUEUE] Waited {wait_time:.1f}s for {request.models}, "
                f"VRAM freed: {vram_available}MB"
            )
        
        # Execute request using existing run_subset() logic
        results = self.run_subset(
            models=request.models,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature
        )
        
        # Remove from active requests
        with self._queue_lock:
            self._active_requests.pop(request_id, None)
        
        return request_id, results
    
    def process_queue(
        self,
        max_requests: int = 10,
        timeout: float = 60.0
    ) -> Dict[str, Dict[str, ModelOutput]]:
        """
        Process multiple requests from the queue.
        
        Args:
            max_requests: Maximum number of requests to process
            timeout: Maximum time to spend processing
            
        Returns:
            Dictionary mapping request_id -> results
        """
        results: Dict[str, Dict[str, ModelOutput]] = {}
        start_time = time.time()
        
        for _ in range(max_requests):
            if time.time() - start_time > timeout:
                break
            
            result = self.process_next_request()
            if result is None:
                break
            
            request_id, request_results = result
            results[request_id] = request_results
        
        return results
    
    def clear_queue(self) -> int:
        """
        Clear all pending requests from the queue.
        
        Returns:
            Number of requests cleared
        """
        cleared = 0
        while True:
            try:
                self._request_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        with self._queue_lock:
            self._active_requests.clear()
        
        return cleared
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_cache(self) -> None:
        """Clear the model instance cache."""
        with self._lock:
            self._model_cache.clear()
    
    def get_successful_outputs(
        self,
        results: Dict[str, ModelOutput]
    ) -> Dict[str, str]:
        """
        Extract only successful outputs from results.
        
        Args:
            results: Dictionary of ModelOutput results
            
        Returns:
            Dictionary mapping model_name -> output text (successful only)
        """
        return {
            name: output.output
            for name, output in results.items()
            if output.success and output.output
        }
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current resource state.
        
        Returns:
            Dictionary with resource information
        """
        summary = {
            "pool_models": self.get_all_models(),
            "max_workers": self.max_workers,
            "cached_models": len(self._model_cache),
            "queue_enabled": self.enable_queue,
            "queue_status": self.get_queue_status() if self.enable_queue else None
        }
        
        if self.gpu_manager is not None:
            stats = self.gpu_manager.get_stats()
            summary["gpu"] = stats.to_dict()
            summary["gpu_pressure"] = self.gpu_manager.get_resource_pressure()
            summary["recommended_parallelism"] = self.gpu_manager.get_recommended_parallelism()
        
        return summary
