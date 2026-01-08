"""
Ollama model loader for local LLM integration with LiteLLM monitoring.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from langchain.callbacks.base import BaseCallbackHandler

try:
    # Use newer langchain-ollama package (better CrewAI/LiteLLM compatibility)
    from langchain_ollama import ChatOllama
    USE_CHAT_OLLAMA = True
except ImportError:
    # Fallback to langchain-community
    from langchain_community.llms import Ollama
    USE_CHAT_OLLAMA = False

try:
    from .litellm_monitor import LiteLLMMonitor
    LITELLM_MONITOR_AVAILABLE = True
except ImportError:
    LITELLM_MONITOR_AVAILABLE = False


@dataclass
class AgentModelConfig:
    """
    Configuration for agent-specific model selection.
    
    Defines optimal models and temperatures for each agent type,
    balancing cost-performance based on agent requirements.
    """
    
    # Default models for each agent type (can be overridden)
    planner_model: str = "cogito:14b"  # Reasoning specialist for strategic planning
    websearch_model: str = "gemma3:12b"  # Fast synthesis for research
    coder_model: str = "deepseek-r1:8b"  # Proven for code generation
    evaluator_model: str = "gemma3:27b"  # Thorough analysis for evaluation
    simulator_model: str = "mistral:instruct"  # Scenario generation
    executor_model: str = "llama3.2:3b"  # Lightweight for execution tasks
    
    # Temperature settings per agent type
    planner_temp: float = 0.7  # Balanced for strategic thinking
    websearch_temp: float = 0.5  # Lower for focused research
    coder_temp: float = 0.3  # Low for deterministic code
    evaluator_temp: float = 0.5  # Moderate for balanced evaluation
    simulator_temp: float = 0.8  # Higher for creative scenarios
    executor_temp: float = 0.4  # Low-moderate for consistent execution
    
    def override_all(self, model_name: str) -> "AgentModelConfig":
        """
        Override all agent models with a single model.
        
        Args:
            model_name: Model to use for all agents
            
        Returns:
            New AgentModelConfig with all models set to model_name
        """
        return AgentModelConfig(
            planner_model=model_name,
            websearch_model=model_name,
            coder_model=model_name,
            evaluator_model=model_name,
            simulator_model=model_name,
            executor_model=model_name,
            planner_temp=self.planner_temp,
            websearch_temp=self.websearch_temp,
            coder_temp=self.coder_temp,
            evaluator_temp=self.evaluator_temp,
            simulator_temp=self.simulator_temp,
            executor_temp=self.executor_temp
        )


class LiteLLMLangChainCallback(BaseCallbackHandler):
    """
    LangChain callback handler that integrates with LiteLLM monitoring.
    
    Tracks all LLM calls made through LangChain for observability.
    """
    
    def __init__(self):
        """Initialize the callback handler."""
        super().__init__()
        self._current_run_tokens = {}
        self._current_run_start = {}
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts."""
        import time
        run_id = kwargs.get('run_id', 'unknown')
        self._current_run_start[run_id] = time.time()
        
        if LiteLLMMonitor.is_enabled():
            logger = LiteLLMMonitor.get_logger()
            if logger and logger.verbose:
                print(f"\n{'='*60}")
                print(f"🚀 LangChain LLM Call Started")
                print(f"   Run ID: {run_id}")
                print(f"   Prompts: {len(prompts)}")
                print(f"{'='*60}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends."""
        import time
        run_id = kwargs.get('run_id', 'unknown')
        
        if run_id in self._current_run_start:
            latency = time.time() - self._current_run_start[run_id]
            del self._current_run_start[run_id]
            
            if LiteLLMMonitor.is_enabled():
                logger = LiteLLMMonitor.get_logger()
                if logger:
                    # Track basic statistics
                    logger.total_requests += 1
                    logger.total_latency += latency
                    
                    # Try to extract token info
                    if hasattr(response, 'llm_output') and response.llm_output:
                        token_usage = response.llm_output.get('token_usage', {})
                        total_tokens = token_usage.get('total_tokens', 0)
                        logger.total_tokens += total_tokens
                    
                    if logger.verbose:
                        print(f"\n✅ LangChain LLM Call Completed")
                        print(f"   Run ID: {run_id}")
                        print(f"   Latency: {latency:.2f}s")
                        print(f"{'='*60}\n")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM errors."""
        run_id = kwargs.get('run_id', 'unknown')
        
        if run_id in self._current_run_start:
            del self._current_run_start[run_id]
        
        if LiteLLMMonitor.is_enabled():
            logger = LiteLLMMonitor.get_logger()
            if logger:
                logger.errors.append({
                    "timestamp": str(kwargs.get('timestamp', 'unknown')),
                    "error": str(error),
                    "run_id": run_id
                })
                
                if logger.verbose:
                    print(f"\n❌ LangChain LLM Error")
                    print(f"   Run ID: {run_id}")
                    print(f"   Error: {error}")
                    print(f"{'='*60}\n")


class OllamaLoader:
    """
    Manages loading and configuration of Ollama models with LiteLLM monitoring.
    
    Provides a clean interface for creating LLM instances with different
    configurations for different agent types. Automatically integrates
    LiteLLM monitoring when enabled.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        enable_monitoring: bool = True,
        agent_model_config: Optional[AgentModelConfig] = None
    ):
        """
        Initialize Ollama loader.
        
        Args:
            base_url: URL of Ollama server
            enable_monitoring: Whether to enable LiteLLM monitoring callbacks
            agent_model_config: Configuration for agent-specific models
        """
        self.base_url = base_url
        self.enable_monitoring = enable_monitoring
        self.agent_config = agent_model_config or AgentModelConfig()
        self._model_cache: Dict[str, Any] = {}
        self._callback_handler: Optional[LiteLLMLangChainCallback] = None
        
        # Initialize callback handler if monitoring is enabled
        if self.enable_monitoring and LITELLM_MONITOR_AVAILABLE:
            self._callback_handler = LiteLLMLangChainCallback()
    
    def create_llm(
        self,
        model_name: str = "deepseek-r1:8b",
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Create an Ollama LLM instance with monitoring enabled.
        
        Args:
            model_name: Name of Ollama model to use
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional parameters for Ollama
            
        Returns:
            Configured Ollama LLM instance (ChatOllama or Ollama) with monitoring callbacks
        """
        cache_key = f"{model_name}_{temperature}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Prepare callbacks
        callbacks = kwargs.pop('callbacks', [])
        if self._callback_handler:
            if isinstance(callbacks, list):
                callbacks.append(self._callback_handler)
            else:
                callbacks = [self._callback_handler]
        
        if USE_CHAT_OLLAMA:
            # Use ChatOllama for better CrewAI compatibility
            # IMPORTANT: CrewAI uses LiteLLM internally, which needs the ollama_chat/ prefix
            # to properly route requests to Ollama
            prefixed_model = f"ollama_chat/{model_name}" if not model_name.startswith("ollama") else model_name
            llm = ChatOllama(
                model=prefixed_model,
                base_url=self.base_url,
                temperature=temperature,
                callbacks=callbacks if callbacks else None,
                **kwargs
            )
        else:
            # Fallback to legacy Ollama
            # IMPORTANT: CrewAI uses LiteLLM internally, which needs the ollama/ prefix
            prefixed_model = f"ollama/{model_name}" if not model_name.startswith("ollama") else model_name
            llm = Ollama(
                model=prefixed_model,
                base_url=self.base_url,
                temperature=temperature,
                callbacks=callbacks if callbacks else None,
                **kwargs
            )
        
        self._model_cache[cache_key] = llm
        return llm
    
    def create_planner_llm(self, model_name: Optional[str] = None):
        """
        Create LLM optimized for strategic planning and task orchestration.
        
        Args:
            model_name: Model to use (defaults to agent_config.planner_model)
            
        Returns:
            Configured Ollama LLM
        """
        model = model_name or self.agent_config.planner_model
        return self.create_llm(model, temperature=self.agent_config.planner_temp)
    
    def create_websearch_llm(self, model_name: Optional[str] = None):
        """
        Create LLM optimized for web research and information synthesis.
        
        Args:
            model_name: Model to use (defaults to agent_config.websearch_model)
            
        Returns:
            Configured Ollama LLM
        """
        model = model_name or self.agent_config.websearch_model
        return self.create_llm(model, temperature=self.agent_config.websearch_temp)
    
    def create_coder_llm(self, model_name: Optional[str] = None):
        """
        Create LLM optimized for code generation.
        
        Args:
            model_name: Model to use (defaults to agent_config.coder_model)
            
        Returns:
            Configured Ollama LLM
        """
        model = model_name or self.agent_config.coder_model
        return self.create_llm(model, temperature=self.agent_config.coder_temp)
    
    def create_evaluator_llm(self, model_name: Optional[str] = None):
        """
        Create LLM optimized for code evaluation.
        
        Args:
            model_name: Model to use (defaults to agent_config.evaluator_model)
            
        Returns:
            Configured Ollama LLM
        """
        model = model_name or self.agent_config.evaluator_model
        return self.create_llm(model, temperature=self.agent_config.evaluator_temp)
    
    def create_simulator_llm(self, model_name: Optional[str] = None):
        """
        Create LLM optimized for scenario simulation and testing.
        
        Args:
            model_name: Model to use (defaults to agent_config.simulator_model)
            
        Returns:
            Configured Ollama LLM
        """
        model = model_name or self.agent_config.simulator_model
        return self.create_llm(model, temperature=self.agent_config.simulator_temp)
    
    def create_executor_llm(self, model_name: Optional[str] = None):
        """
        Create LLM optimized for code execution tasks.
        
        Args:
            model_name: Model to use (defaults to agent_config.executor_model)
            
        Returns:
            Configured Ollama LLM
        """
        model = model_name or self.agent_config.executor_model
        return self.create_llm(model, temperature=self.agent_config.executor_temp)
    
    def create_reasoning_llm(self, model_name: str = "deepseek-r1:8b"):
        """
        Create LLM optimized for reasoning and simulation (legacy method).
        
        Args:
            model_name: Model to use
            
        Returns:
            Configured Ollama LLM
        """
        # Higher temperature for creative scenario exploration
        return self.create_llm(model_name, temperature=0.8)
    
    def validate_connection(self) -> bool:
        """
        Validate connection to Ollama server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Attempt to create a simple LLM instance
            llm = self.create_llm("deepseek-r1:8b")
            # Try a simple prompt
            response = llm.invoke("Test")
            return bool(response)
        except Exception:
            return False
    
    def list_available_models(self) -> list:
        """
        List available Ollama models.
        
        Returns:
            List of model names (placeholder - requires ollama Python API)
        """
        # This is a placeholder - actual implementation would use ollama.list()
        # or make HTTP request to Ollama API
        return ["deepseek-r1:8b", "codellama", "mistral", "mixtral"]
    
    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
    
    # =========================================================================
    # DeepThinker 2.0 Council-Based Methods
    # =========================================================================
    
    def create_council_llm(
        self,
        model_name: str,
        temperature: float,
        **kwargs
    ):
        """
        Create LLM for council-based execution.
        
        Generic method for creating LLMs used by council members.
        
        Args:
            model_name: Name of the Ollama model
            temperature: Sampling temperature for this council member
            **kwargs: Additional parameters
            
        Returns:
            Configured Ollama LLM instance
        """
        return self.create_llm(model_name, temperature=temperature, **kwargs)
    
    def create_arbiter_llm(
        self,
        model_name: str = "gemma3:27b",
        temperature: float = 0.3
    ):
        """
        Create LLM for the Arbiter (final decision-maker).
        
        Uses a large, capable model with low temperature for consistent
        final decisions and contradiction resolution.
        
        Args:
            model_name: Model to use (defaults to gemma3:27b)
            temperature: Low temperature for deterministic decisions
            
        Returns:
            Configured Ollama LLM
        """
        return self.create_llm(model_name, temperature=temperature)
    
    def create_meta_planner_llm(
        self,
        model_name: str = "gemma3:27b",
        temperature: float = 0.5
    ):
        """
        Create LLM for the Meta-Planner (highest-level strategist).
        
        Uses the largest available model with balanced temperature
        for strategic planning and council orchestration decisions.
        
        Args:
            model_name: Model to use (defaults to gemma3:27b)
            temperature: Moderate temperature for balanced strategy
            
        Returns:
            Configured Ollama LLM
        """
        return self.create_llm(model_name, temperature=temperature)
    
    def create_embedding_llm(
        self,
        model_name: str = "qwen3-embedding:4b"
    ):
        """
        Create embedding model for semantic distance consensus.
        
        Note: This returns a model configured for embedding generation.
        For actual embeddings, use get_embeddings() method.
        
        Args:
            model_name: Embedding model to use
            
        Returns:
            Configured embedding model reference
        """
        # Return model name for use with get_embeddings
        return model_name
    
    def get_embeddings(
        self,
        text: str,
        model_name: str = "qwen3-embedding:4b"
    ) -> List[float]:
        """
        Generate embeddings for text using Ollama embedding model.
        
        Uses centralized model_caller for proper resource management.
        
        Args:
            text: Text to embed
            model_name: Embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        from deepthinker.models.model_caller import call_embeddings
        
        embedding = call_embeddings(
            text=text,
            model=model_name,
            timeout=60.0,
            max_retries=2,
            base_url=self.base_url,
        )
        
        # Log to LiteLLM monitor if embedding failed and monitor is available
        if not embedding and LITELLM_MONITOR_AVAILABLE:
            logger = LiteLLMMonitor.get_logger()
            if logger:
                logger.errors.append({
                    "error": "Embedding generation failed (see model_caller logs)",
                    "model": model_name
                })
        
        return embedding
    
    def get_batch_embeddings(
        self,
        texts: List[str],
        model_name: str = "qwen3-embedding:4b"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model_name: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embeddings(text, model_name)
            embeddings.append(embedding)
        return embeddings

