"""
Unified Configuration Manager for DeepThinker.

Provides a single source of truth for configuration across both orchestration
paths (workflow and mission). Supports:
- Centralized model configuration
- Iteration and quality settings
- Environment-based overrides
- Runtime configuration updates
- Configuration validation

Usage:
    from deepthinker.core.config_manager import config_manager, get_config
    
    # Get current configuration
    config = get_config()
    
    # Access specific settings
    model = config.get_model("coder")
    threshold = config.iteration.quality_threshold
    
    # Override from environment
    config_manager.load_from_environment()
    
    # Override at runtime
    config_manager.update({"iteration.max_iterations": 5})
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ConfigSection(str, Enum):
    """Configuration sections."""
    
    MODELS = "models"
    ITERATION = "iteration"
    RESEARCH = "research"
    PLANNING = "planning"
    EXECUTION = "execution"
    GOVERNANCE = "governance"
    MEMORY = "memory"
    SAFETY = "safety"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    """
    Model configuration for all agent/council types.
    
    Centralizes model selection with environment overrides.
    """
    
    # Default models per role
    planner: str = "cogito:14b"
    researcher: str = "gemma3:12b"
    coder: str = "deepseek-r1:8b"
    evaluator: str = "gemma3:27b"
    simulator: str = "mistral:instruct"
    executor: str = "llama3.2:3b"
    arbiter: str = "gemma3:27b"
    meta_planner: str = "cogito:14b"
    embedding: str = "nomic-embed-text"
    
    # Temperatures per role
    planner_temp: float = 0.3
    researcher_temp: float = 0.4
    coder_temp: float = 0.2
    evaluator_temp: float = 0.3
    simulator_temp: float = 0.5
    executor_temp: float = 0.1
    arbiter_temp: float = 0.3
    
    # Global override
    override_all: Optional[str] = None
    
    def get_model(self, role: str) -> str:
        """Get model for a specific role."""
        if self.override_all:
            return self.override_all
        return getattr(self, role, self.coder)
    
    def get_temperature(self, role: str) -> float:
        """Get temperature for a specific role."""
        return getattr(self, f"{role}_temp", 0.3)
    
    def to_agent_model_config(self) -> "AgentModelConfig":
        """Convert to legacy AgentModelConfig for backward compatibility."""
        try:
            from ..models.ollama_loader import AgentModelConfig
            config = AgentModelConfig(
                planner_model=self.get_model("planner"),
                websearch_model=self.get_model("researcher"),
                coder_model=self.get_model("coder"),
                evaluator_model=self.get_model("evaluator"),
                simulator_model=self.get_model("simulator"),
                executor_model=self.get_model("executor"),
            )
            if self.override_all:
                config = config.override_all(self.override_all)
            return config
        except ImportError:
            logger.warning("AgentModelConfig not available")
            return None


@dataclass
class IterationConfig:
    """
    Configuration for iterative refinement.
    
    Used by both workflow and mission orchestration paths.
    """
    
    max_iterations: int = 5
    quality_threshold: float = 7.0
    enabled: bool = True
    
    # Early stopping
    min_improvement: float = 0.1  # Stop if improvement < this
    improvement_window: int = 3    # Number of iterations to check
    
    # Quality safety margin
    quality_safety_margin: float = 0.05
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0 <= self.quality_threshold <= 10:
            raise ValueError("quality_threshold must be between 0 and 10")


@dataclass
class ResearchConfig:
    """
    Configuration for web research phase.
    """
    
    enabled: bool = True
    max_results: int = 5
    timeout: int = 10
    
    # Advanced settings
    min_quality_score: float = 0.3
    deduplicate: bool = True
    extract_snippets: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_results < 1:
            raise ValueError("max_results must be at least 1")
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1")


@dataclass
class PlanningConfig:
    """
    Configuration for planning phase.
    """
    
    enabled: bool = True
    save_plan: bool = False
    plan_output_path: Optional[str] = None
    
    # Planning depth
    max_branches: int = 3
    scenario_count: int = 3


@dataclass
class ExecutionConfig:
    """
    Configuration for code execution.
    """
    
    backend: str = "subprocess"  # "subprocess" or "docker"
    timeout: int = 30
    
    # Docker settings
    docker_memory: str = "512m"
    docker_cpu: float = 1.0
    docker_image: str = "deepthinker-sandbox:latest"
    auto_build_image: bool = True
    
    # Security
    enable_security_scan: bool = True
    sandbox_network: bool = False  # Disable network in sandbox


@dataclass
class GovernanceConfig:
    """
    Configuration for governance and safety layers.
    """
    
    enabled: bool = True
    strict_mode: bool = False  # Fail on missing safety modules
    
    # Thresholds
    warn_threshold: float = 0.3
    block_threshold: float = 0.7
    
    # Features
    enable_claim_validation: bool = True
    enable_phase_guard: bool = True
    enable_proof_packets: bool = False
    
    # Retry policy
    max_retries: int = 2
    retry_on_warn: bool = True


@dataclass
class MemoryConfig:
    """
    Configuration for memory systems.
    """
    
    enabled: bool = True
    
    # Persistence
    persist_state: bool = True
    state_dir: str = ".deepthinker_state"
    
    # Limits
    max_context_tokens: int = 8000
    max_history_items: int = 100
    
    # Compression
    enable_compression: bool = True
    compression_threshold: int = 2000


@dataclass 
class SafetyConfig:
    """
    Configuration for safety features.
    """
    
    # Module availability requirements
    require_governance: bool = False
    require_phase_validator: bool = False
    require_memory_guard: bool = False
    
    # Logging
    log_degradation: bool = True
    log_level: str = "INFO"


@dataclass
class DeepThinkerConfig:
    """
    Complete unified configuration for DeepThinker.
    
    Single source of truth used by both orchestration paths.
    """
    
    models: ModelConfig = field(default_factory=ModelConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    # Global settings
    verbose: bool = False
    debug: bool = False
    ollama_base_url: str = "http://localhost:11434"
    
    def get_model(self, role: str) -> str:
        """Get model for a specific role."""
        return self.models.get_model(role)
    
    def get_temperature(self, role: str) -> float:
        """Get temperature for a specific role."""
        return self.models.get_temperature(role)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepThinkerConfig":
        """Create from dictionary."""
        config = cls()
        
        if "models" in data:
            config.models = ModelConfig(**data["models"])
        if "iteration" in data:
            config.iteration = IterationConfig(**data["iteration"])
        if "research" in data:
            config.research = ResearchConfig(**data["research"])
        if "planning" in data:
            config.planning = PlanningConfig(**data["planning"])
        if "execution" in data:
            config.execution = ExecutionConfig(**data["execution"])
        if "governance" in data:
            config.governance = GovernanceConfig(**data["governance"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "safety" in data:
            config.safety = SafetyConfig(**data["safety"])
        
        # Global settings
        if "verbose" in data:
            config.verbose = data["verbose"]
        if "debug" in data:
            config.debug = data["debug"]
        if "ollama_base_url" in data:
            config.ollama_base_url = data["ollama_base_url"]
        
        return config


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """
    Centralized configuration manager.
    
    Provides:
    - Single source of truth for configuration
    - Environment variable overrides
    - Runtime updates
    - Configuration validation
    - Change callbacks
    """
    
    # Environment variable prefix
    ENV_PREFIX = "DEEPTHINKER_"
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        # Models
        "MODEL_PLANNER": "models.planner",
        "MODEL_RESEARCHER": "models.researcher",
        "MODEL_CODER": "models.coder",
        "MODEL_EVALUATOR": "models.evaluator",
        "MODEL_SIMULATOR": "models.simulator",
        "MODEL_EXECUTOR": "models.executor",
        "MODEL_ARBITER": "models.arbiter",
        "MODEL_ALL": "models.override_all",
        
        # Iteration
        "MAX_ITERATIONS": "iteration.max_iterations",
        "QUALITY_THRESHOLD": "iteration.quality_threshold",
        "ITERATION_ENABLED": "iteration.enabled",
        
        # Research
        "RESEARCH_ENABLED": "research.enabled",
        "RESEARCH_MAX_RESULTS": "research.max_results",
        "RESEARCH_TIMEOUT": "research.timeout",
        
        # Planning
        "PLANNING_ENABLED": "planning.enabled",
        
        # Execution
        "EXECUTION_BACKEND": "execution.backend",
        "EXECUTION_TIMEOUT": "execution.timeout",
        "DOCKER_MEMORY": "execution.docker_memory",
        "DOCKER_CPU": "execution.docker_cpu",
        "SECURITY_SCAN_ENABLED": "execution.enable_security_scan",
        
        # Governance
        "GOVERNANCE_ENABLED": "governance.enabled",
        "GOVERNANCE_STRICT": "governance.strict_mode",
        
        # Memory
        "MEMORY_ENABLED": "memory.enabled",
        "STATE_DIR": "memory.state_dir",
        
        # Global
        "VERBOSE": "verbose",
        "DEBUG": "debug",
        "OLLAMA_API_BASE": "ollama_base_url",
    }
    
    def __init__(self):
        self._config: DeepThinkerConfig = DeepThinkerConfig()
        self._callbacks: List[Callable[[str, Any, Any], None]] = []
        self._frozen: bool = False
        self._initialized: bool = False
    
    @property
    def config(self) -> DeepThinkerConfig:
        """Get current configuration."""
        return self._config
    
    def initialize(
        self,
        config_path: Optional[str] = None,
        load_env: bool = True
    ) -> DeepThinkerConfig:
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to JSON config file
            load_env: Whether to load from environment variables
            
        Returns:
            Initialized configuration
        """
        # Load from file if specified
        if config_path:
            self.load_from_file(config_path)
        
        # Override from environment
        if load_env:
            self.load_from_environment()
        
        self._initialized = True
        logger.info("ConfigManager initialized")
        self._log_config_summary()
        
        return self._config
    
    def load_from_file(self, path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._config = DeepThinkerConfig.from_dict(data)
            logger.info(f"Configuration loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        for env_suffix, config_path in self.ENV_MAPPINGS.items():
            env_var = f"{self.ENV_PREFIX}{env_suffix}"
            value = os.environ.get(env_var)
            
            if value is not None:
                self._set_nested(config_path, self._parse_value(value))
                logger.debug(f"Config {config_path} set from {env_var}")
    
    def _parse_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def _set_nested(self, path: str, value: Any) -> None:
        """
        Set a nested configuration value.
        
        Args:
            path: Dot-separated path (e.g., "models.coder")
            value: Value to set
        """
        parts = path.split('.')
        obj = self._config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        old_value = getattr(obj, parts[-1])
        setattr(obj, parts[-1], value)
        
        # Notify callbacks
        self._notify_change(path, old_value, value)
    
    def _get_nested(self, path: str) -> Any:
        """
        Get a nested configuration value.
        
        Args:
            path: Dot-separated path
            
        Returns:
            Configuration value
        """
        parts = path.split('.')
        obj = self._config
        
        for part in parts:
            obj = getattr(obj, part)
        
        return obj
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.
        
        Args:
            path: Dot-separated path
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        try:
            return self._get_nested(path)
        except AttributeError:
            return default
    
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            path: Dot-separated path
            value: Value to set
            
        Raises:
            RuntimeError: If configuration is frozen
        """
        if self._frozen:
            raise RuntimeError("Configuration is frozen")
        
        self._set_nested(path, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of path -> value updates
            
        Raises:
            RuntimeError: If configuration is frozen
        """
        if self._frozen:
            raise RuntimeError("Configuration is frozen")
        
        for path, value in updates.items():
            self._set_nested(path, value)
    
    def freeze(self) -> None:
        """Freeze configuration to prevent further changes."""
        self._frozen = True
        logger.info("Configuration frozen")
    
    def unfreeze(self) -> None:
        """Unfreeze configuration to allow changes."""
        self._frozen = False
        logger.info("Configuration unfrozen")
    
    def add_change_callback(
        self, 
        callback: Callable[[str, Any, Any], None]
    ) -> None:
        """
        Add a callback for configuration changes.
        
        Args:
            callback: Function(path, old_value, new_value)
        """
        self._callbacks.append(callback)
    
    def _notify_change(self, path: str, old_value: Any, new_value: Any) -> None:
        """Notify callbacks of configuration change."""
        for callback in self._callbacks:
            try:
                callback(path, old_value, new_value)
            except Exception as e:
                logger.error(f"Config change callback failed: {e}")
    
    def _log_config_summary(self) -> None:
        """Log configuration summary."""
        logger.info("=" * 50)
        logger.info("DeepThinker Configuration")
        logger.info("=" * 50)
        logger.info(f"Models:")
        logger.info(f"  Planner: {self._config.models.get_model('planner')}")
        logger.info(f"  Researcher: {self._config.models.get_model('researcher')}")
        logger.info(f"  Coder: {self._config.models.get_model('coder')}")
        logger.info(f"  Evaluator: {self._config.models.get_model('evaluator')}")
        logger.info(f"Iteration: max={self._config.iteration.max_iterations}, threshold={self._config.iteration.quality_threshold}")
        logger.info(f"Research: {'enabled' if self._config.research.enabled else 'disabled'}")
        logger.info(f"Planning: {'enabled' if self._config.planning.enabled else 'disabled'}")
        logger.info(f"Execution: {self._config.execution.backend}")
        logger.info(f"Governance: {'enabled' if self._config.governance.enabled else 'disabled'}")
        logger.info("=" * 50)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self._config.models
    
    def get_iteration_config(self) -> IterationConfig:
        """Get iteration configuration."""
        return self._config.iteration
    
    def get_research_config(self) -> ResearchConfig:
        """Get research configuration."""
        return self._config.research
    
    def get_planning_config(self) -> PlanningConfig:
        """Get planning configuration."""
        return self._config.planning
    
    def get_execution_config(self) -> ExecutionConfig:
        """Get execution configuration."""
        return self._config.execution
    
    def get_governance_config(self) -> GovernanceConfig:
        """Get governance configuration."""
        return self._config.governance
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration."""
        return self._config.memory
    
    def get_safety_config(self) -> SafetyConfig:
        """Get safety configuration."""
        return self._config.safety
    
    def to_legacy_configs(self) -> Dict[str, Any]:
        """
        Convert to legacy configuration objects for backward compatibility.
        
        Returns:
            Dictionary with legacy config objects
        """
        from ..execution.run_workflow import (
            IterationConfig as LegacyIterationConfig,
            ResearchConfig as LegacyResearchConfig,
            PlanningConfig as LegacyPlanningConfig,
        )
        
        return {
            "iteration_config": LegacyIterationConfig(
                max_iterations=self._config.iteration.max_iterations,
                quality_threshold=self._config.iteration.quality_threshold,
                enabled=self._config.iteration.enabled,
            ),
            "research_config": LegacyResearchConfig(
                enabled=self._config.research.enabled,
                max_results=self._config.research.max_results,
                timeout=self._config.research.timeout,
            ),
            "planning_config": LegacyPlanningConfig(
                enabled=self._config.planning.enabled,
                save_plan=self._config.planning.save_plan,
                plan_output_path=self._config.planning.plan_output_path,
            ),
            "agent_model_config": self._config.models.to_agent_model_config(),
        }
    
    def save_to_file(self, path: str) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            path: Output file path
        """
        data = self._config.to_dict()
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")


# =============================================================================
# Global Instance and Convenience Functions
# =============================================================================

# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> DeepThinkerConfig:
    """Get the current configuration."""
    return config_manager.config


def initialize_config(
    config_path: Optional[str] = None,
    load_env: bool = True
) -> DeepThinkerConfig:
    """Initialize the global configuration."""
    return config_manager.initialize(config_path, load_env)


def get_model(role: str) -> str:
    """Get model for a role."""
    return config_manager.config.get_model(role)


def get_temperature(role: str) -> float:
    """Get temperature for a role."""
    return config_manager.config.get_temperature(role)


def set_config(path: str, value: Any) -> None:
    """Set a configuration value."""
    config_manager.set(path, value)


def update_config(updates: Dict[str, Any]) -> None:
    """Update multiple configuration values."""
    config_manager.update(updates)

