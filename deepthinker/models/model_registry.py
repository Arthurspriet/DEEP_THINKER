"""
Model Registry for DeepThinker 2.0.

Central registry of available models with:
- Tier categorization (reasoning, large, medium, small, embedding)
- Capability scoring
- Phase-aware model selection
- Dynamic availability checking

Updated to reflect current Ollama model inventory.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import subprocess

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier categories."""
    REASONING = "reasoning"   # Best for complex reasoning tasks
    LARGE = "large"          # High capability, slower
    MEDIUM = "medium"        # Balanced capability/speed
    SMALL = "small"          # Fast, lower capability
    EMBEDDING = "embedding"  # Embedding models


@dataclass
class ModelInfo:
    """
    Information about a single model.
    
    Attributes:
        name: Model name (e.g., "gemma3:27b")
        tier: Model tier category
        vram_mb: Estimated VRAM usage in MB
        capabilities: Capability scores (0-10)
        default_temperature: Default temperature for this model
        max_tokens: Maximum tokens this model supports
        is_available: Whether model is currently available
    """
    name: str
    tier: ModelTier
    vram_mb: int = 8000
    capabilities: Dict[str, float] = field(default_factory=dict)
    default_temperature: float = 0.5
    max_tokens: int = 4096
    is_available: bool = True


# =============================================================================
# Current Model Registry (Updated from Ollama list)
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # Reasoning tier - best for complex analysis
    "gemma3:27b": ModelInfo(
        name="gemma3:27b",
        tier=ModelTier.REASONING,
        vram_mb=17000,
        capabilities={
            "reasoning": 9,
            "planning": 8,
            "coding": 7,
            "evaluation": 9,
            "creativity": 7,
            "synthesis": 9,
            "research": 8,
        },
        default_temperature=0.5,
        max_tokens=8192,
    ),
    "cogito:14b": ModelInfo(
        name="cogito:14b",
        tier=ModelTier.REASONING,
        vram_mb=9000,
        capabilities={
            "reasoning": 9,
            "planning": 8,
            "coding": 8,
            "evaluation": 8,
            "creativity": 7,
            "synthesis": 8,
            "research": 7,
        },
        default_temperature=0.5,
        max_tokens=8192,
    ),
    "devstral:latest": ModelInfo(
        name="devstral:latest",
        tier=ModelTier.REASONING,
        vram_mb=14000,
        capabilities={
            "reasoning": 8,
            "planning": 7,
            "coding": 9,
            "evaluation": 7,
            "creativity": 6,
            "synthesis": 7,
            "research": 7,
        },
        default_temperature=0.4,
        max_tokens=8192,
    ),
    
    # Large tier
    "gpt-oss:latest": ModelInfo(
        name="gpt-oss:latest",
        tier=ModelTier.LARGE,
        vram_mb=13000,
        capabilities={
            "reasoning": 8,
            "planning": 8,
            "coding": 8,
            "evaluation": 8,
            "creativity": 8,
            "synthesis": 8,
            "research": 8,
        },
        default_temperature=0.5,
        max_tokens=8192,
    ),
    "llava:13b": ModelInfo(
        name="llava:13b",
        tier=ModelTier.LARGE,
        vram_mb=8000,
        capabilities={
            "reasoning": 7,
            "planning": 6,
            "coding": 5,
            "evaluation": 7,
            "creativity": 8,
            "synthesis": 7,
            "research": 7,
            "vision": 9,
        },
        default_temperature=0.5,
        max_tokens=4096,
    ),
    
    # Medium tier - balanced
    "gemma3:12b": ModelInfo(
        name="gemma3:12b",
        tier=ModelTier.MEDIUM,
        vram_mb=8100,
        capabilities={
            "reasoning": 7,
            "planning": 6,
            "coding": 6,
            "evaluation": 7,
            "creativity": 6,
            "synthesis": 7,
            "research": 7,
        },
        default_temperature=0.5,
        max_tokens=8192,
    ),
    "deepseek-r1:8b": ModelInfo(
        name="deepseek-r1:8b",
        tier=ModelTier.MEDIUM,
        vram_mb=5200,
        capabilities={
            "reasoning": 8,
            "planning": 6,
            "coding": 9,
            "evaluation": 7,
            "creativity": 5,
            "synthesis": 6,
            "research": 6,
        },
        default_temperature=0.3,
        max_tokens=8192,
    ),
    "mistral:instruct": ModelInfo(
        name="mistral:instruct",
        tier=ModelTier.MEDIUM,
        vram_mb=4100,
        capabilities={
            "reasoning": 7,
            "planning": 6,
            "coding": 6,
            "evaluation": 5,
            "creativity": 9,
            "synthesis": 7,
            "research": 7,
        },
        default_temperature=0.6,
        max_tokens=8192,
    ),
    
    # Small tier - fast
    "llama3.2:3b": ModelInfo(
        name="llama3.2:3b",
        tier=ModelTier.SMALL,
        vram_mb=2000,
        capabilities={
            "reasoning": 5,
            "planning": 4,
            "coding": 5,
            "evaluation": 5,
            "creativity": 5,
            "synthesis": 5,
            "research": 5,
        },
        default_temperature=0.5,
        max_tokens=4096,
    ),
    "gemma3:4b": ModelInfo(
        name="gemma3:4b",
        tier=ModelTier.SMALL,
        vram_mb=3300,
        capabilities={
            "reasoning": 5,
            "planning": 4,
            "coding": 5,
            "evaluation": 5,
            "creativity": 5,
            "synthesis": 5,
            "research": 5,
        },
        default_temperature=0.5,
        max_tokens=8192,
    ),
    "qwen3:4b": ModelInfo(
        name="qwen3:4b",
        tier=ModelTier.SMALL,
        vram_mb=2500,
        capabilities={
            "reasoning": 5,
            "planning": 4,
            "coding": 5,
            "evaluation": 5,
            "creativity": 5,
            "synthesis": 5,
            "research": 5,
        },
        default_temperature=0.5,
        max_tokens=8192,
    ),
    "llama3.2:1b": ModelInfo(
        name="llama3.2:1b",
        tier=ModelTier.SMALL,
        vram_mb=1300,
        capabilities={
            "reasoning": 4,
            "planning": 3,
            "coding": 4,
            "evaluation": 4,
            "creativity": 4,
            "synthesis": 4,
            "research": 4,
        },
        default_temperature=0.5,
        max_tokens=4096,
    ),
    
    # Embedding tier
    "qwen3-embedding:4b": ModelInfo(
        name="qwen3-embedding:4b",
        tier=ModelTier.EMBEDDING,
        vram_mb=2500,
        capabilities={"embedding": 9},
        default_temperature=0.0,
        max_tokens=8192,
    ),
    "snowflake-arctic-embed:latest": ModelInfo(
        name="snowflake-arctic-embed:latest",
        tier=ModelTier.EMBEDDING,
        vram_mb=670,
        capabilities={"embedding": 8},
        default_temperature=0.0,
        max_tokens=512,
    ),
    "embeddinggemma:latest": ModelInfo(
        name="embeddinggemma:latest",
        tier=ModelTier.EMBEDDING,
        vram_mb=621,
        capabilities={"embedding": 7},
        default_temperature=0.0,
        max_tokens=512,
    ),
}

# Tier groupings for quick access
MODEL_TIERS: Dict[str, List[str]] = {
    "reasoning": ["gemma3:27b", "cogito:14b", "devstral:latest"],
    "large": ["gemma3:27b", "gpt-oss:latest", "cogito:14b", "llava:13b"],
    "medium": ["gemma3:12b", "deepseek-r1:8b", "mistral:instruct", "devstral:latest"],
    "small": ["llama3.2:3b", "gemma3:4b", "qwen3:4b", "llama3.2:1b"],
    "embedding": ["qwen3-embedding:4b", "snowflake-arctic-embed:latest", "embeddinggemma:latest"],
}

# Phase to preferred tier mapping
PHASE_MODEL_PREFERENCES: Dict[str, List[str]] = {
    "reconnaissance": ["small", "medium"],
    "analysis": ["medium", "large"],
    "deep_analysis": ["reasoning", "large"],
    "synthesis": ["large", "reasoning"],
    "implementation": ["medium", "reasoning"],
    "simulation": ["medium"],
    "evaluation": ["medium", "large"],
}


class ModelRegistry:
    """
    Central registry for model management.
    
    Provides:
    - Model lookup by name/tier
    - Phase-aware model selection
    - Capability-based filtering
    - Availability checking
    """
    
    def __init__(
        self,
        check_availability: bool = False,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the registry.
        
        Args:
            check_availability: Whether to check model availability on init
            ollama_base_url: Ollama server URL
        """
        self._models = dict(MODEL_REGISTRY)
        self._tiers = dict(MODEL_TIERS)
        self.ollama_base_url = ollama_base_url
        
        if check_availability:
            self.refresh_availability()
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._models.get(name)
    
    def get_tier(self, tier: str) -> List[str]:
        """Get all model names in a tier."""
        return self._tiers.get(tier, [])
    
    def get_available_in_tier(self, tier: str) -> List[str]:
        """Get available models in a tier."""
        models = self._tiers.get(tier, [])
        return [m for m in models if self._models.get(m, ModelInfo(name=m, tier=ModelTier.SMALL)).is_available]
    
    def get_models_by_capability(
        self,
        capability: str,
        min_score: float = 7.0
    ) -> List[str]:
        """Get models with a capability above threshold."""
        result = []
        for name, info in self._models.items():
            if info.capabilities.get(capability, 0) >= min_score:
                result.append(name)
        return result
    
    def select_for_phase(
        self,
        phase_type: str,
        count: int = 2,
        available_only: bool = True,
        contraction_mode: bool = False
    ) -> List[str]:
        """
        Select models appropriate for a phase.
        
        Args:
            phase_type: Type of phase (reconnaissance, deep_analysis, etc.)
            count: Number of models to select
            available_only: Only return available models
            contraction_mode: Use smallest models only
            
        Returns:
            List of model names
        """
        if contraction_mode:
            # Use smallest available model only
            small_models = self.get_available_in_tier("small") if available_only else self._tiers["small"]
            return small_models[:1] if small_models else ["llama3.2:1b"]
        
        # Get preferred tiers for this phase
        preferred_tiers = PHASE_MODEL_PREFERENCES.get(phase_type, ["medium"])
        
        candidates = []
        for tier in preferred_tiers:
            if available_only:
                tier_models = self.get_available_in_tier(tier)
            else:
                tier_models = self._tiers.get(tier, [])
            candidates.extend(tier_models)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for m in candidates:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        
        return unique[:count]
    
    def select_by_capabilities(
        self,
        required_capabilities: List[str],
        count: int = 2,
        min_score: float = 6.0
    ) -> List[str]:
        """
        Select models that meet capability requirements.
        
        Args:
            required_capabilities: List of required capability names
            count: Number of models to select
            min_score: Minimum score for each capability
            
        Returns:
            List of model names sorted by total score
        """
        scored = []
        for name, info in self._models.items():
            if not info.is_available:
                continue
            
            # Check all capabilities meet minimum
            meets_all = True
            total_score = 0.0
            for cap in required_capabilities:
                score = info.capabilities.get(cap, 0)
                if score < min_score:
                    meets_all = False
                    break
                total_score += score
            
            if meets_all:
                scored.append((name, total_score))
        
        # Sort by total score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in scored[:count]]
    
    def estimate_vram(self, models: List[str]) -> int:
        """Estimate total VRAM needed for a set of models."""
        total = 0
        for name in models:
            info = self._models.get(name)
            if info:
                total += info.vram_mb
        return total
    
    def requires_serialization(self, models: List[str]) -> bool:
        """
        Check if models require serial execution (no parallelism).
        
        REASONING and LARGE tier models must run sequentially to prevent
        GPU thrashing and ensure stable execution.
        
        Args:
            models: List of model names to check
            
        Returns:
            True if any model requires serialization (REASONING or LARGE tier)
        """
        for name in models:
            info = self._models.get(name)
            if info and info.tier in [ModelTier.REASONING, ModelTier.LARGE]:
                return True
        return False
    
    def refresh_availability(self) -> None:
        """Check and update model availability from Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                available = set()
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line.strip():
                        name = line.split()[0]
                        available.add(name)
                
                # Update availability in registry
                for model_name in self._models:
                    self._models[model_name].is_available = model_name in available
                
                logger.info(f"Refreshed model availability: {len(available)} models available")
        except Exception as e:
            logger.warning(f"Failed to check model availability: {e}")
    
    def get_embedding_model(self, preference: str = "qwen3-embedding:4b") -> str:
        """Get the best available embedding model."""
        if preference in self._models and self._models[preference].is_available:
            return preference
        
        # Fallback to any available embedding model
        for name in self._tiers.get("embedding", []):
            if self._models.get(name, ModelInfo(name=name, tier=ModelTier.EMBEDDING)).is_available:
                return name
        
        return "qwen3-embedding:4b"  # Default
    
    def get_pool_config(
        self,
        models: List[str],
        temperature_override: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate pool config tuples for ModelPool.
        
        Args:
            models: List of model names
            temperature_override: Optional temperature to use for all
            
        Returns:
            List of (model_name, temperature) tuples
        """
        config = []
        for name in models:
            info = self._models.get(name)
            if info:
                temp = temperature_override if temperature_override is not None else info.default_temperature
            else:
                temp = temperature_override if temperature_override is not None else 0.5
            config.append((name, temp))
        return config


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry(check_availability: bool = False) -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(check_availability=check_availability)
    return _registry

