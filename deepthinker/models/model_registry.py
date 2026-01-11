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
import re
import subprocess

import requests

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier categories."""
    REASONING = "reasoning"   # Best for complex reasoning tasks
    LARGE = "large"          # High capability, slower
    MEDIUM = "medium"        # Balanced capability/speed
    SMALL = "small"          # Fast, lower capability
    EMBEDDING = "embedding"  # Embedding models


# =============================================================================
# Model Metadata Inference Patterns
# =============================================================================

# Family patterns for capability inference
FAMILY_CAPABILITIES = {
    # family_pattern -> base capabilities dict
    "deepseek": {"coding": 9, "reasoning": 8, "planning": 6, "evaluation": 7, "creativity": 5},
    "coder": {"coding": 9, "reasoning": 6, "planning": 5, "evaluation": 6, "creativity": 4},
    "code": {"coding": 8, "reasoning": 6, "planning": 5, "evaluation": 6, "creativity": 4},
    "cogito": {"reasoning": 9, "planning": 8, "coding": 8, "evaluation": 8, "creativity": 7},
    "gemma": {"reasoning": 7, "planning": 6, "coding": 6, "evaluation": 7, "creativity": 6, "synthesis": 7},
    "llama": {"reasoning": 6, "planning": 5, "coding": 5, "evaluation": 6, "creativity": 6, "research": 6},
    "qwen": {"reasoning": 6, "planning": 5, "coding": 6, "evaluation": 6, "creativity": 5, "research": 6},
    "mistral": {"reasoning": 7, "planning": 6, "coding": 6, "evaluation": 5, "creativity": 9, "synthesis": 7},
    "mixtral": {"reasoning": 8, "planning": 7, "coding": 7, "evaluation": 7, "creativity": 8, "synthesis": 8},
    "phi": {"reasoning": 6, "planning": 5, "coding": 7, "evaluation": 6, "creativity": 5},
    "devstral": {"reasoning": 8, "planning": 7, "coding": 9, "evaluation": 7, "creativity": 6},
    "llava": {"reasoning": 7, "planning": 6, "coding": 5, "evaluation": 7, "creativity": 8, "vision": 9},
    "bakllava": {"reasoning": 6, "planning": 5, "coding": 4, "evaluation": 6, "creativity": 7, "vision": 8},
    "embed": {"embedding": 8},
    "snowflake": {"embedding": 8},
}

# Special tags that modify capabilities
TAG_MODIFIERS = {
    "instruct": {"reasoning": 1, "planning": 1},  # Additive modifiers
    "chat": {"creativity": 1, "synthesis": 1},
    "code": {"coding": 2},
    "coder": {"coding": 2},
    "math": {"reasoning": 2},
    "vision": {"vision": 8},
}


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
        is_known: Whether model is from static registry (True) or auto-discovered (False)
        family: Model family (e.g., "gemma", "llama")
        parameter_size: Size string (e.g., "27b", "8b")
        quantization: Quantization level if known
    """
    name: str
    tier: ModelTier
    vram_mb: int = 8000
    capabilities: Dict[str, float] = field(default_factory=dict)
    default_temperature: float = 0.5
    max_tokens: int = 4096
    is_available: bool = True
    is_known: bool = True
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "tier": self.tier.value if isinstance(self.tier, ModelTier) else self.tier,
            "vram_mb": self.vram_mb,
            "capabilities": self.capabilities,
            "default_temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
            "is_available": self.is_available,
            "is_known": self.is_known,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
        }


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
        ollama_base_url: str = "http://localhost:11434",
        auto_discover: bool = False
    ):
        """
        Initialize the registry.
        
        Args:
            check_availability: Whether to check model availability on init
            ollama_base_url: Ollama server URL
            auto_discover: Whether to auto-discover models from Ollama
        """
        self._models = dict(MODEL_REGISTRY)
        self._tiers = dict(MODEL_TIERS)
        self.ollama_base_url = ollama_base_url
        self._discovered_models: Dict[str, ModelInfo] = {}
        
        if auto_discover:
            self.discover_ollama_models()
        elif check_availability:
            self.refresh_availability()
    
    def discover_ollama_models(self) -> List[str]:
        """
        Discover models from Ollama API and merge into registry.
        
        Models in static registry keep their curated metadata.
        New models get inferred metadata based on name patterns.
        
        Returns:
            List of newly discovered model names
        """
        newly_discovered = []
        available_models = set()
        
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            for model in data.get("models", []):
                name = model.get("name", "")
                if not name:
                    continue
                
                available_models.add(name)
                
                # Check if model is in static registry
                if name in self._models:
                    # Update availability but keep static metadata
                    self._models[name].is_available = True
                    continue
                
                # Check if already discovered
                if name in self._discovered_models:
                    self._discovered_models[name].is_available = True
                    continue
                
                # Infer metadata for new model
                details = model.get("details", {})
                inferred = self.infer_model_metadata(
                    name,
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                )
                inferred.is_available = True
                inferred.is_known = False
                
                # Store in discovered models
                self._discovered_models[name] = inferred
                self._models[name] = inferred
                
                # Add to appropriate tier
                tier_name = inferred.tier.value
                if tier_name not in self._tiers:
                    self._tiers[tier_name] = []
                if name not in self._tiers[tier_name]:
                    self._tiers[tier_name].append(name)
                
                newly_discovered.append(name)
                logger.info(f"Discovered new model: {name} (tier={tier_name})")
            
            # Mark unavailable models
            for model_name in self._models:
                if model_name not in available_models:
                    self._models[model_name].is_available = False
            
            logger.info(
                f"Model discovery complete: {len(available_models)} available, "
                f"{len(newly_discovered)} newly discovered"
            )
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to Ollama at {self.ollama_base_url}")
        except Exception as e:
            logger.warning(f"Error discovering models: {e}")
        
        return newly_discovered
    
    def infer_model_metadata(
        self,
        model_name: str,
        family: Optional[str] = None,
        parameter_size: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> ModelInfo:
        """
        Infer model metadata from name and optional details.
        
        Args:
            model_name: Full model name (e.g., "llama3.2:8b-instruct-q4_0")
            family: Model family from Ollama details
            parameter_size: Parameter size from Ollama details
            quantization: Quantization level from Ollama details
            
        Returns:
            ModelInfo with inferred metadata
        """
        name_lower = model_name.lower()
        
        # Detect family from name if not provided
        detected_family = family
        if not detected_family:
            for fam_pattern in FAMILY_CAPABILITIES.keys():
                if fam_pattern in name_lower:
                    detected_family = fam_pattern
                    break
        
        # Detect parameter size from name
        detected_size = parameter_size
        size_num = None
        if not detected_size:
            size_match = re.search(r":(\d+\.?\d*)b", name_lower)
            if size_match:
                detected_size = size_match.group(1) + "b"
                size_num = float(size_match.group(1))
        else:
            # Parse provided size
            size_match = re.search(r"(\d+\.?\d*)", detected_size)
            if size_match:
                size_num = float(size_match.group(1))
        
        # Determine tier and VRAM based on size
        tier = ModelTier.MEDIUM
        vram_mb = 5000
        capability_mult = 0.7
        
        if size_num is not None:
            if size_num >= 65:
                tier, vram_mb, capability_mult = ModelTier.REASONING, 45000, 1.0
            elif size_num >= 27:
                tier, vram_mb, capability_mult = ModelTier.REASONING, 17000, 0.9
            elif size_num >= 13:
                tier, vram_mb, capability_mult = ModelTier.LARGE, 8000, 0.8
            elif size_num >= 7:
                tier, vram_mb, capability_mult = ModelTier.MEDIUM, 5000, 0.7
            elif size_num >= 3:
                tier, vram_mb, capability_mult = ModelTier.SMALL, 2000, 0.5
            else:
                tier, vram_mb, capability_mult = ModelTier.SMALL, 1000, 0.4
        
        # Check for embedding model
        if "embed" in name_lower or detected_family == "embed" or detected_family == "snowflake":
            tier = ModelTier.EMBEDDING
            capabilities = {"embedding": 8}
        else:
            # Get base capabilities from family
            base_caps = FAMILY_CAPABILITIES.get(detected_family, {
                "reasoning": 5, "planning": 5, "coding": 5, 
                "evaluation": 5, "creativity": 5, "synthesis": 5, "research": 5
            })
            
            # Scale by capability multiplier
            capabilities = {}
            for cap, score in base_caps.items():
                scaled = min(10, round(score * capability_mult + (1 - capability_mult) * 5))
                capabilities[cap] = scaled
            
            # Apply tag modifiers
            for tag, modifiers in TAG_MODIFIERS.items():
                if tag in name_lower:
                    for cap, mod in modifiers.items():
                        if cap in capabilities:
                            capabilities[cap] = min(10, capabilities[cap] + mod)
                        else:
                            capabilities[cap] = mod
        
        # Determine temperature
        if tier == ModelTier.EMBEDDING:
            temperature = 0.0
        elif "code" in name_lower or "coder" in name_lower:
            temperature = 0.3
        elif "instruct" in name_lower:
            temperature = 0.5
        else:
            temperature = 0.5
        
        # Max tokens
        max_tokens = 8192 if size_num and size_num >= 7 else 4096
        
        return ModelInfo(
            name=model_name,
            tier=tier,
            vram_mb=vram_mb,
            capabilities=capabilities,
            default_temperature=temperature,
            max_tokens=max_tokens,
            is_available=True,
            is_known=False,
            family=detected_family,
            parameter_size=detected_size,
            quantization=quantization,
        )
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """
        Get all models (static + discovered).
        
        Returns:
            Dict mapping model name to ModelInfo
        """
        return dict(self._models)
    
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


def get_model_registry(
    check_availability: bool = False,
    auto_discover: bool = False
) -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Args:
        check_availability: Whether to check model availability
        auto_discover: Whether to auto-discover models from Ollama
        
    Returns:
        ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry(
            check_availability=check_availability,
            auto_discover=auto_discover
        )
    elif auto_discover:
        # Refresh discovery if requested
        _registry.discover_ollama_models()
    return _registry


def reset_model_registry() -> None:
    """Reset the global registry instance (useful for testing)."""
    global _registry
    _registry = None

