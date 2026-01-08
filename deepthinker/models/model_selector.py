"""
Model Selector for DeepThinker 2.0.

Provides phase-aware, intelligent model selection:
- Selects appropriate models based on phase type
- Considers time remaining and token budget
- Supports contraction mode for resource exhaustion
- Integrates with ModelRegistry

This replaces the hardcoded model selection with a flexible system.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model_registry import (
    ModelRegistry,
    ModelInfo,
    ModelTier,
    MODEL_REGISTRY,
    MODEL_TIERS,
    PHASE_MODEL_PREFERENCES,
    get_model_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelSelection:
    """
    Result of model selection.
    
    Attributes:
        models: List of selected model names
        temperatures: Corresponding temperatures
        reason: Why these models were selected
        estimated_vram_mb: Estimated VRAM usage
        tier: Primary tier used
    """
    models: List[str]
    temperatures: List[float]
    reason: str
    estimated_vram_mb: int = 0
    tier: str = "medium"
    
    def get_pool_config(self) -> List[Tuple[str, float]]:
        """Get config for ModelPool."""
        return list(zip(self.models, self.temperatures))
    
    def is_contraction_mode(self) -> bool:
        """Check if this is contraction mode (single small model)."""
        return len(self.models) == 1 and self.tier == "small"


class ModelSelector:
    """
    Intelligent model selector for DeepThinker.
    
    Selects models based on:
    - Phase type (reconnaissance -> small, deep_analysis -> reasoning)
    - Time remaining (less time -> smaller models)
    - Uncertainty level (high uncertainty -> multiple models)
    - Token budget (limited budget -> smaller models)
    - Contraction mode (resource exhaustion -> single small)
    """
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        default_temperature: float = 0.5,
        max_concurrent: int = 3
    ):
        """
        Initialize the selector.
        
        Args:
            registry: Model registry to use
            default_temperature: Default temperature if not specified
            max_concurrent: Maximum concurrent models
        """
        self.registry = registry or get_model_registry()
        self.default_temperature = default_temperature
        self.max_concurrent = max_concurrent
    
    def select_for_phase(
        self,
        phase_type: str,
        time_remaining_minutes: float = 60.0,
        uncertainty_level: float = 0.5,
        token_budget: int = 8000,
        contraction_mode: bool = False,
        phase_importance: float = 0.5
    ) -> ModelSelection:
        """
        Select models appropriate for a phase.
        
        Args:
            phase_type: Type of phase (reconnaissance, analysis, etc.)
            time_remaining_minutes: Minutes remaining in mission
            uncertainty_level: Current uncertainty (0-1)
            token_budget: Token budget for this operation
            contraction_mode: If True, use minimal resources
            phase_importance: Importance of the phase (0-1), used for tier floor
            
        Returns:
            ModelSelection with selected models
        """
        # Contraction mode: single smallest model
        if contraction_mode:
            return self._select_contraction_mode()
        
        # Time pressure adjustments
        time_factor = self._compute_time_factor(time_remaining_minutes)
        
        # Determine base tier from phase
        base_tier = self._get_tier_for_phase(phase_type)
        
        # Model-Aware Phase Stabilization: Calculate tier floor based on importance
        min_tier_floor = self._get_tier_floor_for_importance(phase_importance)
        
        # Adjust tier based on time and budget, respecting floor
        adjusted_tier = self._adjust_tier(base_tier, time_factor, token_budget, min_tier_floor)
        
        # Determine model count based on uncertainty
        model_count = self._compute_model_count(uncertainty_level, adjusted_tier)
        
        # Select models
        models = self.registry.select_for_phase(
            phase_type=phase_type,
            count=model_count,
            available_only=True
        )
        
        if not models:
            # Fallback to any available model
            models = self._get_fallback_models()
        
        # Get temperatures
        temperatures = self._get_temperatures(models, phase_type)
        
        # Estimate VRAM
        vram = self.registry.estimate_vram(models)
        
        reason = (
            f"Phase '{phase_type}' with tier '{adjusted_tier}', "
            f"time_factor={time_factor:.1f}, uncertainty={uncertainty_level:.1f}"
        )
        
        logger.debug(f"Model selection: {models} - {reason}")
        
        return ModelSelection(
            models=models,
            temperatures=temperatures,
            reason=reason,
            estimated_vram_mb=vram,
            tier=adjusted_tier,
        )
    
    def select_by_capability(
        self,
        required_capabilities: List[str],
        min_score: float = 7.0,
        count: int = 2
    ) -> ModelSelection:
        """
        Select models by capability requirements.
        
        Args:
            required_capabilities: Required capability names
            min_score: Minimum score for each capability
            count: Number of models to select
            
        Returns:
            ModelSelection
        """
        models = self.registry.select_by_capabilities(
            required_capabilities=required_capabilities,
            count=count,
            min_score=min_score
        )
        
        if not models:
            # Fallback to large tier
            models = self.registry.get_available_in_tier("large")[:count]
        
        temperatures = self._get_temperatures(models, capability=required_capabilities[0])
        vram = self.registry.estimate_vram(models)
        
        return ModelSelection(
            models=models,
            temperatures=temperatures,
            reason=f"capability-based: {required_capabilities}",
            estimated_vram_mb=vram,
            tier="capability_matched",
        )
    
    def select_for_effort(
        self,
        effort_level: str,
        phase_type: str = "default"
    ) -> ModelSelection:
        """
        Select models based on effort level preset.
        
        Args:
            effort_level: One of "quick", "standard", "thorough", "exhaustive"
            phase_type: Phase type for context
            
        Returns:
            ModelSelection
        """
        effort_configs = {
            "quick": {"tier": "small", "count": 1, "temp": 0.3},
            "standard": {"tier": "medium", "count": 2, "temp": 0.5},
            "thorough": {"tier": "large", "count": 2, "temp": 0.5},
            "exhaustive": {"tier": "reasoning", "count": 3, "temp": 0.6},
        }
        
        config = effort_configs.get(effort_level, effort_configs["standard"])
        
        models = self.registry.get_available_in_tier(config["tier"])[:config["count"]]
        
        if not models:
            models = self._get_fallback_models()[:config["count"]]
        
        temperatures = [config["temp"]] * len(models)
        vram = self.registry.estimate_vram(models)
        
        return ModelSelection(
            models=models,
            temperatures=temperatures,
            reason=f"effort_level={effort_level}",
            estimated_vram_mb=vram,
            tier=config["tier"],
        )
    
    def get_embedding_model(self) -> str:
        """Get the best available embedding model."""
        return self.registry.get_embedding_model()
    
    def _select_contraction_mode(self) -> ModelSelection:
        """Select for contraction mode (minimal resources)."""
        small_models = self.registry.get_available_in_tier("small")
        
        if small_models:
            model = small_models[0]
        else:
            model = "llama3.2:1b"  # Ultimate fallback
        
        info = self.registry.get_model(model)
        vram = info.vram_mb if info else 1300
        
        return ModelSelection(
            models=[model],
            temperatures=[0.3],  # Low temp for consistency in contraction
            reason="contraction_mode",
            estimated_vram_mb=vram,
            tier="small",
        )
    
    def _compute_time_factor(self, time_remaining: float) -> float:
        """
        Compute time pressure factor.
        
        0.0 = no time (use smallest)
        1.0 = plenty of time (can use large)
        """
        if time_remaining <= 5:
            return 0.0
        elif time_remaining <= 15:
            return 0.3
        elif time_remaining <= 30:
            return 0.6
        elif time_remaining <= 60:
            return 0.8
        else:
            return 1.0
    
    def _get_tier_for_phase(self, phase_type: str) -> str:
        """Get the base tier for a phase type."""
        phase_lower = phase_type.lower()
        
        if any(kw in phase_lower for kw in ["recon", "gather", "initial", "scout"]):
            return "small"
        elif any(kw in phase_lower for kw in ["deep", "thorough", "comprehensive"]):
            return "reasoning"
        elif any(kw in phase_lower for kw in ["synth", "final", "report", "conclusion"]):
            return "large"
        elif any(kw in phase_lower for kw in ["impl", "code", "build"]):
            return "medium"
        else:
            return "medium"
    
    def _get_tier_floor_for_importance(self, importance: float) -> Optional[str]:
        """
        Get minimum tier floor based on phase importance.
        
        Model-Aware Phase Stabilization: Prevents downgrading below
        the minimum tier required for the importance level.
        
        Args:
            importance: Phase importance score (0.0-1.0)
            
        Returns:
            Minimum tier name, or None if no floor
        """
        if importance >= 0.9:
            return "large"  # Synthesis, critical phases
        elif importance >= 0.7:
            return "medium"  # Implementation, design
        elif importance >= 0.5:
            return "medium"  # Testing, etc.
        else:
            return None  # No floor for low-importance phases
    
    def _adjust_tier(
        self,
        base_tier: str,
        time_factor: float,
        token_budget: int,
        min_tier_floor: Optional[str] = None
    ) -> str:
        """
        Adjust tier based on constraints with optional floor enforcement.
        
        Model-Aware Phase Stabilization: The min_tier_floor parameter prevents
        downgrading below a certain tier regardless of time/budget pressure.
        
        Args:
            base_tier: Initial tier from phase type
            time_factor: Time pressure factor (0-1)
            token_budget: Available token budget
            min_tier_floor: Minimum tier to allow (None = no floor)
            
        Returns:
            Adjusted tier name, respecting floor if set
        """
        tier_order = ["small", "medium", "large", "reasoning"]
        
        try:
            current_idx = tier_order.index(base_tier)
        except ValueError:
            current_idx = 1  # Default to medium
        
        # Calculate floor index if provided
        floor_idx = 0
        if min_tier_floor:
            try:
                floor_idx = tier_order.index(min_tier_floor)
            except ValueError:
                floor_idx = 0
        
        # Reduce tier if low time
        if time_factor < 0.3:
            current_idx = max(floor_idx, current_idx - 2)
        elif time_factor < 0.6:
            current_idx = max(floor_idx, current_idx - 1)
        
        # Reduce tier if low token budget
        if token_budget < 2000:
            current_idx = max(floor_idx, current_idx - 1)
        
        # Ensure we never go below the floor
        current_idx = max(floor_idx, current_idx)
        
        final_tier = tier_order[current_idx]
        
        # Log if floor was enforced
        if min_tier_floor and current_idx == floor_idx and base_tier != min_tier_floor:
            logger.debug(
                f"[MODEL_FLOOR] Tier floor '{min_tier_floor}' enforced "
                f"(would have been lower due to time_factor={time_factor:.2f}, "
                f"token_budget={token_budget})"
            )
        
        return final_tier
    
    def _compute_model_count(
        self,
        uncertainty: float,
        tier: str
    ) -> int:
        """Compute how many models to run based on uncertainty."""
        if tier == "small":
            # Small models are fast, can run more
            if uncertainty > 0.7:
                return min(3, self.max_concurrent)
            return 2
        elif tier == "reasoning":
            # Reasoning models are expensive, run fewer
            if uncertainty > 0.8:
                return 2
            return 1
        else:
            # Standard behavior
            if uncertainty > 0.6:
                return min(3, self.max_concurrent)
            return 2
    
    def _get_temperatures(
        self,
        models: List[str],
        phase_type: str = "",
        capability: str = ""
    ) -> List[float]:
        """Get temperatures for selected models."""
        temperatures = []
        
        for model in models:
            info = self.registry.get_model(model)
            if info:
                temp = info.default_temperature
            else:
                temp = self.default_temperature
            
            # Phase-based adjustments
            if phase_type:
                phase_lower = phase_type.lower()
                if "recon" in phase_lower or "scout" in phase_lower:
                    temp = min(temp, 0.4)  # More deterministic for recon
                elif "creative" in phase_lower or "brainstorm" in phase_lower:
                    temp = max(temp, 0.7)  # More creative
            
            temperatures.append(temp)
        
        return temperatures
    
    def _get_fallback_models(self) -> List[str]:
        """Get fallback models when preferred are unavailable."""
        # Try each tier until we find available models
        for tier in ["medium", "small", "large"]:
            models = self.registry.get_available_in_tier(tier)
            if models:
                return models
        
        # Ultimate fallback - return hardcoded common models
        return ["llama3.2:3b", "gemma3:4b"]


# Global instance
_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get the global model selector instance."""
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector

