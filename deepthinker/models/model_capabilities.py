"""
Model Capabilities Registry for DeepThinker 2.0 Dynamic Council Generator.

Loads model capability definitions from YAML and provides helper functions
for selecting models based on role requirements and constraints.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import yaml, with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


@dataclass
class ModelCapability:
    """Capability profile for a single model."""
    
    name: str
    reasoning: int = 5
    planning: int = 5
    coding: int = 5
    evaluation: int = 5
    creativity: int = 5
    speed: int = 5
    synthesis: int = 5
    research: int = 5
    vram_mb: int = 8000
    tier: str = "medium"
    
    def get_capability(self, capability_name: str) -> int:
        """Get a specific capability score."""
        return getattr(self, capability_name, 5)
    
    def get_score_for_role(self, primary: List[str], secondary: List[str] = None) -> float:
        """
        Calculate weighted score for a role.
        
        Primary capabilities are weighted at 2x, secondary at 1x.
        
        Args:
            primary: List of primary capability names
            secondary: Optional list of secondary capability names
            
        Returns:
            Weighted average score
        """
        if secondary is None:
            secondary = []
        
        total_weight = 0
        total_score = 0
        
        for cap in primary:
            total_score += self.get_capability(cap) * 2
            total_weight += 2
        
        for cap in secondary:
            total_score += self.get_capability(cap) * 1
            total_weight += 1
        
        return total_score / total_weight if total_weight > 0 else 5.0


@dataclass
class RoleRequirement:
    """Requirements for a specific council role."""
    
    role: str
    primary: List[str] = field(default_factory=list)
    secondary: List[str] = field(default_factory=list)
    temperature_range: Tuple[float, float] = (0.4, 0.6)


@dataclass
class PhaseModifier:
    """Phase-specific modifiers for model selection."""
    
    phase: str
    prefer_capabilities: List[str] = field(default_factory=list)
    model_count: int = 2
    use_largest: bool = False


class ModelCapabilitiesRegistry:
    """
    Registry for model capabilities.
    
    Loads capability definitions from YAML and provides methods for
    selecting models based on role requirements and constraints.
    """
    
    _instance: Optional["ModelCapabilitiesRegistry"] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the registry (only once)."""
        if self._initialized:
            return
        
        self.models: Dict[str, ModelCapability] = {}
        self.role_requirements: Dict[str, RoleRequirement] = {}
        self.phase_modifiers: Dict[str, PhaseModifier] = {}
        
        self._load_capabilities()
        self._initialized = True
    
    def _load_capabilities(self) -> None:
        """Load capabilities from YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available - using default capabilities")
            self._load_default_capabilities()
            return
        
        # Find the YAML file
        yaml_path = Path(__file__).parent / "model_capabilities.yaml"
        
        if not yaml_path.exists():
            logger.warning(f"Capabilities file not found at {yaml_path} - using defaults")
            self._load_default_capabilities()
            return
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            self._parse_yaml_data(data)
            logger.info(f"Loaded capabilities for {len(self.models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to load capabilities YAML: {e} - using defaults")
            self._load_default_capabilities()
    
    def _parse_yaml_data(self, data: Dict[str, Any]) -> None:
        """Parse YAML data into registry structures."""
        # Parse models
        models_data = data.get("models", {})
        for model_name, caps in models_data.items():
            if isinstance(caps, dict):
                self.models[model_name] = ModelCapability(
                    name=model_name,
                    reasoning=caps.get("reasoning", 5),
                    planning=caps.get("planning", 5),
                    coding=caps.get("coding", 5),
                    evaluation=caps.get("evaluation", 5),
                    creativity=caps.get("creativity", 5),
                    speed=caps.get("speed", 5),
                    synthesis=caps.get("synthesis", 5),
                    research=caps.get("research", 5),
                    vram_mb=caps.get("vram_mb", 8000),
                    tier=caps.get("tier", "medium"),
                )
        
        # Parse role requirements
        role_data = data.get("role_requirements", {})
        for role, reqs in role_data.items():
            if isinstance(reqs, dict):
                temp_range = reqs.get("temperature_range", [0.4, 0.6])
                self.role_requirements[role] = RoleRequirement(
                    role=role,
                    primary=reqs.get("primary", []),
                    secondary=reqs.get("secondary", []),
                    temperature_range=(temp_range[0], temp_range[1]),
                )
        
        # Parse phase modifiers
        phase_data = data.get("phase_modifiers", {})
        for phase, mods in phase_data.items():
            if isinstance(mods, dict):
                self.phase_modifiers[phase] = PhaseModifier(
                    phase=phase,
                    prefer_capabilities=mods.get("prefer_capabilities", []),
                    model_count=mods.get("model_count", 2),
                    use_largest=mods.get("use_largest", False),
                )
    
    def _load_default_capabilities(self) -> None:
        """Load hardcoded default capabilities."""
        # Default models
        self.models = {
            "gemma3:27b": ModelCapability(
                name="gemma3:27b", reasoning=9, planning=8, coding=7,
                evaluation=9, creativity=7, speed=3, synthesis=9, research=8,
                vram_mb=18000, tier="large"
            ),
            "cogito:14b": ModelCapability(
                name="cogito:14b", reasoning=9, planning=8, coding=8,
                evaluation=8, creativity=7, speed=5, synthesis=8, research=7,
                vram_mb=10000, tier="large"
            ),
            "gemma3:12b": ModelCapability(
                name="gemma3:12b", reasoning=7, planning=6, coding=6,
                evaluation=7, creativity=6, speed=7, synthesis=7, research=7,
                vram_mb=8000, tier="medium"
            ),
            "deepseek-r1:8b": ModelCapability(
                name="deepseek-r1:8b", reasoning=7, planning=5, coding=9,
                evaluation=7, creativity=5, speed=8, synthesis=6, research=6,
                vram_mb=6000, tier="medium"
            ),
            "mistral:instruct": ModelCapability(
                name="mistral:instruct", reasoning=7, planning=6, coding=6,
                evaluation=5, creativity=9, speed=8, synthesis=7, research=7,
                vram_mb=5500, tier="medium"
            ),
            "llama3.2:3b": ModelCapability(
                name="llama3.2:3b", reasoning=5, planning=4, coding=5,
                evaluation=5, creativity=5, speed=9, synthesis=5, research=5,
                vram_mb=2500, tier="small"
            ),
            "llama3.2:1b": ModelCapability(
                name="llama3.2:1b", reasoning=4, planning=3, coding=4,
                evaluation=4, creativity=4, speed=10, synthesis=4, research=4,
                vram_mb=1200, tier="small"
            ),
        }
        
        # Default role requirements
        self.role_requirements = {
            "planner": RoleRequirement("planner", ["reasoning", "planning"], ["synthesis"], (0.6, 0.7)),
            "researcher": RoleRequirement("researcher", ["research", "reasoning"], ["evaluation"], (0.5, 0.6)),
            "coder": RoleRequirement("coder", ["coding", "reasoning"], ["evaluation"], (0.15, 0.25)),
            "evaluator": RoleRequirement("evaluator", ["evaluation", "reasoning"], ["synthesis"], (0.0, 0.1)),
            "simulation": RoleRequirement("simulation", ["creativity", "reasoning"], ["evaluation"], (0.7, 0.8)),
            "optimist": RoleRequirement("optimist", ["creativity", "reasoning"], ["synthesis"], (0.55, 0.65)),
            "skeptic": RoleRequirement("skeptic", ["evaluation", "reasoning"], ["research"], (0.4, 0.5)),
        }
        
        # Default phase modifiers
        self.phase_modifiers = {
            "recon": PhaseModifier("recon", ["research", "speed"], 2, False),
            "analysis_planning": PhaseModifier("analysis_planning", ["reasoning", "planning"], 3, False),
            "deep_analysis": PhaseModifier("deep_analysis", ["reasoning", "evaluation"], 2, False),
            "synthesis": PhaseModifier("synthesis", ["synthesis", "reasoning"], 1, True),
        }
    
    def get_model_capabilities(self, model_name: str) -> Optional[ModelCapability]:
        """
        Get capabilities for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelCapability if found, None otherwise
        """
        return self.models.get(model_name)
    
    def get_all_models(self) -> List[str]:
        """Get list of all registered model names."""
        return list(self.models.keys())
    
    def get_models_by_tier(self, tier: str) -> List[str]:
        """
        Get all models of a specific tier.
        
        Args:
            tier: "small", "medium", or "large"
            
        Returns:
            List of model names in that tier
        """
        return [
            name for name, cap in self.models.items()
            if cap.tier == tier
        ]
    
    def select_models_for_role(
        self,
        role: str,
        top_k: int = 2,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Select the best models for a given role.
        
        Args:
            role: Council role (planner, researcher, coder, etc.)
            top_k: Number of models to select
            constraints: Optional constraints like max_vram, prefer_tier, etc.
            
        Returns:
            List of (model_name, recommended_temperature) tuples
        """
        constraints = constraints or {}
        max_vram = constraints.get("max_vram", float('inf'))
        prefer_tier = constraints.get("prefer_tier")
        exclude_models = constraints.get("exclude_models", [])
        require_tier = constraints.get("require_tier")
        
        # Get role requirements
        role_req = self.role_requirements.get(role)
        if role_req is None:
            logger.warning(f"Unknown role: {role} - using default requirements")
            role_req = RoleRequirement(role, ["reasoning"], [], (0.4, 0.6))
        
        # Score all models
        scored_models: List[Tuple[str, float, ModelCapability]] = []
        
        for model_name, cap in self.models.items():
            # Apply filters
            if model_name in exclude_models:
                continue
            if cap.vram_mb > max_vram:
                continue
            if require_tier and cap.tier != require_tier:
                continue
            
            # Calculate score
            score = cap.get_score_for_role(role_req.primary, role_req.secondary)
            
            # Apply tier preference bonus
            if prefer_tier and cap.tier == prefer_tier:
                score += 1.0
            
            scored_models.append((model_name, score, cap))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # If no models passed filters, fallback to all models (ignore constraints)
        if not scored_models:
            logger.debug(
                f"No models passed filters for role {role} with constraints {constraints}, "
                f"falling back to all available models"
            )
            for model_name, cap in self.models.items():
                score = cap.get_score_for_role(role_req.primary, role_req.secondary)
                scored_models.append((model_name, score, cap))
            scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Select top_k and assign temperatures
        result: List[Tuple[str, float]] = []
        temp_low, temp_high = role_req.temperature_range
        
        for i, (model_name, score, cap) in enumerate(scored_models[:top_k]):
            # Vary temperature slightly within range for diversity
            temp_offset = (i / max(top_k - 1, 1)) * (temp_high - temp_low) if top_k > 1 else 0
            temperature = temp_low + temp_offset
            result.append((model_name, round(temperature, 2)))
        
        return result
    
    def select_models_for_phase(
        self,
        role: str,
        phase: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Select models considering both role and phase requirements.
        
        Args:
            role: Council role
            phase: Mission phase (recon, analysis_planning, deep_analysis, synthesis)
            constraints: Optional constraints
            
        Returns:
            List of (model_name, temperature) tuples
        """
        constraints = constraints or {}
        
        # Get phase modifier
        phase_mod = self.phase_modifiers.get(phase)
        
        # Determine model count
        top_k = constraints.get("model_count", 2)
        if phase_mod:
            top_k = phase_mod.model_count
        
        # Apply phase preferences
        if phase_mod and phase_mod.use_largest:
            constraints["prefer_tier"] = "large"
        
        return self.select_models_for_role(role, top_k=top_k, constraints=constraints)
    
    def get_role_requirements(self, role: str) -> Optional[RoleRequirement]:
        """Get requirements for a role."""
        return self.role_requirements.get(role)
    
    def get_phase_modifier(self, phase: str) -> Optional[PhaseModifier]:
        """Get modifier for a phase."""
        return self.phase_modifiers.get(phase)


# Module-level convenience functions
_registry: Optional[ModelCapabilitiesRegistry] = None


def get_registry() -> ModelCapabilitiesRegistry:
    """Get the singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelCapabilitiesRegistry()
    return _registry


def get_model_capabilities(model_name: str) -> Optional[ModelCapability]:
    """
    Get capabilities for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelCapability if found, None otherwise
    """
    return get_registry().get_model_capabilities(model_name)


def select_models_for_role(
    role: str,
    top_k: int = 2,
    constraints: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float]]:
    """
    Select the best models for a given role.
    
    Args:
        role: Council role (planner, researcher, coder, etc.)
        top_k: Number of models to select
        constraints: Optional constraints
        
    Returns:
        List of (model_name, temperature) tuples
    """
    return get_registry().select_models_for_role(role, top_k, constraints)

