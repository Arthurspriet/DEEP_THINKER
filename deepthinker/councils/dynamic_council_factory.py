"""
Dynamic Council Factory for DeepThinker 2.0.

Builds council configurations at runtime based on mission, phase, and context.
Selects models, temperatures, personas, and consensus types dynamically while
maintaining full compatibility with existing static council architecture.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CouncilDefinition:
    """
    Complete definition for dynamically configuring a council.
    
    Attributes:
        name: Human-readable council name
        council_type: Type identifier (planner, researcher, coder, etc.)
        models: List of (model_name, temperature, persona_name) tuples
        consensus_type: Consensus algorithm to use
        output_type: Expected output type name
        phase: Mission phase this council is configured for
        additional_context: Optional additional context for the council
    """
    
    name: str
    council_type: str
    models: List[Tuple[str, float, Optional[str]]] = field(default_factory=list)
    consensus_type: str = "weighted_blend"
    output_type: str = "str"
    phase: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
    
    def get_model_configs(self) -> List[Tuple[str, float]]:
        """
        Get model configurations as (name, temperature) tuples.
        
        Compatible with existing ModelPool interface.
        """
        return [(model, temp) for model, temp, _ in self.models]
    
    def get_persona_for_model(self, model_name: str) -> Optional[str]:
        """Get the persona assigned to a specific model."""
        for model, _, persona in self.models:
            if model == model_name:
                return persona
        return None
    
    def get_all_personas(self) -> List[str]:
        """Get list of all assigned persona names."""
        return [persona for _, _, persona in self.models if persona]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "name": self.name,
            "council_type": self.council_type,
            "models": [
                {"model": m, "temperature": t, "persona": p}
                for m, t, p in self.models
            ],
            "consensus_type": self.consensus_type,
            "output_type": self.output_type,
            "phase": self.phase,
        }


# Default consensus types for each council type
DEFAULT_CONSENSUS_TYPES = {
    "planner": "weighted_blend",
    "researcher": "voting",
    "coder": "critique_exchange",
    "evaluator": "weighted_blend",
    "simulation": "semantic_distance",
    "optimist": "weighted_blend",
    "skeptic": "weighted_blend",
}

# Default output types for each council type
DEFAULT_OUTPUT_TYPES = {
    "planner": "WorkflowPlan",
    "researcher": "ResearchFindings",
    "coder": "CodeOutput",
    "evaluator": "EvaluationResult",
    "simulation": "SimulationFindings",
    "optimist": "OptimistPerspective",
    "skeptic": "SkepticPerspective",
}


class DynamicCouncilFactory:
    """
    Factory for creating dynamic council configurations.
    
    Uses the model capability registry and persona library to build
    optimal council configurations based on:
    - Mission objective
    - Current phase
    - Council type
    - Difficulty/uncertainty metrics
    - Time/effort constraints
    
    Maintains compatibility with static council configurations by returning
    None when dynamic configuration is not possible or appropriate.
    """
    
    def __init__(
        self,
        capabilities_registry=None,
        persona_loader=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the dynamic council factory.
        
        Args:
            capabilities_registry: Model capabilities registry (auto-loaded if None)
            persona_loader: Persona loader (auto-loaded if None)
            logger: Optional custom logger
        """
        self._logger = logger or logging.getLogger(__name__)
        
        # Lazy-load the capabilities registry
        self._capabilities_registry = capabilities_registry
        self._registry_initialized = capabilities_registry is not None
        
        # Lazy-load the persona loader
        self._persona_loader = persona_loader
        self._loader_initialized = persona_loader is not None
    
    def _ensure_registry(self) -> bool:
        """Ensure the capabilities registry is loaded."""
        if self._registry_initialized:
            return self._capabilities_registry is not None
        
        try:
            from ..models.model_capabilities import get_registry
            self._capabilities_registry = get_registry()
            self._registry_initialized = True
            return True
        except ImportError as e:
            self._logger.warning(f"Failed to import model capabilities: {e}")
            self._registry_initialized = True
            return False
    
    def _ensure_persona_loader(self) -> bool:
        """Ensure the persona loader is loaded."""
        if self._loader_initialized:
            return self._persona_loader is not None
        
        try:
            from ..personas.loader import PersonaLoader
            self._persona_loader = PersonaLoader()
            self._loader_initialized = True
            return True
        except ImportError as e:
            self._logger.warning(f"Failed to import persona loader: {e}")
            self._loader_initialized = True
            return False
    
    def build_council_definition(
        self,
        council_type: str,
        phase: str,
        mission_objective: str,
        difficulty: Optional[float] = None,
        uncertainty: Optional[float] = None,
        time_budget: Optional[float] = None,
        available_vram: Optional[int] = None,
        prefer_personas: Optional[List[str]] = None,
    ) -> Optional[CouncilDefinition]:
        """
        Build a dynamic council definition.
        
        Args:
            council_type: Type of council (planner, researcher, etc.)
            phase: Current mission phase
            mission_objective: The mission objective text
            difficulty: Optional difficulty score (0-1)
            uncertainty: Optional uncertainty score (0-1)
            time_budget: Optional remaining time in minutes
            available_vram: Optional available GPU VRAM in MB
            prefer_personas: Optional list of preferred persona names
            
        Returns:
            CouncilDefinition if successful, None to fall back to static config
        """
        try:
            # Ensure dependencies are loaded
            if not self._ensure_registry():
                self._logger.warning("Capabilities registry not available - using static config")
                return None
            
            # Normalize phase name to internal format
            normalized_phase = self._normalize_phase(phase)
            
            # Determine constraints based on context
            constraints = self._build_constraints(
                difficulty=difficulty,
                uncertainty=uncertainty,
                time_budget=time_budget,
                available_vram=available_vram,
            )
            
            # Select models based on role and phase
            models_with_temps = self._select_models(
                council_type=council_type,
                phase=normalized_phase,
                constraints=constraints,
            )
            
            if not models_with_temps:
                self._logger.warning(f"No models selected for {council_type} - using static config")
                return None
            
            # Assign personas to models
            models_with_personas = self._assign_personas(
                council_type=council_type,
                models_with_temps=models_with_temps,
                prefer_personas=prefer_personas,
            )
            
            # Determine consensus type
            consensus_type = self._select_consensus(
                council_type=council_type,
                phase=normalized_phase,
                uncertainty=uncertainty,
            )
            
            # Build the council definition
            definition = CouncilDefinition(
                name=f"dynamic_{council_type}",
                council_type=council_type,
                models=models_with_personas,
                consensus_type=consensus_type,
                output_type=DEFAULT_OUTPUT_TYPES.get(council_type, "str"),
                phase=phase,
                additional_context={
                    "difficulty": difficulty,
                    "uncertainty": uncertainty,
                    "time_budget": time_budget,
                }
            )
            
            self._log_definition(definition)
            return definition
            
        except Exception as e:
            self._logger.error(f"Failed to build council definition: {e}")
            return None
    
    def _normalize_phase(self, phase: str) -> str:
        """Normalize phase name to internal format."""
        phase_lower = phase.lower()
        
        if any(kw in phase_lower for kw in ["recon", "context", "gather"]):
            return "recon"
        elif any(kw in phase_lower for kw in ["analysis", "plan", "design", "strategy"]):
            return "analysis_planning"
        elif any(kw in phase_lower for kw in ["deep", "implement", "code", "build"]):
            return "deep_analysis"
        elif any(kw in phase_lower for kw in ["synthesis", "report", "final", "summary"]):
            return "synthesis"
        else:
            return "analysis_planning"  # Default
    
    def _build_constraints(
        self,
        difficulty: Optional[float],
        uncertainty: Optional[float],
        time_budget: Optional[float],
        available_vram: Optional[int],
    ) -> Dict[str, Any]:
        """Build constraints dictionary for model selection."""
        constraints = {}
        
        # VRAM constraint - only apply if we have meaningful available memory
        # Ignore if available_vram is too low (would filter out all models)
        MIN_VRAM_THRESHOLD = 1000  # 1GB minimum to apply constraint
        if available_vram is not None and available_vram >= MIN_VRAM_THRESHOLD:
            constraints["max_vram"] = available_vram
        elif available_vram is not None and available_vram < MIN_VRAM_THRESHOLD:
            self._logger.debug(
                f"Ignoring VRAM constraint ({available_vram}MB) - below minimum threshold"
            )
        
        # Time pressure adjustments
        if time_budget is not None:
            if time_budget < 2.0:
                # Very limited time - use small/fast models
                constraints["require_tier"] = "small"
            elif time_budget < 5.0:
                # Limited time - prefer medium models
                constraints["prefer_tier"] = "medium"
        
        # Difficulty/uncertainty adjustments
        if difficulty is not None and uncertainty is not None:
            if difficulty > 0.7 and uncertainty > 0.7:
                # Hard and uncertain - use more models
                constraints["model_count"] = 3
                constraints["prefer_tier"] = "large"
            elif difficulty < 0.3 and uncertainty < 0.3:
                # Easy and certain - fewer models, faster
                constraints["model_count"] = 1
        
        return constraints
    
    def _select_models(
        self,
        council_type: str,
        phase: str,
        constraints: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Select models for the council, filtering out embedding models."""
        if self._capabilities_registry is None:
            return []
        
        # Get model count (default 2)
        model_count = constraints.pop("model_count", 2)
        
        # Use phase-aware selection if available
        models = self._capabilities_registry.select_models_for_phase(
            role=council_type,
            phase=phase,
            constraints=constraints.copy(),  # Use copy to preserve original
        )
        
        # If no models found with constraints, try without constraints
        if not models:
            self._logger.debug(
                f"No models found with constraints for {council_type}/{phase}, "
                f"retrying without constraints"
            )
            models = self._capabilities_registry.select_models_for_role(
                role=council_type,
                top_k=model_count,
                constraints=None,  # No constraints
            )
        
        # Filter out embedding models for non-embedding councils
        # Embedding models cannot do text generation
        if council_type != "embedding":
            models = self._filter_out_embedding_models(models)
        
        # Limit to requested count
        return models[:model_count]
    
    def _filter_out_embedding_models(
        self,
        models: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Filter out embedding-only models that cannot do text generation.
        
        Embedding models like qwen3-embedding, snowflake-arctic-embed, etc.
        should not be used for text generation councils.
        
        Args:
            models: List of (model_name, temperature) tuples
            
        Returns:
            Filtered list without embedding models
        """
        embedding_patterns = [
            "embedding",
            "embed",
            "arctic-embed",
            "nomic-embed",
            "mxbai-embed",
            "bge-",
            "e5-",
        ]
        
        filtered = []
        for model_name, temp in models:
            model_lower = model_name.lower()
            is_embedding = any(pattern in model_lower for pattern in embedding_patterns)
            
            if is_embedding:
                self._logger.debug(
                    f"Filtering out embedding model {model_name} from text generation council"
                )
            else:
                filtered.append((model_name, temp))
        
        if len(filtered) < len(models):
            self._logger.info(
                f"Filtered {len(models) - len(filtered)} embedding models from selection"
            )
        
        return filtered
    
    def _assign_personas(
        self,
        council_type: str,
        models_with_temps: List[Tuple[str, float]],
        prefer_personas: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, Optional[str]]]:
        """Assign personas to models."""
        # Get default personas for this council type
        if prefer_personas:
            persona_names = prefer_personas
        elif self._ensure_persona_loader() and self._persona_loader is not None:
            persona_names = self._persona_loader.get_default_for_council(council_type)
        else:
            persona_names = []
        
        result = []
        for i, (model, temp) in enumerate(models_with_temps):
            # Assign persona if available
            persona = persona_names[i] if i < len(persona_names) else None
            result.append((model, temp, persona))
        
        return result
    
    def _select_consensus(
        self,
        council_type: str,
        phase: str,
        uncertainty: Optional[float] = None,
    ) -> str:
        """Select consensus algorithm based on context."""
        # Start with default for council type
        consensus = DEFAULT_CONSENSUS_TYPES.get(council_type, "weighted_blend")
        
        # Phase-specific overrides
        if phase == "synthesis":
            # Synthesis phase benefits from weighted blend
            consensus = "weighted_blend"
        elif phase == "deep_analysis" and uncertainty is not None and uncertainty > 0.7:
            # High uncertainty - use semantic distance to filter outliers
            consensus = "semantic_distance"
        
        return consensus
    
    def _log_definition(self, definition: CouncilDefinition) -> None:
        """Log the council definition for debugging with Rich formatting if available."""
        model_strs = [
            f"{m}@{t:.2f}" + (f"[{p}]" if p else "")
            for m, t, p in definition.models
        ]
        
        # Try to use verbose logger for Rich output
        try:
            from ..cli import verbose_logger
            if verbose_logger and verbose_logger.enabled:
                verbose_logger.log_council_definition(definition.to_dict())
                return
        except ImportError:
            pass
        
        # Fallback to standard logging
        self._logger.info(
            f"[DynamicCouncilFactory] Built {definition.council_type} council: "
            f"models={model_strs}, consensus={definition.consensus_type}, "
            f"phase={definition.phase}"
        )
        
        # Also log details at debug level
        self._logger.debug(
            f"  Difficulty: {definition.additional_context.get('difficulty') if definition.additional_context else 'N/A'}, "
            f"Uncertainty: {definition.additional_context.get('uncertainty') if definition.additional_context else 'N/A'}, "
            f"Time budget: {definition.additional_context.get('time_budget') if definition.additional_context else 'N/A'}"
        )
    
    def get_default_definition(self, council_type: str) -> CouncilDefinition:
        """
        Get a default definition for a council type.
        
        This is a fallback when dynamic configuration is not desired
        but you still want a CouncilDefinition object.
        
        Args:
            council_type: Type of council
            
        Returns:
            Default CouncilDefinition
        """
        # Use default models from model_capabilities if available
        default_models = []
        
        if self._ensure_registry() and self._capabilities_registry is not None:
            models_temps = self._capabilities_registry.select_models_for_role(
                council_type, top_k=2
            )
            default_models = [(m, t, None) for m, t in models_temps]
        
        # Fallback hardcoded defaults
        if not default_models:
            default_models = [
                ("gemma3:12b", 0.5, None),
                ("cogito:14b", 0.5, None),
            ]
        
        return CouncilDefinition(
            name=f"default_{council_type}",
            council_type=council_type,
            models=default_models,
            consensus_type=DEFAULT_CONSENSUS_TYPES.get(council_type, "weighted_blend"),
            output_type=DEFAULT_OUTPUT_TYPES.get(council_type, "str"),
        )

