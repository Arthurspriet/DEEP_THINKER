"""
Council Model Configuration for DeepThinker 2.0.

Defines per-council model pools with associated temperature settings.
Each council uses multiple LLMs that collaborate, critique, vote, and reach consensus.
"""

from typing import List, Tuple
from dataclasses import dataclass, field


# Model pool definitions: List of (model_name, temperature) tuples
# Temperature controls creativity/determinism balance per model

PLANNER_MODELS: List[Tuple[str, float]] = [
    ("gemma3:27b", 0.7),      # High reasoning, balanced creativity
    ("cogito:14b", 0.6),       # Strong reasoning specialist
    ("gemma3:12b", 0.65),      # Efficient reasoning
]

CODER_MODELS: List[Tuple[str, float]] = [
    ("deepseek-r1:8b", 0.2),   # Code specialist, low temp for precision
    ("cogito:14b", 0.25),      # Reasoning-enhanced coding
    ("gemma3:12b", 0.15),      # Deterministic code generation
]

EVALUATOR_MODELS: List[Tuple[str, float]] = [
    ("gemma3:27b", 0.0),       # Deterministic evaluation
    ("cogito:14b", 0.1),       # Near-deterministic analysis
    ("gemma3:12b", 0.05),      # Consistent scoring
]

SIMULATION_MODELS: List[Tuple[str, float]] = [
    ("mistral:instruct", 0.8), # Creative scenario generation
    ("gemma3:27b", 0.7),       # Balanced edge-case exploration
]

RESEARCHER_MODELS: List[Tuple[str, float]] = [
    ("gemma3:12b", 0.5),       # Balanced research synthesis
    ("cogito:14b", 0.6),       # Reasoning-focused research
]

# Multi-view council models (for autonomous reasoning)
OPTIMIST_MODELS: List[Tuple[str, float]] = [
    ("gemma3:12b", 0.6),       # Balanced positive analysis
    ("cogito:14b", 0.55),      # Reasoning with optimistic lens
]

SKEPTIC_MODELS: List[Tuple[str, float]] = [
    ("gemma3:12b", 0.4),       # More deterministic critical analysis
    ("cogito:14b", 0.45),      # Rigorous skeptical reasoning
]

# Single-model roles (highest reasoning depth)
ARBITER_MODEL: str = "gemma3:27b"
META_PLANNER_MODEL: str = "gemma3:27b"

# Embedding model for semantic distance consensus
EMBEDDING_MODEL: str = "qwen3-embedding:4b"


@dataclass
class CouncilModelPool:
    """
    Configuration for a council's model pool.
    
    Attributes:
        name: Council name identifier
        models: List of (model_name, temperature) tuples
        default_consensus: Default consensus algorithm for this council
    """
    
    name: str
    models: List[Tuple[str, float]] = field(default_factory=list)
    default_consensus: str = "voting"
    
    def get_model_names(self) -> List[str]:
        """Return list of model names only."""
        return [model for model, _ in self.models]
    
    def get_temperatures(self) -> List[float]:
        """Return list of temperatures only."""
        return [temp for _, temp in self.models]
    
    def get_model_config(self, model_name: str) -> Tuple[str, float]:
        """Get specific model configuration by name."""
        for model, temp in self.models:
            if model == model_name:
                return (model, temp)
        raise ValueError(f"Model {model_name} not found in pool {self.name}")


@dataclass 
class CouncilConfig:
    """
    Complete configuration for all councils in DeepThinker 2.0.
    
    Provides centralized access to all council model pools and settings.
    Includes multi-view councils for autonomous reasoning.
    """
    
    planner: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="planner",
        models=PLANNER_MODELS,
        default_consensus="weighted_blend"
    ))
    
    researcher: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="researcher", 
        models=RESEARCHER_MODELS,
        default_consensus="voting"
    ))
    
    coder: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="coder",
        models=CODER_MODELS,
        default_consensus="critique_exchange"
    ))
    
    evaluator: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="evaluator",
        models=EVALUATOR_MODELS,
        default_consensus="weighted_blend"
    ))
    
    simulation: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="simulation",
        models=SIMULATION_MODELS,
        default_consensus="semantic_distance"
    ))
    
    # Multi-view councils for autonomous reasoning
    optimist: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="optimist",
        models=OPTIMIST_MODELS,
        default_consensus="weighted_blend"
    ))
    
    skeptic: CouncilModelPool = field(default_factory=lambda: CouncilModelPool(
        name="skeptic",
        models=SKEPTIC_MODELS,
        default_consensus="weighted_blend"
    ))
    
    arbiter_model: str = ARBITER_MODEL
    meta_planner_model: str = META_PLANNER_MODEL
    embedding_model: str = EMBEDDING_MODEL
    
    def get_council(self, council_name: str) -> CouncilModelPool:
        """Get council configuration by name."""
        councils = {
            "planner": self.planner,
            "researcher": self.researcher,
            "coder": self.coder,
            "evaluator": self.evaluator,
            "simulation": self.simulation,
            "optimist": self.optimist,
            "skeptic": self.skeptic,
        }
        if council_name not in councils:
            raise ValueError(f"Unknown council: {council_name}")
        return councils[council_name]
    
    def get_all_councils(self) -> List[CouncilModelPool]:
        """Return all council configurations."""
        return [
            self.planner,
            self.researcher,
            self.coder,
            self.evaluator,
            self.simulation,
            self.optimist,
            self.skeptic,
        ]
    
    def get_multiview_councils(self) -> List[CouncilModelPool]:
        """Return multi-view council configurations."""
        return [
            self.optimist,
            self.skeptic,
        ]


# Global default configuration
DEFAULT_COUNCIL_CONFIG = CouncilConfig()

