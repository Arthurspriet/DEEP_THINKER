"""
Configuration structures for simulation and scenario testing.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import json


@dataclass
class NoiseConfig:
    """
    Configuration for noise injection in scenarios.
    
    Attributes:
        type: Type of noise ("label_flip")
        probability: Probability of noise application (0-1)
    """
    
    type: str
    probability: float = 0.0
    
    def __post_init__(self):
        """Validate noise configuration."""
        valid_types = ["label_flip"]
        if self.type not in valid_types:
            raise ValueError(f"noise type must be one of {valid_types}, got '{self.type}'")
        
        if not 0 <= self.probability <= 1:
            raise ValueError(f"probability must be between 0 and 1, got {self.probability}")


@dataclass
class ScenarioConfig:
    """
    Configuration for a single simulation scenario.
    
    Attributes:
        name: Unique scenario identifier
        description: Human-readable description
        data_path: Optional path to dataset (None = use base dataset)
        data_filter: Optional pandas query string to filter data
        test_split_ratio: Train/test split ratio (None = use base config)
        noise_config: Optional noise injection configuration
        parameter_overrides: Optional dict of custom parameters
    """
    
    name: str
    description: str = ""
    data_path: Optional[str] = None
    data_filter: Optional[str] = None
    test_split_ratio: Optional[float] = None
    noise_config: Optional[NoiseConfig] = None
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate scenario configuration."""
        if not self.name:
            raise ValueError("scenario name cannot be empty")
        
        if self.test_split_ratio is not None:
            if not 0 < self.test_split_ratio < 1:
                raise ValueError(
                    f"test_split_ratio must be between 0 and 1, got {self.test_split_ratio}"
                )
        
        # Convert noise_config dict to NoiseConfig if needed
        if isinstance(self.noise_config, dict):
            self.noise_config = NoiseConfig(**self.noise_config)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        """Create ScenarioConfig from dictionary."""
        return cls(**data)


@dataclass
class SimulationConfig:
    """
    Top-level configuration for simulation engine.
    
    Attributes:
        mode: Simulation mode ("none", "basic", "scenarios")
        scenarios: List of scenario configurations
        validation_split: Validation split ratio for basic mode
        random_seed: Random seed for reproducibility
    """
    
    mode: str = "none"
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    validation_split: float = 0.1
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate simulation configuration."""
        valid_modes = ["none", "basic", "scenarios"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")
        
        if self.mode == "scenarios" and not self.scenarios:
            raise ValueError("mode 'scenarios' requires at least one scenario definition")
        
        if not 0 < self.validation_split < 1:
            raise ValueError(
                f"validation_split must be between 0 and 1, got {self.validation_split}"
            )
        
        # Convert scenario dicts to ScenarioConfig objects if needed
        for i, scenario in enumerate(self.scenarios):
            if isinstance(scenario, dict):
                self.scenarios[i] = ScenarioConfig.from_dict(scenario)
    
    def is_enabled(self) -> bool:
        """Check if simulation is enabled."""
        return self.mode != "none"
    
    @classmethod
    def from_json_file(cls, file_path: str) -> "SimulationConfig":
        """
        Load simulation configuration from JSON file.
        
        Args:
            file_path: Path to JSON configuration file
            
        Returns:
            SimulationConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Simulation config file not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def create_basic_mode(cls, validation_split: float = 0.1) -> "SimulationConfig":
        """
        Create a basic mode configuration with default validation scenario.
        
        Args:
            validation_split: Validation split ratio
            
        Returns:
            SimulationConfig for basic mode
        """
        return cls(
            mode="basic",
            validation_split=validation_split,
            scenarios=[
                ScenarioConfig(
                    name="validation",
                    description="Held-out validation set performance"
                )
            ]
        )
    
    @classmethod
    def create_disabled(cls) -> "SimulationConfig":
        """Create a disabled simulation configuration."""
        return cls(mode="none")

