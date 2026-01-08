"""
Configuration for dataset-based evaluation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DockerConfig:
    """
    Configuration for Docker-based secure code execution.
    
    Attributes:
        image_name: Docker image to use for execution (legacy, use execution_profile)
        memory_limit: Memory limit (legacy, use execution_profile)
        cpu_limit: CPU limit (legacy, use execution_profile)
        enable_security_scanning: Whether to scan code (legacy, use execution_profile)
        auto_build_image: Automatically build image if not found
        execution_profile: Execution profile name (preferred, defaults to "SAFE_ML")
    """
    
    image_name: str = "deepthinker-sandbox:latest"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    enable_security_scanning: bool = True
    auto_build_image: bool = True
    execution_profile: Optional[str] = "SAFE_ML"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.cpu_limit <= 0:
            raise ValueError(f"cpu_limit must be positive, got {self.cpu_limit}")


@dataclass
class DataConfig:
    """
    Configuration for evaluating generated code on real datasets.
    
    Attributes:
        data_path: Path to dataset file (CSV or JSON)
        task_type: Type of ML task ("classification" or "regression")
        target_column: Name of target column (None = last column)
        test_split_ratio: Fraction of data for testing (0-1)
        metric_weight: Weight for metrics in combined score (0-1)
        random_seed: Random seed for reproducibility
        execution_timeout: Maximum execution time in seconds
        execution_backend: Backend for code execution ("subprocess" or "docker")
        docker_config: Configuration for Docker execution (if backend is "docker")
    """
    
    data_path: str
    task_type: str
    target_column: Optional[str] = None
    test_split_ratio: float = 0.2
    metric_weight: float = 0.5
    random_seed: int = 42
    execution_timeout: int = 30
    execution_backend: str = "subprocess"
    docker_config: Optional[DockerConfig] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate task type
        valid_task_types = ["classification", "regression"]
        if self.task_type not in valid_task_types:
            raise ValueError(
                f"task_type must be one of {valid_task_types}, got '{self.task_type}'"
            )
        
        # Validate test split ratio
        if not 0 < self.test_split_ratio < 1:
            raise ValueError(
                f"test_split_ratio must be between 0 and 1, got {self.test_split_ratio}"
            )
        
        # Validate metric weight
        if not 0 <= self.metric_weight <= 1:
            raise ValueError(
                f"metric_weight must be between 0 and 1, got {self.metric_weight}"
            )
        
        # Validate timeout
        if self.execution_timeout <= 0:
            raise ValueError(
                f"execution_timeout must be positive, got {self.execution_timeout}"
            )
        
        # Validate execution backend
        valid_backends = ["subprocess", "docker"]
        if self.execution_backend not in valid_backends:
            raise ValueError(
                f"execution_backend must be one of {valid_backends}, got '{self.execution_backend}'"
            )
        
        # Initialize default docker config if using docker backend
        if self.execution_backend == "docker" and self.docker_config is None:
            self.docker_config = DockerConfig()
    
    def is_enabled(self) -> bool:
        """Check if metric evaluation is enabled (data_path provided)."""
        return bool(self.data_path)

