"""
Execution profile definitions for tiered capability system.

Profiles define resource limits, security policies, and capabilities
for code execution environments.
"""

from dataclasses import dataclass
from typing import Literal, List


@dataclass
class ExecutionProfile:
    """
    Defines execution environment capabilities and constraints.
    
    Attributes:
        name: Profile name (e.g., "SAFE_ML", "GPU_ML")
        ram_limit: Memory limit (e.g., "512m", "32g")
        cpu_limit: CPU limit as fraction (e.g., 1.0 = 1 CPU)
        gpu_enabled: Whether GPU access is enabled
        network_policy: Network access policy (none, allowlist, proxy, full)
        network_allowlist: Allowed network domains (if allowlist policy)
        docker_image: Docker image to use for execution
        security_scan_level: Security scanning strictness (strict, warn, log, disabled)
        allowed_languages: List of allowed programming languages
        max_execution_time: Maximum execution time in seconds
    """
    
    name: str
    ram_limit: str
    cpu_limit: float
    gpu_enabled: bool
    network_policy: Literal["none", "allowlist", "proxy", "full"]
    network_allowlist: List[str]
    docker_image: str
    security_scan_level: Literal["strict", "warn", "log", "disabled"]
    allowed_languages: List[str]
    max_execution_time: int
    
    def __post_init__(self):
        """Validate profile configuration."""
        if self.cpu_limit <= 0:
            raise ValueError(f"cpu_limit must be positive, got {self.cpu_limit}")
        if self.max_execution_time <= 0:
            raise ValueError(f"max_execution_time must be positive, got {self.max_execution_time}")
        if self.network_policy == "allowlist" and not self.network_allowlist:
            raise ValueError("allowlist network_policy requires network_allowlist")
        if not self.allowed_languages:
            raise ValueError("allowed_languages cannot be empty")
    
    def allows_language(self, language: str) -> bool:
        """Check if a language is allowed in this profile."""
        return language.lower() in [lang.lower() for lang in self.allowed_languages]
    
    def has_network_access(self) -> bool:
        """Check if profile allows any network access."""
        return self.network_policy != "none"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "ram_limit": self.ram_limit,
            "cpu_limit": self.cpu_limit,
            "gpu_enabled": self.gpu_enabled,
            "network_policy": self.network_policy,
            "network_allowlist": self.network_allowlist,
            "docker_image": self.docker_image,
            "security_scan_level": self.security_scan_level,
            "allowed_languages": self.allowed_languages,
            "max_execution_time": self.max_execution_time,
        }

