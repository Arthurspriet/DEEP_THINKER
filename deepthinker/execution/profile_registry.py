"""
Execution profile registry for loading and managing execution profiles.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .execution_profile import ExecutionProfile


class ExecutionProfileRegistry:
    """
    Registry for execution profiles loaded from configuration.
    
    Loads profiles from YAML/JSON files and provides access to them.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize profile registry.
        
        Args:
            config_path: Path to profiles config file. If None, uses default location.
        """
        if config_path is None:
            # Default to profiles.yaml in execution directory
            config_path = Path(__file__).parent / "profiles.yaml"
        
        self.config_path = Path(config_path)
        self._profiles: Dict[str, ExecutionProfile] = {}
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load profiles from configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Profile config file not found: {self.config_path}"
            )
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if "profiles" not in config:
            raise ValueError("Config file must contain 'profiles' key")
        
        for name, profile_data in config["profiles"].items():
            profile_data["name"] = name
            profile = ExecutionProfile(**profile_data)
            self._profiles[name] = profile
    
    def get_profile(self, name: str) -> ExecutionProfile:
        """
        Get execution profile by name.
        
        Args:
            name: Profile name (e.g., "SAFE_ML")
            
        Returns:
            ExecutionProfile instance
            
        Raises:
            KeyError: If profile not found
        """
        if name not in self._profiles:
            raise KeyError(
                f"Profile '{name}' not found. Available: {list(self._profiles.keys())}"
            )
        return self._profiles[name]
    
    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self._profiles.keys())
    
    def has_profile(self, name: str) -> bool:
        """Check if a profile exists."""
        return name in self._profiles
    
    def get_default_profile(self) -> ExecutionProfile:
        """Get the default SAFE_ML profile."""
        return self.get_profile("SAFE_ML")


# Global registry instance
_default_registry: Optional[ExecutionProfileRegistry] = None


def get_default_registry() -> ExecutionProfileRegistry:
    """Get or create the default profile registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ExecutionProfileRegistry()
    return _default_registry

