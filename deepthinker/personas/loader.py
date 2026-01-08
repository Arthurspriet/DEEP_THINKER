"""
Persona Loader for DeepThinker 2.0 Dynamic Council Generator.

Loads persona definitions from markdown files and provides helper
functions for persona assignment to council members.
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# Default persona assignments per council type
DEFAULT_COUNCIL_PERSONAS: Dict[str, List[str]] = {
    "planner": ["strategist", "systems_thinker"],
    "researcher": ["evidence_hunter", "systems_thinker"],
    "coder": ["code_architect", "security_auditor"],
    "evaluator": ["skeptic", "evidence_hunter"],
    "simulation": ["stress_tester", "skeptic"],
    "optimist": ["optimist", "strategist"],
    "skeptic": ["skeptic", "evidence_hunter"],
}


class PersonaLoader:
    """
    Loader for persona definitions.
    
    Caches loaded personas for efficiency.
    """
    
    _instance: Optional["PersonaLoader"] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern for loader."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the loader."""
        if self._initialized:
            return
        
        self._cache: Dict[str, str] = {}
        self._personas_dir = Path(__file__).parent
        self._scan_available_personas()
        self._initialized = True
    
    def _scan_available_personas(self) -> None:
        """Scan for available persona files."""
        self._available_personas: List[str] = []
        
        if not self._personas_dir.exists():
            logger.warning(f"Personas directory not found: {self._personas_dir}")
            return
        
        for path in self._personas_dir.glob("*.md"):
            persona_name = path.stem
            self._available_personas.append(persona_name)
        
        logger.debug(f"Found {len(self._available_personas)} personas")
    
    def load(self, name: str) -> Optional[str]:
        """
        Load a persona by name.
        
        Args:
            name: Persona name (without .md extension)
            
        Returns:
            Persona text content, or None if not found
        """
        # Check cache
        if name in self._cache:
            return self._cache[name]
        
        # Try to load from file
        persona_path = self._personas_dir / f"{name}.md"
        
        if not persona_path.exists():
            logger.warning(f"Persona not found: {name}")
            return None
        
        try:
            with open(persona_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            self._cache[name] = content
            return content
            
        except Exception as e:
            logger.error(f"Failed to load persona {name}: {e}")
            return None
    
    def get_available(self) -> List[str]:
        """Get list of available persona names."""
        return list(self._available_personas)
    
    def get_default_for_council(self, council_type: str) -> List[str]:
        """
        Get default persona names for a council type.
        
        Args:
            council_type: Type of council (planner, researcher, etc.)
            
        Returns:
            List of default persona names for this council
        """
        return DEFAULT_COUNCIL_PERSONAS.get(council_type, [])
    
    def load_defaults_for_council(self, council_type: str) -> List[str]:
        """
        Load all default personas for a council type.
        
        Args:
            council_type: Type of council
            
        Returns:
            List of loaded persona texts
        """
        persona_names = self.get_default_for_council(council_type)
        personas = []
        
        for name in persona_names:
            content = self.load(name)
            if content:
                personas.append(content)
        
        return personas
    
    def clear_cache(self) -> None:
        """Clear the persona cache."""
        self._cache.clear()
    
    def reload(self) -> None:
        """Reload all personas by clearing cache and rescanning."""
        self.clear_cache()
        self._scan_available_personas()


# Module-level singleton instance
_loader: Optional[PersonaLoader] = None


def _get_loader() -> PersonaLoader:
    """Get the singleton loader instance."""
    global _loader
    if _loader is None:
        _loader = PersonaLoader()
    return _loader


def load_persona(name: str) -> Optional[str]:
    """
    Load a persona by name.
    
    Args:
        name: Persona name (without .md extension)
        
    Returns:
        Persona text content, or None if not found
    """
    return _get_loader().load(name)


def get_available_personas() -> List[str]:
    """Get list of available persona names."""
    return _get_loader().get_available()


def get_default_personas_for_council(council_type: str) -> List[str]:
    """
    Get default persona names for a council type.
    
    Args:
        council_type: Type of council (planner, researcher, etc.)
        
    Returns:
        List of default persona names for this council
    """
    return _get_loader().get_default_for_council(council_type)

