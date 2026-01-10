"""
Persona Loader for DeepThinker 2.0 Dynamic Council Generator.

Loads persona definitions from markdown files and provides helper
functions for persona assignment to council members.

Enhanced with domain support for knowledge routing:
- Personas can specify knowledge domains in YAML frontmatter
- Domains are used by KnowledgeRouter to filter relevant knowledge

Frontmatter format:
---
domains: [economy, government]
knowledge_priority: high
---
Persona content here...
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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

# Default domain mappings for personas without frontmatter
# Aligned with KnowledgeRouter's PERSONA_DOMAIN_MAPPING
DEFAULT_PERSONA_DOMAINS: Dict[str, List[str]] = {
    "evidence_hunter": ["all"],
    "skeptic": ["military", "terrorism", "transnational", "government"],
    "strategist": ["economy", "government", "people", "introduction"],
    "systems_thinker": ["environment", "energy", "transportation", "communications"],
    "code_architect": [],
    "security_auditor": ["military", "terrorism", "communications", "government"],
    "economist": ["economy", "energy", "transportation", "people"],
    "optimist": ["economy", "introduction", "people", "environment"],
    "stress_tester": ["military", "terrorism", "transnational", "environment"],
    "reductionist": ["introduction", "geography", "people"],
}

# Frontmatter regex pattern
FRONTMATTER_PATTERN = re.compile(
    r'^---\s*\n(.*?)\n---\s*\n',
    re.DOTALL
)


@dataclass
class PersonaWithDomains:
    """
    A persona with associated knowledge domains.
    
    Attributes:
        name: Persona name (e.g., "strategist")
        text: Full persona prompt text
        domains: List of knowledge domains this persona should receive
        knowledge_priority: Priority level for knowledge injection ("high", "medium", "low")
        metadata: Additional metadata from frontmatter
    """
    name: str
    text: str
    domains: List[str] = field(default_factory=list)
    knowledge_priority: str = "medium"
    metadata: Dict[str, str] = field(default_factory=dict)
    
    @property
    def has_domains(self) -> bool:
        """Check if persona has domain specifications."""
        return len(self.domains) > 0
    
    @property
    def wants_all_knowledge(self) -> bool:
        """Check if persona wants all knowledge domains."""
        return "all" in self.domains
    
    def get_prompt_text(self) -> str:
        """Get the persona text for prompt injection."""
        return self.text


def _parse_frontmatter(content: str) -> Tuple[Dict[str, any], str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Args:
        content: Full file content
        
    Returns:
        Tuple of (frontmatter_dict, remaining_content)
    """
    match = FRONTMATTER_PATTERN.match(content)
    
    if not match:
        return {}, content
    
    frontmatter_text = match.group(1)
    remaining = content[match.end():]
    
    # Simple YAML parsing (avoid dependency on pyyaml)
    frontmatter = {}
    
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if ':' not in line:
            continue
        
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        # Parse list values: [item1, item2]
        if value.startswith('[') and value.endswith(']'):
            items = value[1:-1].split(',')
            value = [item.strip().strip('"\'') for item in items if item.strip()]
        # Parse quoted strings
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        
        frontmatter[key] = value
    
    return frontmatter, remaining.strip()


class PersonaLoader:
    """
    Loader for persona definitions.
    
    Enhanced with domain support:
    - Parses YAML frontmatter for domain specifications
    - Returns structured PersonaWithDomains objects
    - Falls back to default domain mappings
    
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
        
        # Cache for raw text content
        self._cache: Dict[str, str] = {}
        # Cache for structured personas with domains
        self._structured_cache: Dict[str, PersonaWithDomains] = {}
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
        Load a persona by name (text only, for backward compatibility).
        
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
            
            # Parse frontmatter and extract just the text
            _, text_content = _parse_frontmatter(content)
            
            self._cache[name] = text_content
            return text_content
            
        except Exception as e:
            logger.error(f"Failed to load persona {name}: {e}")
            return None
    
    def load_with_domains(self, name: str) -> Optional[PersonaWithDomains]:
        """
        Load a persona with domain information.
        
        Parses YAML frontmatter for domain specifications.
        Falls back to default domain mappings if no frontmatter.
        
        Args:
            name: Persona name (without .md extension)
            
        Returns:
            PersonaWithDomains object, or None if not found
        """
        # Check structured cache
        if name in self._structured_cache:
            return self._structured_cache[name]
        
        # Try to load from file
        persona_path = self._personas_dir / f"{name}.md"
        
        if not persona_path.exists():
            logger.warning(f"Persona not found: {name}")
            return None
        
        try:
            with open(persona_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Parse frontmatter
            frontmatter, text_content = _parse_frontmatter(content)
            
            # Extract domains (from frontmatter or defaults)
            domains = frontmatter.get('domains', [])
            if not domains:
                domains = DEFAULT_PERSONA_DOMAINS.get(name, ["all"])
            
            # Extract knowledge priority
            knowledge_priority = frontmatter.get('knowledge_priority', 'medium')
            
            # Build metadata dict (excluding domains and priority)
            metadata = {
                k: v for k, v in frontmatter.items()
                if k not in ('domains', 'knowledge_priority')
            }
            
            persona = PersonaWithDomains(
                name=name,
                text=text_content,
                domains=domains,
                knowledge_priority=knowledge_priority,
                metadata=metadata,
            )
            
            # Cache both structured and text versions
            self._structured_cache[name] = persona
            self._cache[name] = text_content
            
            logger.debug(f"Loaded persona '{name}' with domains: {domains}")
            
            return persona
            
        except Exception as e:
            logger.error(f"Failed to load persona {name}: {e}")
            return None
    
    def get_domains_for_persona(self, name: str) -> List[str]:
        """
        Get knowledge domains for a persona.
        
        Args:
            name: Persona name
            
        Returns:
            List of domain names
        """
        persona = self.load_with_domains(name)
        if persona:
            return persona.domains
        return DEFAULT_PERSONA_DOMAINS.get(name, ["all"])
    
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
    
    def load_defaults_with_domains(
        self,
        council_type: str
    ) -> List[PersonaWithDomains]:
        """
        Load all default personas with domain info for a council type.
        
        Args:
            council_type: Type of council
            
        Returns:
            List of PersonaWithDomains objects
        """
        persona_names = self.get_default_for_council(council_type)
        personas = []
        
        for name in persona_names:
            persona = self.load_with_domains(name)
            if persona:
                personas.append(persona)
        
        return personas
    
    def clear_cache(self) -> None:
        """Clear the persona cache."""
        self._cache.clear()
        self._structured_cache.clear()
    
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


def load_persona_with_domains(name: str) -> Optional[PersonaWithDomains]:
    """
    Load a persona with domain information.
    
    Args:
        name: Persona name (without .md extension)
        
    Returns:
        PersonaWithDomains object, or None if not found
    """
    return _get_loader().load_with_domains(name)


def get_domains_for_persona(name: str) -> List[str]:
    """
    Get knowledge domains for a persona.
    
    Args:
        name: Persona name
        
    Returns:
        List of domain names
    """
    return _get_loader().get_domains_for_persona(name)


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


def get_default_personas_with_domains(council_type: str) -> List[PersonaWithDomains]:
    """
    Get default personas with domain info for a council type.
    
    Args:
        council_type: Type of council
        
    Returns:
        List of PersonaWithDomains objects
    """
    return _get_loader().load_defaults_with_domains(council_type)
