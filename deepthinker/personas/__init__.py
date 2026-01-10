"""
Persona Library for DeepThinker 2.0 Dynamic Council Generator.

Personas are prompt-level role definitions that can be applied to model
calls to specialize their behavior for specific analytical perspectives.

Enhanced with domain support for knowledge routing:
- Personas can specify knowledge domains in YAML frontmatter
- Domains are used by KnowledgeRouter to filter relevant knowledge
"""

from .loader import (
    load_persona,
    load_persona_with_domains,
    get_available_personas,
    get_default_personas_for_council,
    get_default_personas_with_domains,
    get_domains_for_persona,
    PersonaLoader,
    PersonaWithDomains,
    DEFAULT_PERSONA_DOMAINS,
)

__all__ = [
    "load_persona",
    "load_persona_with_domains",
    "get_available_personas",
    "get_default_personas_for_council",
    "get_default_personas_with_domains",
    "get_domains_for_persona",
    "PersonaLoader",
    "PersonaWithDomains",
    "DEFAULT_PERSONA_DOMAINS",
]

