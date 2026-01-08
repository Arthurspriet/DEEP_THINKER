"""
Persona Library for DeepThinker 2.0 Dynamic Council Generator.

Personas are prompt-level role definitions that can be applied to model
calls to specialize their behavior for specific analytical perspectives.
"""

from .loader import (
    load_persona,
    get_available_personas,
    get_default_personas_for_council,
    PersonaLoader,
)

__all__ = [
    "load_persona",
    "get_available_personas",
    "get_default_personas_for_council",
    "PersonaLoader",
]

