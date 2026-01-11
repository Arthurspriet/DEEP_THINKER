"""
System Prompts for DeepThinker 2.0 Councils.

Provides role-specific identity prompts for each council and special agent.
Also provides context-aware output instructions for machine vs human outputs.
"""

from pathlib import Path
from typing import Dict, Optional

# Import output configuration for context-aware instructions
from .output_config import (
    OutputContext,
    INTERNAL_INSTRUCTIONS,
    HUMAN_INSTRUCTIONS,
    INTERNAL_CONCISE_INSTRUCTIONS,
    get_output_instructions,
    get_json_schema_instruction,
)


def _load_prompt(name: str) -> str:
    """Load a prompt file by name."""
    prompt_dir = Path(__file__).parent
    prompt_file = prompt_dir / f"{name}.txt"
    
    if prompt_file.exists():
        return prompt_file.read_text()
    return ""


# Lazy-loaded prompts
_PROMPTS: Dict[str, str] = {}


def get_prompt(name: str) -> str:
    """
    Get a system prompt by name.
    
    Args:
        name: Prompt name (e.g., "planner_council", "arbiter")
        
    Returns:
        Prompt text
    """
    if name not in _PROMPTS:
        _PROMPTS[name] = _load_prompt(name)
    return _PROMPTS[name]


def get_planner_council_prompt() -> str:
    """Get planner council system prompt."""
    return get_prompt("planner_council")


def get_coder_council_prompt() -> str:
    """Get coder council system prompt."""
    return get_prompt("coder_council")


def get_evaluator_council_prompt() -> str:
    """Get evaluator council system prompt."""
    return get_prompt("evaluator_council")


def get_simulation_council_prompt() -> str:
    """Get simulation council system prompt."""
    return get_prompt("simulation_council")


def get_researcher_council_prompt() -> str:
    """Get researcher council system prompt."""
    return get_prompt("researcher_council")


def get_arbiter_prompt() -> str:
    """Get arbiter system prompt."""
    return get_prompt("arbiter")


def get_meta_planner_prompt() -> str:
    """Get meta-planner system prompt."""
    return get_prompt("meta_planner")


def wrap_prompt_with_output_context(
    prompt: str,
    context: OutputContext = OutputContext.INTERNAL,
    concise: bool = False,
) -> str:
    """
    Wrap a prompt with context-appropriate output instructions.
    
    Appends output formatting instructions to the end of the prompt
    based on whether the output is for internal (machine) or human consumption.
    
    Args:
        prompt: The base prompt text
        context: Output context (INTERNAL or HUMAN)
        concise: If True and INTERNAL context, use ultra-concise instructions
        
    Returns:
        Prompt with output instructions appended
    """
    output_instructions = get_output_instructions(context, concise)
    return f"{prompt}\n\n{output_instructions}"


__all__ = [
    # Prompt loading
    "get_prompt",
    "get_planner_council_prompt",
    "get_coder_council_prompt",
    "get_evaluator_council_prompt",
    "get_simulation_council_prompt",
    "get_researcher_council_prompt",
    "get_arbiter_prompt",
    "get_meta_planner_prompt",
    # Output context configuration
    "OutputContext",
    "INTERNAL_INSTRUCTIONS",
    "HUMAN_INSTRUCTIONS",
    "INTERNAL_CONCISE_INSTRUCTIONS",
    "get_output_instructions",
    "get_json_schema_instruction",
    "wrap_prompt_with_output_context",
]

