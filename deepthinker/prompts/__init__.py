"""
System Prompts for DeepThinker 2.0 Councils.

Provides role-specific identity prompts for each council and special agent.
"""

from pathlib import Path
from typing import Dict


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


__all__ = [
    "get_prompt",
    "get_planner_council_prompt",
    "get_coder_council_prompt",
    "get_evaluator_council_prompt",
    "get_simulation_council_prompt",
    "get_researcher_council_prompt",
    "get_arbiter_prompt",
    "get_meta_planner_prompt",
]

