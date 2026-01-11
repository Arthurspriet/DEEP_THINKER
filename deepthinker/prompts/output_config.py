"""
Output configuration for context-aware model instructions.

Provides different formatting instructions depending on whether output
is for machine parsing (internal) or human reading (external).
"""

from enum import Enum
from typing import Optional


class OutputContext(Enum):
    """Context for model output - determines formatting instructions."""
    
    INTERNAL = "internal"  # Machine-to-machine, JSON parsing, council outputs
    HUMAN = "human"        # Final output for users, reports, explanations


# Instructions for internal/machine outputs - optimized for parsing and token efficiency
INTERNAL_INSTRUCTIONS = """
OUTPUT RULES (STRICT):
- Return ONLY valid JSON. No markdown, no code blocks, no explanations.
- No prose before or after the JSON object.
- Be maximally concise. Omit optional fields if empty.
- Do not wrap JSON in ```json``` or any code blocks.
- Do not include explanatory text or commentary.
- Example of correct output: {"key": "value"}
- Example of INCORRECT output: ```json\n{"key": "value"}\n```
"""

# Instructions for human-readable outputs
HUMAN_INSTRUCTIONS = """
OUTPUT RULES:
- Format your response clearly for human reading.
- Use markdown formatting where helpful for structure and readability.
- Include explanations and context where appropriate.
- Organize information with headers, lists, and emphasis as needed.
"""

# Concise mode - even more aggressive token optimization
INTERNAL_CONCISE_INSTRUCTIONS = """
OUTPUT RULES (ULTRA-STRICT):
- Return ONLY the JSON object. Nothing else.
- No markdown. No code blocks. No explanations. No commentary.
- Minimal whitespace. No pretty-printing unless schema requires it.
- Omit all optional/empty fields.
- Single line JSON preferred when under 200 chars.
"""


def get_output_instructions(
    context: OutputContext,
    concise: bool = False
) -> str:
    """
    Get formatting instructions based on output context.
    
    Args:
        context: Whether output is for internal (machine) or human consumption
        concise: If True and context is INTERNAL, use ultra-concise instructions
        
    Returns:
        Instruction string to prepend/append to prompts
    """
    if context == OutputContext.HUMAN:
        return HUMAN_INSTRUCTIONS
    
    # Internal context
    if concise:
        return INTERNAL_CONCISE_INSTRUCTIONS
    return INTERNAL_INSTRUCTIONS


def get_json_schema_instruction(schema_hint: Optional[str] = None) -> str:
    """
    Get JSON schema enforcement instruction.
    
    Args:
        schema_hint: Optional JSON schema or example to include
        
    Returns:
        Instruction string for JSON schema compliance
    """
    base = "Respond with valid JSON matching the specified schema."
    if schema_hint:
        return f"{base}\n\nExpected format:\n{schema_hint}"
    return base


__all__ = [
    "OutputContext",
    "INTERNAL_INSTRUCTIONS",
    "HUMAN_INSTRUCTIONS", 
    "INTERNAL_CONCISE_INSTRUCTIONS",
    "get_output_instructions",
    "get_json_schema_instruction",
]

