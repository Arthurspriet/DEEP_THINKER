"""
Blinding Module for Constitution.

Provides sanitization functions to remove routing/model identifiers
from judge inputs, ensuring blinded evaluation.

Judges should NOT see:
- Which model produced text
- Which routing strategy/councils were used
- Which policy or bandit arm was selected

Judges CAN see:
- Objective contract
- Artifact text
- Evidence objects (sanitized)
- Phase name (generic)
"""

import hashlib
import re
import logging
from typing import Any, Dict, List, Optional, Set

from .config import ConstitutionConfig, get_constitution_config
from .constitution_spec import ConstitutionSpec, get_default_spec

logger = logging.getLogger(__name__)

# Known model identifiers to remove
MODEL_IDENTIFIERS = {
    # Ollama models
    "llama3", "llama3.1", "llama3.2", "llama3.3",
    "qwen", "qwen2", "qwen2.5", "qwen3",
    "gemma", "gemma2", "gemma3",
    "phi", "phi3", "phi4",
    "mistral", "mixtral",
    "deepseek", "deepseek-r1",
    "cogito", "cogito:14b", "cogito:32b",
    "command-r", "command-r-plus",
    # OpenAI models
    "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5",
    "o1", "o1-preview", "o1-mini", "o3", "o3-mini",
    # Anthropic models
    "claude", "claude-3", "claude-3.5", "claude-sonnet", "claude-opus",
    # Model size indicators
    ":1b", ":3b", ":4b", ":7b", ":8b", ":12b", ":14b", ":32b", ":70b", ":72b",
    "small", "medium", "large", "xlarge",
    "SMALL", "MEDIUM", "LARGE", "XLARGE",
}

# Known council identifiers
COUNCIL_IDENTIFIERS = {
    "research_council", "researcher_council",
    "planner_council",
    "coder_council",
    "evaluator_council",
    "simulation_council",
    "synthesis_council",
    "evidence_council",
    "explorer_council",
    "optimist_council", "OptimistCouncil",
    "skeptic_council", "SkepticCouncil",
}

# Routing/strategy identifiers
ROUTING_IDENTIFIERS = {
    "bandit", "thompson", "ucb", "epsilon-greedy",
    "ml_router", "router",
    "model_tier", "council_set",
    "arm_selected", "arm:",
}

# Policy identifiers
POLICY_IDENTIFIERS = {
    "scorecard_policy", "stop_policy",
    "escalate_policy", "governance",
}

# Regex patterns for model references
MODEL_PATTERNS = [
    # Model with size: "cogito:14b", "llama3.2:7b"
    re.compile(r'\b([a-zA-Z0-9_-]+:[0-9]+b)\b', re.IGNORECASE),
    # Ollama-style: "model: llama3"
    re.compile(r'model:\s*["\']?([a-zA-Z0-9_.-]+)["\']?', re.IGNORECASE),
    # Used model: "using llama3.2"
    re.compile(r'using\s+([a-zA-Z0-9_.-]+)\s+model', re.IGNORECASE),
    # Selected model: "selected: gpt-4"
    re.compile(r'selected:\s*["\']?([a-zA-Z0-9_.-]+)["\']?', re.IGNORECASE),
]

# Regex patterns for council references
COUNCIL_PATTERNS = [
    re.compile(r'\b([a-zA-Z_]*council[a-zA-Z_]*)\b', re.IGNORECASE),
    re.compile(r'council:\s*["\']?([a-zA-Z0-9_]+)["\']?', re.IGNORECASE),
]

# Regex patterns for routing references
ROUTING_PATTERNS = [
    re.compile(r'arm:\s*["\']?([a-zA-Z0-9_]+)["\']?', re.IGNORECASE),
    re.compile(r'tier:\s*["\']?([a-zA-Z0-9_]+)["\']?', re.IGNORECASE),
    re.compile(r'route:\s*["\']?([a-zA-Z0-9_]+)["\']?', re.IGNORECASE),
]


def sanitize_for_judge(
    text: str,
    config: Optional[ConstitutionConfig] = None,
) -> str:
    """
    Sanitize text for judge input by removing identifying information.
    
    Removes:
    - Model identifiers (llama3, gpt-4, etc.)
    - Council identifiers
    - Routing/bandit identifiers
    - Policy identifiers
    
    Args:
        text: Text to sanitize
        config: Optional constitution config
        
    Returns:
        Sanitized text
    """
    config = config or get_constitution_config()
    
    if not config.blinding_enabled:
        return text
    
    if not text:
        return text
    
    result = text
    
    # Remove known model identifiers
    for identifier in MODEL_IDENTIFIERS:
        # Case-insensitive word boundary replacement
        pattern = re.compile(r'\b' + re.escape(identifier) + r'\b', re.IGNORECASE)
        result = pattern.sub("[model]", result)
    
    # Remove known council identifiers
    for identifier in COUNCIL_IDENTIFIERS:
        pattern = re.compile(r'\b' + re.escape(identifier) + r'\b', re.IGNORECASE)
        result = pattern.sub("[council]", result)
    
    # Remove known routing identifiers
    for identifier in ROUTING_IDENTIFIERS:
        pattern = re.compile(r'\b' + re.escape(identifier) + r'\b', re.IGNORECASE)
        result = pattern.sub("[routing]", result)
    
    # Remove known policy identifiers
    for identifier in POLICY_IDENTIFIERS:
        pattern = re.compile(r'\b' + re.escape(identifier) + r'\b', re.IGNORECASE)
        result = pattern.sub("[policy]", result)
    
    # Apply regex patterns
    for pattern in MODEL_PATTERNS:
        result = pattern.sub("[model]", result)
    
    for pattern in COUNCIL_PATTERNS:
        result = pattern.sub("[council]", result)
    
    for pattern in ROUTING_PATTERNS:
        result = pattern.sub("[routing]", result)
    
    return result


def sanitize_metadata(
    metadata: Dict[str, Any],
    config: Optional[ConstitutionConfig] = None,
) -> Dict[str, Any]:
    """
    Sanitize metadata dict by removing/redacting identifying fields.
    
    Args:
        metadata: Metadata dictionary to sanitize
        config: Optional constitution config
        
    Returns:
        Sanitized metadata dictionary
    """
    config = config or get_constitution_config()
    
    if not config.blinding_enabled:
        return metadata
    
    # Fields to completely remove
    fields_to_remove = {
        "models_used", "model", "model_tier", "model_name",
        "councils_used", "council", "council_set",
        "routing_decision", "bandit_arm", "arm_selected",
        "policy_decision", "policy_action",
    }
    
    # Fields to hash if present
    fields_to_hash = {
        "mission_id",  # Keep but hash for privacy
    }
    
    result = {}
    
    for key, value in metadata.items():
        if key in fields_to_remove:
            continue
        elif key in fields_to_hash:
            if isinstance(value, str):
                result[key] = _hash_value(value)
            else:
                result[key] = value
        elif isinstance(value, str):
            result[key] = sanitize_for_judge(value, config)
        elif isinstance(value, dict):
            result[key] = sanitize_metadata(value, config)
        elif isinstance(value, list):
            result[key] = [
                sanitize_for_judge(v, config) if isinstance(v, str)
                else sanitize_metadata(v, config) if isinstance(v, dict)
                else v
                for v in value
            ]
        else:
            result[key] = value
    
    return result


def sanitize_evidence(
    evidence: List[Dict[str, Any]],
    config: Optional[ConstitutionConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Sanitize evidence objects for judge input.
    
    Keeps:
    - evidence_type
    - content_excerpt (sanitized)
    - source (may be hashed)
    - confidence
    
    Removes:
    - Internal IDs
    - Raw references
    - Processing metadata
    
    Args:
        evidence: List of evidence dictionaries
        config: Optional constitution config
        
    Returns:
        List of sanitized evidence dictionaries
    """
    config = config or get_constitution_config()
    
    if not config.blinding_enabled:
        return evidence
    
    allowed_fields = {"evidence_type", "content_excerpt", "confidence", "source"}
    
    sanitized = []
    for ev in evidence:
        sanitized_ev = {}
        for key in allowed_fields:
            if key in ev:
                value = ev[key]
                if isinstance(value, str):
                    sanitized_ev[key] = sanitize_for_judge(value, config)
                else:
                    sanitized_ev[key] = value
        sanitized.append(sanitized_ev)
    
    return sanitized


def create_blinded_judge_input(
    objective: str,
    output: str,
    phase_name: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[ConstitutionConfig] = None,
) -> Dict[str, Any]:
    """
    Create a fully blinded input dictionary for judge evaluation.
    
    Args:
        objective: Mission objective
        output: Artifact text to evaluate
        phase_name: Phase name (kept as-is)
        evidence: Optional list of evidence objects
        metadata: Optional metadata to include
        config: Optional constitution config
        
    Returns:
        Dictionary suitable for judge input
    """
    config = config or get_constitution_config()
    
    blinded = {
        "objective": sanitize_for_judge(objective, config),
        "output": sanitize_for_judge(output, config),
        "phase_name": phase_name,
    }
    
    if evidence:
        blinded["evidence"] = sanitize_evidence(evidence, config)
    
    if metadata:
        blinded["metadata"] = sanitize_metadata(metadata, config)
    
    return blinded


def _hash_value(value: str, length: int = 12) -> str:
    """Hash a value for privacy."""
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def is_identifier_present(text: str) -> bool:
    """
    Check if any model/routing identifiers are present in text.
    
    Useful for testing blinding effectiveness.
    
    Args:
        text: Text to check
        
    Returns:
        True if identifiers are found
    """
    text_lower = text.lower()
    
    # Check known identifiers
    for identifier in MODEL_IDENTIFIERS:
        if identifier.lower() in text_lower:
            return True
    
    for identifier in COUNCIL_IDENTIFIERS:
        if identifier.lower() in text_lower:
            return True
    
    for identifier in ROUTING_IDENTIFIERS:
        if identifier.lower() in text_lower:
            return True
    
    # Check patterns
    for pattern in MODEL_PATTERNS:
        if pattern.search(text):
            return True
    
    return False


