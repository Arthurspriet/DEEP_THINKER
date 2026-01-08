"""
Council Output Contract for DeepThinker 2.0.

Defines the standardized output structure that all councils must produce.
This ensures predictable, structured outputs across all phases.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CouncilOutputContract:
    """
    Standardized output contract for all councils.
    
    Every council postprocess() method should produce output
    that can be normalized to this contract.
    
    Attributes:
        summary: Brief summary of the output (required)
        key_points: List of key findings/points
        recommendations: Actionable recommendations
        sources_suggested: Suggested sources or references
        raw_output: Full raw output text
        confidence: Confidence score 0-1
        council_name: Name of the council that produced this
        phase_name: Name of the phase (optional)
        iteration: Iteration number
        metadata: Additional metadata
    """
    summary: str
    key_points: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    sources_suggested: List[str] = field(default_factory=list)
    raw_output: str = ""
    confidence: float = 0.5
    council_name: str = ""
    phase_name: str = ""
    iteration: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def truncated_summary(self, max_chars: int = 500) -> str:
        """Get truncated summary for logging."""
        if len(self.summary) <= max_chars:
            return self.summary
        return self.summary[:max_chars] + "..."
    
    def is_empty(self) -> bool:
        """Check if output is essentially empty."""
        return (
            not self.summary.strip() and 
            not self.key_points and 
            not self.raw_output.strip()
        )
    
    def char_count(self) -> int:
        """Get total character count of output."""
        return (
            len(self.summary) +
            sum(len(p) for p in self.key_points) +
            sum(len(r) for r in self.recommendations) +
            len(self.raw_output)
        )


def normalize_output_to_contract(
    output: Any,
    council_name: str = "",
    phase_name: str = "",
    iteration: int = 1
) -> CouncilOutputContract:
    """
    Normalize any council output to the standard contract.
    
    Handles various output types:
    - CouncilOutputContract (pass-through)
    - Dict with known keys
    - Dataclass with known fields
    - String (raw output)
    
    Args:
        output: The raw output from a council
        council_name: Name of the producing council
        phase_name: Name of the phase
        iteration: Current iteration number
        
    Returns:
        Normalized CouncilOutputContract
    """
    if output is None:
        return CouncilOutputContract(
            summary="",
            raw_output="",
            council_name=council_name,
            phase_name=phase_name,
            iteration=iteration,
        )
    
    # Already a contract
    if isinstance(output, CouncilOutputContract):
        output.council_name = council_name or output.council_name
        output.phase_name = phase_name or output.phase_name
        output.iteration = iteration
        return output
    
    # Dictionary
    if isinstance(output, dict):
        return _normalize_dict_output(output, council_name, phase_name, iteration)
    
    # String
    if isinstance(output, str):
        return _normalize_string_output(output, council_name, phase_name, iteration)
    
    # Dataclass or object with attributes
    return _normalize_object_output(output, council_name, phase_name, iteration)


def _normalize_dict_output(
    output: Dict[str, Any],
    council_name: str,
    phase_name: str,
    iteration: int
) -> CouncilOutputContract:
    """Normalize dictionary output to contract."""
    
    # Extract summary from various possible keys
    summary = ""
    for key in ["summary", "output", "result", "content", "text"]:
        if key in output and output[key]:
            summary = str(output[key])
            break
    
    # Extract key points
    key_points = []
    for key in ["key_points", "keyPoints", "findings", "key_findings", "points"]:
        if key in output and isinstance(output[key], list):
            key_points = [str(p) for p in output[key]]
            break
    
    # Extract recommendations
    recommendations = []
    for key in ["recommendations", "suggestions", "actions", "next_steps"]:
        if key in output and isinstance(output[key], list):
            recommendations = [str(r) for r in output[key]]
            break
    
    # Extract sources
    sources = []
    for key in ["sources_suggested", "sources", "references", "citations"]:
        if key in output and isinstance(output[key], list):
            sources = [str(s) for s in output[key]]
            break
    
    # Extract raw output
    raw_output = output.get("raw_output", output.get("raw", ""))
    if not raw_output and summary:
        raw_output = summary
    
    # Extract confidence
    confidence = 0.5
    for key in ["confidence", "confidence_score", "score"]:
        if key in output:
            try:
                val = float(output[key])
                confidence = val if val <= 1.0 else val / 10.0
                break
            except (ValueError, TypeError):
                pass
    
    # Extract metadata
    metadata = {}
    for key in ["metadata", "meta", "extra"]:
        if key in output and isinstance(output[key], dict):
            metadata = output[key]
            break
    
    return CouncilOutputContract(
        summary=summary,
        key_points=key_points,
        recommendations=recommendations,
        sources_suggested=sources,
        raw_output=str(raw_output),
        confidence=confidence,
        council_name=council_name,
        phase_name=phase_name,
        iteration=iteration,
        metadata=metadata,
    )


def _normalize_string_output(
    output: str,
    council_name: str,
    phase_name: str,
    iteration: int
) -> CouncilOutputContract:
    """Normalize string output to contract."""
    
    # Try to extract structure from the string
    key_points = _extract_bullet_points(output, ["key point", "key finding", "finding"])
    recommendations = _extract_bullet_points(output, ["recommend", "suggestion", "action"])
    sources = _extract_bullet_points(output, ["source", "reference"])
    
    # Extract confidence if mentioned
    confidence = 0.5
    conf_match = re.search(r'confidence[:\s]*([0-9.]+)', output, re.IGNORECASE)
    if conf_match:
        try:
            val = float(conf_match.group(1))
            confidence = val if val <= 1.0 else val / 10.0
        except ValueError:
            pass
    
    # Generate summary from first paragraph or first 500 chars
    summary = output
    first_para = output.split('\n\n')[0] if '\n\n' in output else output
    if len(first_para) < 500:
        summary = first_para
    else:
        summary = output[:500] + "..."
    
    return CouncilOutputContract(
        summary=summary,
        key_points=key_points,
        recommendations=recommendations,
        sources_suggested=sources,
        raw_output=output,
        confidence=confidence,
        council_name=council_name,
        phase_name=phase_name,
        iteration=iteration,
    )


def _normalize_object_output(
    output: Any,
    council_name: str,
    phase_name: str,
    iteration: int
) -> CouncilOutputContract:
    """Normalize dataclass/object output to contract."""
    
    # Try to extract from common attribute names
    summary = ""
    for attr in ["summary", "output", "result", "content", "text"]:
        if hasattr(output, attr):
            val = getattr(output, attr)
            if val:
                summary = str(val)
                break
    
    key_points = []
    for attr in ["key_points", "keyPoints", "findings", "key_findings"]:
        if hasattr(output, attr):
            val = getattr(output, attr)
            if isinstance(val, list):
                key_points = [str(p) for p in val]
                break
    
    recommendations = []
    for attr in ["recommendations", "suggestions", "actions"]:
        if hasattr(output, attr):
            val = getattr(output, attr)
            if isinstance(val, list):
                recommendations = [str(r) for r in val]
                break
    
    sources = []
    for attr in ["sources_suggested", "sources", "references"]:
        if hasattr(output, attr):
            val = getattr(output, attr)
            if isinstance(val, list):
                sources = [str(s) for s in val]
                break
    
    raw_output = ""
    for attr in ["raw_output", "raw", "full_output"]:
        if hasattr(output, attr):
            val = getattr(output, attr)
            if val:
                raw_output = str(val)
                break
    
    if not raw_output:
        raw_output = str(output)
    
    confidence = 0.5
    for attr in ["confidence", "confidence_score"]:
        if hasattr(output, attr):
            try:
                val = float(getattr(output, attr))
                confidence = val if val <= 1.0 else val / 10.0
                break
            except (ValueError, TypeError):
                pass
    
    return CouncilOutputContract(
        summary=summary or str(output)[:500],
        key_points=key_points,
        recommendations=recommendations,
        sources_suggested=sources,
        raw_output=raw_output,
        confidence=confidence,
        council_name=council_name,
        phase_name=phase_name,
        iteration=iteration,
    )


def _extract_bullet_points(text: str, section_keywords: List[str]) -> List[str]:
    """
    Extract bullet points from a section of text.
    
    Args:
        text: Full text to search
        section_keywords: Keywords that identify the section
        
    Returns:
        List of extracted bullet points
    """
    points = []
    lines = text.split('\n')
    in_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if entering a relevant section
        if any(kw in line_lower for kw in section_keywords):
            in_section = True
            continue
        
        # Check if leaving section (new header)
        if in_section and (line.startswith('#') or line.startswith('**')):
            in_section = False
            continue
        
        # Extract bullet points
        if in_section and line.strip().startswith(('-', '*', '•', '1', '2', '3')):
            content = line.strip().lstrip('-*•0123456789.) ')
            if content and len(content) > 5:
                points.append(content)
    
    return points[:10]  # Limit to 10 points


def validate_output_contract(output: CouncilOutputContract) -> tuple:
    """
    Validate an output contract.
    
    Args:
        output: CouncilOutputContract to validate
        
    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []
    
    # Check for empty output
    if output.is_empty():
        issues.append("Output is empty")
    
    # Check summary
    if not output.summary or len(output.summary.strip()) < 10:
        issues.append("Summary is missing or too short")
    
    # Check confidence range
    if not 0.0 <= output.confidence <= 1.0:
        issues.append(f"Confidence {output.confidence} out of range [0, 1]")
    
    # Warn on very large output
    char_count = output.char_count()
    if char_count > 50000:
        issues.append(f"Output very large ({char_count} chars)")
    
    return len(issues) == 0, issues

