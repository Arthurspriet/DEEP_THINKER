"""
Depth Evaluator for DeepThinker.

Provides model-agnostic depth measurement for phase outputs using
observable artifact heuristics. Depth is computed from structural
indicators rather than token counts or chain-of-thought.

Design principles:
- Model-agnostic: Works across any LLM output
- Observable: Based only on output text, not internals
- Non-blocking: Never fails, returns safe defaults
- Backward-compatible: All fields optional
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Depth Targets per Phase Type
# =============================================================================

DEPTH_TARGETS: Dict[str, float] = {
    "reconnaissance": 0.4,   # Exploratory, breadth over depth
    "recon": 0.4,            # Alias
    "research": 0.6,         # Should surface multiple angles
    "analysis": 0.7,         # Core reasoning phase
    "deep_analysis": 0.85,   # Explicitly deep
    "synthesis": 0.75,       # Must synthesize trade-offs
    "planning": 0.5,         # Planning is structured, not deep
    "design": 0.6,           # Design requires trade-off awareness
    "implementation": 0.5,   # Code-focused, depth less relevant
    "coding": 0.5,           # Alias
    "testing": 0.4,          # Testing is verification, not depth
    "simulation": 0.6,       # Should explore scenarios
    "review": 0.5,           # Review is validation
    "default": 0.5,          # Baseline
}


# =============================================================================
# Depth Indicator Patterns
# =============================================================================

# Perspective indicators - markers of multiple viewpoints
PERSPECTIVE_PATTERNS = [
    r"\bfrom\s+(?:a|an|the)?\s*\w+\s+perspective\b",
    r"\bon\s+the\s+other\s+hand\b",
    r"\balternatively\b",
    r"\banother\s+view\b",
    r"\bconversely\b",
    r"\bin\s+contrast\b",
    r"\bsome\s+argue\b",
    r"\bwhile\s+others\b",
    r"\bdifferent\s+stakeholders\b",
    r"\bmultiple\s+perspectives\b",
]

# Trade-off indicators - explicit acknowledgment of trade-offs
TRADEOFF_PATTERNS = [
    r"\bhowever\b",
    r"\btrade-?off\b",
    r"\bversus\b",
    r"\bbut\s+this\s+means\b",
    r"\bat\s+the\s+cost\s+of\b",
    r"\bbalancing\b",
    r"\bcompromise\b",
    r"\btension\s+between\b",
    r"\bon\s+one\s+hand\b",
    r"\bpros\s+and\s+cons\b",
]

# Counterargument indicators - challenges and refutations
COUNTERARGUMENT_PATTERNS = [
    r"\bcritics\s+argue\b",
    r"\blimitation\s+is\b",
    r"\brisk\s+of\b",
    r"\bweakness\b",
    r"\bdrawback\b",
    r"\bconcern\s+is\b",
    r"\bchallenge\s+is\b",
    r"\bpotential\s+issue\b",
    r"\bcounter-?argument\b",
    r"\bskeptics\b",
    r"\bdownside\b",
]

# Uncertainty indicators - explicit acknowledgment of uncertainty
UNCERTAINTY_PATTERNS = [
    r"\blikely\b",
    r"\bapproximately\b",
    r"\buncertain\b",
    r"\bdepends\s+on\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bmay\s+or\s+may\s+not\b",
    r"\bit\s+remains\s+unclear\b",
    r"\bmore\s+research\s+needed\b",
    r"\bestimated\b",
    r"\btentative\b",
]

# Synthesis indicators - evidence of integration
SYNTHESIS_PATTERNS = [
    r"\bcombining\s+these\b",
    r"\bthis\s+suggests\b",
    r"\btaken\s+together\b",
    r"\bin\s+summary\b",
    r"\boverall\b",
    r"\bsynthesizing\b",
    r"\bintegrating\b",
    r"\bthe\s+key\s+insight\b",
    r"\bto\s+conclude\b",
    r"\bin\s+conclusion\b",
    r"\bholistically\b",
]


@dataclass
class DepthIndicators:
    """
    Raw counts and derived scores for each depth indicator.
    
    Attributes:
        perspective_count: Number of distinct perspective markers found
        tradeoff_count: Number of trade-off acknowledgments
        counterargument_count: Number of challenge/refutation patterns
        uncertainty_count: Number of uncertainty acknowledgments
        synthesis_count: Number of synthesis markers
        total_words: Total word count for density calculations
    """
    perspective_count: int = 0
    tradeoff_count: int = 0
    counterargument_count: int = 0
    uncertainty_count: int = 0
    synthesis_count: int = 0
    total_words: int = 0
    
    @property
    def counterargument_density(self) -> float:
        """Counterarguments per 100 words."""
        if self.total_words == 0:
            return 0.0
        return (self.counterargument_count / self.total_words) * 100
    
    @property
    def synthesis_present(self) -> float:
        """Binary indicator (0 or 1) for synthesis presence."""
        return 1.0 if self.synthesis_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/artifacts."""
        return {
            "perspective_count": self.perspective_count,
            "tradeoff_count": self.tradeoff_count,
            "counterargument_count": self.counterargument_count,
            "counterargument_density": self.counterargument_density,
            "uncertainty_count": self.uncertainty_count,
            "synthesis_count": self.synthesis_count,
            "synthesis_present": self.synthesis_present,
            "total_words": self.total_words,
        }


def _count_pattern_matches(text: str, patterns: list) -> int:
    """Count unique matches for a list of regex patterns."""
    text_lower = text.lower()
    total = 0
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            total += len(matches)
        except re.error:
            continue
    return total


def extract_depth_indicators(output: str) -> DepthIndicators:
    """
    Extract depth indicators from text output.
    
    Args:
        output: The text to analyze
        
    Returns:
        DepthIndicators with counts for each category
    """
    if not output or not isinstance(output, str):
        return DepthIndicators()
    
    try:
        # Count words
        words = output.split()
        total_words = len(words)
        
        # Count each indicator type
        indicators = DepthIndicators(
            perspective_count=_count_pattern_matches(output, PERSPECTIVE_PATTERNS),
            tradeoff_count=_count_pattern_matches(output, TRADEOFF_PATTERNS),
            counterargument_count=_count_pattern_matches(output, COUNTERARGUMENT_PATTERNS),
            uncertainty_count=_count_pattern_matches(output, UNCERTAINTY_PATTERNS),
            synthesis_count=_count_pattern_matches(output, SYNTHESIS_PATTERNS),
            total_words=total_words,
        )
        
        return indicators
        
    except Exception as e:
        logger.warning(f"Error extracting depth indicators: {e}")
        return DepthIndicators()


def compute_depth_score(
    output: str,
    phase_type: Optional[str] = None,
    expected_perspectives: int = 3
) -> float:
    """
    Compute a depth score from 0.0 to 1.0 for the given output.
    
    The score is a weighted combination of:
    - Perspective count (25%)
    - Trade-off presence (20%)
    - Counterargument density (20%)
    - Uncertainty acknowledgment (15%)
    - Evidence synthesis (20%)
    
    Args:
        output: The text to analyze
        phase_type: Optional phase type for context (currently unused)
        expected_perspectives: Expected number of perspectives for normalization
        
    Returns:
        Depth score between 0.0 and 1.0
    """
    indicators = extract_depth_indicators(output)
    
    if indicators.total_words == 0:
        return 0.0
    
    try:
        # Perspective score: normalized by expected count
        perspective_score = min(1.0, indicators.perspective_count / max(1, expected_perspectives))
        
        # Trade-off score: normalized by expecting at least 2
        tradeoff_score = min(1.0, indicators.tradeoff_count / 2)
        
        # Counterargument density score: expecting ~0.1 per 100 words = 1 per 1000 words
        # For a 500-word output, expecting ~0.5 counterarguments
        density_target = 0.1  # per 100 words
        counterargument_score = min(1.0, indicators.counterargument_density / density_target)
        
        # Uncertainty score: normalized by expecting at least 3
        uncertainty_score = min(1.0, indicators.uncertainty_count / 3)
        
        # Synthesis score: binary presence
        synthesis_score = indicators.synthesis_present
        
        # Weighted combination
        depth_score = (
            0.25 * perspective_score +
            0.20 * tradeoff_score +
            0.20 * counterargument_score +
            0.15 * uncertainty_score +
            0.20 * synthesis_score
        )
        
        # Clamp to [0, 1]
        depth_score = max(0.0, min(1.0, depth_score))
        
        logger.debug(
            f"[DEPTH] Computed depth={depth_score:.2f} "
            f"(persp={perspective_score:.2f}, tradeoff={tradeoff_score:.2f}, "
            f"counter={counterargument_score:.2f}, uncert={uncertainty_score:.2f}, "
            f"synth={synthesis_score:.2f})"
        )
        
        return depth_score
        
    except Exception as e:
        logger.warning(f"Error computing depth score: {e}")
        return 0.0


def get_depth_target(phase_type: str) -> float:
    """
    Get the depth target for a given phase type.
    
    Args:
        phase_type: The type of phase (e.g., "research", "analysis")
        
    Returns:
        Target depth score between 0.0 and 1.0
    """
    if not phase_type:
        return DEPTH_TARGETS["default"]
    
    # Normalize phase type
    phase_type_lower = phase_type.lower().strip()
    
    # Direct match
    if phase_type_lower in DEPTH_TARGETS:
        return DEPTH_TARGETS[phase_type_lower]
    
    # Partial match (e.g., "deep_analysis_phase" matches "deep_analysis")
    for key, value in DEPTH_TARGETS.items():
        if key in phase_type_lower or phase_type_lower in key:
            return value
    
    return DEPTH_TARGETS["default"]


def compute_depth_gap(depth_achieved: float, depth_target: float) -> float:
    """
    Compute the gap between achieved and target depth.
    
    Args:
        depth_achieved: The computed depth score
        depth_target: The target depth for the phase
        
    Returns:
        Gap (0.0 if target met or exceeded, positive otherwise)
    """
    return max(0.0, depth_target - depth_achieved)


def get_weakest_indicator(indicators: DepthIndicators) -> str:
    """
    Identify which depth indicator is weakest (for enrichment targeting).
    
    Args:
        indicators: The computed depth indicators
        
    Returns:
        Name of the weakest indicator category
    """
    scores = {
        "perspective": min(1.0, indicators.perspective_count / 3),
        "tradeoff": min(1.0, indicators.tradeoff_count / 2),
        "counterargument": min(1.0, indicators.counterargument_density / 0.1),
        "synthesis": indicators.synthesis_present,
    }
    
    # Find minimum
    weakest = min(scores, key=scores.get)
    return weakest


def select_enrichment_type(indicators: DepthIndicators) -> str:
    """
    Select the appropriate enrichment pass type based on weakest indicator.
    
    Args:
        indicators: The computed depth indicators
        
    Returns:
        Enrichment type: 'skeptic_expansion', 'counterfactual_analysis',
        'synthesis_refinement', or 'alternative_framing'
    """
    weakest = get_weakest_indicator(indicators)
    
    enrichment_map = {
        "perspective": "alternative_framing",
        "tradeoff": "counterfactual_analysis",
        "counterargument": "skeptic_expansion",
        "synthesis": "synthesis_refinement",
    }
    
    return enrichment_map.get(weakest, "skeptic_expansion")


# Enrichment prompts for each type
ENRICHMENT_PROMPTS: Dict[str, str] = {
    "skeptic_expansion": (
        "Building on the analysis above, identify 2-3 key weaknesses, "
        "counterarguments, or limitations that a skeptical critic might raise. "
        "Be specific about what could go wrong or what has been overlooked."
    ),
    "counterfactual_analysis": (
        "Consider the main assumptions in the analysis above. "
        "What if these assumptions were partially or fully incorrect? "
        "Explore alternative scenarios and their implications."
    ),
    "synthesis_refinement": (
        "Synthesize the key points from the analysis above into a unified framework. "
        "Explicitly identify trade-offs, tensions, and how different factors interact. "
        "Provide an integrated perspective that balances competing considerations."
    ),
    "alternative_framing": (
        "Reframe the analysis above from a different stakeholder perspective. "
        "How would someone with opposing priorities or values view this? "
        "What aspects would they emphasize or de-emphasize?"
    ),
}


def get_enrichment_prompt(enrichment_type: str) -> str:
    """
    Get the prompt for a specific enrichment pass type.
    
    Args:
        enrichment_type: Type of enrichment pass
        
    Returns:
        Prompt text for the enrichment
    """
    return ENRICHMENT_PROMPTS.get(enrichment_type, ENRICHMENT_PROMPTS["skeptic_expansion"])






