"""
Step Tier Policy for Epistemic Hardening.

Enforces model tier requirements by step type to prevent weak models
from performing truth-critical work.

Key principles:
- Parsing/formatting: small models allowed
- Search planning: medium or better required
- Claim adjudication: strong models required (reasoning or large)
- Synthesis: strong models required
- VRAM pressure may delay, but never downgrade truth steps

Usage:
    from step_tier_policy import StepType, get_minimum_tier, can_use_model
    
    # Check if a model can handle a step
    if can_use_model(ModelTier.SMALL, StepType.CLAIM_ADJUDICATION):
        # This will be False - small models can't adjudicate
        pass
    
    # Get minimum tier for a step
    min_tier = get_minimum_tier(StepType.SYNTHESIS)  # Returns REASONING or LARGE
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from .model_registry import ModelTier

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """
    Types of processing steps with different model requirements.
    
    Steps are categorized by their truth-criticality:
    - PARSING/FORMATTING: Low criticality, any model works
    - SEARCH_PLANNING: Medium criticality, need reasonable quality
    - CLAIM_ADJUDICATION: High criticality, requires strong models
    - SYNTHESIS: High criticality, requires strong models
    - EVIDENCE_EXTRACTION: High criticality, requires accuracy
    - GROUNDING: High criticality, truth-critical
    """
    PARSING = "parsing"
    FORMATTING = "formatting"
    SEARCH_PLANNING = "search_planning"
    CLAIM_ADJUDICATION = "claim_adjudication"
    SYNTHESIS = "synthesis"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    GROUNDING = "grounding"
    EXPLORATION = "exploration"
    SUMMARIZATION = "summarization"


# Tier precedence (higher index = stronger model)
TIER_PRECEDENCE = {
    ModelTier.SMALL: 1,
    ModelTier.MEDIUM: 2,
    ModelTier.LARGE: 3,
    ModelTier.REASONING: 4,
    ModelTier.EMBEDDING: 0,  # Special case - embedding only
}


# Model tiers allowed for each step type (in order of preference)
STEP_TIER_REQUIREMENTS: Dict[StepType, List[ModelTier]] = {
    # Low criticality - any model works
    StepType.PARSING: [
        ModelTier.SMALL, 
        ModelTier.MEDIUM, 
        ModelTier.LARGE, 
        ModelTier.REASONING
    ],
    StepType.FORMATTING: [
        ModelTier.SMALL, 
        ModelTier.MEDIUM, 
        ModelTier.LARGE, 
        ModelTier.REASONING
    ],
    
    # Medium criticality - need decent quality
    StepType.SEARCH_PLANNING: [
        ModelTier.MEDIUM, 
        ModelTier.LARGE, 
        ModelTier.REASONING
    ],
    StepType.SUMMARIZATION: [
        ModelTier.MEDIUM, 
        ModelTier.LARGE, 
        ModelTier.REASONING
    ],
    StepType.EXPLORATION: [
        ModelTier.MEDIUM, 
        ModelTier.LARGE, 
        ModelTier.REASONING
    ],
    
    # High criticality - truth-critical, require strong models
    StepType.CLAIM_ADJUDICATION: [
        ModelTier.REASONING, 
        ModelTier.LARGE
    ],
    StepType.SYNTHESIS: [
        ModelTier.REASONING, 
        ModelTier.LARGE
    ],
    StepType.EVIDENCE_EXTRACTION: [
        ModelTier.REASONING, 
        ModelTier.LARGE
    ],
    StepType.GROUNDING: [
        ModelTier.REASONING, 
        ModelTier.LARGE
    ],
}


# Steps that are truth-critical and should never be downgraded
TRUTH_CRITICAL_STEPS: Set[StepType] = {
    StepType.CLAIM_ADJUDICATION,
    StepType.GROUNDING,
    StepType.EVIDENCE_EXTRACTION,
    StepType.SYNTHESIS,
}


def get_minimum_tier(step_type: StepType) -> ModelTier:
    """
    Get the minimum required model tier for a step type.
    
    Args:
        step_type: The type of step
        
    Returns:
        The minimum ModelTier required
    """
    allowed_tiers = STEP_TIER_REQUIREMENTS.get(step_type, [ModelTier.MEDIUM])
    
    if not allowed_tiers:
        return ModelTier.MEDIUM
    
    # Return the tier with lowest precedence (minimum allowed)
    min_tier = min(allowed_tiers, key=lambda t: TIER_PRECEDENCE.get(t, 2))
    return min_tier


def can_use_model(model_tier: ModelTier, step_type: StepType) -> bool:
    """
    Check if a model tier is sufficient for a step type.
    
    Args:
        model_tier: The tier of the model to check
        step_type: The type of step to perform
        
    Returns:
        True if the model tier is sufficient
    """
    allowed_tiers = STEP_TIER_REQUIREMENTS.get(step_type, [ModelTier.MEDIUM])
    return model_tier in allowed_tiers


def is_truth_critical(step_type: StepType) -> bool:
    """
    Check if a step type is truth-critical.
    
    Truth-critical steps should never have their model tier downgraded
    due to VRAM pressure - they should be delayed instead.
    
    Args:
        step_type: The type of step
        
    Returns:
        True if the step is truth-critical
    """
    return step_type in TRUTH_CRITICAL_STEPS


def get_allowed_tiers(step_type: StepType) -> List[ModelTier]:
    """
    Get all allowed model tiers for a step type.
    
    Args:
        step_type: The type of step
        
    Returns:
        List of allowed ModelTiers
    """
    return STEP_TIER_REQUIREMENTS.get(step_type, [ModelTier.MEDIUM])


def get_preferred_tier(step_type: StepType) -> ModelTier:
    """
    Get the preferred (strongest) model tier for a step type.
    
    Args:
        step_type: The type of step
        
    Returns:
        The preferred ModelTier
    """
    allowed_tiers = STEP_TIER_REQUIREMENTS.get(step_type, [ModelTier.MEDIUM])
    
    if not allowed_tiers:
        return ModelTier.MEDIUM
    
    # Return the tier with highest precedence (preferred)
    return max(allowed_tiers, key=lambda t: TIER_PRECEDENCE.get(t, 2))


@dataclass
class TierDecision:
    """
    Result of a tier policy check.
    
    Attributes:
        allowed: Whether the model tier is allowed
        step_type: The step type being evaluated
        requested_tier: The requested model tier
        minimum_tier: The minimum tier required
        is_truth_critical: Whether the step is truth-critical
        message: Explanation of the decision
        should_delay: Whether to delay (vs downgrade) if needed
    """
    allowed: bool
    step_type: StepType
    requested_tier: ModelTier
    minimum_tier: ModelTier
    is_truth_critical: bool
    message: str
    should_delay: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "allowed": self.allowed,
            "step_type": self.step_type.value,
            "requested_tier": self.requested_tier.value,
            "minimum_tier": self.minimum_tier.value,
            "is_truth_critical": self.is_truth_critical,
            "message": self.message,
            "should_delay": self.should_delay,
        }


def evaluate_tier_policy(
    step_type: StepType,
    requested_tier: ModelTier,
    vram_pressure: str = "low"
) -> TierDecision:
    """
    Evaluate whether a model tier is appropriate for a step.
    
    This is the main policy entry point that considers:
    - Whether the tier meets minimum requirements
    - Whether the step is truth-critical
    - Current VRAM pressure
    
    Args:
        step_type: The type of step to perform
        requested_tier: The model tier being requested
        vram_pressure: Current VRAM pressure ("low", "medium", "high", "critical")
        
    Returns:
        TierDecision with policy evaluation results
    """
    minimum_tier = get_minimum_tier(step_type)
    allowed = can_use_model(requested_tier, step_type)
    is_critical = is_truth_critical(step_type)
    
    # Determine message
    if allowed:
        message = f"Model tier {requested_tier.value} is allowed for {step_type.value}"
        should_delay = False
    else:
        if is_critical:
            message = (
                f"Model tier {requested_tier.value} is insufficient for truth-critical "
                f"step {step_type.value}. Minimum tier: {minimum_tier.value}. "
                f"VRAM pressure may delay, but will not downgrade."
            )
            should_delay = True
        else:
            message = (
                f"Model tier {requested_tier.value} is below minimum {minimum_tier.value} "
                f"for step {step_type.value}."
            )
            should_delay = False
    
    # Check if VRAM pressure should affect decision
    if vram_pressure in ("high", "critical") and is_critical and not allowed:
        message += f" Will delay due to {vram_pressure} VRAM pressure rather than downgrade."
        should_delay = True
    
    decision = TierDecision(
        allowed=allowed,
        step_type=step_type,
        requested_tier=requested_tier,
        minimum_tier=minimum_tier,
        is_truth_critical=is_critical,
        message=message,
        should_delay=should_delay,
    )
    
    if not allowed:
        logger.warning(f"[STEP TIER POLICY] {message}")
    
    return decision


def infer_step_type(
    phase_name: str,
    action_description: str = ""
) -> StepType:
    """
    Infer the step type from context.
    
    Args:
        phase_name: Name of the current phase
        action_description: Description of what's being done
        
    Returns:
        Inferred StepType
    """
    phase_lower = phase_name.lower()
    action_lower = action_description.lower()
    
    combined = f"{phase_lower} {action_lower}"
    
    # Truth-critical patterns
    if any(p in combined for p in ["adjudicat", "evaluat", "judg", "contest"]):
        return StepType.CLAIM_ADJUDICATION
    
    if any(p in combined for p in ["synthes", "conclud", "final", "report"]):
        return StepType.SYNTHESIS
    
    if any(p in combined for p in ["ground", "verif", "validat"]):
        return StepType.GROUNDING
    
    if any(p in combined for p in ["extract", "evidence", "source"]):
        return StepType.EVIDENCE_EXTRACTION
    
    # Medium criticality patterns
    if any(p in combined for p in ["search", "query", "plan"]):
        return StepType.SEARCH_PLANNING
    
    if any(p in combined for p in ["explor", "recon", "discover"]):
        return StepType.EXPLORATION
    
    if any(p in combined for p in ["summar", "compress", "condense"]):
        return StepType.SUMMARIZATION
    
    # Low criticality patterns
    if any(p in combined for p in ["pars", "format", "structur", "json"]):
        return StepType.PARSING
    
    # Default based on phase
    if "synthesis" in phase_lower or "final" in phase_lower:
        return StepType.SYNTHESIS
    
    if "analysis" in phase_lower or "deep" in phase_lower:
        return StepType.CLAIM_ADJUDICATION
    
    if "recon" in phase_lower or "research" in phase_lower:
        return StepType.EXPLORATION
    
    # Safe default
    return StepType.EXPLORATION


def get_tier_policy_summary() -> Dict[str, Any]:
    """
    Get a summary of the tier policy configuration.
    
    Returns:
        Dictionary with policy summary
    """
    return {
        "truth_critical_steps": [s.value for s in TRUTH_CRITICAL_STEPS],
        "tier_precedence": {
            tier.value: precedence 
            for tier, precedence in TIER_PRECEDENCE.items()
        },
        "step_requirements": {
            step.value: [t.value for t in tiers]
            for step, tiers in STEP_TIER_REQUIREMENTS.items()
        },
    }

