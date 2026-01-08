"""
Multi-View Utilities for DeepThinker 2.0.

Provides utilities for extracting disagreements between Optimist and Skeptic
councils and converting them into iteration-driving context.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MultiViewDisagreement:
    """
    Represents disagreements between Optimist and Skeptic perspectives.
    
    These disagreements drive iteration by identifying areas needing
    more investigation or evidence.
    """
    
    # Quantitative measures
    agreement_score: float  # 0-1, how much they agree (0 = total disagreement)
    
    # Disagreement categories
    disputed_claims: List[str] = field(default_factory=list)  # Claims one accepts, other rejects
    contested_risks: List[str] = field(default_factory=list)  # Risks Skeptic sees, Optimist dismisses
    contested_opportunities: List[str] = field(default_factory=list)  # Opportunities Optimist sees, Skeptic dismisses
    
    # Derived iteration drivers
    unresolved_questions: List[str] = field(default_factory=list)  # Questions from disagreement
    evidence_requests: List[str] = field(default_factory=list)  # Evidence that would resolve disputes
    next_focus_areas: List[str] = field(default_factory=list)  # Areas needing more analysis
    
    # Raw data
    optimist_confidence: float = 0.5
    skeptic_confidence: float = 0.5
    
    def get_iteration_drivers(self) -> Dict[str, List[str]]:
        """Get all iteration-driving elements."""
        return {
            "unresolved_questions": self.unresolved_questions,
            "evidence_requests": self.evidence_requests,
            "next_focus_areas": self.next_focus_areas
        }
    
    def should_continue_iteration(self) -> bool:
        """Check if disagreement warrants more iteration."""
        # Continue if significant disagreement exists
        if self.agreement_score < 0.6:
            return True
        # Continue if there are contested items
        if len(self.contested_risks) > 2 or len(self.contested_opportunities) > 2:
            return True
        # Continue if evidence is needed
        if self.evidence_requests:
            return True
        return False
    
    def summary(self) -> str:
        """Generate a summary of the disagreement."""
        return (
            f"Agreement: {self.agreement_score:.1%} | "
            f"Disputed: {len(self.disputed_claims)} claims, "
            f"{len(self.contested_risks)} risks, "
            f"{len(self.contested_opportunities)} opportunities | "
            f"Questions: {len(self.unresolved_questions)}"
        )


def extract_disagreements(
    optimist_output: Any,
    skeptic_output: Any
) -> MultiViewDisagreement:
    """
    Extract disagreements between Optimist and Skeptic perspectives.
    
    Args:
        optimist_output: OptimistPerspective object
        skeptic_output: SkepticPerspective object
        
    Returns:
        MultiViewDisagreement with extracted disagreements
    """
    # Handle None outputs - return with default agreement score
    if optimist_output is None or skeptic_output is None:
        return MultiViewDisagreement(agreement_score=0.5)  # Unknown agreement
    
    disagreement = MultiViewDisagreement(agreement_score=0.5)  # Will be updated
    
    # Extract confidence scores
    disagreement.optimist_confidence = getattr(optimist_output, 'confidence', 0.5)
    disagreement.skeptic_confidence = getattr(skeptic_output, 'confidence', 0.5)
    
    # Get key claims from each
    optimist_claims = _get_optimist_claims(optimist_output)
    skeptic_concerns = _get_skeptic_concerns(skeptic_output)
    
    # Identify contested items
    disagreement.contested_opportunities = _find_contested_optimist_claims(
        optimist_claims, skeptic_concerns
    )
    disagreement.contested_risks = _find_contested_skeptic_claims(
        skeptic_concerns, optimist_claims
    )
    
    # Generate disputed claims (asymmetric assertions)
    disagreement.disputed_claims = _generate_disputed_claims(
        optimist_output, skeptic_output
    )
    
    # Calculate agreement score
    disagreement.agreement_score = _calculate_agreement_score(
        disagreement.optimist_confidence,
        disagreement.skeptic_confidence,
        len(disagreement.contested_risks),
        len(disagreement.contested_opportunities)
    )
    
    # Generate iteration drivers from disagreements
    disagreement.unresolved_questions = _generate_questions_from_disputes(
        disagreement.disputed_claims,
        disagreement.contested_risks,
        disagreement.contested_opportunities
    )
    
    disagreement.evidence_requests = _generate_evidence_requests(
        disagreement.contested_risks,
        disagreement.contested_opportunities
    )
    
    disagreement.next_focus_areas = _generate_focus_areas(
        disagreement.disputed_claims,
        disagreement.contested_risks
    )
    
    logger.info(f"Multi-view disagreement: {disagreement.summary()}")
    
    return disagreement


def _get_optimist_claims(output: Any) -> List[str]:
    """Extract key claims from OptimistPerspective."""
    claims = []
    
    if hasattr(output, 'opportunities'):
        claims.extend(output.opportunities[:3])
    if hasattr(output, 'strengths'):
        claims.extend(output.strengths[:3])
    if hasattr(output, 'success_factors'):
        claims.extend(output.success_factors[:2])
    if hasattr(output, 'best_case_outcome') and output.best_case_outcome:
        claims.append(output.best_case_outcome[:100])
    
    return claims


def _get_skeptic_concerns(output: Any) -> List[str]:
    """Extract key concerns from SkepticPerspective."""
    concerns = []
    
    if hasattr(output, 'risks'):
        concerns.extend(output.risks[:3])
    if hasattr(output, 'weaknesses'):
        concerns.extend(output.weaknesses[:3])
    if hasattr(output, 'failure_modes'):
        concerns.extend(output.failure_modes[:2])
    if hasattr(output, 'worst_case_outcome') and output.worst_case_outcome:
        concerns.append(output.worst_case_outcome[:100])
    
    return concerns


def _find_contested_optimist_claims(
    claims: List[str],
    concerns: List[str]
) -> List[str]:
    """Find optimist claims that skeptic would contest."""
    contested = []
    
    # Simple heuristic: optimist claims that relate to skeptic concerns
    for claim in claims:
        claim_lower = claim.lower()
        for concern in concerns:
            concern_lower = concern.lower()
            # Check for semantic overlap
            if _has_semantic_overlap(claim_lower, concern_lower):
                contested.append(claim)
                break
    
    return contested[:5]


def _find_contested_skeptic_claims(
    concerns: List[str],
    claims: List[str]
) -> List[str]:
    """Find skeptic concerns that optimist would dismiss."""
    contested = []
    
    for concern in concerns:
        concern_lower = concern.lower()
        for claim in claims:
            claim_lower = claim.lower()
            if _has_semantic_overlap(concern_lower, claim_lower):
                contested.append(concern)
                break
    
    return contested[:5]


def _has_semantic_overlap(text1: str, text2: str) -> bool:
    """Check if two texts have semantic overlap (simple keyword match)."""
    # Extract significant words (>4 chars)
    words1 = set(w for w in text1.split() if len(w) > 4)
    words2 = set(w for w in text2.split() if len(w) > 4)
    
    # Check for overlap
    overlap = words1 & words2
    return len(overlap) >= 2 or (len(overlap) >= 1 and len(words1) < 5)


def _generate_disputed_claims(
    optimist: Any,
    skeptic: Any
) -> List[str]:
    """Generate list of disputed claims from both perspectives."""
    disputed = []
    
    # High confidence optimist vs high confidence skeptic = dispute
    opt_conf = getattr(optimist, 'confidence', 0.5)
    skep_conf = getattr(skeptic, 'confidence', 0.5)
    
    if abs(opt_conf - skep_conf) > 0.3:
        # One is much more confident - this is a disputed area
        if opt_conf > skep_conf:
            disputed.append("Optimist's confidence may be overestimated")
        else:
            disputed.append("Skeptic's concerns may be overestimated")
    
    # Add specific disputes from success factors vs failure modes
    success_factors = getattr(optimist, 'success_factors', [])
    failure_modes = getattr(skeptic, 'failure_modes', [])
    
    for sf in success_factors[:2]:
        disputed.append(f"Will this succeed? Optimist: Yes ({sf[:50]})")
    
    for fm in failure_modes[:2]:
        disputed.append(f"Will this fail? Skeptic: Yes ({fm[:50]})")
    
    return disputed[:5]


def _calculate_agreement_score(
    opt_conf: float,
    skep_conf: float,
    num_contested_risks: int,
    num_contested_opportunities: int
) -> float:
    """
    Calculate agreement score between perspectives.
    
    Higher score = more agreement.
    Lower score = more disagreement = needs more iteration.
    """
    # Base score from confidence similarity
    conf_diff = abs(opt_conf - skep_conf)
    conf_score = 1.0 - conf_diff
    
    # Penalty for contested items
    contested_total = num_contested_risks + num_contested_opportunities
    contest_penalty = min(0.4, contested_total * 0.1)
    
    agreement = max(0.0, min(1.0, conf_score - contest_penalty))
    
    return agreement


def _generate_questions_from_disputes(
    disputed: List[str],
    contested_risks: List[str],
    contested_opportunities: List[str]
) -> List[str]:
    """Generate research questions from disputed areas."""
    questions = []
    
    for risk in contested_risks[:2]:
        questions.append(f"Is '{risk[:50]}' a real risk or is it manageable?")
    
    for opp in contested_opportunities[:2]:
        questions.append(f"Is '{opp[:50]}' a real opportunity or wishful thinking?")
    
    for dispute in disputed[:2]:
        if "?" not in dispute:
            questions.append(f"How do we resolve: {dispute[:60]}?")
    
    return questions[:5]


def _generate_evidence_requests(
    contested_risks: List[str],
    contested_opportunities: List[str]
) -> List[str]:
    """Generate evidence requests from contested items."""
    requests = []
    
    for risk in contested_risks[:2]:
        requests.append(f"Evidence about likelihood of: {risk[:50]}")
    
    for opp in contested_opportunities[:2]:
        requests.append(f"Evidence supporting: {opp[:50]}")
    
    return requests[:5]


def _generate_focus_areas(
    disputed: List[str],
    contested_risks: List[str]
) -> List[str]:
    """Generate focus areas for next iteration."""
    areas = []
    
    # Focus on the most contested risks
    for risk in contested_risks[:2]:
        areas.append(f"Risk analysis: {risk[:40]}")
    
    # Focus on disputed claims
    for dispute in disputed[:2]:
        areas.append(f"Investigation needed: {dispute[:40]}")
    
    return areas[:5]


def calculate_multiview_agreement(
    optimist_output: Any,
    skeptic_output: Any
) -> float:
    """
    Calculate simple agreement score between perspectives.
    
    Returns:
        Agreement score from 0 (disagree) to 1 (agree)
    """
    if optimist_output is None or skeptic_output is None:
        return 0.5  # Unknown
    
    opt_conf = getattr(optimist_output, 'confidence', 0.5)
    skep_conf = getattr(skeptic_output, 'confidence', 0.5)
    
    # Basic confidence-based agreement
    conf_diff = abs(opt_conf - skep_conf)
    return 1.0 - conf_diff

