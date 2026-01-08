"""
Answer Confidence Header - Generates confidence metadata for final output.
"""

import logging
from typing import List, Dict

from ..schemas import Claim, ConfidenceScore, ConfidenceHeader

logger = logging.getLogger(__name__)


class AnswerConfidenceHeader:
    """
    Generates confidence metadata for final output.
    
    Includes:
    - Overall confidence score
    - Verified vs unverified claims count
    - Assumptions count
    """
    
    def __init__(self):
        """Initialize confidence header generator."""
        pass
    
    def generate_header(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        verified_claims: Dict[str, bool],
        uncertainty_threshold: float = 0.6
    ) -> ConfidenceHeader:
        """
        Generate confidence header for final output.
        
        Args:
            claims: List of all claims
            confidence_scores: Confidence scores for claims
            verified_claims: Dict mapping claim_id -> bool (True if verified)
            uncertainty_threshold: Threshold below which claims are considered uncertain
            
        Returns:
            ConfidenceHeader with metadata
        """
        # Count verified vs unverified
        verified_count = sum(1 for v in verified_claims.values() if v)
        unverified_count = len(verified_claims) - verified_count
        
        # Count assumptions
        assumptions_count = sum(
            1 for claim in claims 
            if claim.claim_type == "assumption"
        )
        
        # Calculate overall confidence (average of all confidence scores)
        if confidence_scores:
            overall_confidence = sum(score.score for score in confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.5  # Neutral if no scores
        
        # Collect uncertainty markers
        uncertainty_markers = []
        for claim, score in zip(claims, confidence_scores):
            if score.score < uncertainty_threshold:
                uncertainty_markers.append(f"{claim.id}: {claim.text[:50]}...")
        
        # Generate formatted header
        header_lines = [
            "=" * 80,
            "CONFIDENCE METADATA",
            "=" * 80,
            f"Overall Confidence: {overall_confidence:.2%}",
            f"Verified Claims: {verified_count}",
            f"Unverified Claims: {unverified_count}",
            f"Assumptions: {assumptions_count}",
        ]
        
        if uncertainty_markers:
            header_lines.append(f"\nUncertainty Markers ({len(uncertainty_markers)}):")
            for marker in uncertainty_markers[:5]:  # Limit to first 5
                header_lines.append(f"  - {marker}")
        
        header_lines.append("=" * 80)
        
        formatted_header = "\n".join(header_lines)
        
        header = ConfidenceHeader(
            overall_confidence=overall_confidence,
            verified_claims_count=verified_count,
            unverified_claims_count=unverified_count,
            assumptions_count=assumptions_count,
            uncertainty_markers=uncertainty_markers,
            formatted_header=formatted_header
        )
        
        logger.info(
            f"Generated confidence header: confidence={overall_confidence:.2%}, "
            f"verified={verified_count}, unverified={unverified_count}"
        )
        
        return header

