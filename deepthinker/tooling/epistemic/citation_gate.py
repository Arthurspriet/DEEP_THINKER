"""
Mandatory Citation Gate - Enforces citation requirements before final output.
"""

import logging
from typing import List, Dict, Set, Optional

from ..schemas import Claim, ConfidenceScore
from ..schemas import UnverifiedClaimError

logger = logging.getLogger(__name__)


class CitationGate:
    """
    Enforces citation requirements before final output.
    
    Hard constraint: No claim passes without:
    - Trusted memory reference OR
    - Web evidence OR
    - Explicit uncertainty label
    """
    
    def __init__(self, require_citations: bool = True):
        """
        Initialize citation gate.
        
        Args:
            require_citations: Whether to enforce citation requirements (default: True)
        """
        self.require_citations = require_citations
    
    def check_claims(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        memory_references: Optional[Dict[str, List[str]]] = None,
        web_evidence: Optional[Dict[str, List[str]]] = None,
        allow_uncertainty_labels: bool = True
    ) -> Dict[str, bool]:
        """
        Check if claims have required citations.
        
        Args:
            claims: List of claims to check
            confidence_scores: Confidence scores for claims
            memory_references: Optional dict mapping claim_id -> list of memory evidence IDs
            web_evidence: Optional dict mapping claim_id -> list of web source URLs
            allow_uncertainty_labels: Whether uncertainty-labeled claims can pass
            
        Returns:
            Dict mapping claim_id -> bool (True if verified, False if unverified)
            
        Raises:
            UnverifiedClaimError: If require_citations=True and unverified claims found
        """
        if not self.require_citations:
            # Citations not required - all claims pass
            return {claim.id: True for claim in claims}
        
        memory_references = memory_references or {}
        web_evidence = web_evidence or {}
        
        verified = {}
        unverified_claims = []
        
        for claim, confidence in zip(claims, confidence_scores):
            claim_id = claim.id
            
            # Check if claim has memory reference
            has_memory = claim_id in memory_references and len(memory_references[claim_id]) > 0
            
            # Check if claim has web evidence
            has_web = claim_id in web_evidence and len(web_evidence[claim_id]) > 0
            
            # Check if claim is explicitly marked as uncertain
            is_uncertain = (
                allow_uncertainty_labels and
                (claim.claim_type == "uncertainty" or confidence.risk_level == "HIGH")
            )
            
            # Claim is verified if it has any of the above
            is_verified = has_memory or has_web or is_uncertain
            
            verified[claim_id] = is_verified
            
            if not is_verified:
                unverified_claims.append((claim, confidence))
        
        # If unverified claims found and citations required, raise error
        if unverified_claims and self.require_citations:
            error_claims = [
                f"{claim.id}: {claim.text[:100]}..." 
                for claim, _ in unverified_claims[:5]  # Limit to first 5
            ]
            raise UnverifiedClaimError(
                claim_id=unverified_claims[0][0].id,
                claim_text=unverified_claims[0][0].text,
                reason=f"Found {len(unverified_claims)} unverified claims without citations: {', '.join(error_claims)}"
            )
        
        return verified
    
    def filter_verified_claims(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        memory_references: Optional[Dict[str, List[str]]] = None,
        web_evidence: Optional[Dict[str, List[str]]] = None,
        allow_uncertainty_labels: bool = True
    ) -> List[Claim]:
        """
        Filter claims to only include verified ones.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores
            memory_references: Optional memory references
            web_evidence: Optional web evidence
            allow_uncertainty_labels: Whether to allow uncertainty labels
            
        Returns:
            List of verified claims only
        """
        verified_map = self.check_claims(
            claims=claims,
            confidence_scores=confidence_scores,
            memory_references=memory_references,
            web_evidence=web_evidence,
            allow_uncertainty_labels=allow_uncertainty_labels
        )
        
        return [claim for claim in claims if verified_map.get(claim.id, False)]
    
    def mark_unverified_claims(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        memory_references: Optional[Dict[str, List[str]]] = None,
        web_evidence: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Mark unverified claims with uncertainty labels.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores
            memory_references: Optional memory references
            web_evidence: Optional web evidence
            
        Returns:
            List of claim IDs that were marked as uncertain
        """
        verified_map = self.check_claims(
            claims=claims,
            confidence_scores=confidence_scores,
            memory_references=memory_references,
            web_evidence=web_evidence,
            allow_uncertainty_labels=True  # Allow uncertainty to pass
        )
        
        # Mark unverified claims
        marked = []
        for claim in claims:
            if not verified_map.get(claim.id, False):
                # Update claim type to uncertainty
                claim.claim_type = "uncertainty"
                marked.append(claim.id)
        
        return marked

