"""
Claim Extractor for Proof Packets.

Bridges the existing ClaimValidator from epistemics to the Proof Packet
claim format. Extracts atomic claims from output text and normalizes them
to ClaimEntry format.
"""

import logging
from typing import List, Optional

from ..epistemics.claim_validator import (
    Claim,
    ClaimType,
    ClaimValidator,
    get_claim_validator,
    Source,
)
from .proof_packet import ClaimEntry, ClaimTypeProof

logger = logging.getLogger(__name__)


# Maximum confidence for claims without evidence
MAX_CONFIDENCE_NO_EVIDENCE = 0.3


class ProofClaimExtractor:
    """
    Extracts and normalizes claims for Proof Packets.
    
    Uses the existing ClaimValidator from epistemics module to parse
    claims from LLM output, then converts them to the Proof Packet
    ClaimEntry format.
    
    Usage:
        extractor = ProofClaimExtractor()
        claims = extractor.extract_claims(output_text, sources)
    """
    
    def __init__(
        self,
        min_grounded_ratio: float = 0.6,
        strict_mode: bool = False,
    ):
        """
        Initialize the claim extractor.
        
        Args:
            min_grounded_ratio: Minimum ratio of grounded claims
            strict_mode: Whether to use strict validation
        """
        self._validator = get_claim_validator(
            min_grounded_ratio=min_grounded_ratio,
            strict_mode=strict_mode,
        )
    
    def extract_claims(
        self,
        output_text: str,
        sources: Optional[List[Source]] = None,
        cap_ungrounded_confidence: bool = True,
    ) -> List[ClaimEntry]:
        """
        Extract claims from output text and convert to ClaimEntry format.
        
        Args:
            output_text: Raw LLM output text
            sources: Optional list of available sources
            cap_ungrounded_confidence: Whether to cap confidence for ungrounded claims
            
        Returns:
            List of ClaimEntry objects
        """
        if not output_text or not output_text.strip():
            logger.debug("[PROOF_CLAIMS] Empty output text, no claims extracted")
            return []
        
        # Parse claims using existing validator
        raw_claims = self._validator.parse_claims(output_text, sources)
        
        if not raw_claims:
            logger.debug("[PROOF_CLAIMS] No claims parsed from output")
            return []
        
        # Convert to ClaimEntry format
        claim_entries = []
        for claim in raw_claims:
            entry = self._convert_claim(claim, cap_ungrounded_confidence)
            claim_entries.append(entry)
        
        logger.debug(f"[PROOF_CLAIMS] Extracted {len(claim_entries)} claims from output")
        return claim_entries
    
    def _convert_claim(
        self,
        claim: Claim,
        cap_ungrounded_confidence: bool = True,
    ) -> ClaimEntry:
        """
        Convert an epistemics Claim to a proof packet ClaimEntry.
        
        Args:
            claim: Raw claim from ClaimValidator
            cap_ungrounded_confidence: Whether to cap confidence for ungrounded claims
            
        Returns:
            ClaimEntry for proof packet
        """
        # Map claim type
        claim_type = self._map_claim_type(claim.claim_type)
        
        # Determine confidence
        confidence = claim.confidence
        
        # Cap confidence for ungrounded claims
        if cap_ungrounded_confidence and not claim.is_grounded():
            confidence = min(confidence, MAX_CONFIDENCE_NO_EVIDENCE)
        
        # Normalize text (truncate to 500 chars)
        normalized_text = claim.text[:500].strip()
        
        return ClaimEntry(
            claim_id=claim.id,
            normalized_text=normalized_text,
            claim_type=claim_type,
            confidence_estimate=confidence,
        )
    
    def _map_claim_type(self, claim_type: ClaimType) -> ClaimTypeProof:
        """
        Map epistemics ClaimType to proof packet ClaimTypeProof.
        
        Mapping:
        - FACT -> FACT
        - INFERENCE -> INFERENCE
        - SPECULATION -> ASSUMPTION (speculation = unvalidated assumption)
        
        Args:
            claim_type: Original claim type
            
        Returns:
            Proof packet claim type
        """
        mapping = {
            ClaimType.FACT: ClaimTypeProof.FACT,
            ClaimType.INFERENCE: ClaimTypeProof.INFERENCE,
            ClaimType.SPECULATION: ClaimTypeProof.ASSUMPTION,
        }
        return mapping.get(claim_type, ClaimTypeProof.INFERENCE)
    
    def extract_with_validation(
        self,
        output_text: str,
        sources: Optional[List[Source]] = None,
    ) -> tuple[List[ClaimEntry], float, bool]:
        """
        Extract claims and return validation info.
        
        Args:
            output_text: Raw LLM output text
            sources: Optional list of available sources
            
        Returns:
            Tuple of (claims, grounded_ratio, is_valid)
        """
        claims = self.extract_claims(output_text, sources)
        
        if not claims:
            return [], 1.0, True
        
        # Parse raw claims for validation
        raw_claims = self._validator.parse_claims(output_text, sources)
        validation = self._validator.validate(raw_claims)
        
        return claims, validation.grounded_ratio, validation.is_valid
    
    def get_claim_by_id(
        self,
        claims: List[ClaimEntry],
        claim_id: str,
    ) -> Optional[ClaimEntry]:
        """
        Find a claim by ID.
        
        Args:
            claims: List of claim entries
            claim_id: ID to search for
            
        Returns:
            ClaimEntry if found, None otherwise
        """
        for claim in claims:
            if claim.claim_id == claim_id:
                return claim
        return None
    
    def get_claims_by_type(
        self,
        claims: List[ClaimEntry],
        claim_type: ClaimTypeProof,
    ) -> List[ClaimEntry]:
        """
        Filter claims by type.
        
        Args:
            claims: List of claim entries
            claim_type: Type to filter by
            
        Returns:
            Filtered list of claims
        """
        return [c for c in claims if c.claim_type == claim_type]
    
    def compute_claim_stats(
        self,
        claims: List[ClaimEntry],
    ) -> dict:
        """
        Compute statistics about extracted claims.
        
        Args:
            claims: List of claim entries
            
        Returns:
            Dictionary with claim statistics
        """
        if not claims:
            return {
                "total_claims": 0,
                "by_type": {},
                "average_confidence": 0.0,
                "low_confidence_count": 0,
            }
        
        by_type = {}
        for claim_type in ClaimTypeProof:
            count = len([c for c in claims if c.claim_type == claim_type])
            if count > 0:
                by_type[claim_type.value] = count
        
        avg_confidence = sum(c.confidence_estimate for c in claims) / len(claims)
        low_confidence = sum(1 for c in claims if c.confidence_estimate < 0.5)
        
        return {
            "total_claims": len(claims),
            "by_type": by_type,
            "average_confidence": avg_confidence,
            "low_confidence_count": low_confidence,
        }


# Global extractor instance
_extractor: Optional[ProofClaimExtractor] = None


def get_proof_claim_extractor(
    min_grounded_ratio: float = 0.6,
    strict_mode: bool = False,
) -> ProofClaimExtractor:
    """Get the global proof claim extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ProofClaimExtractor(
            min_grounded_ratio=min_grounded_ratio,
            strict_mode=strict_mode,
        )
    return _extractor


