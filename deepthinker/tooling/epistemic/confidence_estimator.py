"""
Claim Confidence Estimator - Multi-signal confidence scoring for claims.
"""

import logging
import re
from typing import Dict, List, Optional, Any

from ..schemas import Claim, ConfidenceScore

logger = logging.getLogger(__name__)


class ClaimConfidenceTool:
    """
    Estimates confidence in claims using multiple signals.
    
    Signals:
    - Council agreement level (from consensus engine)
    - Memory presence (check RAG stores)
    - Linguistic uncertainty markers
    """
    
    # Linguistic uncertainty markers (lower confidence)
    UNCERTAINTY_MARKERS = [
        "may", "might", "could", "possibly", "perhaps", "maybe",
        "suggests", "indicates", "appears", "seems", "likely",
        "probably", "uncertain", "unclear", "unknown", "speculative",
        "tentative", "unverified", "unconfirmed"
    ]
    
    # High confidence markers
    HIGH_CONFIDENCE_MARKERS = [
        "proves", "confirms", "demonstrates", "shows", "is", "are",
        "was", "were", "has", "have", "contains", "includes"
    ]
    
    # Confidence threshold
    UNTRUSTED_THRESHOLD = 0.6
    
    def __init__(self, memory_manager: Optional[Any] = None):
        """
        Initialize confidence estimator.
        
        Args:
            memory_manager: Optional MemoryManager for checking memory presence
        """
        self.memory_manager = memory_manager
    
    def estimate_confidence(
        self,
        claim: Claim,
        council_agreement: Optional[float] = None,
        check_memory: bool = True
    ) -> ConfidenceScore:
        """
        Estimate confidence for a claim.
        
        Args:
            claim: Claim to estimate confidence for
            council_agreement: Optional agreement level from consensus (0.0-1.0)
            check_memory: Whether to check if claim exists in memory
            
        Returns:
            ConfidenceScore with score, risk level, and signal breakdown
        """
        signals: Dict[str, Any] = {}
        score = 0.5  # Start with neutral
        
        # Signal 1: Council agreement
        if council_agreement is not None:
            signals["council_agreement"] = council_agreement
            score += (council_agreement - 0.5) * 0.3  # Weight: 30%
        else:
            signals["council_agreement"] = None
        
        # Signal 2: Memory presence
        memory_presence = False
        if check_memory and self.memory_manager:
            memory_presence = self._check_memory_presence(claim)
            signals["memory_presence"] = memory_presence
            if memory_presence:
                score += 0.2  # Boost if in memory
        else:
            signals["memory_presence"] = False
        
        # Signal 3: Linguistic uncertainty markers
        linguistic_uncertainty = self._detect_linguistic_uncertainty(claim.text)
        signals["linguistic_uncertainty"] = linguistic_uncertainty
        if linguistic_uncertainty:
            score -= 0.2  # Penalty for uncertainty markers
        else:
            # Check for high confidence markers
            has_high_confidence_markers = any(
                marker in claim.text.lower() 
                for marker in self.HIGH_CONFIDENCE_MARKERS
            )
            if has_high_confidence_markers:
                score += 0.1
        
        # Signal 4: Claim type
        claim_type_weights = {
            "factual": 0.0,  # Neutral
            "inference": -0.1,  # Slight penalty
            "assumption": -0.2,  # Penalty
            "uncertainty": -0.3,  # Larger penalty
        }
        type_penalty = claim_type_weights.get(claim.claim_type, 0.0)
        signals["claim_type"] = claim.claim_type
        signals["type_penalty"] = type_penalty
        score += type_penalty
        
        # Normalize to [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Determine risk level
        if score >= 0.7:
            risk_level = "LOW"
        elif score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return ConfidenceScore(
            score=score,
            risk_level=risk_level,
            signals=signals,
            claim_id=claim.id,
            council_agreement=council_agreement,
            memory_presence=memory_presence,
            linguistic_uncertainty=linguistic_uncertainty
        )
    
    def _check_memory_presence(self, claim: Claim) -> bool:
        """
        Check if claim or similar claim exists in memory.
        
        Args:
            claim: Claim to check
            
        Returns:
            True if claim found in memory
        """
        if not self.memory_manager:
            return False
        
        try:
            # Try to retrieve similar content from RAG store
            # Use claim text as query
            query = claim.text[:200]  # Limit query length
            
            # Check mission RAG store
            if hasattr(self.memory_manager, 'mission_rag'):
                results = self.memory_manager.mission_rag.search(
                    query=query,
                    top_k=3
                )
                if results:
                    # Check if any result is similar to claim
                    for result in results:
                        # Simple similarity check (could be improved with embeddings)
                        result_text = result.get('text', '').lower()
                        claim_text = claim.text.lower()
                        # Check for significant overlap
                        if len(set(result_text.split()) & set(claim_text.split())) >= 3:
                            return True
            
            # Check global RAG store
            if hasattr(self.memory_manager, 'global_rag'):
                results = self.memory_manager.global_rag.search(
                    query=query,
                    top_k=3
                )
                if results:
                    for result in results:
                        result_text = result.get('text', '').lower()
                        claim_text = claim.text.lower()
                        if len(set(result_text.split()) & set(claim_text.split())) >= 3:
                            return True
        except Exception as e:
            logger.warning(f"Error checking memory presence: {e}")
        
        return False
    
    def _detect_linguistic_uncertainty(self, text: str) -> bool:
        """
        Detect linguistic uncertainty markers in text.
        
        Args:
            text: Text to check
            
        Returns:
            True if uncertainty markers found
        """
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.UNCERTAINTY_MARKERS)
    
    def is_untrusted(self, confidence_score: ConfidenceScore) -> bool:
        """
        Check if claim is untrusted (below threshold).
        
        Args:
            confidence_score: Confidence score to check
            
        Returns:
            True if claim is untrusted
        """
        return confidence_score.score < self.UNTRUSTED_THRESHOLD
    
    def estimate_batch(
        self,
        claims: List[Claim],
        council_agreements: Optional[Dict[str, float]] = None,
        check_memory: bool = True
    ) -> List[ConfidenceScore]:
        """
        Estimate confidence for multiple claims.
        
        Args:
            claims: List of claims
            council_agreements: Optional dict mapping claim_id -> agreement level
            check_memory: Whether to check memory
            
        Returns:
            List of confidence scores
        """
        scores = []
        for claim in claims:
            agreement = council_agreements.get(claim.id) if council_agreements else None
            score = self.estimate_confidence(
                claim=claim,
                council_agreement=agreement,
                check_memory=check_memory
            )
            scores.append(score)
        return scores

