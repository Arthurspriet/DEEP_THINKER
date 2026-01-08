"""
Search Justification Generator - Generates explicit justifications for web searches.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from ..schemas import Claim, ConfidenceScore, SearchJustification

logger = logging.getLogger(__name__)


class SearchJustificationGenerator:
    """
    Generates explicit justifications for web searches.
    
    Rule: If justification exists → search MUST execute (no silent skip).
    """
    
    def __init__(self):
        """Initialize the justification generator."""
        pass
    
    def generate_justifications(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        min_confidence_threshold: float = 0.6
    ) -> List[SearchJustification]:
        """
        Generate search justifications for low-confidence claims.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores for claims
            min_confidence_threshold: Minimum confidence to require search (default: 0.6)
            
        Returns:
            List of search justifications
        """
        justifications = []
        
        # Create confidence map
        confidence_map = {score.claim_id: score for score in confidence_scores}
        
        for claim in claims:
            confidence = confidence_map.get(claim.id)
            if not confidence:
                continue
            
            # Only generate justification for low-confidence claims
            if confidence.score < min_confidence_threshold:
                queries = self._generate_queries(claim)
                rationale = self._generate_rationale(claim, confidence)
                priority = self._determine_priority(confidence)
                
                justification = SearchJustification(
                    claim_id=claim.id,
                    queries=queries,
                    rationale=rationale,
                    priority=priority,
                    created_at=datetime.utcnow()
                )
                justifications.append(justification)
        
        logger.info(f"Generated {len(justifications)} search justifications")
        return justifications
    
    def _generate_queries(self, claim: Claim) -> List[str]:
        """
        Generate search queries from a claim.
        
        Args:
            claim: Claim to generate queries for
            
        Returns:
            List of search query strings
        """
        queries = []
        
        # Extract key terms from claim
        text = claim.text
        
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "may", "might", "can", "this", "that", "these", "those"
        }
        
        # Extract key phrases (2-4 word combinations)
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Generate query from claim text (first 100 chars)
        if len(text) > 20:
            # Use first sentence or first 100 chars
            first_sentence = text.split('.')[0].strip()
            if len(first_sentence) > 20 and len(first_sentence) < 100:
                queries.append(first_sentence)
            else:
                queries.append(text[:100].strip())
        
        # Generate query from key terms
        if len(meaningful_words) >= 3:
            # Take first 5 meaningful words
            key_terms = " ".join(meaningful_words[:5])
            if len(key_terms) > 10:
                queries.append(key_terms)
        
        # Generate query from claim type context
        if claim.claim_type == "factual":
            # For factual claims, try to extract subject + predicate
            # Simple heuristic: take first noun phrase
            words = text.split()
            if len(words) >= 3:
                # Take first 4-6 words as query
                query = " ".join(words[:6])
                if len(query) > 15:
                    queries.append(query)
        
        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 5:
                seen.add(q_lower)
                unique_queries.append(q.strip())
        
        # Ensure at least one query
        if not unique_queries:
            unique_queries.append(text[:100].strip())
        
        return unique_queries[:3]  # Max 3 queries per claim
    
    def _generate_rationale(self, claim: Claim, confidence: ConfidenceScore) -> str:
        """
        Generate rationale for why search is needed.
        
        Args:
            claim: Claim being searched
            confidence: Confidence score
            
        Returns:
            Rationale string
        """
        reasons = []
        
        # Low confidence
        if confidence.score < 0.4:
            reasons.append("Very low confidence score")
        elif confidence.score < 0.6:
            reasons.append("Below confidence threshold")
        
        # Risk level
        if confidence.risk_level == "HIGH":
            reasons.append("High risk of hallucination")
        elif confidence.risk_level == "MEDIUM":
            reasons.append("Medium risk of inaccuracy")
        
        # Missing memory
        if not confidence.memory_presence:
            reasons.append("No supporting evidence in memory")
        
        # Linguistic uncertainty
        if confidence.linguistic_uncertainty:
            reasons.append("Contains uncertainty markers")
        
        # Claim type
        if claim.claim_type == "factual":
            reasons.append("Factual claim requires verification")
        elif claim.claim_type == "inference":
            reasons.append("Inference requires supporting evidence")
        
        if not reasons:
            reasons.append("Claim requires external validation")
        
        rationale = f"Search required: {', '.join(reasons)}. Claim: {claim.text[:150]}..."
        return rationale
    
    def _determine_priority(self, confidence: ConfidenceScore) -> str:
        """
        Determine search priority based on confidence.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Priority level: "low", "medium", or "high"
        """
        if confidence.score < 0.3 or confidence.risk_level == "HIGH":
            return "high"
        elif confidence.score < 0.5:
            return "medium"
        else:
            return "low"
    
    def enforce_mandatory_execution(
        self,
        justifications: List[SearchJustification],
        executed_searches: Dict[str, bool]
    ) -> List[SearchJustification]:
        """
        Enforce mandatory execution - return justifications that were not executed.
        
        Args:
            justifications: List of generated justifications
            executed_searches: Dict mapping claim_id -> bool (True if search executed)
            
        Returns:
            List of justifications that were NOT executed (violations)
        """
        violations = []
        for justification in justifications:
            if not executed_searches.get(justification.claim_id, False):
                violations.append(justification)
                logger.warning(
                    f"MANDATORY SEARCH VIOLATION: Claim {justification.claim_id} "
                    f"has justification but search was not executed. "
                    f"Rationale: {justification.rationale}"
                )
        
        return violations

