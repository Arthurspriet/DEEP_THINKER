"""
Counterfactual Checker - Detects fragile reasoning by checking what fails if claims are false.
"""

import logging
import re
from typing import List, Dict

from ..schemas import Claim, ConfidenceScore, CounterfactualResult

logger = logging.getLogger(__name__)


class CounterfactualChecker:
    """
    Lightweight pass: "What fails if this claim is false?"
    
    Flags weak conclusions (high fragility).
    """
    
    # Fragile claim patterns (claims that would break reasoning if false)
    FRAGILE_PATTERNS = [
        r'\b(?:only|solely|exclusively|must|required|necessary|essential)\b',
        r'\b(?:always|never|all|none|every)\b',
        r'\b(?:proves|confirms|demonstrates|shows conclusively)\b',
    ]
    
    # Dependencies keywords (claims that other claims depend on)
    DEPENDENCY_KEYWORDS = [
        "because", "since", "due to", "as a result", "therefore", "thus",
        "consequently", "hence", "follows from", "depends on", "requires"
    ]
    
    def __init__(self):
        """Initialize counterfactual checker."""
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.FRAGILE_PATTERNS
        ]
    
    def check_fragility(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        claim_dependencies: Optional[Dict[str, List[str]]] = None
    ) -> CounterfactualResult:
        """
        Check fragility of claims using counterfactual analysis.
        
        Args:
            claims: List of claims to check
            confidence_scores: Confidence scores for claims
            claim_dependencies: Optional dict mapping claim_id -> list of dependent claim IDs
            
        Returns:
            CounterfactualResult with fragile claims and risk score
        """
        fragile_claims = []
        failure_scenarios = []
        
        # Create confidence map
        confidence_map = {score.claim_id: score for score in confidence_scores}
        
        for claim in claims:
            confidence = confidence_map.get(claim.id)
            if not confidence:
                continue
            
            fragility_score = 0.0
            
            # Check for fragile patterns
            for pattern in self._compiled_patterns:
                if pattern.search(claim.text):
                    fragility_score += 0.3
                    break
            
            # Check for dependencies
            if claim_dependencies:
                dependents = claim_dependencies.get(claim.id, [])
                if dependents:
                    fragility_score += 0.2 * min(len(dependents), 3)  # Cap at 0.6
            
            # Check confidence (low confidence = more fragile)
            if confidence.score < 0.5:
                fragility_score += 0.3
            
            # Check for dependency keywords in text
            text_lower = claim.text.lower()
            if any(keyword in text_lower for keyword in self.DEPENDENCY_KEYWORDS):
                fragility_score += 0.2
            
            # Normalize to [0, 1]
            fragility_score = min(1.0, fragility_score)
            
            # If fragility is high, mark as fragile
            if fragility_score > 0.5:
                fragile_claims.append(claim.id)
                
                # Generate failure scenario
                scenario = self._generate_failure_scenario(claim, confidence, fragility_score)
                failure_scenarios.append(scenario)
        
        # Calculate overall risk score
        if fragile_claims:
            risk_score = len(fragile_claims) / len(claims)
        else:
            risk_score = 0.0
        
        result = CounterfactualResult(
            fragile_claims=fragile_claims,
            risk_score=risk_score,
            failure_scenarios=failure_scenarios
        )
        
        logger.info(
            f"Counterfactual check: {len(fragile_claims)} fragile claims, "
            f"risk_score={risk_score:.2f}"
        )
        
        return result
    
    def _generate_failure_scenario(
        self,
        claim: Claim,
        confidence: ConfidenceScore,
        fragility_score: float
    ) -> str:
        """
        Generate a failure scenario description.
        
        Args:
            claim: Fragile claim
            confidence: Confidence score
            fragility_score: Calculated fragility score
            
        Returns:
            Failure scenario string
        """
        parts = [f"If claim '{claim.id}' is false:"]
        
        if confidence.score < 0.5:
            parts.append("Low confidence suggests high uncertainty")
        
        if fragility_score > 0.7:
            parts.append("High fragility - reasoning may collapse")
        elif fragility_score > 0.5:
            parts.append("Moderate fragility - some reasoning may be affected")
        
        parts.append(f"Claim text: {claim.text[:100]}...")
        
        return " ".join(parts)

