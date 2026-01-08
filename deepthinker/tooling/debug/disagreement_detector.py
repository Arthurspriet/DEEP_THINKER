"""
Council Disagreement Detector - Measures divergence between council outputs.
"""

import logging
from typing import List, Dict, Any, Optional

from ..schemas import DisagreementReport, Claim, ConfidenceScore

logger = logging.getLogger(__name__)


class CouncilDisagreementDetector:
    """
    Measures divergence between council outputs.
    
    Requires explicit resolution or abstention.
    """
    
    def __init__(self, divergence_threshold: float = 0.3):
        """
        Initialize disagreement detector.
        
        Args:
            divergence_threshold: Threshold above which disagreement requires resolution
        """
        self.divergence_threshold = divergence_threshold
    
    def detect_disagreement(
        self,
        council_outputs: List[Dict[str, Any]],
        claims: Optional[List[Claim]] = None,
        confidence_scores: Optional[List[ConfidenceScore]] = None
    ) -> DisagreementReport:
        """
        Detect disagreements between council outputs.
        
        Args:
            council_outputs: List of council output dicts with keys:
                - council_name: str
                - output: str
                - confidence: float (optional)
            claims: Optional list of claims extracted from outputs
            confidence_scores: Optional confidence scores for claims
            
        Returns:
            DisagreementReport with divergence score and unresolved items
        """
        if len(council_outputs) < 2:
            return DisagreementReport(
                divergence_score=0.0,
                unresolved=[],
                agreement_areas=[],
                disagreement_areas=[],
                requires_arbitration=False
            )
        
        # Extract outputs and confidences
        outputs = [co.get("output", "") for co in council_outputs]
        confidences = [
            co.get("confidence", 0.5) 
            for co in council_outputs
        ]
        
        # Calculate divergence based on output similarity
        divergence_scores = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = self._calculate_similarity(outputs[i], outputs[j])
                divergence = 1.0 - similarity
                divergence_scores.append(divergence)
        
        # Average divergence
        avg_divergence = (
            sum(divergence_scores) / len(divergence_scores)
            if divergence_scores else 0.0
        )
        
        # Calculate confidence divergence
        if confidences:
            confidence_range = max(confidences) - min(confidences)
            # Normalize to [0, 1]
            confidence_divergence = min(1.0, confidence_range)
            # Combine with output divergence
            avg_divergence = (avg_divergence + confidence_divergence) / 2
        
        # Identify agreement and disagreement areas
        agreement_areas = []
        disagreement_areas = []
        unresolved = []
        
        if claims and confidence_scores:
            # Group claims by confidence
            claim_confidence_map = {
                score.claim_id: score.score 
                for score in confidence_scores
            }
            
            for claim in claims:
                confidence = claim_confidence_map.get(claim.id, 0.5)
                
                # High confidence = agreement area
                if confidence > 0.7:
                    agreement_areas.append(claim.id)
                # Low confidence = disagreement area
                elif confidence < 0.5:
                    disagreement_areas.append(claim.id)
                    unresolved.append(claim.id)
        
        # Determine if arbitration required
        requires_arbitration = avg_divergence >= self.divergence_threshold
        
        report = DisagreementReport(
            divergence_score=avg_divergence,
            unresolved=unresolved,
            agreement_areas=agreement_areas,
            disagreement_areas=disagreement_areas,
            requires_arbitration=requires_arbitration
        )
        
        logger.info(
            f"Detected disagreement: divergence={avg_divergence:.2f}, "
            f"unresolved={len(unresolved)}, requires_arbitration={requires_arbitration}"
        )
        
        return report
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap (Jaccard similarity)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def requires_resolution(self, report: DisagreementReport) -> bool:
        """
        Check if disagreement requires explicit resolution.
        
        Args:
            report: Disagreement report
            
        Returns:
            True if resolution required
        """
        return report.requires_arbitration or len(report.unresolved) > 0

