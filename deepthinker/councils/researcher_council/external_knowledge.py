"""
External Knowledge Artifact for Research Findings.

Tracks web search results and evidence strength for anti-hallucination enforcement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ExternalKnowledge:
    """
    Tracks external knowledge gathered via web search.
    
    Attributes:
        queries: Search queries that were executed
        sources: URLs from search results
        evidence_strength: Strength of evidence found ("weak" | "moderate" | "strong")
        confidence_delta: Change in confidence from external validation (-0.2 to +0.3)
        coverage: How well evidence covers the knowledge gap ("partial" | "adequate")
        search_failed: Whether search execution failed
    """
    
    queries: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    evidence_strength: str = "weak"  # "weak" | "moderate" | "strong"
    confidence_delta: float = 0.0  # -0.2 to +0.3
    coverage: str = "partial"  # "partial" | "adequate"
    search_failed: bool = False
    
    def calculate_evidence_strength(
        self, 
        result_count: int, 
        has_urls: bool,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Calculate evidence strength based on search results and source quality.
        
        Phase 6.3: Uses source quality scores, not just count.
        
        Args:
            result_count: Number of search results returned
            has_urls: Whether results contain valid URLs
            sources: Optional list of source dicts with quality_score and quality_tier
        """
        if self.search_failed or result_count == 0:
            self.evidence_strength = "weak"
            self.confidence_delta = -0.2
            return
        
        # Calculate average quality score if sources provided
        avg_quality = 0.5  # Default medium quality
        if sources and len(sources) > 0:
            quality_scores = [s.get('quality_score', 0.5) for s in sources if isinstance(s, dict)]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Phase 6.3: Evidence strength based on count AND quality
        # Penalize low-quality sources
        if avg_quality < 0.4:
            # Low quality sources cap at moderate
            if result_count >= 3 and has_urls:
                self.evidence_strength = "moderate"
                self.confidence_delta = +0.1
                self.coverage = "partial"
            elif result_count >= 1 and has_urls:
                self.evidence_strength = "weak"
                self.confidence_delta = 0.0
            else:
                self.evidence_strength = "weak"
                self.confidence_delta = 0.0
        elif result_count >= 3 and has_urls and avg_quality >= 0.7:
            # Strong: 3+ results AND high quality
            self.evidence_strength = "strong"
            self.confidence_delta = +0.3
            self.coverage = "adequate"
        elif result_count >= 2 and has_urls and avg_quality >= 0.5:
            # Moderate: 2+ results AND medium+ quality
            self.evidence_strength = "moderate"
            self.confidence_delta = +0.1
            self.coverage = "partial"
        elif result_count >= 1 and has_urls:
            # Weak: at least 1 result
            self.evidence_strength = "weak"
            self.confidence_delta = 0.0
        else:
            self.evidence_strength = "weak"
            self.confidence_delta = 0.0

