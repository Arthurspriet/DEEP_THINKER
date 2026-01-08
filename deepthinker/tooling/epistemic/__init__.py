"""
Epistemic Control Tools - Anti-Hallucination Core.
"""

from .claim_extractor import ClaimExtractorTool
from .confidence_estimator import ClaimConfidenceTool
from .citation_gate import CitationGate, UnverifiedClaimError

__all__ = [
    "ClaimExtractorTool",
    "ClaimConfidenceTool",
    "CitationGate",
    "UnverifiedClaimError",
]

