"""
Claim Extraction Tool - Extracts atomic, verifiable claims from council outputs.
"""

import re
import uuid
import logging
from typing import List, Optional
from datetime import datetime

from ..schemas import Claim

logger = logging.getLogger(__name__)


class ClaimExtractorTool:
    """
    Extracts atomic, verifiable claims from council outputs.
    
    Uses regex patterns and heuristics to identify factual claims,
    inferences, assumptions, and uncertainty markers.
    """
    
    # Patterns for factual claims
    FACTUAL_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:is|are|was|were|has|have|contains|includes|shows|indicates|demonstrates|proves|confirms|reveals|suggests|states|reports|finds|according to)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:\d+%|\d+ percent|\d+ of|\d+ out of)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:study|research|analysis|data|evidence|statistics|survey|report)[^.!?]*[.!?])',
    ]
    
    # Patterns for inferences
    INFERENCE_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:therefore|thus|hence|consequently|it follows|implies|suggests that|indicates that)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:likely|probably|possibly|may|might|could|appears|seems)[^.!?]*[.!?])',
    ]
    
    # Patterns for assumptions
    ASSUMPTION_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:assuming|presuming|supposing|if we assume|given that)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:we assume|it is assumed|assuming)[^.!?]*[.!?])',
    ]
    
    # Patterns for uncertainty
    UNCERTAINTY_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:uncertain|unclear|unknown|unverified|unconfirmed|speculative|tentative)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:may|might|could|possibly|perhaps|maybe)[^.!?]*[.!?])',
    ]
    
    def __init__(self):
        """Initialize the claim extractor."""
        self._compiled_patterns = {
            "factual": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.FACTUAL_PATTERNS],
            "inference": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.INFERENCE_PATTERNS],
            "assumption": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.ASSUMPTION_PATTERNS],
            "uncertainty": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.UNCERTAINTY_PATTERNS],
        }
    
    def extract_claims(
        self,
        text: str,
        source_council: Optional[str] = None,
        source_phase: Optional[str] = None,
        context_window: int = 100
    ) -> List[Claim]:
        """
        Extract atomic claims from text.
        
        Args:
            text: Text to extract claims from
            source_council: Name of council that produced the text
            source_phase: Phase name where text was produced
            context_window: Number of characters to include as context around each claim
            
        Returns:
            List of Claim objects with stable IDs
        """
        claims = []
        seen_texts = set()  # Deduplicate similar claims
        
        # Extract claims by type
        for claim_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    claim_text = match.group(1).strip()
                    
                    # Skip if too short or too long
                    if len(claim_text) < 20 or len(claim_text) > 500:
                        continue
                    
                    # Deduplicate
                    normalized = claim_text.lower().strip()
                    if normalized in seen_texts:
                        continue
                    seen_texts.add(normalized)
                    
                    # Extract context
                    start = max(0, match.start() - context_window)
                    end = min(len(text), match.end() + context_window)
                    context = text[start:end].strip()
                    
                    # Generate stable ID
                    claim_id = self._generate_claim_id(claim_text, claim_type)
                    
                    claim = Claim(
                        id=claim_id,
                        text=claim_text,
                        context=context,
                        claim_type=claim_type,
                        source_council=source_council,
                        source_phase=source_phase,
                        created_at=datetime.utcnow()
                    )
                    claims.append(claim)
        
        # Also extract simple declarative sentences that might be claims
        # (fallback for patterns that don't match)
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 300:
                continue
            
            # Check if it's a declarative sentence (starts with capital, ends with period)
            if re.match(r'^[A-Z].*\.$', sentence):
                normalized = sentence.lower().strip()
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    claim_id = self._generate_claim_id(sentence, "factual")
                    claim = Claim(
                        id=claim_id,
                        text=sentence,
                        context=sentence,
                        claim_type="factual",
                        source_council=source_council,
                        source_phase=source_phase,
                        created_at=datetime.utcnow()
                    )
                    claims.append(claim)
        
        logger.debug(f"Extracted {len(claims)} claims from text (length: {len(text)})")
        return claims
    
    def _generate_claim_id(self, claim_text: str, claim_type: str) -> str:
        """
        Generate a stable ID for a claim.
        
        Uses hash of normalized text + type to ensure same claim gets same ID.
        """
        import hashlib
        normalized = claim_text.lower().strip()
        hash_input = f"{claim_type}:{normalized}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"claim_{claim_type}_{hash_hex}"
    
    def extract_from_council_output(
        self,
        output: str,
        council_name: str,
        phase_name: Optional[str] = None
    ) -> List[Claim]:
        """
        Convenience method to extract claims from council output.
        
        Args:
            output: Council output text
            council_name: Name of the council
            phase_name: Optional phase name
            
        Returns:
            List of extracted claims
        """
        return self.extract_claims(
            text=output,
            source_council=council_name,
            source_phase=phase_name
        )

