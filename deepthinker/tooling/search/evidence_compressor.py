"""
Evidence Compressor Tool - Converts raw web search results into council-consumable format.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from ..schemas import CompressedEvidence

logger = logging.getLogger(__name__)


class EvidenceCompressorTool:
    """
    Converts raw web search pages into council-consumable format:
    - Quoted snippets (preserve context)
    - Bullet evidence points
    - Source reliability scores
    """
    
    # Reliability indicators (positive)
    HIGH_RELIABILITY_INDICATORS = [
        "study", "research", "peer-reviewed", "journal", "academic",
        "university", "government", "official", "data", "statistics"
    ]
    
    # Reliability indicators (negative)
    LOW_RELIABILITY_INDICATORS = [
        "opinion", "blog", "forum", "reddit", "social media", "unverified"
    ]
    
    def __init__(self, max_snippets_per_source: int = 3, max_snippet_length: int = 200):
        """
        Initialize evidence compressor.
        
        Args:
            max_snippets_per_source: Maximum snippets to extract per source
            max_snippet_length: Maximum length of each snippet
        """
        self.max_snippets_per_source = max_snippets_per_source
        self.max_snippet_length = max_snippet_length
    
    def compress_evidence(
        self,
        claim_id: str,
        raw_results: List[Dict[str, Any]],
        claim_text: Optional[str] = None
    ) -> CompressedEvidence:
        """
        Compress raw web search results into evidence format.
        
        Args:
            claim_id: ID of claim being verified
            raw_results: List of raw search results (from DuckDuckGo or similar)
            claim_text: Optional claim text for relevance scoring
            
        Returns:
            CompressedEvidence with snippets, sources, and reliability
        """
        snippets = []
        sources = []
        reliability_scores = []
        
        for result in raw_results:
            # Extract source URL
            url = result.get('href', '') or result.get('url', '')
            if url:
                sources.append(url)
            
            # Extract title and body
            title = result.get('title', '') or result.get('title', '')
            body = result.get('body', '') or result.get('snippet', '') or result.get('description', '')
            
            # Generate snippets from body
            source_snippets = self._extract_snippets(
                text=body,
                claim_text=claim_text,
                max_snippets=self.max_snippets_per_source
            )
            
            # If no body snippets, use title
            if not source_snippets and title:
                source_snippets = [title[:self.max_snippet_length]]
            
            snippets.extend(source_snippets)
            
            # Calculate reliability for this source
            reliability = self._calculate_reliability(url, title, body)
            reliability_scores.append(reliability)
        
        # Calculate overall reliability (average)
        overall_reliability = (
            sum(reliability_scores) / len(reliability_scores)
            if reliability_scores else 0.5
        )
        
        # Calculate relevance score if claim text provided
        relevance_score = 0.5
        if claim_text:
            relevance_score = self._calculate_relevance(snippets, claim_text)
        
        # Limit total snippets
        snippets = snippets[:10]  # Max 10 snippets total
        
        evidence = CompressedEvidence(
            claim_id=claim_id,
            snippets=snippets,
            sources=sources,
            reliability=overall_reliability,
            relevance_score=relevance_score
        )
        
        logger.debug(
            f"Compressed evidence for claim {claim_id}: "
            f"{len(snippets)} snippets, {len(sources)} sources, "
            f"reliability={overall_reliability:.2f}"
        )
        
        return evidence
    
    def _extract_snippets(
        self,
        text: str,
        claim_text: Optional[str] = None,
        max_snippets: int = 3
    ) -> List[str]:
        """
        Extract relevant snippets from text.
        
        Args:
            text: Text to extract from
            claim_text: Optional claim text for relevance
            max_snippets: Maximum snippets to extract
            
        Returns:
            List of snippet strings
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            # Fallback: use first N characters
            if len(text) > self.max_snippet_length:
                return [text[:self.max_snippet_length] + "..."]
            return [text]
        
        # If claim text provided, prioritize sentences with overlapping terms
        if claim_text:
            claim_words = set(claim_text.lower().split())
            scored_sentences = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(claim_words & sentence_words)
                score = overlap / max(len(claim_words), 1)
                scored_sentences.append((score, sentence))
            
            # Sort by relevance
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            sentences = [s for _, s in scored_sentences]
        
        # Extract snippets
        snippets = []
        for sentence in sentences[:max_snippets]:
            if len(sentence) > self.max_snippet_length:
                # Truncate at word boundary
                truncated = sentence[:self.max_snippet_length]
                last_space = truncated.rfind(' ')
                if last_space > self.max_snippet_length * 0.8:
                    truncated = truncated[:last_space]
                snippets.append(truncated + "...")
            else:
                snippets.append(sentence)
        
        return snippets
    
    def _calculate_reliability(self, url: str, title: str, body: str) -> float:
        """
        Calculate reliability score for a source.
        
        Args:
            url: Source URL
            title: Source title
            body: Source body text
            
        Returns:
            Reliability score (0.0 to 1.0)
        """
        score = 0.5  # Start neutral
        
        text_lower = (url + " " + title + " " + body).lower()
        
        # Positive indicators
        for indicator in self.HIGH_RELIABILITY_INDICATORS:
            if indicator in text_lower:
                score += 0.1
        
        # Negative indicators
        for indicator in self.LOW_RELIABILITY_INDICATORS:
            if indicator in text_lower:
                score -= 0.15
        
        # Domain-based scoring
        if any(domain in url.lower() for domain in ['.edu', '.gov', '.org']):
            score += 0.1
        elif '.com' in url.lower() and 'blog' in url.lower():
            score -= 0.1
        
        # Normalize to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _calculate_relevance(self, snippets: List[str], claim_text: str) -> float:
        """
        Calculate relevance score of snippets to claim.
        
        Args:
            snippets: List of snippet strings
            claim_text: Claim text to compare against
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not snippets:
            return 0.0
        
        claim_words = set(claim_text.lower().split())
        
        total_overlap = 0
        total_words = 0
        
        for snippet in snippets:
            snippet_words = set(snippet.lower().split())
            overlap = len(claim_words & snippet_words)
            total_overlap += overlap
            total_words += len(snippet_words)
        
        if total_words == 0:
            return 0.0
        
        # Relevance = overlap ratio
        relevance = min(1.0, total_overlap / max(len(claim_words), 1))
        
        return relevance

