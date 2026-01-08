"""
WebSearch Tool - Enables agents to search the web using DuckDuckGo.
"""

from typing import List, Dict, Any, Optional, Type, Tuple
from pydantic import Field
from crewai.tools import BaseTool
from ddgs import DDGS
import time
import re


class WebSearchTool(BaseTool):
    """
    A tool that allows agents to search the web using DuckDuckGo.
    
    This tool provides free web search capabilities without requiring API keys.
    It's particularly useful for finding documentation, code examples, and best practices.
    """
    
    name: str = "Web Search"
    description: str = (
        "Search the web for information, documentation, code examples, and best practices. "
        "Use this when you need to find current information, API documentation, "
        "library usage examples, or coding patterns. "
        "Input should be a search query string."
    )
    max_results: int = Field(default=5, description="Maximum number of search results to return")
    timeout: int = Field(default=10, description="Timeout in seconds for search requests")
    
    def __init__(self, **kwargs):
        """Initialize with empty search results storage."""
        super().__init__(**kwargs)
        self._last_search_results: List[Dict[str, Any]] = []  # Store structured results with quality scores
    
    def _score_source_quality(self, url: str, title: str, snippet: str) -> Tuple[float, str]:
        """
        Score source quality based on domain and content indicators.
        
        Phase 6.2: Domain-based trust tiers for anti-hallucination.
        
        Args:
            url: Source URL
            title: Page title
            snippet: Content snippet
            
        Returns:
            Tuple of (quality_score: float, tier: str)
            - quality_score: 0.0-1.0 (higher = more trustworthy)
            - tier: "HIGH", "MEDIUM", "LOW", or "VERY_LOW"
        """
        url_lower = url.lower()
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        # Extract domain from URL
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url_lower)
        domain = domain_match.group(1) if domain_match else ""
        
        # High trust domains (0.9-1.0)
        high_trust_patterns = [
            r'\.gov\b',  # Government sites
            r'\.edu\b',  # Educational institutions
            r'\.ac\.',   # Academic domains (e.g., .ac.uk)
            r'nih\.gov', r'cdc\.gov', r'fda\.gov',  # Specific gov agencies
            r'mit\.edu', r'stanford\.edu', r'harvard\.edu',  # Top universities
            r'wikipedia\.org',  # Generally reliable
        ]
        for pattern in high_trust_patterns:
            if re.search(pattern, url_lower):
                # Check for academic/research indicators
                if any(word in snippet_lower for word in ['study', 'research', 'peer-reviewed', 'journal']):
                    return (1.0, "HIGH")
                return (0.9, "HIGH")
        
        # Medium trust domains (0.6-0.8)
        # News sites and established organizations
        medium_trust_news = [
            r'reuters\.com', r'bbc\.com', r'bbc\.co\.uk', r'ap\.org',
            r'wsj\.com', r'nytimes\.com', r'theguardian\.com', r'cnn\.com',
            r'npr\.org', r'pbs\.org', r'propublica\.org'
        ]
        for pattern in medium_trust_news:
            if re.search(pattern, url_lower):
                return (0.75, "MEDIUM")
        
        # .org domains (could be think tanks or non-profits)
        if re.search(r'\.org\b', url_lower):
            # Check for think tank indicators
            think_tank_indicators = ['think tank', 'policy', 'research institute', 'foundation']
            if any(indicator in snippet_lower or indicator in title_lower for indicator in think_tank_indicators):
                return (0.8, "MEDIUM")
            return (0.65, "MEDIUM")
        
        # Low trust domains (0.3-0.5)
        low_trust_patterns = [
            r'reddit\.com', r'stackoverflow\.com', r'stackexchange\.com',  # Forums/Q&A
            r'quora\.com', r'youtube\.com', r'medium\.com',  # User-generated content
            r'blogspot\.', r'wordpress\.com', r'blog\.',  # Blogs
            r'twitter\.com', r'facebook\.com', r'linkedin\.com',  # Social media
        ]
        for pattern in low_trust_patterns:
            if re.search(pattern, url_lower):
                return (0.4, "LOW")
        
        # Very low trust (0.0-0.2)
        suspicious_patterns = [
            r'\.tk\b', r'\.ml\b', r'\.ga\b',  # Suspicious TLDs
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, url_lower):
                return (0.1, "VERY_LOW")
        
        # Default: .com sites (medium-low trust)
        if re.search(r'\.com\b', url_lower):
            # Boost if it's a known reputable site
            if any(re.search(pattern, url_lower) for pattern in ['arxiv', 'pubmed', 'scholar']):
                return (0.85, "MEDIUM")
            return (0.5, "MEDIUM")
        
        # Unknown domain - default to medium-low
        return (0.5, "MEDIUM")
    
    def _run(self, query: str) -> str:
        """
        Execute a web search query.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results as a string
        """
        try:
            # Initialize DuckDuckGo search
            ddgs = DDGS(timeout=self.timeout)
            
            # Perform search
            results = list(ddgs.text(
                query,
                max_results=self.max_results
            ))
            
            if not results:
                return f"No results found for query: {query}"
            
            # Format results with quality scoring (Phase 6.4)
            formatted_results = [f"Search Results for: {query}\n"]
            formatted_results.append("=" * 80 + "\n")
            
            scored_results = []
            for result in results:
                title = result.get('title', 'No title')
                snippet = result.get('body', 'No description')
                url = result.get('href', 'No URL')
                
                # Score source quality
                quality_score, quality_tier = self._score_source_quality(url, title, snippet)
                
                # Store structured result
                scored_results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'quality_score': quality_score,
                    'quality_tier': quality_tier
                })
                
                # Format with quality annotation
                quality_label = {
                    'HIGH': '[HIGH QUALITY]',
                    'MEDIUM': '[MEDIUM]',
                    'LOW': '[LOW QUALITY]',
                    'VERY_LOW': '[VERY LOW QUALITY]'
                }.get(quality_tier, '[UNKNOWN]')
                
                formatted_results.append(f"{len(scored_results)}. {title} {quality_label}")
                formatted_results.append(f"   URL: {url}")
                formatted_results.append(f"   {snippet}")
                formatted_results.append("")
            
            # Store structured results for later use (Phase 6.3)
            self._last_search_results = scored_results
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"


def create_websearch_tool(
    max_results: int = 5,
    timeout: int = 10
) -> WebSearchTool:
    """
    Factory function to create a WebSearch tool.
    
    Args:
        max_results: Maximum number of search results to return
        timeout: Timeout in seconds for search requests
        
    Returns:
        Configured WebSearchTool instance
    """
    return WebSearchTool(
        max_results=max_results,
        timeout=timeout
    )

