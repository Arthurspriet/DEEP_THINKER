"""
arXiv API Client for DeepThinker.

Provides HTTP client for arXiv API with rate limiting,
Atom feed parsing, and download capabilities.

Endpoint: https://export.arxiv.org/api/query
Documentation: https://info.arxiv.org/help/api/basics.html
"""

import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from .cache import ArxivCache, get_arxiv_cache
from .config import ArxivConfig, get_arxiv_config
from .models import ArxivEvidence, ArxivPaper

logger = logging.getLogger(__name__)

# Try to import feedparser, fall back to basic XML parsing
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning(
        "[ARXIV] feedparser not installed, using basic XML parsing. "
        "Install with: pip install feedparser"
    )


class RateLimiter:
    """
    Simple rate limiter for API calls.
    
    Ensures minimum interval between calls.
    Thread-safe.
    """
    
    def __init__(self, min_interval_sec: float):
        """
        Initialize rate limiter.
        
        Args:
            min_interval_sec: Minimum seconds between calls
        """
        self.min_interval = min_interval_sec
        self._last_call: float = 0.0
        self._lock = threading.Lock()
    
    def wait(self) -> float:
        """
        Wait until rate limit allows next call.
        
        Returns:
            Actual wait time in seconds
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            wait_time = max(0, self.min_interval - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self._last_call = time.time()
            return wait_time
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        with self._lock:
            self._last_call = 0.0


class ArxivClient:
    """
    HTTP client for arXiv API.
    
    Provides search, fetch by ID, and download functionality
    with rate limiting and caching.
    
    Usage:
        client = ArxivClient()
        
        # Search for papers
        papers = client.search("cat:cs.CL AND ti:alignment")
        
        # Get specific paper
        paper = client.get_by_id("2501.01234v1")
        
        # Download PDF
        path, sha256 = client.download_pdf("2501.01234")
    """
    
    def __init__(
        self,
        config: Optional[ArxivConfig] = None,
        cache: Optional[ArxivCache] = None,
    ):
        """
        Initialize the arXiv client.
        
        Args:
            config: ArxivConfig instance (uses global if None)
            cache: ArxivCache instance (uses global if None)
        """
        self.config = config or get_arxiv_config()
        self.cache = cache or get_arxiv_cache(self.config)
        
        # Rate limiters (separate for API and downloads)
        self._api_limiter = RateLimiter(self.config.api_interval_sec)
        self._dl_limiter = RateLimiter(self.config.dl_interval_sec)
        
        # HTTP session
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.config.user_agent,
        })
    
    def search(
        self,
        query: str,
        start: int = 0,
        max_results: int = 10,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List[ArxivPaper], ArxivEvidence]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query (arXiv query syntax)
            start: Starting index for pagination
            max_results: Maximum results to return
            sort_by: Sort field ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending", "descending")
            
        Returns:
            Tuple of (list of ArxivPaper, ArxivEvidence)
            
        Raises:
            ArxivError: On API error
        """
        # Clamp max_results
        max_results = min(max_results, self.config.max_results)
        
        # Build query parameters
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
        }
        if sort_by:
            params["sortBy"] = sort_by
        if sort_order:
            params["sortOrder"] = sort_order
        
        url = f"{self.config.api_base_url}?{urlencode(params)}"
        
        # Rate limit
        wait_time = self._api_limiter.wait()
        if wait_time > 0:
            logger.debug(f"[ARXIV] Rate limited, waited {wait_time:.2f}s")
        
        # Make request
        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ArxivError(f"API request failed: {e}") from e
        
        # Parse response
        papers = self._parse_feed(response.text)
        
        # Check for empty parse on successful response
        if len(papers) == 0 and response.status_code == 200 and len(response.text) > 0:
            logger.warning(
                "arxiv_parse_empty",
                extra={
                    "request_url": url,
                    "status_code": response.status_code,
                    "content_length": len(response.text),
                    "query": query,
                }
            )
            raise ArxivParseError(
                "arXiv returned empty results for query. "
                "The query may be malformed or no papers match."
            )
        
        # Create evidence
        evidence = ArxivEvidence.for_search(
            request_url=url,
            result_count=len(papers),
            query=query,
        )
        
        logger.info(f"[ARXIV] Search '{query[:50]}...' returned {len(papers)} papers")
        
        return papers, evidence
    
    def get_by_id(self, arxiv_id: str) -> Tuple[Optional[ArxivPaper], ArxivEvidence]:
        """
        Get paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., "2501.01234" or "2501.01234v1")
            
        Returns:
            Tuple of (ArxivPaper or None, ArxivEvidence)
            
        Raises:
            ArxivError: On API error
        """
        # Parse ID and version
        base_id, version = self._parse_arxiv_id(arxiv_id)
        
        # Build URL
        id_to_fetch = f"{base_id}{version}" if version else base_id
        params = {"id_list": id_to_fetch}
        url = f"{self.config.api_base_url}?{urlencode(params)}"
        
        # Rate limit
        wait_time = self._api_limiter.wait()
        if wait_time > 0:
            logger.debug(f"[ARXIV] Rate limited, waited {wait_time:.2f}s")
        
        # Make request
        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ArxivError(f"API request failed: {e}") from e
        
        # Parse response
        papers = self._parse_feed(response.text)
        paper = papers[0] if papers else None
        
        # Log warning if HTTP 200 with content but no parsed paper
        # (unlike search, get_by_id returns None for not-found rather than raising)
        if paper is None and response.status_code == 200 and len(response.text) > 0:
            logger.warning(
                "arxiv_parse_empty",
                extra={
                    "request_url": url,
                    "status_code": response.status_code,
                    "content_length": len(response.text),
                    "arxiv_id": arxiv_id,
                }
            )
        
        # Create evidence
        evidence = ArxivEvidence.for_paper(
            arxiv_id=base_id,
            version=version,
            request_url=url,
            title=paper.title if paper else "",
        )
        
        if paper:
            logger.info(f"[ARXIV] Fetched {arxiv_id}: {paper.title[:50]}...")
        else:
            logger.warning(f"[ARXIV] Paper not found: {arxiv_id}")
        
        return paper, evidence
    
    def download_pdf(
        self,
        arxiv_id: str,
        out_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[str, str, ArxivEvidence]:
        """
        Download PDF for paper.
        
        Args:
            arxiv_id: arXiv paper ID
            out_path: Optional output path (uses cache if None)
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (local_path, sha256, ArxivEvidence)
            
        Raises:
            ArxivError: On download error
        """
        return self._download(arxiv_id, "pdf", out_path, use_cache)
    
    def download_source(
        self,
        arxiv_id: str,
        out_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[str, str, ArxivEvidence]:
        """
        Download source (e-print) for paper.
        
        Args:
            arxiv_id: arXiv paper ID
            out_path: Optional output path (uses cache if None)
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (local_path, sha256, ArxivEvidence)
            
        Raises:
            ArxivError: On download error
        """
        return self._download(arxiv_id, "source", out_path, use_cache)
    
    def _download(
        self,
        arxiv_id: str,
        kind: str,
        out_path: Optional[str],
        use_cache: bool,
    ) -> Tuple[str, str, ArxivEvidence]:
        """
        Internal download implementation.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            out_path: Optional output path
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (local_path, sha256, ArxivEvidence)
        """
        base_id, version = self._parse_arxiv_id(arxiv_id)
        
        # Check cache first
        if use_cache:
            cached = self.cache.get(base_id, kind, version)
            if cached:
                logger.info(f"[ARXIV] Cache hit for {arxiv_id} {kind}")
                evidence = ArxivEvidence.for_download(
                    arxiv_id=base_id,
                    version=version,
                    request_url=cached.get("request_url", ""),
                    local_path=cached["local_path"],
                    sha256=cached["sha256"],
                    content_type=kind,
                )
                evidence.metadata["cache_hit"] = True
                return cached["local_path"], cached["sha256"], evidence
        
        # Build download URL
        id_for_url = f"{base_id}{version}" if version else base_id
        if kind == "pdf":
            url = f"{self.config.pdf_base_url}/{id_for_url}.pdf"
        else:  # source
            url = f"{self.config.source_base_url}/{id_for_url}"
        
        # Rate limit (use download limiter)
        wait_time = self._dl_limiter.wait()
        if wait_time > 0:
            logger.debug(f"[ARXIV] Download rate limited, waited {wait_time:.2f}s")
        
        # Download
        try:
            response = self._session.get(url, timeout=120, stream=True)
            response.raise_for_status()
            content = response.content
        except requests.RequestException as e:
            raise ArxivError(f"Download failed: {e}") from e
        
        # Store in cache
        metadata = self.cache.put(
            arxiv_id=base_id,
            kind=kind,
            version=version,
            content=content,
            request_url=url,
        )
        
        if metadata is None:
            raise ArxivError("Failed to cache downloaded file")
        
        local_path = metadata["local_path"]
        sha256 = metadata["sha256"]
        
        # Copy to out_path if specified
        if out_path:
            import shutil
            shutil.copy2(local_path, out_path)
            local_path = out_path
        
        # Create evidence
        evidence = ArxivEvidence.for_download(
            arxiv_id=base_id,
            version=version,
            request_url=url,
            local_path=local_path,
            sha256=sha256,
            content_type=kind,
        )
        
        logger.info(
            f"[ARXIV] Downloaded {arxiv_id} {kind} "
            f"({len(content)} bytes, sha256={sha256[:16]}...)"
        )
        
        return local_path, sha256, evidence
    
    def _parse_arxiv_id(self, arxiv_id: str) -> Tuple[str, Optional[str]]:
        """
        Parse arXiv ID into base ID and version.
        
        Args:
            arxiv_id: Full or partial arXiv ID
            
        Returns:
            Tuple of (base_id, version or None)
        """
        # Match patterns like "2501.01234v1" or "2501.01234"
        # Also handles old format like "hep-th/9901001v2"
        match = re.match(r"^(.+?)(v\d+)?$", arxiv_id)
        if match:
            base_id = match.group(1)
            version = match.group(2)  # None if no version
            return base_id, version
        return arxiv_id, None
    
    def _parse_feed(self, xml_content: str) -> List[ArxivPaper]:
        """
        Parse Atom feed XML into ArxivPaper objects.
        
        Args:
            xml_content: Raw XML string
            
        Returns:
            List of ArxivPaper objects
        """
        if HAS_FEEDPARSER:
            return self._parse_with_feedparser(xml_content)
        else:
            return self._parse_with_xml(xml_content)
    
    def _parse_with_feedparser(self, xml_content: str) -> List[ArxivPaper]:
        """Parse using feedparser library."""
        feed = feedparser.parse(xml_content)
        papers = []
        
        for entry in feed.entries:
            # Extract ID and version
            arxiv_url = entry.get("id", "")
            # URL format: http://arxiv.org/abs/2501.01234v1
            match = re.search(r"/abs/(.+?)(?:v(\d+))?$", arxiv_url)
            if match:
                base_id = match.group(1)
                version = f"v{match.group(2)}" if match.group(2) else None
            else:
                # Fallback: try to extract from arxiv:id tag
                base_id = entry.get("arxiv_id", arxiv_url.split("/")[-1])
                base_id, version = self._parse_arxiv_id(base_id)
            
            # Parse dates
            published = None
            updated = None
            if entry.get("published"):
                try:
                    published = datetime.fromisoformat(
                        entry.published.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            if entry.get("updated"):
                try:
                    updated = datetime.fromisoformat(
                        entry.updated.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            
            # Extract authors
            authors = []
            for author in entry.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract categories
            categories = []
            primary_category = ""
            for tag in entry.get("tags", []):
                term = tag.get("term", "")
                if term:
                    categories.append(term)
                    if not primary_category:
                        primary_category = term
            
            # Also check arxiv:primary_category
            if hasattr(entry, "arxiv_primary_category"):
                primary_category = entry.arxiv_primary_category.get("term", primary_category)
            
            # Extract links
            links = {}
            for link in entry.get("links", []):
                href = link.get("href", "")
                link_type = link.get("type", "")
                rel = link.get("rel", "")
                
                if "pdf" in href or link_type == "application/pdf":
                    links["pdf"] = href
                elif rel == "alternate":
                    links["abs"] = href
            
            paper = ArxivPaper(
                id=base_id,
                version=version,
                title=entry.get("title", "").replace("\n", " ").strip(),
                authors=authors,
                abstract=entry.get("summary", "").strip(),
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                links=links,
                doi=getattr(entry, "arxiv_doi", None),
                journal_ref=getattr(entry, "arxiv_journal_ref", None),
                comment=getattr(entry, "arxiv_comment", None),
            )
            papers.append(paper)
        
        return papers
    
    def _parse_with_xml(self, xml_content: str) -> List[ArxivPaper]:
        """Parse using basic xml.etree (fallback if feedparser unavailable)."""
        import xml.etree.ElementTree as ET
        
        # Define namespaces
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"[ARXIV] XML parse error: {e}")
            return []
        
        papers = []
        
        for entry in root.findall("atom:entry", ns):
            # Extract ID
            id_elem = entry.find("atom:id", ns)
            arxiv_url = id_elem.text if id_elem is not None else ""
            match = re.search(r"/abs/(.+?)(?:v(\d+))?$", arxiv_url)
            if match:
                base_id = match.group(1)
                version = f"v{match.group(2)}" if match.group(2) else None
            else:
                base_id = arxiv_url.split("/")[-1] if arxiv_url else ""
                base_id, version = self._parse_arxiv_id(base_id)
            
            # Extract title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.replace("\n", " ").strip() if title_elem is not None and title_elem.text else ""
            
            # Extract abstract
            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
            
            # Extract authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            
            # Extract dates
            published = None
            updated = None
            pub_elem = entry.find("atom:published", ns)
            if pub_elem is not None and pub_elem.text:
                try:
                    published = datetime.fromisoformat(pub_elem.text.replace("Z", "+00:00"))
                except ValueError:
                    pass
            upd_elem = entry.find("atom:updated", ns)
            if upd_elem is not None and upd_elem.text:
                try:
                    updated = datetime.fromisoformat(upd_elem.text.replace("Z", "+00:00"))
                except ValueError:
                    pass
            
            # Extract categories
            categories = []
            primary_category = ""
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    categories.append(term)
                    if not primary_category:
                        primary_category = term
            
            # Extract links
            links = {}
            for link in entry.findall("atom:link", ns):
                href = link.get("href", "")
                link_type = link.get("type", "")
                if "pdf" in href or link_type == "application/pdf":
                    links["pdf"] = href
                elif link.get("rel") == "alternate":
                    links["abs"] = href
            
            paper = ArxivPaper(
                id=base_id,
                version=version,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                links=links,
            )
            papers.append(paper)
        
        return papers
    
    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
    
    def __enter__(self) -> "ArxivClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ArxivError(Exception):
    """Exception raised for arXiv API errors."""
    pass


class ArxivParseError(ArxivError):
    """
    Raised when HTTP 200 but parsing yields no entries.
    
    This typically indicates a malformed query or an unexpected
    response format from arXiv API.
    """
    pass


# Global client instance
_client: Optional[ArxivClient] = None


def get_arxiv_client(config: Optional[ArxivConfig] = None) -> ArxivClient:
    """
    Get global arXiv client instance.
    
    Args:
        config: Optional config override
        
    Returns:
        ArxivClient instance
    """
    global _client
    if _client is None or config is not None:
        _client = ArxivClient(config)
    return _client


def reset_arxiv_client() -> None:
    """Reset global client instance (for testing)."""
    global _client
    if _client is not None:
        _client.close()
    _client = None


