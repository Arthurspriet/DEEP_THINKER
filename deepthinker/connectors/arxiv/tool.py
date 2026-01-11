"""
arXiv Tool Functions for DeepThinker.

Provides high-level tool-callable functions for arXiv operations
with evidence tracking and feature flag checks.

These functions are designed to be invoked as tools during missions
and return structured results with provenance information.

Hardening v1 (2026-01):
- Added structured audit logging for all tool calls (arxiv_tool_call event)
- Metadata-only by default: should_download_arxiv() required for PDF/source
- Client raises ArxivParseError on HTTP 200 with empty results
- ARXIV evidence remains confidence-neutral (regression tested)
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .client import ArxivClient, ArxivError, get_arxiv_client
from .config import ArxivConfig, get_arxiv_config, is_arxiv_enabled
from .models import ArxivEvidence, ArxivPaper

logger = logging.getLogger(__name__)


class ArxivDisabledError(Exception):
    """Raised when arXiv connector is not enabled."""
    pass


def _check_enabled() -> None:
    """Check if arXiv is enabled, raise if not."""
    if not is_arxiv_enabled():
        raise ArxivDisabledError(
            "arXiv connector is disabled. Set DEEPTHINKER_ARXIV_ENABLED=true to enable."
        )


def arxiv_search(
    query: str,
    max_results: int = 10,
    start: int = 0,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search arXiv for papers.
    
    Tool-callable function that searches arXiv and returns results
    with evidence objects for provenance tracking.
    
    Args:
        query: arXiv search query (supports arXiv query syntax)
            Examples:
            - "cat:cs.CL AND ti:alignment"
            - "au:Bengio"
            - "abs:transformer attention"
        max_results: Maximum number of results (default: 10, max: 50)
        start: Starting index for pagination (default: 0)
        sort_by: Sort field ("relevance", "lastUpdatedDate", "submittedDate")
        sort_order: Sort order ("ascending", "descending")
        
    Returns:
        Dictionary with:
        - papers: List of paper dictionaries
        - evidence: Evidence dictionary
        - count: Number of results
        - query: Original query
        - error: Error message if failed (None if success)
        
    Example:
        >>> result = arxiv_search("cat:cs.CL AND ti:alignment", max_results=5)
        >>> for paper in result["papers"]:
        ...     print(f"{paper['arxiv_id']}: {paper['title']}")
    """
    start_time = time.time()
    logger.info(
        "arxiv_tool_call",
        extra={
            "action": "search",
            "query": query,
            "max_results": max_results,
            "start": start,
        }
    )
    
    try:
        _check_enabled()
    except ArxivDisabledError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "search",
                "query": query,
                "count": 0,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        return {
            "papers": [],
            "evidence": None,
            "count": 0,
            "query": query,
            "error": str(e),
        }
    
    try:
        client = get_arxiv_client()
        papers, evidence = client.search(
            query=query,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "search",
                "query": query,
                "count": len(papers),
                "duration_ms": duration_ms,
                "evidence_id": evidence.evidence_id if evidence else None,
            }
        )
        
        return {
            "papers": [p.to_dict() for p in papers],
            "evidence": evidence.to_dict(),
            "count": len(papers),
            "query": query,
            "error": None,
        }
        
    except ArxivError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "search",
                "query": query,
                "count": 0,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        logger.error(f"[ARXIV_TOOL] Search failed: {e}")
        return {
            "papers": [],
            "evidence": None,
            "count": 0,
            "query": query,
            "error": str(e),
        }
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "search",
                "query": query,
                "count": 0,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        logger.exception(f"[ARXIV_TOOL] Unexpected error in search: {e}")
        return {
            "papers": [],
            "evidence": None,
            "count": 0,
            "query": query,
            "error": f"Unexpected error: {e}",
        }


def arxiv_get(arxiv_id: str) -> Dict[str, Any]:
    """
    Get paper metadata by arXiv ID.
    
    Tool-callable function that fetches a single paper's metadata
    with evidence for provenance tracking.
    
    Args:
        arxiv_id: arXiv paper ID (e.g., "2501.01234" or "2501.01234v1")
        
    Returns:
        Dictionary with:
        - paper: Paper dictionary (or None if not found)
        - evidence: Evidence dictionary
        - found: Whether paper was found
        - arxiv_id: Requested ID
        - error: Error message if failed (None if success)
        
    Example:
        >>> result = arxiv_get("2501.01234v1")
        >>> if result["found"]:
        ...     print(result["paper"]["title"])
    """
    start_time = time.time()
    logger.info(
        "arxiv_tool_call",
        extra={
            "action": "get",
            "arxiv_id": arxiv_id,
        }
    )
    
    try:
        _check_enabled()
    except ArxivDisabledError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "get",
                "arxiv_id": arxiv_id,
                "found": False,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        return {
            "paper": None,
            "evidence": None,
            "found": False,
            "arxiv_id": arxiv_id,
            "error": str(e),
        }
    
    try:
        client = get_arxiv_client()
        paper, evidence = client.get_by_id(arxiv_id)
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "get",
                "arxiv_id": arxiv_id,
                "found": paper is not None,
                "duration_ms": duration_ms,
                "evidence_id": evidence.evidence_id if evidence else None,
            }
        )
        
        return {
            "paper": paper.to_dict() if paper else None,
            "evidence": evidence.to_dict(),
            "found": paper is not None,
            "arxiv_id": arxiv_id,
            "error": None,
        }
        
    except ArxivError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "get",
                "arxiv_id": arxiv_id,
                "found": False,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        logger.error(f"[ARXIV_TOOL] Get failed: {e}")
        return {
            "paper": None,
            "evidence": None,
            "found": False,
            "arxiv_id": arxiv_id,
            "error": str(e),
        }
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "get",
                "arxiv_id": arxiv_id,
                "found": False,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        logger.exception(f"[ARXIV_TOOL] Unexpected error in get: {e}")
        return {
            "paper": None,
            "evidence": None,
            "found": False,
            "arxiv_id": arxiv_id,
            "error": f"Unexpected error: {e}",
        }


def arxiv_download(
    arxiv_id: str,
    kind: str = "pdf",
    out_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Download PDF or source for a paper.
    
    Tool-callable function that downloads paper content with
    caching, deduplication, and evidence tracking.
    
    Args:
        arxiv_id: arXiv paper ID (e.g., "2501.01234" or "2501.01234v1")
        kind: Content type to download ("pdf" or "source")
        out_path: Optional output path (uses cache if None)
        use_cache: Whether to use cached version if available
        
    Returns:
        Dictionary with:
        - local_path: Path to downloaded file
        - sha256: SHA256 hash of content
        - evidence: Evidence dictionary
        - cached: Whether result was from cache
        - arxiv_id: Requested ID
        - kind: Content type
        - error: Error message if failed (None if success)
        
    Example:
        >>> result = arxiv_download("2501.01234", kind="pdf")
        >>> if not result["error"]:
        ...     print(f"Downloaded to: {result['local_path']}")
    """
    start_time = time.time()
    logger.info(
        "arxiv_tool_call",
        extra={
            "action": "download",
            "arxiv_id": arxiv_id,
            "kind": kind,
            "use_cache": use_cache,
        }
    )
    
    # Validate kind
    if kind not in ("pdf", "source"):
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "download",
                "arxiv_id": arxiv_id,
                "kind": kind,
                "cache_hit": False,
                "duration_ms": duration_ms,
                "error": f"Invalid kind: {kind}",
            }
        )
        return {
            "local_path": None,
            "sha256": None,
            "evidence": None,
            "cached": False,
            "arxiv_id": arxiv_id,
            "kind": kind,
            "error": f"Invalid kind: {kind}. Must be 'pdf' or 'source'.",
        }
    
    try:
        _check_enabled()
    except ArxivDisabledError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "download",
                "arxiv_id": arxiv_id,
                "kind": kind,
                "cache_hit": False,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        return {
            "local_path": None,
            "sha256": None,
            "evidence": None,
            "cached": False,
            "arxiv_id": arxiv_id,
            "kind": kind,
            "error": str(e),
        }
    
    try:
        client = get_arxiv_client()
        
        if kind == "pdf":
            local_path, sha256, evidence = client.download_pdf(
                arxiv_id, out_path=out_path, use_cache=use_cache
            )
        else:
            local_path, sha256, evidence = client.download_source(
                arxiv_id, out_path=out_path, use_cache=use_cache
            )
        
        cached = evidence.metadata.get("cache_hit", False)
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "download",
                "arxiv_id": arxiv_id,
                "kind": kind,
                "cache_hit": cached,
                "duration_ms": duration_ms,
                "evidence_id": evidence.evidence_id if evidence else None,
            }
        )
        
        return {
            "local_path": local_path,
            "sha256": sha256,
            "evidence": evidence.to_dict(),
            "cached": cached,
            "arxiv_id": arxiv_id,
            "kind": kind,
            "error": None,
        }
        
    except ArxivError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "download",
                "arxiv_id": arxiv_id,
                "kind": kind,
                "cache_hit": False,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        logger.error(f"[ARXIV_TOOL] Download failed: {e}")
        return {
            "local_path": None,
            "sha256": None,
            "evidence": None,
            "cached": False,
            "arxiv_id": arxiv_id,
            "kind": kind,
            "error": str(e),
        }
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "arxiv_tool_call",
            extra={
                "action": "download",
                "arxiv_id": arxiv_id,
                "kind": kind,
                "cache_hit": False,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        logger.exception(f"[ARXIV_TOOL] Unexpected error in download: {e}")
        return {
            "local_path": None,
            "sha256": None,
            "evidence": None,
            "cached": False,
            "arxiv_id": arxiv_id,
            "kind": kind,
            "error": f"Unexpected error: {e}",
        }


def arxiv_batch_get(arxiv_ids: List[str]) -> Dict[str, Any]:
    """
    Get metadata for multiple papers.
    
    Convenience function to fetch multiple papers in sequence
    (rate-limited per arXiv API requirements).
    
    Args:
        arxiv_ids: List of arXiv IDs to fetch
        
    Returns:
        Dictionary with:
        - papers: Dict mapping arxiv_id -> paper dict (or None)
        - evidence: List of evidence dictionaries
        - found_count: Number of papers found
        - total_requested: Total papers requested
        - errors: Dict mapping arxiv_id -> error message
    """
    try:
        _check_enabled()
    except ArxivDisabledError as e:
        return {
            "papers": {},
            "evidence": [],
            "found_count": 0,
            "total_requested": len(arxiv_ids),
            "errors": {aid: str(e) for aid in arxiv_ids},
        }
    
    papers = {}
    evidence_list = []
    errors = {}
    
    for arxiv_id in arxiv_ids:
        result = arxiv_get(arxiv_id)
        
        if result["error"]:
            errors[arxiv_id] = result["error"]
            papers[arxiv_id] = None
        else:
            papers[arxiv_id] = result["paper"]
            if result["evidence"]:
                evidence_list.append(result["evidence"])
    
    found_count = sum(1 for p in papers.values() if p is not None)
    
    return {
        "papers": papers,
        "evidence": evidence_list,
        "found_count": found_count,
        "total_requested": len(arxiv_ids),
        "errors": errors,
    }


def get_arxiv_tool_status() -> Dict[str, Any]:
    """
    Get status of arXiv tool.
    
    Returns:
        Dictionary with:
        - enabled: Whether arXiv is enabled
        - ingest_enabled: Whether RAG ingestion is enabled
        - cache_stats: Cache statistics
        - config: Current configuration
    """
    config = get_arxiv_config()
    
    status = {
        "enabled": config.enabled,
        "ingest_enabled": config.ingest_enabled,
        "config": {
            "api_interval_sec": config.api_interval_sec,
            "dl_interval_sec": config.dl_interval_sec,
            "cache_dir": config.cache_dir,
            "max_results": config.max_results,
        },
    }
    
    if config.enabled:
        from .cache import get_arxiv_cache
        cache = get_arxiv_cache()
        status["cache_stats"] = cache.get_stats()
    else:
        status["cache_stats"] = None
    
    return status


