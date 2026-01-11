"""
arXiv Connector for DeepThinker.

Provides search, metadata retrieval, and PDF/source download capabilities
for arXiv papers with full provenance tracking.

Usage:
    from deepthinker.connectors.arxiv import arxiv_search, arxiv_get, arxiv_download
    
    # Search for papers
    results = arxiv_search("cat:cs.CL AND ti:alignment", max_results=10)
    
    # Get specific paper
    paper = arxiv_get("2501.01234v1")
    
    # Download PDF
    download = arxiv_download("2501.01234", kind="pdf")

CLI:
    python -m deepthinker.connectors.arxiv search "query"
    python -m deepthinker.connectors.arxiv get 2501.01234
    python -m deepthinker.connectors.arxiv download 2501.01234 --kind pdf
"""

from .config import ArxivConfig, get_arxiv_config, is_arxiv_enabled
from .models import ArxivPaper, ArxivEvidence
from .client import ArxivError, ArxivParseError
from .tool import arxiv_search, arxiv_get, arxiv_download

__all__ = [
    # Config
    "ArxivConfig",
    "get_arxiv_config",
    "is_arxiv_enabled",
    # Models
    "ArxivPaper",
    "ArxivEvidence",
    # Exceptions
    "ArxivError",
    "ArxivParseError",
    # Tool functions
    "arxiv_search",
    "arxiv_get",
    "arxiv_download",
]


