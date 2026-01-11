"""
Tools for DeepThinker agents.
"""

from .websearch_tool import WebSearchTool
from .search_triggers import (
    SearchTriggerManager,
    SearchDecision,
    should_trigger_search,
    get_phase_search_quota,
    SEARCH_TRIGGER_KEYWORDS,
    PHASE_SEARCH_QUOTAS,
    ARXIV_TRIGGER_KEYWORDS,
    should_use_arxiv,
    get_arxiv_search_queries,
)

__all__ = [
    "WebSearchTool",
    "SearchTriggerManager",
    "SearchDecision",
    "should_trigger_search",
    "get_phase_search_quota",
    "SEARCH_TRIGGER_KEYWORDS",
    "PHASE_SEARCH_QUOTAS",
    # arXiv support
    "ARXIV_TRIGGER_KEYWORDS",
    "should_use_arxiv",
    "get_arxiv_search_queries",
]

