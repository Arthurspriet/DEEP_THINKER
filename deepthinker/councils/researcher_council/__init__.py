"""
Researcher Council for DeepThinker 2.0.

Research council that synthesizes information from multiple LLMs
using voting consensus.

Includes WebSearchGate for mandatory search enforcement in factual domains.
"""

from .researcher_council import (
    ResearcherCouncil,
    ResearchContext,
    ResearchFindings,
    WebSearchGate,
    WebSearchGateResult,
    FactualDomain,
    get_web_search_gate,
)

__all__ = [
    "ResearcherCouncil",
    "ResearchContext",
    "ResearchFindings",
    "WebSearchGate",
    "WebSearchGateResult",
    "FactualDomain",
    "get_web_search_gate",
]

