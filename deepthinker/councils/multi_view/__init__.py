"""
Multi-View Councils for DeepThinker 2.0.

Provides contrasting perspectives (optimistic vs skeptical) for
enhanced reasoning and disagreement resolution.

Enhanced with:
- True divergence between Optimist and Skeptic prompts
- Disagreement extraction utilities for iteration driving
- Multi-view agreement scoring
"""

from .optimist_council import OptimistCouncil, OptimistContext, OptimistPerspective
from .skeptic_council import SkepticCouncil, SkepticContext, SkepticPerspective
from .multiview_utils import (
    MultiViewDisagreement,
    extract_disagreements,
    calculate_multiview_agreement,
)

__all__ = [
    "OptimistCouncil",
    "OptimistContext", 
    "OptimistPerspective",
    "SkepticCouncil",
    "SkepticContext",
    "SkepticPerspective",
    "MultiViewDisagreement",
    "extract_disagreements",
    "calculate_multiview_agreement",
]

