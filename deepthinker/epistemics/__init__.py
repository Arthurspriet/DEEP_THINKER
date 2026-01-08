"""
Epistemics Module for DeepThinker.

Provides claim validation, evidence tracking, and epistemic risk assessment
to enforce evidence-grounded reasoning throughout mission execution.

Components:
- Claim: Structured representation of factual assertions
- ClaimValidator: Validates claims against evidence requirements
- EpistemicRiskScore: Quantifies hallucination and speculation risk
"""

from .claim_validator import (
    Claim,
    ClaimType,
    ClaimStatus,
    ClaimValidationResult,
    ClaimValidator,
    EpistemicRiskScore,
    Source,
    get_claim_validator,
)
from .focus_area_manager import (
    FocusAreaStatus,
    FocusArea,
    FocusAreaDecision,
    FocusAreaManager,
)
from .claim_registry import (
    ClaimRegistry,
    ClaimContestRecord,
)
from .claim_graph import (
    ClaimGraph,
    ClaimNode,
    ClaimEdge,
    EdgeType,
    rank_claims_by_load_bearing,
)
from .contradiction_detector import (
    ContradictionDetector,
    ContradictionResult,
    get_contradiction_detector,
)

__all__ = [
    "Claim",
    "ClaimType",
    "ClaimStatus",
    "ClaimValidationResult",
    "ClaimValidator",
    "EpistemicRiskScore",
    "Source",
    "get_claim_validator",
    # Focus Area Management (Epistemic Hardening Phase 4)
    "FocusAreaStatus",
    "FocusArea",
    "FocusAreaDecision",
    "FocusAreaManager",
    # Claim Registry (Epistemic Hardening Phase 5)
    "ClaimRegistry",
    "ClaimContestRecord",
    # Claim Graph (Sprint 2 - Metrics)
    "ClaimGraph",
    "ClaimNode",
    "ClaimEdge",
    "EdgeType",
    "rank_claims_by_load_bearing",
    # Contradiction Detection (Sprint 2 - Metrics)
    "ContradictionDetector",
    "ContradictionResult",
    "get_contradiction_detector",
]

