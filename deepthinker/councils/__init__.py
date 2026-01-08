"""
Council System for DeepThinker 2.0.

Provides council-based multi-LLM orchestration where multiple models
collaborate, critique, vote, and reach consensus for each task phase.

Includes multi-view councils for autonomous reasoning:
- OptimistCouncil: Generates strongly positive interpretations
- SkepticCouncil: Generates strongly critical interpretations

DeepThinker 2.0 Additions:
- ExplorerCouncil: Fast, broad reconnaissance (new in 2.0)
- EvidenceCouncil: Narrow, deep evidence gathering (new in 2.0)

Dynamic Council Generator:
- CouncilDefinition: Runtime council configuration
- DynamicCouncilFactory: Builds councils based on mission/phase/context
"""

from .base_council import BaseCouncil
from .planner_council import PlannerCouncil
from .researcher_council import ResearcherCouncil
from .coder_council import CoderCouncil
from .evaluator_council import EvaluatorCouncil
from .simulation_council import SimulationCouncil

# Dynamic council generator
try:
    from .dynamic_council_factory import CouncilDefinition, DynamicCouncilFactory
    DYNAMIC_COUNCIL_AVAILABLE = True
except ImportError:
    DYNAMIC_COUNCIL_AVAILABLE = False
    CouncilDefinition = None
    DynamicCouncilFactory = None

# Multi-view councils for autonomous reasoning
try:
    from .multi_view import OptimistCouncil, SkepticCouncil
    MULTIVIEW_AVAILABLE = True
except ImportError:
    MULTIVIEW_AVAILABLE = False
    OptimistCouncil = None
    SkepticCouncil = None

# New 2.0 councils - ExplorerCouncil and EvidenceCouncil
try:
    from .explorer_council import ExplorerCouncil, ExplorerContext, ExplorerOutput
    EXPLORER_AVAILABLE = True
except ImportError:
    EXPLORER_AVAILABLE = False
    ExplorerCouncil = None
    ExplorerContext = None
    ExplorerOutput = None

try:
    from .evidence_council import (
        EvidenceCouncil,
        EvidenceContext,
        EvidenceOutput,
        EvidenceItem,
        Citation,
    )
    EVIDENCE_AVAILABLE = True
except ImportError:
    EVIDENCE_AVAILABLE = False
    EvidenceCouncil = None
    EvidenceContext = None
    EvidenceOutput = None
    EvidenceItem = None
    Citation = None

__all__ = [
    "BaseCouncil",
    "PlannerCouncil",
    "ResearcherCouncil",
    "CoderCouncil",
    "EvaluatorCouncil",
    "SimulationCouncil",
    "OptimistCouncil",
    "SkepticCouncil",
    "CouncilDefinition",
    "DynamicCouncilFactory",
    # New 2.0 councils
    "ExplorerCouncil",
    "ExplorerContext",
    "ExplorerOutput",
    "EvidenceCouncil",
    "EvidenceContext",
    "EvidenceOutput",
    "EvidenceItem",
    "Citation",
    # Availability flags
    "MULTIVIEW_AVAILABLE",
    "DYNAMIC_COUNCIL_AVAILABLE",
    "EXPLORER_AVAILABLE",
    "EVIDENCE_AVAILABLE",
]

