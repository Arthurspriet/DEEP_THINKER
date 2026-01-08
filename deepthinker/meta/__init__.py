"""
Meta-Cognition Engine for DeepThinker 2.0.

Provides reflective reasoning, hypothesis management, internal debate,
plan revision, and reasoning supervision capabilities to support
long-running autonomous missions.

This module augments the mission pipeline with a meta-layer that wraps
each phase with deeper reasoning:

    phase → council → META → revise state → next phase

Components:
    - ReflectionEngine: Analyzes phase outputs for assumptions, weaknesses, etc.
    - HypothesisManager: Manages a dynamic DAG of reasoning hypotheses
    - DebateEngine: Runs internal debates between LLM personas
    - PlanReviser: Produces plan revisions based on meta-cognition results
    - MetaController: Orchestrates all meta-cognition components
    - ReasoningSupervisor: Central brain for meta-level decisions (NEW in 2.0)
        - Analyzes phase/mission metrics (difficulty, uncertainty, progress)
        - Creates depth contracts for councils
        - Decides when to trigger multi-view reasoning
        - Detects reasoning loops and stagnation
        - Plans deepening loops
"""

from .reflection import ReflectionEngine
from .hypotheses import HypothesisManager
from .debate import DebateEngine
from .plan_revision import PlanReviser
from .meta_controller import MetaController
from .supervisor import (
    ReasoningSupervisor,
    PhaseMetrics,
    MissionMetrics,
    DepthContract,
    DeepeningPlan,
    LoopDetection,
)

__all__ = [
    "ReflectionEngine",
    "HypothesisManager",
    "DebateEngine",
    "PlanReviser",
    "MetaController",
    "ReasoningSupervisor",
    "PhaseMetrics",
    "MissionMetrics",
    "DepthContract",
    "DeepeningPlan",
    "LoopDetection",
]

