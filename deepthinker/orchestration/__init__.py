"""
Orchestration Learning Layer for DeepThinker.

Provides resource-aware orchestration that learns which decisions
were worth the cost, optimizing for quality per GPU-second/token.
"""

from .outcome_logger import PhaseOutcome
from .orchestration_store import OrchestrationStore
from .policy_memory import PolicyMemory, CouncilPhaseStats
from .marginal_utility_gate import MarginalUtilityGate, MarginalUtilityDecision
from .council_value_estimator import CouncilValueEstimator
from .phase_time_allocator import (
    PhaseTimeAllocator,
    PhaseTimeBudget,
    TimeAllocation,
    create_allocator_from_store,
)

__all__ = [
    "PhaseOutcome",
    "OrchestrationStore",
    "PolicyMemory",
    "CouncilPhaseStats",
    "MarginalUtilityGate",
    "MarginalUtilityDecision",
    "CouncilValueEstimator",
    "PhaseTimeAllocator",
    "PhaseTimeBudget",
    "TimeAllocation",
    "create_allocator_from_store",
]

