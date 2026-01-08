"""
Normative Control Layer for DeepThinker.

This module provides a constitutional governance layer that judges outputs
and governs phase progression without generating text, reasoning, or storing memory.

Key components:
- NormativeController: Main entry point for governance evaluation
- NormativeVerdict: Result containing status, violations, and recommended actions
- RuleEngine: Deterministic rule evaluation
- Violation: Typed violation with severity scoring

The layer is:
- Constitutional (can block progress)
- Adaptive (severity depends on violation type)
- Config-driven
- Resource-aware
- Lightweight (VRAM-safe, no LLM usage)
"""

from .violation import Violation, ViolationType, create_violation
from .phase_contracts import (
    GOVERNANCE_PHASE_CONTRACTS,
    GovernancePhaseContract,
    get_governance_contract,
    EvidenceBudget,
    get_evidence_budget,
    DEFAULT_EVIDENCE_BUDGETS,
)
from .rule_engine import RuleEngine, GovernanceConfig, load_governance_config
from .normative_layer import (
    NormativeController,
    NormativeVerdict,
    VerdictStatus,
    RecommendedAction,
)

__all__ = [
    # Core controller
    "NormativeController",
    "NormativeVerdict",
    "VerdictStatus",
    "RecommendedAction",
    # Violations
    "Violation",
    "ViolationType",
    "create_violation",
    # Rule engine
    "RuleEngine",
    "GovernanceConfig",
    "load_governance_config",
    # Phase contracts
    "GOVERNANCE_PHASE_CONTRACTS",
    "GovernancePhaseContract",
    "get_governance_contract",
    # Evidence budgets (Epistemic Hardening Phase 3)
    "EvidenceBudget",
    "get_evidence_budget",
    "DEFAULT_EVIDENCE_BUDGETS",
]

