"""
Bio Prior Integration Helpers.

Provides helper functions for integrating bio prior signals with
the existing DeepThinker systems.

Soft v1 Scope:
- Only `depth_budget_delta` is applied
- All other signals are log-only
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .signals import PressureSignals

if TYPE_CHECKING:
    from ..meta.supervisor import DeepeningPlan

logger = logging.getLogger(__name__)


# Fields that are actually applied in soft v1
V1_APPLIED_FIELDS = {"depth_budget_delta"}

# Fields that are log-only in v1 (future PRs)
V1_LOG_ONLY_FIELDS = {
    "redundancy_check",
    "force_falsification_step",
    "branch_pruning_suggested",
    "retrieval_diversify",
    "council_diversity_min",
    "confidence_penalty_delta",
    "exploration_bias_delta",
}


def apply_bio_pressures_to_deepening_plan(
    plan: "DeepeningPlan",
    signals: PressureSignals,
    max_deepening_rounds_limit: int = 5,
) -> List[str]:
    """
    Apply bio prior pressure signals to a DeepeningPlan.
    
    Soft v1 ONLY applies:
    - depth_budget_delta: Modifies max_deepening_rounds within bounds
    
    Everything else is logged but NOT applied in v1.
    
    Args:
        plan: DeepeningPlan to modify
        signals: PressureSignals from bio prior engine
        max_deepening_rounds_limit: Maximum allowed deepening rounds
        
    Returns:
        List of field names that were actually applied
    """
    applied_fields: List[str] = []
    
    # Apply depth_budget_delta to max_deepening_rounds
    if signals.depth_budget_delta != 0:
        original_rounds = plan.max_deepening_rounds
        new_rounds = plan.max_deepening_rounds + signals.depth_budget_delta
        
        # Clamp to safe bounds [1, max_deepening_rounds_limit]
        new_rounds = max(1, min(max_deepening_rounds_limit, new_rounds))
        
        if new_rounds != original_rounds:
            plan.max_deepening_rounds = new_rounds
            applied_fields.append("depth_budget_delta")
            
            logger.info(
                f"[BIO_PRIORS] Applied depth_budget_delta: "
                f"{original_rounds} -> {new_rounds} rounds "
                f"(delta: {signals.depth_budget_delta})"
            )
    
    # Log other signals (NOT applied in v1)
    _log_unapplied_signals(signals)
    
    return applied_fields


def compute_would_apply_diff(
    plan: "DeepeningPlan",
    signals: PressureSignals,
    max_deepening_rounds_limit: int = 5,
) -> Dict[str, Any]:
    """
    Compute what WOULD be applied without actually applying.
    
    Used in shadow mode to log the diff.
    
    Args:
        plan: DeepeningPlan to analyze
        signals: PressureSignals from bio prior engine
        max_deepening_rounds_limit: Maximum allowed deepening rounds
        
    Returns:
        Dictionary with would_apply diff for all signals
    """
    diff: Dict[str, Any] = {
        "v1_applied": {},
        "v1_log_only": {},
    }
    
    # depth_budget_delta (would be applied in v1)
    if signals.depth_budget_delta != 0:
        original_rounds = plan.max_deepening_rounds
        new_rounds = plan.max_deepening_rounds + signals.depth_budget_delta
        new_rounds = max(1, min(max_deepening_rounds_limit, new_rounds))
        
        diff["v1_applied"]["depth_budget_delta"] = {
            "current": original_rounds,
            "would_become": new_rounds,
            "delta": signals.depth_budget_delta,
            "would_change": new_rounds != original_rounds,
        }
    
    # Log-only signals (would NOT be applied in v1)
    if signals.redundancy_check:
        diff["v1_log_only"]["redundancy_check"] = {
            "value": True,
            "reason": "TODO: Requires diagnostic step system",
        }
    
    if signals.force_falsification_step:
        diff["v1_log_only"]["force_falsification_step"] = {
            "value": True,
            "reason": "TODO: Requires step injection capability",
        }
    
    if signals.branch_pruning_suggested:
        diff["v1_log_only"]["branch_pruning_suggested"] = {
            "value": True,
            "reason": "TODO: Requires pruning flag in planner",
        }
    
    if signals.retrieval_diversify:
        diff["v1_log_only"]["retrieval_diversify"] = {
            "value": True,
            "reason": "TODO: Requires retrieval system integration",
        }
    
    if signals.council_diversity_min != 1:
        diff["v1_log_only"]["council_diversity_min"] = {
            "value": signals.council_diversity_min,
            "reason": "TODO: Requires council config integration",
        }
    
    if signals.confidence_penalty_delta != 0.0:
        diff["v1_log_only"]["confidence_penalty_delta"] = {
            "value": signals.confidence_penalty_delta,
            "reason": "TODO: Requires confidence system integration",
        }
    
    if signals.exploration_bias_delta != 0.0:
        diff["v1_log_only"]["exploration_bias_delta"] = {
            "value": signals.exploration_bias_delta,
            "reason": "TODO: Requires exploration config integration",
        }
    
    return diff


def _log_unapplied_signals(signals: PressureSignals) -> None:
    """Log signals that are NOT applied in v1."""
    unapplied = []
    
    if signals.redundancy_check:
        unapplied.append("redundancy_check=True")
    
    if signals.force_falsification_step:
        unapplied.append("force_falsification_step=True")
    
    if signals.branch_pruning_suggested:
        unapplied.append("branch_pruning_suggested=True")
    
    if signals.retrieval_diversify:
        unapplied.append("retrieval_diversify=True")
    
    if signals.council_diversity_min != 1:
        unapplied.append(f"council_diversity_min={signals.council_diversity_min}")
    
    if signals.confidence_penalty_delta != 0.0:
        unapplied.append(f"confidence_penalty_delta={signals.confidence_penalty_delta}")
    
    if signals.exploration_bias_delta != 0.0:
        unapplied.append(f"exploration_bias_delta={signals.exploration_bias_delta}")
    
    if unapplied:
        logger.debug(
            f"[BIO_PRIORS] Log-only signals (v1 not applied): {', '.join(unapplied)}"
        )


def get_v1_scope_info() -> Dict[str, Any]:
    """
    Get information about soft v1 scope.
    
    Returns:
        Dictionary with applied and log-only fields for v1
    """
    return {
        "version": "v1",
        "applied_fields": list(V1_APPLIED_FIELDS),
        "log_only_fields": list(V1_LOG_ONLY_FIELDS),
        "note": "Only depth_budget_delta is applied in v1. Other signals require dedicated injection points.",
    }

