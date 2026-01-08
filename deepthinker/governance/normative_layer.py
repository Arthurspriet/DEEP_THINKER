"""
Normative Control Layer for DeepThinker.

Main entry point for governance evaluation. The NormativeController
produces verdicts (ALLOW/WARN/BLOCK) based on deterministic rule evaluation.

Design principles:
- Does NOT generate text
- Does NOT reason
- Does NOT store memory
- Judges outputs and governs progression
- Enforceable even if the model disagrees
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from .violation import Violation, ViolationType
from .rule_engine import RuleEngine, GovernanceConfig, load_governance_config
from .phase_contracts import get_governance_contract, get_evidence_budget, EvidenceBudget

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState
    from ..decisions.decision_emitter import DecisionEmitter

logger = logging.getLogger(__name__)


class VerdictStatus(str, Enum):
    """
    Governance verdict status.
    
    ALLOW: Phase output is admissible, proceed to next phase
    WARN: Phase output has issues, proceed with penalties applied
    BLOCK: Phase output is inadmissible, must retry or take corrective action
    """
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


class RecommendedAction(str, Enum):
    """
    Recommended corrective action for governance violations.
    """
    NONE = "NONE"
    RETRY_PHASE = "RETRY_PHASE"
    FORCE_WEB_SEARCH = "FORCE_WEB_SEARCH"
    SCOPE_REDUCTION = "SCOPE_REDUCTION"


@dataclass
class NormativeVerdict:
    """
    Result of normative governance evaluation.
    
    Attributes:
        status: ALLOW, WARN, or BLOCK
        violations: List of detected violations
        recommended_action: Suggested corrective action
        confidence_penalty: Amount to reduce stated confidence
        epistemic_risk: Computed epistemic risk score (0-1)
        aggregate_severity: Combined severity of all violations
        phase_name: Phase that was evaluated
        can_retry: Whether the phase can be retried
        max_retries: Maximum retries allowed given current resources
        governance_summary: Summary dict for reporting
        escalation_hint: Model-Aware Phase Stabilization hint for model escalation
    """
    
    status: VerdictStatus
    violations: List[Violation] = field(default_factory=list)
    recommended_action: RecommendedAction = RecommendedAction.NONE
    confidence_penalty: float = 0.0
    epistemic_risk: float = 0.0
    aggregate_severity: float = 0.0
    phase_name: str = ""
    can_retry: bool = True
    max_retries: int = 2
    governance_summary: Dict[str, Any] = field(default_factory=dict)
    escalation_hint: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "violations": [v.to_dict() for v in self.violations],
            "recommended_action": self.recommended_action.value,
            "confidence_penalty": self.confidence_penalty,
            "epistemic_risk": self.epistemic_risk,
            "aggregate_severity": self.aggregate_severity,
            "phase_name": self.phase_name,
            "can_retry": self.can_retry,
            "max_retries": self.max_retries,
            "governance_summary": self.governance_summary,
            "escalation_hint": self.escalation_hint,
        }
    
    def has_hard_violations(self) -> bool:
        """Check if any hard violations exist."""
        return any(v.is_hard for v in self.violations)
    
    def get_hard_violations(self) -> List[Violation]:
        """Get all hard violations."""
        return [v for v in self.violations if v.is_hard]
    
    def get_violations_by_type(self, violation_type: ViolationType) -> List[Violation]:
        """Get violations of a specific type."""
        return [v for v in self.violations if v.type == violation_type]
    
    def get_violation_types(self) -> List[str]:
        """Get list of violation type strings for escalation signal."""
        return [v.type.value for v in self.violations]


class NormativeController:
    """
    Main governance controller.
    
    Evaluates phase outputs against governance rules and produces
    verdicts that determine whether execution can proceed.
    
    Usage:
        controller = NormativeController()
        verdict = controller.evaluate(
            phase_name="synthesis",
            phase_output=phase.artifacts,
            mission_state=state
        )
        
        if verdict.status == VerdictStatus.BLOCK:
            # Handle blocked phase
            pass
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        gpu_manager: Optional[Any] = None,
        decision_emitter: Optional["DecisionEmitter"] = None,
    ):
        """
        Initialize the normative controller.
        
        Args:
            config_path: Optional path to governance config YAML
            gpu_manager: Optional GPU manager for resource-aware governance
            decision_emitter: Optional DecisionEmitter for accountability logging
        """
        self.config = load_governance_config(config_path)
        self.rule_engine = RuleEngine(self.config)
        self.gpu_manager = gpu_manager
        self._decision_emitter = decision_emitter
        
        # Track phase retries for escalation
        self._phase_retry_counts: Dict[str, int] = {}
        
        # Track blocked phases for reporting
        self._blocked_phases: List[str] = []
        self._total_violations: int = 0
    
    def evaluate(
        self,
        phase_name: str,
        phase_output: Dict[str, Any],
        mission_state: Optional["MissionState"] = None,
    ) -> NormativeVerdict:
        """
        Evaluate phase output against governance rules.
        
        This is the main entry point. It:
        1. Runs the rule engine to detect violations
        2. Aggregates violations into severity score
        3. Determines verdict status (ALLOW/WARN/BLOCK)
        4. Computes confidence penalty and epistemic risk
        5. Determines recommended corrective action
        
        Args:
            phase_name: Name of the phase being evaluated
            phase_output: Phase artifacts/output dictionary
            mission_state: Optional mission state for context
            
        Returns:
            NormativeVerdict with status and details
        """
        # Get current GPU pressure for resource-aware evaluation
        gpu_pressure = self._get_gpu_pressure()
        
        # Run rule engine
        violations = self.rule_engine.evaluate_rules(
            phase_name=phase_name,
            phase_output=phase_output,
            mission_state=mission_state,
            gpu_pressure=gpu_pressure,
        )
        
        # Epistemic Hardening Phase 3: Check evidence budget
        evidence_budget_violations = self._check_evidence_budget(
            phase_name=phase_name,
            phase_output=phase_output,
            mission_state=mission_state,
        )
        violations.extend(evidence_budget_violations)
        
        # Compute aggregate severity
        aggregate_severity = self._compute_aggregate_severity(violations, phase_name)
        
        # Determine verdict status
        status = self._determine_status(violations, aggregate_severity, phase_name)
        
        # Compute epistemic risk
        epistemic_risk = self._compute_epistemic_risk(violations, mission_state)
        
        # Compute confidence penalty
        confidence_penalty = self._compute_confidence_penalty(
            phase_output, epistemic_risk, mission_state
        )
        
        # Determine recommended action
        recommended_action = self._determine_action(violations, status)
        
        # Check retry limits
        can_retry, max_retries = self._check_retry_limits(phase_name, gpu_pressure)
        
        # Build governance summary for reporting
        governance_summary = self._build_summary(
            violations, status, epistemic_risk, phase_name
        )
        
        # Model-Aware Phase Stabilization: Build escalation hint for model routing
        escalation_hint = self._build_escalation_hint(
            violations=violations,
            status=status,
            aggregate_severity=aggregate_severity,
            phase_name=phase_name,
        )
        
        # Track statistics
        self._total_violations += len(violations)
        if status == VerdictStatus.BLOCK:
            self._blocked_phases.append(phase_name)
        
        # Log verdict
        self._log_verdict(phase_name, status, violations, aggregate_severity)
        
        # Build the verdict object
        verdict = NormativeVerdict(
            status=status,
            violations=violations,
            recommended_action=recommended_action,
            confidence_penalty=confidence_penalty,
            epistemic_risk=epistemic_risk,
            aggregate_severity=aggregate_severity,
            phase_name=phase_name,
            can_retry=can_retry,
            max_retries=max_retries,
            governance_summary=governance_summary,
            escalation_hint=escalation_hint,
        )
        
        # Decision Accountability: Emit GOVERNANCE_INTERVENTION decision record
        self._emit_governance_decision(verdict, mission_state)
        
        return verdict
    
    def record_retry(self, phase_name: str) -> None:
        """Record a phase retry for escalation tracking."""
        self._phase_retry_counts[phase_name] = self._phase_retry_counts.get(phase_name, 0) + 1
    
    def get_retry_count(self, phase_name: str) -> int:
        """Get retry count for a phase."""
        return self._phase_retry_counts.get(phase_name, 0)
    
    def reset_phase_tracking(self, phase_name: str) -> None:
        """Reset tracking for a phase (e.g., after successful completion)."""
        self._phase_retry_counts.pop(phase_name, None)
    
    def set_decision_emitter(self, emitter: "DecisionEmitter") -> None:
        """
        Set the decision emitter for accountability logging.
        
        Args:
            emitter: DecisionEmitter instance
        """
        self._decision_emitter = emitter
    
    def _emit_governance_decision(
        self,
        verdict: NormativeVerdict,
        mission_state: Optional["MissionState"],
    ) -> Optional[str]:
        """
        Emit a GOVERNANCE_INTERVENTION decision record for accountability.
        
        Only emits for BLOCK and WARN verdicts (not ALLOW).
        
        Args:
            verdict: The governance verdict produced
            mission_state: Current mission state
            
        Returns:
            decision_id if emitted, None otherwise
        """
        if not self._decision_emitter or not mission_state:
            return None
        
        # Only record interventions (not ALLOWs)
        if verdict.status == VerdictStatus.ALLOW:
            return None
        
        try:
            # Get phase type from contract if available
            phase_type = "unknown"
            try:
                contract = get_governance_contract(verdict.phase_name)
                phase_type = contract.phase_type if hasattr(contract, 'phase_type') else "unknown"
            except Exception:
                pass
            
            decision_id = self._decision_emitter.emit_governance_intervention(
                mission_id=mission_state.mission_id,
                phase_id=verdict.phase_name,
                phase_type=phase_type,
                verdict_status=verdict.status.value,
                violation_types=verdict.get_violation_types(),
                aggregate_severity=verdict.aggregate_severity,
                recommended_action=verdict.recommended_action.value,
                can_retry=verdict.can_retry,
                triggered_by=mission_state.last_model_decision_id,
            )
            
            # Track in mission state
            if decision_id:
                mission_state.set_last_governance_decision(decision_id)
            
            return decision_id
            
        except Exception as e:
            logger.debug(f"[DECISION] Failed to emit governance decision: {e}")
            return None
    
    def get_governance_report(self) -> Dict[str, Any]:
        """
        Get governance summary report for final mission output.
        
        Returns summary only (not verbose violation list) as per spec.
        """
        return {
            "epistemic_risk_score": self._get_latest_epistemic_risk(),
            "total_violations": self._total_violations,
            "phases_blocked": len(self._blocked_phases),
            "blocked_phase_names": self._blocked_phases.copy(),
        }
    
    def reset(self) -> None:
        """Reset controller state for new mission."""
        self._phase_retry_counts.clear()
        self._blocked_phases.clear()
        self._total_violations = 0
        self.rule_engine.clear_history()
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _get_gpu_pressure(self) -> str:
        """Get current GPU pressure level."""
        if self.gpu_manager is None:
            return "low"
        
        try:
            return self.gpu_manager.get_resource_pressure()
        except Exception:
            return "low"
    
    def _check_evidence_budget(
        self,
        phase_name: str,
        phase_output: Dict[str, Any],
        mission_state: Optional["MissionState"],
    ) -> List[Violation]:
        """
        Check if evidence budget is met for the phase.
        
        Epistemic Hardening Phase 3: Enforce minimum evidence requirements.
        
        Args:
            phase_name: Name of the phase
            phase_output: Phase artifacts/output dictionary
            mission_state: Optional mission state for context
            
        Returns:
            List of violations if budget not met
        """
        violations = []
        
        # Check if evidence budgets are enabled
        if mission_state is not None:
            constraints = getattr(mission_state, "constraints", None)
            if constraints and not getattr(constraints, "enable_evidence_budgets", False):
                return violations
        
        # Get evidence budget for this phase
        budget = get_evidence_budget(phase_name)
        
        if not budget.enforce:
            return violations
        
        # Extract source count from phase output and mission state
        sources_count = 0
        grounded_claims_count = 0
        
        # Check phase output for sources (with fallback for sources_suggested)
        for sources_key in ["sources", "sources_suggested"]:
            if sources_key in phase_output:
                sources = phase_output[sources_key]
                if isinstance(sources, list):
                    sources_count = len(sources)
                elif isinstance(sources, int):
                    sources_count = sources
                break  # Use first matching key
        
        # Check phase output for grounded claims
        if "grounded_claims" in phase_output:
            claims = phase_output["grounded_claims"]
            if isinstance(claims, list):
                grounded_claims_count = len(claims)
            elif isinstance(claims, int):
                grounded_claims_count = claims
        
        # Also check mission state epistemic telemetry
        if mission_state is not None:
            telemetry = getattr(mission_state, "epistemic_telemetry", {})
            
            # Get sources for this phase
            sources_per_phase = telemetry.get("sources_per_phase", {})
            phase_sources = sources_per_phase.get(phase_name, 0)
            sources_count = max(sources_count, phase_sources)
            
            # Estimate grounded claims from grounded_claim_ratio
            grounded_ratio = telemetry.get("grounded_claim_ratio", 0.0)
            claim_results = telemetry.get("claim_validation_results", {})
            total_claims = claim_results.get("total_claims", 0)
            if total_claims > 0 and grounded_claims_count == 0:
                grounded_claims_count = int(total_claims * grounded_ratio)
        
        # Check budget
        is_met, reason = budget.is_met(sources_count, grounded_claims_count)
        
        if not is_met:
            shortfall = budget.get_shortfall(sources_count, grounded_claims_count)
            
            violation = Violation(
                type=ViolationType.EPISTEMIC_BUDGET_NOT_MET,
                severity=0.7,
                description=f"Evidence budget not met: {reason}",
                phase_name=phase_name,
                is_hard=True,  # Evidence budget violations block phase
                details={
                    "sources_count": sources_count,
                    "grounded_claims_count": grounded_claims_count,
                    "budget": budget.to_dict(),
                    "shortfall": shortfall,
                }
            )
            violations.append(violation)
            
            logger.warning(
                f"[EVIDENCE BUDGET] Phase '{phase_name}' budget not met: {reason}. "
                f"Shortfall: {shortfall}"
            )
        
        return violations
    
    def _compute_aggregate_severity(
        self,
        violations: List[Violation],
        phase_name: str,
    ) -> float:
        """
        Compute aggregate severity from violations.
        
        Uses weighted average with phase-specific strictness.
        """
        if not violations:
            return 0.0
        
        # Sum severities
        total_severity = sum(v.severity for v in violations)
        
        # Weighted by violation count (diminishing returns)
        count_factor = min(1.0, len(violations) / 5)
        base_severity = total_severity / len(violations)
        
        # Apply phase strictness
        phase_strictness = self.config.get_phase_strictness(phase_name)
        
        # Combine: base * (1 + count_factor) * strictness
        aggregate = base_severity * (1 + count_factor * 0.5) * phase_strictness
        
        return min(1.0, aggregate)
    
    def _determine_status(
        self,
        violations: List[Violation],
        aggregate_severity: float,
        phase_name: str,
    ) -> VerdictStatus:
        """Determine verdict status based on violations and severity."""
        # No violations = ALLOW
        if not violations:
            return VerdictStatus.ALLOW
        
        # Any hard violation = BLOCK
        if any(v.is_hard for v in violations):
            return VerdictStatus.BLOCK
        
        # Check escalation
        retry_count = self._phase_retry_counts.get(phase_name, 0)
        threshold = self.config.escalation.get("repeated_violation_threshold", 2)
        escalate = self.config.escalation.get("escalate_from_warn_to_block", True)
        
        if escalate and retry_count >= threshold:
            logger.warning(
                f"Escalating to BLOCK for phase '{phase_name}' after {retry_count} retries"
            )
            return VerdictStatus.BLOCK
        
        # Check severity thresholds
        if aggregate_severity >= self.config.block_threshold:
            return VerdictStatus.BLOCK
        
        if aggregate_severity >= self.config.warn_threshold:
            return VerdictStatus.WARN
        
        return VerdictStatus.ALLOW
    
    def _compute_epistemic_risk(
        self,
        violations: List[Violation],
        mission_state: Optional["MissionState"],
    ) -> float:
        """
        Compute epistemic risk score.
        
        Combines violation-based risk with mission state telemetry.
        """
        # Base risk from violations
        epistemic_violations = [
            v for v in violations
            if v.type.value.startswith("epistemic_")
        ]
        
        if epistemic_violations:
            violation_risk = sum(v.severity for v in epistemic_violations) / len(epistemic_violations)
        else:
            violation_risk = 0.0
        
        # Get existing risk from mission state
        state_risk = 0.0
        if mission_state is not None:
            telemetry = getattr(mission_state, "epistemic_telemetry", {})
            state_risk = telemetry.get("epistemic_risk", 0.0)
        
        # Combine with weighted average
        combined_risk = 0.6 * state_risk + 0.4 * violation_risk
        
        return min(1.0, combined_risk)
    
    def _compute_confidence_penalty(
        self,
        phase_output: Dict[str, Any],
        epistemic_risk: float,
        mission_state: Optional["MissionState"],
    ) -> float:
        """
        Compute confidence penalty to apply.
        
        Implements: final_confidence = min(stated_confidence, 1 - epistemic_risk)
        """
        # Get stated confidence
        stated_confidence = self._extract_stated_confidence(phase_output)
        
        if stated_confidence is None:
            # Try from mission state
            if mission_state is not None:
                convergence = getattr(mission_state, "convergence_state", None)
                if convergence:
                    stated_confidence = getattr(convergence, "confidence_score", None)
        
        if stated_confidence is None:
            return 0.0
        
        # Clamp confidence by epistemic risk
        max_allowed = 1.0 - epistemic_risk
        
        if stated_confidence > max_allowed:
            penalty = stated_confidence - max_allowed
            return penalty
        
        return 0.0
    
    def _determine_action(
        self,
        violations: List[Violation],
        status: VerdictStatus,
    ) -> RecommendedAction:
        """Determine recommended corrective action."""
        if status == VerdictStatus.ALLOW:
            return RecommendedAction.NONE
        
        # Check configured action mappings
        action_configs = self.config.recommended_actions
        
        # Count violations by type prefix
        type_counts: Dict[str, int] = {}
        for v in violations:
            prefix = v.type.value.split("_")[0]
            type_counts[prefix] = type_counts.get(prefix, 0) + 1
        
        # Check epistemic dominant
        if "epistemic_dominant" in action_configs:
            cfg = action_configs["epistemic_dominant"]
            if type_counts.get("epistemic", 0) >= cfg.get("min_count", 2):
                return RecommendedAction.FORCE_WEB_SEARCH
        
        # Check structural violation
        if "structural_violation" in action_configs:
            cfg = action_configs["structural_violation"]
            structural_types = {ViolationType.STRUCTURAL_MISSING_SCENARIOS, ViolationType.STRUCTURAL_MALFORMED_SCENARIO}
            if any(v.type in structural_types for v in violations):
                return RecommendedAction.RETRY_PHASE
        
        # Check excessive violations
        if "excessive_violations" in action_configs:
            cfg = action_configs["excessive_violations"]
            if len(violations) >= cfg.get("min_total_violations", 5):
                return RecommendedAction.SCOPE_REDUCTION
        
        # Default for BLOCK: retry phase
        if status == VerdictStatus.BLOCK:
            return RecommendedAction.RETRY_PHASE
        
        return RecommendedAction.NONE
    
    def _check_retry_limits(
        self,
        phase_name: str,
        gpu_pressure: str,
    ) -> tuple:
        """Check if retry is allowed given current state."""
        retry_count = self._phase_retry_counts.get(phase_name, 0)
        
        # Get max retries from resource modifier
        modifier = self.config.get_resource_modifier(gpu_pressure)
        max_retries = modifier.get("max_retries", 2)
        
        can_retry = retry_count < max_retries
        
        return can_retry, max_retries
    
    def _build_summary(
        self,
        violations: List[Violation],
        status: VerdictStatus,
        epistemic_risk: float,
        phase_name: str,
    ) -> Dict[str, Any]:
        """Build governance summary for reporting."""
        return {
            "phase_name": phase_name,
            "status": status.value,
            "violation_count": len(violations),
            "hard_violation_count": sum(1 for v in violations if v.is_hard),
            "epistemic_risk": epistemic_risk,
            "violation_types": list(set(v.type.value for v in violations)),
        }
    
    def _build_escalation_hint(
        self,
        violations: List[Violation],
        status: VerdictStatus,
        aggregate_severity: float,
        phase_name: str,
    ) -> Dict[str, Any]:
        """
        Build escalation hint for model-aware retry logic.
        
        Model-Aware Phase Stabilization: This hint is passed to the model
        supervisor to enable intelligent model escalation on retries.
        
        Args:
            violations: List of detected violations
            status: Verdict status
            aggregate_severity: Combined severity score
            phase_name: Name of the phase
            
        Returns:
            Escalation hint dictionary
        """
        if status != VerdictStatus.BLOCK:
            return {}
        
        # Categorize violations for escalation guidance
        violation_categories = {}
        for v in violations:
            category = v.type.value.split("_")[0]  # e.g., "epistemic", "structural"
            violation_categories[category] = violation_categories.get(category, 0) + 1
        
        # Determine if this seems like a model capability issue
        # Structural violations and high-severity epistemic issues suggest model upgrade
        suggests_model_upgrade = (
            "structural" in violation_categories or
            aggregate_severity >= 0.7 or
            violation_categories.get("epistemic", 0) >= 2
        )
        
        return {
            "violation_types": [v.type.value for v in violations],
            "violation_categories": violation_categories,
            "aggregate_severity": aggregate_severity,
            "suggests_model_upgrade": suggests_model_upgrade,
            "retry_count": self._phase_retry_counts.get(phase_name, 0),
        }
    
    def _log_verdict(
        self,
        phase_name: str,
        status: VerdictStatus,
        violations: List[Violation],
        aggregate_severity: float,
    ) -> None:
        """Log verdict for observability."""
        level = logging.INFO if status == VerdictStatus.ALLOW else logging.WARNING
        
        hard_count = sum(1 for v in violations if v.is_hard)
        
        logger.log(
            level,
            f"[GOVERNANCE] Phase '{phase_name}': {status.value} "
            f"(violations={len(violations)}, hard={hard_count}, severity={aggregate_severity:.2f})"
        )
        
        if violations and logger.isEnabledFor(logging.DEBUG):
            for v in violations[:5]:  # Limit log spam
                logger.debug(f"  - {v}")
    
    def _extract_stated_confidence(self, phase_output: Dict[str, Any]) -> Optional[float]:
        """Extract stated confidence from phase output."""
        for key in ["confidence", "confidence_score", "stated_confidence"]:
            if key in phase_output:
                value = phase_output[key]
                if isinstance(value, (int, float)):
                    return float(value)
        return None
    
    def _get_latest_epistemic_risk(self) -> float:
        """Get latest epistemic risk from rule engine history."""
        violations = self.rule_engine.get_violation_history()
        epistemic_violations = [
            v for v in violations
            if v.type.value.startswith("epistemic_")
        ]
        
        if not epistemic_violations:
            return 0.0
        
        return sum(v.severity for v in epistemic_violations) / len(epistemic_violations)


# Global controller instance (optional singleton)
_controller: Optional[NormativeController] = None


def get_normative_controller(
    config_path: Optional[str] = None,
    gpu_manager: Optional[Any] = None,
) -> NormativeController:
    """
    Get or create global normative controller instance.
    
    Args:
        config_path: Optional path to governance config
        gpu_manager: Optional GPU manager
        
    Returns:
        NormativeController instance
    """
    global _controller
    
    if _controller is None:
        _controller = NormativeController(
            config_path=config_path,
            gpu_manager=gpu_manager,
        )
    
    return _controller

