"""
Decision Emitter for DeepThinker.

Provides a stateless, opt-in, non-blocking interface for emitting
decision records at key choice points in the system.

Now includes async review queue integration (Sprint 4).
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .decision_record import DecisionRecord, DecisionType

if TYPE_CHECKING:
    from .decision_store import DecisionStore
    from ..review.review_queue import ReviewQueue

logger = logging.getLogger(__name__)


class DecisionEmitter:
    """
    Stateless emitter for Decision Records.
    
    Key properties:
    - Opt-in: Only emits if enabled and store is provided
    - Non-blocking: Catches all exceptions internally
    - Stateless: No internal state, all context passed in
    
    Usage:
        emitter = DecisionEmitter(store=decision_store, enabled=True)
        
        decision_id = emitter.emit(
            decision_type=DecisionType.MODEL_SELECTION,
            mission_id="abc-123",
            phase_id="Research",
            phase_type="reconnaissance",
            options_considered=["gemma3:12b", "cogito:14b"],
            selected_option="cogito:14b",
            constraints_snapshot={"time_remaining": 45.2, "importance": 0.9},
            rationale="High importance phase, adequate time remaining",
            confidence=0.85,
        )
    """
    
    def __init__(
        self,
        store: Optional["DecisionStore"] = None,
        enabled: bool = True,
        review_queue: Optional["ReviewQueue"] = None,
    ):
        """
        Initialize the decision emitter.
        
        Args:
            store: DecisionStore for persistence (optional)
            enabled: Whether emission is enabled (default True)
            review_queue: Optional ReviewQueue for async review
        """
        self._store = store
        self._enabled = enabled
        self._review_queue = review_queue
        self._review_types: set = set()  # Decision types to review
    
    @property
    def enabled(self) -> bool:
        """Check if emitter is enabled and has a store."""
        return self._enabled and self._store is not None
    
    def enable(self) -> None:
        """Enable decision emission."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable decision emission."""
        self._enabled = False
    
    def set_store(self, store: "DecisionStore") -> None:
        """
        Set the decision store.
        
        Args:
            store: DecisionStore instance for persistence
        """
        self._store = store
    
    def set_review_queue(self, queue: "ReviewQueue") -> None:
        """
        Set the review queue for async review.
        
        Args:
            queue: ReviewQueue instance
        """
        self._review_queue = queue
    
    def configure_review(
        self,
        review_routing: bool = True,
        review_alignment: bool = True,
        review_phase_termination: bool = True,
        review_governance: bool = False,
    ) -> None:
        """
        Configure which decision types to queue for review.
        
        Args:
            review_routing: Review MODEL_SELECTION decisions
            review_alignment: Review ALIGNMENT_ACTION decisions
            review_phase_termination: Review PHASE_TERMINATION decisions
            review_governance: Review GOVERNANCE_INTERVENTION decisions
        """
        self._review_types = set()
        
        if review_routing:
            self._review_types.add(DecisionType.MODEL_SELECTION)
        if review_alignment:
            # Alignment actions would be a new DecisionType
            pass
        if review_phase_termination:
            self._review_types.add(DecisionType.PHASE_TERMINATION)
        if review_governance:
            self._review_types.add(DecisionType.GOVERNANCE_INTERVENTION)
    
    def emit(
        self,
        decision_type: DecisionType,
        mission_id: str,
        phase_id: str,
        phase_type: str,
        options_considered: List[str],
        selected_option: str,
        constraints_snapshot: Dict[str, Any],
        rationale: str,
        confidence: float = 0.8,
        triggered_by: Optional[str] = None,
    ) -> Optional[str]:
        """
        Emit a decision record.
        
        Creates a DecisionRecord and appends it to the mission's decision log.
        Non-blocking: catches all exceptions and logs warnings.
        
        Args:
            decision_type: Category of decision
            mission_id: Parent mission identifier
            phase_id: Phase name where decision occurred
            phase_type: Type of phase (reconnaissance, synthesis, etc.)
            options_considered: Abstract labels of alternatives
            selected_option: The choice that was made
            constraints_snapshot: State at decision time
            rationale: Brief justification (NOT chain-of-thought)
            confidence: Decision confidence (0.0-1.0)
            triggered_by: ID of decision that caused this one
            
        Returns:
            decision_id if emitted successfully, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            # Generate unique ID
            decision_id = str(uuid.uuid4())
            
            # Create record
            record = DecisionRecord(
                decision_id=decision_id,
                decision_type=decision_type,
                timestamp=datetime.utcnow(),
                mission_id=mission_id,
                phase_id=phase_id,
                phase_type=phase_type,
                options_considered=options_considered,
                selected_option=selected_option,
                rationale=rationale,
                confidence=confidence,
                constraints_snapshot=constraints_snapshot,
                triggered_by_decision_id=triggered_by,
            )
            
            # Persist
            self._store.write(record)
            
            # Async review hook (non-blocking)
            self._maybe_enqueue_review(record)
            
            # Log for observability
            logger.debug(
                f"[DECISION] Emitted {decision_type.value} decision {decision_id[:8]} "
                f"for phase {phase_id}"
            )
            
            return decision_id
            
        except Exception as e:
            # Non-blocking: log warning and continue
            logger.warning(
                f"[DECISION] Failed to emit {decision_type.value} decision: {e}"
            )
            return None
    
    def emit_model_selection(
        self,
        mission_id: str,
        phase_id: str,
        phase_type: str,
        models_considered: List[str],
        models_selected: List[str],
        time_remaining: float,
        importance: float,
        gpu_pressure: str,
        downgraded: bool,
        reason: str,
        triggered_by: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convenience method for MODEL_SELECTION decisions.
        
        Args:
            mission_id: Parent mission identifier
            phase_id: Phase name
            phase_type: Type of phase
            models_considered: Models that were options
            models_selected: Models that were chosen
            time_remaining: Minutes remaining in mission
            importance: Phase importance (0-1)
            gpu_pressure: Current GPU pressure level
            downgraded: Whether selection was downgraded
            reason: Supervisor's reason string
            triggered_by: ID of triggering decision
            
        Returns:
            decision_id if emitted successfully
        """
        return self.emit(
            decision_type=DecisionType.MODEL_SELECTION,
            mission_id=mission_id,
            phase_id=phase_id,
            phase_type=phase_type,
            options_considered=models_considered,
            selected_option=", ".join(models_selected),
            constraints_snapshot={
                "time_remaining_minutes": time_remaining,
                "importance": importance,
                "gpu_pressure": gpu_pressure,
                "downgraded": downgraded,
            },
            rationale=reason,
            confidence=0.9 if not downgraded else 0.7,
            triggered_by=triggered_by,
        )
    
    def emit_governance_intervention(
        self,
        mission_id: str,
        phase_id: str,
        phase_type: str,
        verdict_status: str,
        violation_types: List[str],
        aggregate_severity: float,
        recommended_action: str,
        can_retry: bool,
        triggered_by: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convenience method for GOVERNANCE_INTERVENTION decisions.
        
        Args:
            mission_id: Parent mission identifier
            phase_id: Phase name
            phase_type: Type of phase
            verdict_status: ALLOW/WARN/BLOCK
            violation_types: Types of violations detected
            aggregate_severity: Combined severity score
            recommended_action: Suggested corrective action
            can_retry: Whether phase can be retried
            triggered_by: ID of triggering decision
            
        Returns:
            decision_id if emitted successfully
        """
        return self.emit(
            decision_type=DecisionType.GOVERNANCE_INTERVENTION,
            mission_id=mission_id,
            phase_id=phase_id,
            phase_type=phase_type,
            options_considered=["ALLOW", "WARN", "BLOCK"],
            selected_option=verdict_status,
            constraints_snapshot={
                "violation_types": violation_types,
                "aggregate_severity": aggregate_severity,
                "recommended_action": recommended_action,
                "can_retry": can_retry,
            },
            rationale=f"{len(violation_types)} violations, severity={aggregate_severity:.2f}",
            confidence=min(1.0, aggregate_severity + 0.3),
            triggered_by=triggered_by,
        )
    
    def emit_retry_escalation(
        self,
        mission_id: str,
        phase_id: str,
        phase_type: str,
        from_models: List[str],
        to_models: List[str],
        retry_count: int,
        escalation_reason: str,
        triggered_by: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convenience method for RETRY_ESCALATION decisions.
        
        Args:
            mission_id: Parent mission identifier
            phase_id: Phase name
            phase_type: Type of phase
            from_models: Models that failed
            to_models: Models to try next
            retry_count: Current retry number
            escalation_reason: Why escalation is happening
            triggered_by: ID of governance decision that triggered this
            
        Returns:
            decision_id if emitted successfully
        """
        return self.emit(
            decision_type=DecisionType.RETRY_ESCALATION,
            mission_id=mission_id,
            phase_id=phase_id,
            phase_type=phase_type,
            options_considered=from_models + to_models,
            selected_option=", ".join(to_models),
            constraints_snapshot={
                "from_models": from_models,
                "to_models": to_models,
                "retry_count": retry_count,
            },
            rationale=escalation_reason,
            confidence=0.7,
            triggered_by=triggered_by,
        )
    
    def emit_phase_termination(
        self,
        mission_id: str,
        phase_id: str,
        phase_type: str,
        status: str,
        outcome_cause: str,
        time_remaining: float,
        quality_score: Optional[float],
        retry_count: int,
        governance_severity: float,
        triggered_by: Optional[str] = None,
        # Depth control fields (optional)
        depth_achieved: Optional[float] = None,
        depth_target: Optional[float] = None,
        depth_gap: Optional[float] = None,
        enrichment_passes: int = 0,
        time_unused_seconds: float = 0.0,
        depth_pressure_applied: bool = False,
    ) -> Optional[str]:
        """
        Convenience method for PHASE_TERMINATION decisions.
        
        Args:
            mission_id: Parent mission identifier
            phase_id: Phase name
            phase_type: Type of phase
            status: Final phase status (completed, failed, etc.)
            outcome_cause: Primary cause from OutcomeCause enum
            time_remaining: Minutes remaining at phase end
            quality_score: Quality score if available
            retry_count: Number of retries attempted
            governance_severity: Last governance severity
            triggered_by: ID of decision that led here
            depth_achieved: Final computed depth score (optional)
            depth_target: Phase-specific depth target (optional)
            depth_gap: Remaining depth gap (optional)
            enrichment_passes: Number of enrichment passes performed
            time_unused_seconds: Time remaining at phase end in seconds
            depth_pressure_applied: Whether depth pressure was active
            
        Returns:
            decision_id if emitted successfully
        """
        constraints = {
            "outcome_cause": outcome_cause,
            "time_remaining_at_end": time_remaining,
            "quality_score": quality_score,
            "retry_count": retry_count,
            "governance_severity": governance_severity,
        }
        
        # Add depth control fields if provided
        if depth_achieved is not None:
            constraints["depth_achieved"] = depth_achieved
        if depth_target is not None:
            constraints["depth_target"] = depth_target
        if depth_gap is not None:
            constraints["depth_gap"] = depth_gap
        if enrichment_passes > 0:
            constraints["enrichment_passes"] = enrichment_passes
        if time_unused_seconds > 0:
            constraints["time_unused_seconds"] = time_unused_seconds
        if depth_pressure_applied:
            constraints["depth_pressure_applied"] = depth_pressure_applied
        
        return self.emit(
            decision_type=DecisionType.PHASE_TERMINATION,
            mission_id=mission_id,
            phase_id=phase_id,
            phase_type=phase_type,
            options_considered=["completed", "completed_degraded", "failed", "skipped"],
            selected_option=status,
            constraints_snapshot=constraints,
            rationale=f"Outcome cause: {outcome_cause}",
            confidence=0.95 if status == "completed" else 0.8,
            triggered_by=triggered_by,
        )
    
    def emit_empty_output_escalation(
        self,
        mission_id: str,
        phase_id: str,
        phase_type: str,
        council_name: str,
        empty_fields: List[str],
        from_model: str,
        to_model: str,
        triggered_by: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convenience method for EMPTY_OUTPUT_ESCALATION decisions.
        
        Args:
            mission_id: Parent mission identifier
            phase_id: Phase name
            phase_type: Type of phase
            council_name: Name of council that detected empty output
            empty_fields: Which structured fields were empty
            from_model: Original model that produced empty output
            to_model: Escalation target model
            triggered_by: ID of triggering decision
            
        Returns:
            decision_id if emitted successfully
        """
        return self.emit(
            decision_type=DecisionType.EMPTY_OUTPUT_ESCALATION,
            mission_id=mission_id,
            phase_id=phase_id,
            phase_type=phase_type,
            options_considered=[from_model, to_model],
            selected_option=to_model,
            constraints_snapshot={
                "council_name": council_name,
                "empty_fields": empty_fields,
                "from_model": from_model,
                "to_model": to_model,
            },
            rationale=f"Empty {', '.join(empty_fields)} from {from_model}, escalating to {to_model}",
            confidence=0.6,
            triggered_by=triggered_by,
        )
    
    def attribute_cost(
        self,
        mission_id: str,
        decision_id: str,
        hardware_cost: float,
    ) -> bool:
        """
        Attribute hardware cost to an existing decision.
        
        Called post-execution to fill in hardware_cost_attributed.
        
        Args:
            mission_id: Mission containing the decision
            decision_id: ID of decision to update
            hardware_cost: Cost to attribute
            
        Returns:
            True if attribution succeeded
        """
        if not self.enabled:
            return False
        
        try:
            return self._store.update_cost(mission_id, decision_id, hardware_cost)
        except Exception as e:
            logger.warning(
                f"[DECISION] Failed to attribute cost to {decision_id[:8]}: {e}"
            )
            return False
    
    def _maybe_enqueue_review(self, record: DecisionRecord) -> None:
        """
        Maybe enqueue decision for async review.
        
        Non-blocking: fires and forgets.
        
        Args:
            record: DecisionRecord to potentially queue
        """
        # Check if review queue is configured
        if self._review_queue is None:
            return
        
        # Check if this decision type should be reviewed
        if record.decision_type not in self._review_types:
            return
        
        try:
            # Enqueue for review (non-blocking)
            queue_id = self._review_queue.enqueue(record.to_dict())
            
            if queue_id:
                logger.debug(
                    f"[DECISION] Queued for review: {record.decision_id[:8]} -> {queue_id}"
                )
        except Exception as e:
            # Non-blocking: log and continue
            logger.debug(f"[DECISION] Review queue enqueue failed (non-critical): {e}")

