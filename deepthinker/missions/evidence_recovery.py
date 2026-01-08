"""
Evidence Recovery Mode for DeepThinker Epistemic Hardening.

When epistemic grounding is insufficient, this module provides a recovery
mechanism that converts deepening operations into evidence gathering.

Entry conditions:
- grounded_claim_ratio < 0.2
- sources_per_phase == 0
- Repeated epistemic gate failures

Exit conditions:
- Minimum evidence budget reached
- Evidence target met
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .mission_types import MissionState

logger = logging.getLogger(__name__)


class RecoveryEntryReason(str, Enum):
    """Reasons for entering evidence recovery mode."""
    LOW_GROUNDED_RATIO = "low_grounded_claim_ratio"
    NO_SOURCES = "no_sources_in_phase"
    REPEATED_GATE_FAILURES = "repeated_epistemic_gate_failures"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class EvidenceRecoveryState:
    """
    Tracks evidence recovery mode state for a mission.
    
    When active, all deepening operations are converted to:
    - Evidence search (web search, document retrieval)
    - Claim extraction from search results
    - Citation attachment to existing claims
    
    Attributes:
        active: Whether recovery mode is currently active
        entry_reason: Why recovery mode was entered
        evidence_target: Number of sources/claims to collect
        evidence_collected: Number of sources/claims collected so far
        gate_failure_count: Number of consecutive gate failures
        phases_in_recovery: Phases that have been in recovery mode
        recovery_attempts: Total recovery attempts in this mission
        max_recovery_attempts: Maximum allowed recovery attempts
    """
    active: bool = False
    entry_reason: RecoveryEntryReason = RecoveryEntryReason.LOW_GROUNDED_RATIO
    evidence_target: int = 5
    evidence_collected: int = 0
    gate_failure_count: int = 0
    phases_in_recovery: List[str] = field(default_factory=list)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    # Thresholds for entry conditions
    GROUNDED_RATIO_THRESHOLD: float = 0.2
    GATE_FAILURE_THRESHOLD: int = 2
    
    def should_enter_recovery(
        self,
        grounded_claim_ratio: float,
        sources_in_phase: int,
        gate_failures: int
    ) -> tuple:
        """
        Check if recovery mode should be activated.
        
        Args:
            grounded_claim_ratio: Ratio of grounded claims (0-1)
            sources_in_phase: Number of sources in current phase
            gate_failures: Number of consecutive gate failures
            
        Returns:
            Tuple of (should_enter: bool, reason: RecoveryEntryReason)
        """
        if self.active:
            return False, self.entry_reason
        
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.warning(
                f"Max recovery attempts ({self.max_recovery_attempts}) reached, "
                "cannot enter recovery mode"
            )
            return False, RecoveryEntryReason.MANUAL_TRIGGER
        
        # Check conditions in order of priority
        if grounded_claim_ratio < self.GROUNDED_RATIO_THRESHOLD:
            return True, RecoveryEntryReason.LOW_GROUNDED_RATIO
        
        if sources_in_phase == 0:
            return True, RecoveryEntryReason.NO_SOURCES
        
        if gate_failures >= self.GATE_FAILURE_THRESHOLD:
            return True, RecoveryEntryReason.REPEATED_GATE_FAILURES
        
        return False, RecoveryEntryReason.MANUAL_TRIGGER
    
    def enter_recovery(
        self,
        reason: RecoveryEntryReason,
        phase_name: str,
        evidence_target: int = 5
    ) -> None:
        """
        Enter evidence recovery mode.
        
        Args:
            reason: Why recovery is being entered
            phase_name: Current phase name
            evidence_target: How many sources/claims to collect
        """
        self.active = True
        self.entry_reason = reason
        self.evidence_target = evidence_target
        self.evidence_collected = 0
        self.recovery_attempts += 1
        
        if phase_name not in self.phases_in_recovery:
            self.phases_in_recovery.append(phase_name)
        
        logger.info(
            f"[EVIDENCE RECOVERY] Entered recovery mode: "
            f"reason={reason.value}, target={evidence_target}, "
            f"attempt={self.recovery_attempts}/{self.max_recovery_attempts}"
        )
    
    def record_evidence(self, count: int = 1) -> None:
        """
        Record evidence collection progress.
        
        Args:
            count: Number of evidence items collected
        """
        self.evidence_collected += count
        logger.debug(
            f"[EVIDENCE RECOVERY] Collected {count} evidence items, "
            f"total={self.evidence_collected}/{self.evidence_target}"
        )
    
    def can_exit_recovery(self) -> tuple:
        """
        Check if recovery mode can be exited.
        
        Returns:
            Tuple of (can_exit: bool, reason: str)
        """
        if not self.active:
            return True, "Not in recovery mode"
        
        if self.evidence_collected >= self.evidence_target:
            return True, f"Evidence target met ({self.evidence_collected}/{self.evidence_target})"
        
        return False, f"Evidence target not met ({self.evidence_collected}/{self.evidence_target})"
    
    def exit_recovery(self) -> None:
        """Exit evidence recovery mode."""
        if not self.active:
            return
        
        logger.info(
            f"[EVIDENCE RECOVERY] Exiting recovery mode: "
            f"collected={self.evidence_collected}/{self.evidence_target}"
        )
        
        self.active = False
        self.gate_failure_count = 0
    
    def record_gate_failure(self) -> None:
        """Record an epistemic gate failure."""
        self.gate_failure_count += 1
        logger.warning(
            f"[EVIDENCE RECOVERY] Gate failure recorded, "
            f"count={self.gate_failure_count}/{self.GATE_FAILURE_THRESHOLD}"
        )
    
    def reset_gate_failures(self) -> None:
        """Reset gate failure counter after successful gate pass."""
        self.gate_failure_count = 0
    
    def get_recovery_actions(self) -> List[str]:
        """
        Get the actions to take during recovery mode.
        
        Returns:
            List of action strings describing recovery operations
        """
        if not self.active:
            return []
        
        return [
            "web_search",
            "claim_extraction",
            "citation_attachment",
            "source_validation",
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "active": self.active,
            "entry_reason": self.entry_reason.value if self.active else None,
            "evidence_target": self.evidence_target,
            "evidence_collected": self.evidence_collected,
            "gate_failure_count": self.gate_failure_count,
            "phases_in_recovery": self.phases_in_recovery,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRecoveryState":
        """Create from dictionary."""
        state = cls()
        state.active = data.get("active", False)
        if data.get("entry_reason"):
            state.entry_reason = RecoveryEntryReason(data["entry_reason"])
        state.evidence_target = data.get("evidence_target", 5)
        state.evidence_collected = data.get("evidence_collected", 0)
        state.gate_failure_count = data.get("gate_failure_count", 0)
        state.phases_in_recovery = data.get("phases_in_recovery", [])
        state.recovery_attempts = data.get("recovery_attempts", 0)
        state.max_recovery_attempts = data.get("max_recovery_attempts", 3)
        return state


def check_evidence_recovery_entry(
    mission_state: "MissionState",
    phase_name: str
) -> tuple:
    """
    Check if evidence recovery mode should be entered.
    
    Convenience function that extracts telemetry from mission state
    and checks recovery conditions.
    
    Args:
        mission_state: Current mission state
        phase_name: Current phase name
        
    Returns:
        Tuple of (should_enter: bool, reason: RecoveryEntryReason)
    """
    telemetry = getattr(mission_state, "epistemic_telemetry", {})
    
    grounded_ratio = telemetry.get("grounded_claim_ratio", 1.0)
    sources_per_phase = telemetry.get("sources_per_phase", {})
    sources_in_phase = sources_per_phase.get(phase_name, 0)
    
    recovery_state = getattr(mission_state, "evidence_recovery_state", None)
    if recovery_state is None:
        recovery_state = EvidenceRecoveryState()
    
    return recovery_state.should_enter_recovery(
        grounded_claim_ratio=grounded_ratio,
        sources_in_phase=sources_in_phase,
        gate_failures=recovery_state.gate_failure_count
    )

