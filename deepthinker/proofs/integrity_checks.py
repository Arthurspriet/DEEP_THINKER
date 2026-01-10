"""
Integrity Checks for Proof Packets.

Enforces the non-negotiable invariants:
1. Conservation of Evidence - Confidence cannot increase without new evidence
2. Monotonic Uncertainty Under Compression - Compression cannot reduce uncertainty
3. No-Free-Lunch Depth - Depth without new evidence/contradiction resolution is flagged
4. Metric Divergence - Scorecard vs proof metrics divergence detection

All checks are designed to be:
- Deterministic
- Laptop-friendly (no heavy computation)
- Structural (not heuristic)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .proof_packet import (
    ClaimEntry,
    ContradictionEntry,
    EvidenceBinding,
    IntegrityFlags,
    ProofPacket,
    ResolutionStatus,
    UncertaintyEntry,
)

logger = logging.getLogger(__name__)


# Threshold for confidence increase to be considered significant
CONFIDENCE_INCREASE_THRESHOLD = 0.05

# Threshold for uncertainty decrease to be considered significant
UNCERTAINTY_DECREASE_THRESHOLD = 0.05

# Threshold for metric divergence to be flagged
METRIC_DIVERGENCE_THRESHOLD = 0.2


@dataclass
class IntegrityCheckResult:
    """
    Result of a single integrity check.
    
    Attributes:
        passed: Whether the check passed
        violation_type: Type of violation if failed
        details: Detailed information about the violation
        affected_claims: IDs of claims involved in violation
    """
    passed: bool
    violation_type: str = ""
    details: str = ""
    affected_claims: List[str] = None
    
    def __post_init__(self):
        if self.affected_claims is None:
            self.affected_claims = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violation_type": self.violation_type,
            "details": self.details,
            "affected_claims": self.affected_claims,
        }


class ProofIntegrityChecker:
    """
    Checks Proof Packet invariants.
    
    Provides four core checks:
    1. check_evidence_conservation - Confidence cannot increase without new evidence
    2. check_monotonic_uncertainty - Compression cannot reduce uncertainty
    3. check_no_free_lunch - Depth without progress is flagged
    4. check_metric_divergence - Scorecard vs proof metrics
    
    Usage:
        checker = ProofIntegrityChecker()
        flags = checker.check_all(current_packet, prev_packet)
        if flags.has_violations:
            handle_violations(flags)
    """
    
    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_INCREASE_THRESHOLD,
        uncertainty_threshold: float = UNCERTAINTY_DECREASE_THRESHOLD,
        divergence_threshold: float = METRIC_DIVERGENCE_THRESHOLD,
    ):
        """
        Initialize the integrity checker.
        
        Args:
            confidence_threshold: Threshold for significant confidence increase
            uncertainty_threshold: Threshold for significant uncertainty decrease
            divergence_threshold: Threshold for metric divergence
        """
        self._confidence_threshold = confidence_threshold
        self._uncertainty_threshold = uncertainty_threshold
        self._divergence_threshold = divergence_threshold
    
    def check_all(
        self,
        current: ProofPacket,
        prev: Optional[ProofPacket] = None,
        is_compression: bool = False,
        depth_increased: bool = False,
        scorecard: Optional[Dict[str, float]] = None,
    ) -> IntegrityFlags:
        """
        Run all integrity checks and return flags.
        
        Args:
            current: Current proof packet
            prev: Previous proof packet (if any)
            is_compression: Whether this is a compression operation
            depth_increased: Whether depth/escalation increased
            scorecard: Optional scorecard metrics for divergence check
            
        Returns:
            IntegrityFlags with all violation flags set
        """
        flags = IntegrityFlags()
        violation_details = {}
        
        # Check 1: Conservation of Evidence
        conservation_result = self.check_evidence_conservation(current, prev)
        if not conservation_result.passed:
            flags.evidence_conservation_violation = True
            violation_details["evidence_conservation"] = conservation_result.to_dict()
            logger.warning(
                f"[INTEGRITY] Evidence conservation violation: {conservation_result.details}"
            )
        
        # Check 2: Monotonic Uncertainty (only during compression)
        if is_compression:
            uncertainty_result = self.check_monotonic_uncertainty(current, prev)
            if not uncertainty_result.passed:
                flags.monotonic_uncertainty_violation = True
                violation_details["monotonic_uncertainty"] = uncertainty_result.to_dict()
                logger.warning(
                    f"[INTEGRITY] Monotonic uncertainty violation: {uncertainty_result.details}"
                )
        
        # Check 3: No-Free-Lunch Depth
        if depth_increased:
            nfl_result = self.check_no_free_lunch(current, prev)
            if not nfl_result.passed:
                flags.no_free_lunch_depth_violation = True
                violation_details["no_free_lunch"] = nfl_result.to_dict()
                logger.warning(
                    f"[INTEGRITY] No-free-lunch violation: {nfl_result.details}"
                )
        
        # Check 4: Metric Divergence (if scorecard provided)
        if scorecard is not None:
            divergence_result = self.check_metric_divergence(current, scorecard)
            if not divergence_result.passed:
                flags.metric_divergence_flag = True
                violation_details["metric_divergence"] = divergence_result.to_dict()
                logger.warning(
                    f"[INTEGRITY] Metric divergence: {divergence_result.details}"
                )
        
        flags.violation_details = violation_details
        
        return flags
    
    def check_evidence_conservation(
        self,
        current: ProofPacket,
        prev: Optional[ProofPacket],
    ) -> IntegrityCheckResult:
        """
        Check that claim confidence does not increase without new evidence.
        
        Invariant: For any claim that exists in both packets, if confidence
        increased by more than threshold, there must be new evidence bound.
        
        Args:
            current: Current proof packet
            prev: Previous proof packet
            
        Returns:
            IntegrityCheckResult
        """
        if prev is None:
            return IntegrityCheckResult(passed=True)
        
        affected_claims = []
        
        for claim in current.claim_set:
            prev_claim = prev.get_claim(claim.claim_id)
            
            if prev_claim is None:
                # New claim, no conservation check needed
                continue
            
            # Check for significant confidence increase
            confidence_delta = claim.confidence_estimate - prev_claim.confidence_estimate
            
            if confidence_delta > self._confidence_threshold:
                # Confidence increased - check for new evidence
                prev_binding = prev.get_evidence_for_claim(claim.claim_id)
                curr_binding = current.get_evidence_for_claim(claim.claim_id)
                
                prev_evidence: Set[str] = set()
                curr_evidence: Set[str] = set()
                
                if prev_binding:
                    prev_evidence = set(prev_binding.evidence_ids)
                if curr_binding:
                    curr_evidence = set(curr_binding.evidence_ids)
                
                new_evidence = curr_evidence - prev_evidence
                
                if not new_evidence:
                    # Confidence increased without new evidence - violation!
                    affected_claims.append(claim.claim_id)
        
        if affected_claims:
            return IntegrityCheckResult(
                passed=False,
                violation_type="evidence_conservation",
                details=f"{len(affected_claims)} claim(s) had confidence increase without new evidence",
                affected_claims=affected_claims,
            )
        
        return IntegrityCheckResult(passed=True)
    
    def check_monotonic_uncertainty(
        self,
        current: ProofPacket,
        prev: Optional[ProofPacket],
    ) -> IntegrityCheckResult:
        """
        Check that compression does not reduce uncertainty.
        
        Invariant: During memory compression/distillation, uncertainty
        cannot decrease unless explicitly validated.
        
        Args:
            current: Current proof packet
            prev: Previous proof packet
            
        Returns:
            IntegrityCheckResult
        """
        if prev is None:
            return IntegrityCheckResult(passed=True)
        
        affected_claims = []
        
        for uncertainty in current.uncertainty_declarations:
            prev_uncertainty = prev.get_uncertainty_for_claim(uncertainty.claim_id)
            
            if prev_uncertainty is None:
                continue
            
            # Check for significant uncertainty decrease
            uncertainty_delta = prev_uncertainty.epistemic_uncertainty - uncertainty.epistemic_uncertainty
            
            if uncertainty_delta > self._uncertainty_threshold:
                # Uncertainty decreased during compression - violation!
                affected_claims.append(uncertainty.claim_id)
        
        if affected_claims:
            return IntegrityCheckResult(
                passed=False,
                violation_type="monotonic_uncertainty",
                details=f"{len(affected_claims)} claim(s) had uncertainty decrease during compression",
                affected_claims=affected_claims,
            )
        
        return IntegrityCheckResult(passed=True)
    
    def check_no_free_lunch(
        self,
        current: ProofPacket,
        prev: Optional[ProofPacket],
    ) -> IntegrityCheckResult:
        """
        Check that depth/escalation produces progress.
        
        Invariant: Depth increase without new evidence OR contradiction
        resolution must be flagged.
        
        Args:
            current: Current proof packet
            prev: Previous proof packet
            
        Returns:
            IntegrityCheckResult
        """
        if prev is None:
            # First packet, cannot have free lunch
            return IntegrityCheckResult(passed=True)
        
        # Count new evidence
        prev_evidence_ids: Set[str] = set()
        curr_evidence_ids: Set[str] = set()
        
        for binding in prev.evidence_bindings:
            prev_evidence_ids.update(binding.evidence_ids)
        for binding in current.evidence_bindings:
            curr_evidence_ids.update(binding.evidence_ids)
        
        new_evidence_count = len(curr_evidence_ids - prev_evidence_ids)
        
        # Count resolved contradictions
        prev_unresolved = sum(
            1 for c in prev.contradiction_status
            if c.contradictions_found > 0 and c.resolution_status == ResolutionStatus.UNRESOLVED
        )
        curr_unresolved = sum(
            1 for c in current.contradiction_status
            if c.contradictions_found > 0 and c.resolution_status == ResolutionStatus.UNRESOLVED
        )
        
        contradictions_resolved = max(0, prev_unresolved - curr_unresolved)
        
        # Free lunch: no new evidence AND no contradictions resolved
        if new_evidence_count == 0 and contradictions_resolved == 0:
            return IntegrityCheckResult(
                passed=False,
                violation_type="no_free_lunch",
                details="Depth increased without new evidence or contradiction resolution",
                affected_claims=[],
            )
        
        return IntegrityCheckResult(passed=True)
    
    def check_metric_divergence(
        self,
        current: ProofPacket,
        scorecard: Dict[str, float],
    ) -> IntegrityCheckResult:
        """
        Check for divergence between scorecard and proof metrics.
        
        Compares proof packet summary metrics against scorecard metrics
        to detect potential gaming or metric inflation.
        
        Args:
            current: Current proof packet
            scorecard: Scorecard metrics dictionary
            
        Returns:
            IntegrityCheckResult
        """
        divergences = []
        
        # Compare evidence grounding
        if "evidence_grounding" in scorecard:
            proof_coverage = current.evidence_coverage_ratio
            scorecard_grounding = scorecard["evidence_grounding"]
            
            if abs(proof_coverage - scorecard_grounding) > self._divergence_threshold:
                divergences.append(
                    f"evidence: proof={proof_coverage:.2f} vs scorecard={scorecard_grounding:.2f}"
                )
        
        # Compare consistency (inverse of contradiction ratio)
        if "consistency" in scorecard:
            proof_consistency = 1.0 - (
                current.unresolved_contradiction_count / max(current.claim_count, 1)
            )
            scorecard_consistency = scorecard["consistency"]
            
            if abs(proof_consistency - scorecard_consistency) > self._divergence_threshold:
                divergences.append(
                    f"consistency: proof={proof_consistency:.2f} vs scorecard={scorecard_consistency:.2f}"
                )
        
        # Compare overall confidence
        if "overall" in scorecard:
            proof_confidence = current.average_confidence
            scorecard_overall = scorecard["overall"]
            
            # Flag if scorecard is significantly higher than proof evidence suggests
            if scorecard_overall - proof_confidence > self._divergence_threshold:
                divergences.append(
                    f"confidence: proof={proof_confidence:.2f} vs scorecard={scorecard_overall:.2f}"
                )
        
        if divergences:
            return IntegrityCheckResult(
                passed=False,
                violation_type="metric_divergence",
                details=f"Divergence detected: {'; '.join(divergences)}",
                affected_claims=[],
            )
        
        return IntegrityCheckResult(passed=True)
    
    def compute_integrity_summary(
        self,
        flags: IntegrityFlags,
    ) -> Dict[str, Any]:
        """
        Compute a summary of integrity check results.
        
        Args:
            flags: IntegrityFlags from check_all
            
        Returns:
            Summary dictionary
        """
        return {
            "has_violations": flags.has_violations,
            "violation_count": flags.violation_count,
            "violations": {
                "evidence_conservation": flags.evidence_conservation_violation,
                "monotonic_uncertainty": flags.monotonic_uncertainty_violation,
                "no_free_lunch_depth": flags.no_free_lunch_depth_violation,
                "metric_divergence": flags.metric_divergence_flag,
            },
            "details": flags.violation_details,
        }


def find_matching_claim(
    packet: ProofPacket,
    claim_id: str,
    use_semantic: bool = False,
) -> Optional[ClaimEntry]:
    """
    Find a claim in a packet by ID.
    
    For future enhancement: use semantic similarity for claim matching
    to handle reformulated claims.
    
    Args:
        packet: Proof packet to search
        claim_id: Claim ID to find
        use_semantic: Whether to use semantic matching (future)
        
    Returns:
        Matching ClaimEntry or None
    """
    # Direct ID match
    return packet.get_claim(claim_id)


# Global checker instance
_checker: Optional[ProofIntegrityChecker] = None


def get_integrity_checker(
    confidence_threshold: float = CONFIDENCE_INCREASE_THRESHOLD,
    uncertainty_threshold: float = UNCERTAINTY_DECREASE_THRESHOLD,
) -> ProofIntegrityChecker:
    """Get the global integrity checker instance."""
    global _checker
    if _checker is None:
        _checker = ProofIntegrityChecker(
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold,
        )
    return _checker


