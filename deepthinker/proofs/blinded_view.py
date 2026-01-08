"""
Blinded View Generator for Proof Packets.

Creates anonymized views of Proof Packets for unbiased evaluation.
Removes or hashes all identifiers that could reveal:
- Which model produced the output
- Which council ran
- Which policy was chosen
- Routing strategy names

Preserves:
- Claims and their content
- Evidence bindings and coverage
- Contradiction status
- Uncertainty declarations
- Integrity flags
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .proof_packet import (
    ClaimEntry,
    ContradictionEntry,
    EvidenceBinding,
    IntegrityFlags,
    ProofPacket,
    UncertaintyEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class BlindedProofView:
    """
    Anonymized view of a Proof Packet for blinded evaluation.
    
    All model/council/policy identifiers are removed or hashed.
    Evaluators cannot determine the source of the output.
    
    Attributes:
        blinded_id: Anonymized packet ID
        proof_version: Version of proof packet schema
        timestamp: When packet was created
        claim_set: Claims with normalized text
        evidence_bindings: Evidence bindings (source IDs hashed)
        contradiction_status: Contradiction entries
        uncertainty_declarations: Uncertainty entries
        integrity_flags: Integrity flags (unchanged)
        summary: Summary statistics
    """
    blinded_id: str = ""
    proof_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    claim_set: List[ClaimEntry] = field(default_factory=list)
    evidence_bindings: List[EvidenceBinding] = field(default_factory=list)
    contradiction_status: List[ContradictionEntry] = field(default_factory=list)
    uncertainty_declarations: List[UncertaintyEntry] = field(default_factory=list)
    integrity_flags: IntegrityFlags = field(default_factory=IntegrityFlags)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "blinded_id": self.blinded_id,
            "proof_version": self.proof_version,
            "timestamp": self.timestamp.isoformat(),
            "claim_set": [c.to_dict() for c in self.claim_set],
            "evidence_bindings": [e.to_dict() for e in self.evidence_bindings],
            "contradiction_status": [c.to_dict() for c in self.contradiction_status],
            "uncertainty_declarations": [u.to_dict() for u in self.uncertainty_declarations],
            "integrity_flags": self.integrity_flags.to_dict(),
            "summary": self.summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlindedProofView":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            blinded_id=data.get("blinded_id", ""),
            proof_version=data.get("proof_version", "1.0"),
            timestamp=timestamp,
            claim_set=[ClaimEntry.from_dict(c) for c in data.get("claim_set", [])],
            evidence_bindings=[EvidenceBinding.from_dict(e) for e in data.get("evidence_bindings", [])],
            contradiction_status=[ContradictionEntry.from_dict(c) for c in data.get("contradiction_status", [])],
            uncertainty_declarations=[UncertaintyEntry.from_dict(u) for u in data.get("uncertainty_declarations", [])],
            integrity_flags=IntegrityFlags.from_dict(data.get("integrity_flags", {})),
            summary=data.get("summary", {}),
        )
    
    @property
    def claim_count(self) -> int:
        """Total number of claims."""
        return len(self.claim_set)
    
    @property
    def evidence_coverage_ratio(self) -> float:
        """Ratio of claims with evidence."""
        if not self.claim_set:
            return 0.0
        with_evidence = sum(1 for b in self.evidence_bindings if b.coverage_score > 0)
        return with_evidence / len(self.claim_set)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence across claims."""
        if not self.claim_set:
            return 0.0
        return sum(c.confidence_estimate for c in self.claim_set) / len(self.claim_set)
    
    @property
    def average_uncertainty(self) -> float:
        """Average uncertainty across claims."""
        if not self.uncertainty_declarations:
            return 0.5
        return sum(u.epistemic_uncertainty for u in self.uncertainty_declarations) / len(self.uncertainty_declarations)


def generate_blinded_view(packet: ProofPacket) -> BlindedProofView:
    """
    Generate a blinded view of a Proof Packet.
    
    Removes/hashes all identifying information while preserving
    the substantive content needed for evaluation.
    
    Args:
        packet: Original ProofPacket
        
    Returns:
        BlindedProofView for anonymous evaluation
    """
    # Generate blinded ID from original packet ID
    blinded_id = _hash_id(packet.packet_id)
    
    # Blind claim IDs while preserving content
    blinded_claims = []
    claim_id_map: Dict[str, str] = {}  # original -> blinded mapping
    
    for claim in packet.claim_set:
        blinded_claim_id = _hash_id(claim.claim_id)
        claim_id_map[claim.claim_id] = blinded_claim_id
        
        blinded_claims.append(ClaimEntry(
            claim_id=blinded_claim_id,
            normalized_text=claim.normalized_text,
            claim_type=claim.claim_type,
            confidence_estimate=claim.confidence_estimate,
        ))
    
    # Blind evidence bindings
    blinded_bindings = []
    for binding in packet.evidence_bindings:
        blinded_claim_id = claim_id_map.get(binding.claim_id, _hash_id(binding.claim_id))
        blinded_evidence_ids = [_hash_id(eid) for eid in binding.evidence_ids]
        
        blinded_bindings.append(EvidenceBinding(
            claim_id=blinded_claim_id,
            evidence_ids=blinded_evidence_ids,
            evidence_type=binding.evidence_type,
            coverage_score=binding.coverage_score,
        ))
    
    # Blind contradiction status
    blinded_contradictions = []
    for entry in packet.contradiction_status:
        blinded_claim_id = claim_id_map.get(entry.claim_id, _hash_id(entry.claim_id))
        blinded_contradicting = [
            claim_id_map.get(cid, _hash_id(cid))
            for cid in entry.contradicting_claim_ids
        ]
        
        blinded_contradictions.append(ContradictionEntry(
            claim_id=blinded_claim_id,
            contradiction_checked=entry.contradiction_checked,
            contradictions_found=entry.contradictions_found,
            resolution_status=entry.resolution_status,
            contradicting_claim_ids=blinded_contradicting,
        ))
    
    # Blind uncertainty declarations
    blinded_uncertainties = []
    for entry in packet.uncertainty_declarations:
        blinded_claim_id = claim_id_map.get(entry.claim_id, _hash_id(entry.claim_id))
        
        blinded_uncertainties.append(UncertaintyEntry(
            claim_id=blinded_claim_id,
            epistemic_uncertainty=entry.epistemic_uncertainty,
            source_of_uncertainty=entry.source_of_uncertainty,
        ))
    
    # Integrity flags are preserved as-is (no identifying info)
    # But we strip violation details that might contain IDs
    blinded_flags = IntegrityFlags(
        evidence_conservation_violation=packet.integrity_flags.evidence_conservation_violation,
        monotonic_uncertainty_violation=packet.integrity_flags.monotonic_uncertainty_violation,
        no_free_lunch_depth_violation=packet.integrity_flags.no_free_lunch_depth_violation,
        metric_divergence_flag=packet.integrity_flags.metric_divergence_flag,
        violation_details={},  # Strip details
    )
    
    # Compute summary for easy evaluation
    summary = {
        "claim_count": len(blinded_claims),
        "evidence_coverage_ratio": packet.evidence_coverage_ratio,
        "average_confidence": packet.average_confidence,
        "average_uncertainty": packet.average_uncertainty,
        "contradiction_count": packet.contradiction_count,
        "unresolved_contradictions": packet.unresolved_contradiction_count,
        "has_integrity_violations": packet.integrity_flags.has_violations,
        "integrity_violation_count": packet.integrity_flags.violation_count,
    }
    
    return BlindedProofView(
        blinded_id=blinded_id,
        proof_version=packet.metadata.proof_version,
        timestamp=packet.metadata.timestamp,
        claim_set=blinded_claims,
        evidence_bindings=blinded_bindings,
        contradiction_status=blinded_contradictions,
        uncertainty_declarations=blinded_uncertainties,
        integrity_flags=blinded_flags,
        summary=summary,
    )


def _hash_id(original_id: str) -> str:
    """
    Hash an ID for anonymization.
    
    Uses SHA-256 truncated to 12 chars for readability.
    
    Args:
        original_id: Original identifier
        
    Returns:
        Hashed identifier
    """
    if not original_id:
        return ""
    
    hash_bytes = hashlib.sha256(original_id.encode('utf-8')).hexdigest()[:12]
    return f"blind_{hash_bytes}"


def compare_blinded_views(
    view1: BlindedProofView,
    view2: BlindedProofView,
) -> Dict[str, Any]:
    """
    Compare two blinded views for evaluation.
    
    Computes comparative metrics without knowing which models
    produced which output.
    
    Args:
        view1: First blinded view
        view2: Second blinded view
        
    Returns:
        Comparison metrics dictionary
    """
    return {
        "claim_count_diff": view1.claim_count - view2.claim_count,
        "evidence_coverage_diff": view1.evidence_coverage_ratio - view2.evidence_coverage_ratio,
        "confidence_diff": view1.average_confidence - view2.average_confidence,
        "uncertainty_diff": view1.average_uncertainty - view2.average_uncertainty,
        "view1_violations": view1.integrity_flags.violation_count,
        "view2_violations": view2.integrity_flags.violation_count,
        "view1_summary": view1.summary,
        "view2_summary": view2.summary,
    }


class BlindedEvaluator:
    """
    Evaluates blinded proof views without knowing the source.
    
    Provides methods for ranking and scoring blinded views
    based on their structural properties.
    
    Usage:
        evaluator = BlindedEvaluator()
        score = evaluator.score_view(blinded_view)
        ranking = evaluator.rank_views([view1, view2, view3])
    """
    
    def __init__(
        self,
        evidence_weight: float = 0.3,
        confidence_weight: float = 0.2,
        uncertainty_penalty: float = 0.2,
        integrity_penalty: float = 0.3,
    ):
        """
        Initialize the evaluator.
        
        Args:
            evidence_weight: Weight for evidence coverage
            confidence_weight: Weight for confidence
            uncertainty_penalty: Penalty weight for uncertainty
            integrity_penalty: Penalty weight for integrity violations
        """
        self.evidence_weight = evidence_weight
        self.confidence_weight = confidence_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.integrity_penalty = integrity_penalty
    
    def score_view(self, view: BlindedProofView) -> float:
        """
        Score a blinded view.
        
        Higher scores indicate better structural quality.
        
        Args:
            view: BlindedProofView to score
            
        Returns:
            Score in [0, 1]
        """
        # Positive components
        evidence_score = view.evidence_coverage_ratio * self.evidence_weight
        confidence_score = view.average_confidence * self.confidence_weight
        
        # Negative components (penalties)
        uncertainty_penalty = view.average_uncertainty * self.uncertainty_penalty
        
        # Integrity penalty: -0.1 per violation
        violation_penalty = min(
            1.0,
            view.integrity_flags.violation_count * 0.1
        ) * self.integrity_penalty
        
        # Combine
        score = evidence_score + confidence_score - uncertainty_penalty - violation_penalty
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def rank_views(self, views: List[BlindedProofView]) -> List[tuple]:
        """
        Rank multiple blinded views by score.
        
        Args:
            views: List of blinded views
            
        Returns:
            List of (view, score) tuples, sorted by score descending
        """
        scored = [(view, self.score_view(view)) for view in views]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def evaluate_against_threshold(
        self,
        view: BlindedProofView,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate a view against a quality threshold.
        
        Args:
            view: BlindedProofView to evaluate
            threshold: Minimum score threshold
            
        Returns:
            Evaluation result dictionary
        """
        score = self.score_view(view)
        passed = score >= threshold
        
        return {
            "passed": passed,
            "score": score,
            "threshold": threshold,
            "margin": score - threshold,
            "claim_count": view.claim_count,
            "evidence_coverage": view.evidence_coverage_ratio,
            "average_confidence": view.average_confidence,
            "average_uncertainty": view.average_uncertainty,
            "integrity_violations": view.integrity_flags.violation_count,
        }

