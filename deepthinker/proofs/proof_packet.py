"""
Proof Packet Core Data Structures.

Defines the ProofPacket and all its component dataclasses for
Proof-Carrying Reasoning (PCR) in DeepThinker.

A Proof Packet contains 7 sections:
1. Claim Set - Atomic claims with types and confidence
2. Evidence Bindings - Links claims to evidence objects
3. Contradiction Status - Contradiction checking results per claim
4. Uncertainty Declaration - Epistemic uncertainty per claim
5. Decision Trace - IDs of routing/escalation/alignment/tool decisions
6. Integrity Flags - Invariant violation indicators
7. Metadata - Version, mission, phase, timestamp, model hash
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ClaimTypeProof(str, Enum):
    """Types of claims in a Proof Packet."""
    FACT = "fact"
    INFERENCE = "inference"
    RECOMMENDATION = "recommendation"
    ASSUMPTION = "assumption"


class EvidenceTypeProof(str, Enum):
    """Types of evidence sources."""
    WEB = "web"
    CODE = "code"
    MEMORY = "memory"
    SIMULATION = "simulation"
    ASSUMPTION = "assumption"


class ResolutionStatus(str, Enum):
    """Resolution status for contradictions."""
    UNRESOLVED = "unresolved"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"


class UncertaintySource(str, Enum):
    """Sources of epistemic uncertainty."""
    MISSING_EVIDENCE = "missing_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"
    EXTRAPOLATION = "extrapolation"
    MODEL_LIMITATION = "model_limitation"
    TEMPORAL_DECAY = "temporal_decay"


@dataclass
class ClaimEntry:
    """
    An atomic claim extracted from output.
    
    Attributes:
        claim_id: Unique identifier for the claim
        normalized_text: Normalized claim text (max 500 chars)
        claim_type: Type of claim (fact, inference, recommendation, assumption)
        confidence_estimate: Model-side confidence before judge (0-1)
    """
    claim_id: str
    normalized_text: str
    claim_type: ClaimTypeProof
    confidence_estimate: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "normalized_text": self.normalized_text,
            "claim_type": self.claim_type.value,
            "confidence_estimate": self.confidence_estimate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEntry":
        return cls(
            claim_id=data.get("claim_id", ""),
            normalized_text=data.get("normalized_text", ""),
            claim_type=ClaimTypeProof(data.get("claim_type", "inference")),
            confidence_estimate=data.get("confidence_estimate", 0.5),
        )


@dataclass
class EvidenceBinding:
    """
    Binding between a claim and its supporting evidence.
    
    Attributes:
        claim_id: ID of the claim this evidence supports
        evidence_ids: List of EvidenceObject IDs
        evidence_type: Type of evidence (web, code, memory, etc.)
        coverage_score: How directly evidence supports the claim (0-1)
    """
    claim_id: str
    evidence_ids: List[str] = field(default_factory=list)
    evidence_type: EvidenceTypeProof = EvidenceTypeProof.ASSUMPTION
    coverage_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "evidence_ids": self.evidence_ids,
            "evidence_type": self.evidence_type.value,
            "coverage_score": self.coverage_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBinding":
        return cls(
            claim_id=data.get("claim_id", ""),
            evidence_ids=data.get("evidence_ids", []),
            evidence_type=EvidenceTypeProof(data.get("evidence_type", "assumption")),
            coverage_score=data.get("coverage_score", 0.0),
        )


@dataclass
class ContradictionEntry:
    """
    Contradiction status for a claim.
    
    Attributes:
        claim_id: ID of the claim
        contradiction_checked: Whether contradiction check was run
        contradictions_found: Number of contradictions found
        resolution_status: How contradictions were resolved
        contradicting_claim_ids: IDs of claims that contradict this one
    """
    claim_id: str
    contradiction_checked: bool = False
    contradictions_found: int = 0
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    contradicting_claim_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "contradiction_checked": self.contradiction_checked,
            "contradictions_found": self.contradictions_found,
            "resolution_status": self.resolution_status.value,
            "contradicting_claim_ids": self.contradicting_claim_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContradictionEntry":
        return cls(
            claim_id=data.get("claim_id", ""),
            contradiction_checked=data.get("contradiction_checked", False),
            contradictions_found=data.get("contradictions_found", 0),
            resolution_status=ResolutionStatus(data.get("resolution_status", "unresolved")),
            contradicting_claim_ids=data.get("contradicting_claim_ids", []),
        )


@dataclass
class UncertaintyEntry:
    """
    Uncertainty declaration for a claim.
    
    Attributes:
        claim_id: ID of the claim
        epistemic_uncertainty: Uncertainty level (0-1, higher = more uncertain)
        source_of_uncertainty: Primary source of uncertainty
    """
    claim_id: str
    epistemic_uncertainty: float = 0.5
    source_of_uncertainty: UncertaintySource = UncertaintySource.MODEL_LIMITATION
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "source_of_uncertainty": self.source_of_uncertainty.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyEntry":
        return cls(
            claim_id=data.get("claim_id", ""),
            epistemic_uncertainty=data.get("epistemic_uncertainty", 0.5),
            source_of_uncertainty=UncertaintySource(
                data.get("source_of_uncertainty", "model_limitation")
            ),
        )


@dataclass
class DecisionTrace:
    """
    Minimal decision trace - IDs only, no text.
    
    Attributes:
        routing_decision_ids: IDs of routing decisions
        escalation_decision_ids: IDs of escalation decisions
        alignment_action_ids: IDs of alignment corrective actions
        tool_event_ids: IDs of tool usage events
    """
    routing_decision_ids: List[str] = field(default_factory=list)
    escalation_decision_ids: List[str] = field(default_factory=list)
    alignment_action_ids: List[str] = field(default_factory=list)
    tool_event_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_decision_ids": self.routing_decision_ids,
            "escalation_decision_ids": self.escalation_decision_ids,
            "alignment_action_ids": self.alignment_action_ids,
            "tool_event_ids": self.tool_event_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionTrace":
        return cls(
            routing_decision_ids=data.get("routing_decision_ids", []),
            escalation_decision_ids=data.get("escalation_decision_ids", []),
            alignment_action_ids=data.get("alignment_action_ids", []),
            tool_event_ids=data.get("tool_event_ids", []),
        )
    
    @property
    def total_decisions(self) -> int:
        """Total number of decisions traced."""
        return (
            len(self.routing_decision_ids) +
            len(self.escalation_decision_ids) +
            len(self.alignment_action_ids) +
            len(self.tool_event_ids)
        )


@dataclass
class IntegrityFlags:
    """
    Invariant violation flags.
    
    All flags are False by default (no violations).
    True indicates a violation was detected.
    
    Attributes:
        evidence_conservation_violation: Confidence increased without new evidence
        monotonic_uncertainty_violation: Uncertainty decreased during compression
        no_free_lunch_depth_violation: Depth increased without progress
        metric_divergence_flag: Scorecard metrics diverge from proof metrics
    """
    evidence_conservation_violation: bool = False
    monotonic_uncertainty_violation: bool = False
    no_free_lunch_depth_violation: bool = False
    metric_divergence_flag: bool = False
    
    # Detailed violation info
    violation_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_conservation_violation": self.evidence_conservation_violation,
            "monotonic_uncertainty_violation": self.monotonic_uncertainty_violation,
            "no_free_lunch_depth_violation": self.no_free_lunch_depth_violation,
            "metric_divergence_flag": self.metric_divergence_flag,
            "violation_details": self.violation_details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrityFlags":
        return cls(
            evidence_conservation_violation=data.get("evidence_conservation_violation", False),
            monotonic_uncertainty_violation=data.get("monotonic_uncertainty_violation", False),
            no_free_lunch_depth_violation=data.get("no_free_lunch_depth_violation", False),
            metric_divergence_flag=data.get("metric_divergence_flag", False),
            violation_details=data.get("violation_details", {}),
        )
    
    @property
    def has_violations(self) -> bool:
        """Check if any violations exist."""
        return (
            self.evidence_conservation_violation or
            self.monotonic_uncertainty_violation or
            self.no_free_lunch_depth_violation or
            self.metric_divergence_flag
        )
    
    @property
    def violation_count(self) -> int:
        """Count total violations."""
        return sum([
            self.evidence_conservation_violation,
            self.monotonic_uncertainty_violation,
            self.no_free_lunch_depth_violation,
            self.metric_divergence_flag,
        ])


@dataclass
class ProofPacketMetadata:
    """
    Proof Packet metadata.
    
    Attributes:
        proof_version: Version of proof packet schema
        mission_id: Parent mission ID
        phase_id: Phase name
        timestamp: When packet was created
        generation_model_hash: Hashed model identifier (not raw)
        blinded_evaluation_ready: Whether packet can be blindly evaluated
    """
    proof_version: str = "1.0"
    mission_id: str = ""
    phase_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    generation_model_hash: str = ""
    blinded_evaluation_ready: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_version": self.proof_version,
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "timestamp": self.timestamp.isoformat(),
            "generation_model_hash": self.generation_model_hash,
            "blinded_evaluation_ready": self.blinded_evaluation_ready,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofPacketMetadata":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            proof_version=data.get("proof_version", "1.0"),
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            timestamp=timestamp,
            generation_model_hash=data.get("generation_model_hash", ""),
            blinded_evaluation_ready=data.get("blinded_evaluation_ready", True),
        )


@dataclass
class ProofPacket:
    """
    Complete Proof Packet - the core PCR artifact.
    
    Contains 7 sections:
    1. claim_set: Atomic claims extracted from output
    2. evidence_bindings: Links claims to evidence
    3. contradiction_status: Per-claim contradiction info
    4. uncertainty_declarations: Per-claim uncertainty
    5. decision_trace: IDs of decisions that led here
    6. integrity_flags: Invariant violation indicators
    7. metadata: Version, IDs, timestamps
    
    Attributes:
        packet_id: Unique identifier for this packet
        claim_set: List of extracted claims
        evidence_bindings: List of evidence bindings
        contradiction_status: List of contradiction entries
        uncertainty_declarations: List of uncertainty entries
        decision_trace: Decision trace object
        integrity_flags: Integrity flags object
        metadata: Packet metadata
    """
    packet_id: str = ""
    claim_set: List[ClaimEntry] = field(default_factory=list)
    evidence_bindings: List[EvidenceBinding] = field(default_factory=list)
    contradiction_status: List[ContradictionEntry] = field(default_factory=list)
    uncertainty_declarations: List[UncertaintyEntry] = field(default_factory=list)
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)
    integrity_flags: IntegrityFlags = field(default_factory=IntegrityFlags)
    metadata: ProofPacketMetadata = field(default_factory=ProofPacketMetadata)
    
    def __post_init__(self):
        """Generate packet ID if not provided."""
        if not self.packet_id:
            self.packet_id = f"pp_{uuid.uuid4().hex[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "packet_id": self.packet_id,
            "claim_set": [c.to_dict() for c in self.claim_set],
            "evidence_bindings": [e.to_dict() for e in self.evidence_bindings],
            "contradiction_status": [c.to_dict() for c in self.contradiction_status],
            "uncertainty_declarations": [u.to_dict() for u in self.uncertainty_declarations],
            "decision_trace": self.decision_trace.to_dict(),
            "integrity_flags": self.integrity_flags.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofPacket":
        """Create from dictionary."""
        return cls(
            packet_id=data.get("packet_id", ""),
            claim_set=[ClaimEntry.from_dict(c) for c in data.get("claim_set", [])],
            evidence_bindings=[EvidenceBinding.from_dict(e) for e in data.get("evidence_bindings", [])],
            contradiction_status=[ContradictionEntry.from_dict(c) for c in data.get("contradiction_status", [])],
            uncertainty_declarations=[UncertaintyEntry.from_dict(u) for u in data.get("uncertainty_declarations", [])],
            decision_trace=DecisionTrace.from_dict(data.get("decision_trace", {})),
            integrity_flags=IntegrityFlags.from_dict(data.get("integrity_flags", {})),
            metadata=ProofPacketMetadata.from_dict(data.get("metadata", {})),
        )
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_claim(self, claim_id: str) -> Optional[ClaimEntry]:
        """Get a claim by ID."""
        for claim in self.claim_set:
            if claim.claim_id == claim_id:
                return claim
        return None
    
    def get_evidence_for_claim(self, claim_id: str) -> Optional[EvidenceBinding]:
        """Get evidence binding for a claim."""
        for binding in self.evidence_bindings:
            if binding.claim_id == claim_id:
                return binding
        return None
    
    def get_contradiction_for_claim(self, claim_id: str) -> Optional[ContradictionEntry]:
        """Get contradiction status for a claim."""
        for entry in self.contradiction_status:
            if entry.claim_id == claim_id:
                return entry
        return None
    
    def get_uncertainty_for_claim(self, claim_id: str) -> Optional[UncertaintyEntry]:
        """Get uncertainty declaration for a claim."""
        for entry in self.uncertainty_declarations:
            if entry.claim_id == claim_id:
                return entry
        return None
    
    # =========================================================================
    # Summary Methods
    # =========================================================================
    
    @property
    def claim_count(self) -> int:
        """Total number of claims."""
        return len(self.claim_set)
    
    @property
    def evidence_coverage_ratio(self) -> float:
        """Ratio of claims with evidence (coverage > 0)."""
        if not self.claim_set:
            return 0.0
        with_evidence = sum(
            1 for b in self.evidence_bindings
            if b.coverage_score > 0
        )
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
    
    @property
    def contradiction_count(self) -> int:
        """Total contradictions found."""
        return sum(c.contradictions_found for c in self.contradiction_status)
    
    @property
    def unresolved_contradiction_count(self) -> int:
        """Count of unresolved contradictions."""
        return sum(
            1 for c in self.contradiction_status
            if c.contradictions_found > 0 and c.resolution_status == ResolutionStatus.UNRESOLVED
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the proof packet."""
        return {
            "packet_id": self.packet_id,
            "mission_id": self.metadata.mission_id,
            "phase_id": self.metadata.phase_id,
            "claim_count": self.claim_count,
            "evidence_coverage_ratio": self.evidence_coverage_ratio,
            "average_confidence": self.average_confidence,
            "average_uncertainty": self.average_uncertainty,
            "contradiction_count": self.contradiction_count,
            "unresolved_contradictions": self.unresolved_contradiction_count,
            "has_integrity_violations": self.integrity_flags.has_violations,
            "integrity_violation_count": self.integrity_flags.violation_count,
            "decision_count": self.decision_trace.total_decisions,
        }


def hash_model_identifier(model_name: str) -> str:
    """
    Hash a model identifier for blinded evaluation.
    
    Args:
        model_name: Raw model name (e.g., "cogito:14b")
        
    Returns:
        Hashed identifier (e.g., "sha256:abc123...")
    """
    if not model_name:
        return ""
    
    hash_bytes = hashlib.sha256(model_name.encode('utf-8')).hexdigest()[:16]
    return f"sha256:{hash_bytes}"


