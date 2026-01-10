"""
Proof-Carrying Reasoning (PCR) - Proof Packet v1.

This module provides structural accountability for DeepThinker outputs
through machine-readable Proof Packets that bind claims to evidence,
track contradictions, declare uncertainty, and enforce invariants.

A Proof Packet is:
- Machine-readable
- Minimal
- Compositional
- Blind-evaluable
- Orthogonal to generation logic

Key Components:
- ProofPacket: Core data structure with 7 sections
- ProofPacketBuilder: Assembles packets from phase outputs
- ProofStore: JSONL persistence layer
- ProofIntegrityChecker: Enforces invariants
- BlindedProofView: Anonymized view for unbiased evaluation
"""

from .proof_packet import (
    ProofPacket,
    ClaimEntry,
    ClaimTypeProof,
    EvidenceBinding,
    EvidenceTypeProof,
    ContradictionEntry,
    ResolutionStatus,
    UncertaintyEntry,
    UncertaintySource,
    DecisionTrace,
    IntegrityFlags,
    ProofPacketMetadata,
)
from .claim_extractor import ProofClaimExtractor
from .evidence_binder import EvidenceBinder
from .integrity_checks import ProofIntegrityChecker
from .builder import ProofPacketBuilder
from .proof_store import ProofStore
from .blinded_view import BlindedProofView, generate_blinded_view

__all__ = [
    # Core dataclasses
    "ProofPacket",
    "ClaimEntry",
    "ClaimTypeProof",
    "EvidenceBinding",
    "EvidenceTypeProof",
    "ContradictionEntry",
    "ResolutionStatus",
    "UncertaintyEntry",
    "UncertaintySource",
    "DecisionTrace",
    "IntegrityFlags",
    "ProofPacketMetadata",
    # Components
    "ProofClaimExtractor",
    "EvidenceBinder",
    "ProofIntegrityChecker",
    "ProofPacketBuilder",
    "ProofStore",
    # Blinded evaluation
    "BlindedProofView",
    "generate_blinded_view",
]


