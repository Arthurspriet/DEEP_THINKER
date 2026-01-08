"""
Proof Packet Builder.

Orchestrates the assembly of Proof Packets from phase outputs and
arbiter decisions. Coordinates claim extraction, evidence binding,
contradiction detection, and integrity checking.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .proof_packet import (
    ClaimEntry,
    ContradictionEntry,
    DecisionTrace,
    EvidenceBinding,
    IntegrityFlags,
    ProofPacket,
    ProofPacketMetadata,
    ResolutionStatus,
    UncertaintyEntry,
    UncertaintySource,
    hash_model_identifier,
)
from .claim_extractor import ProofClaimExtractor, get_proof_claim_extractor
from .evidence_binder import EvidenceBinder, get_evidence_binder
from .integrity_checks import ProofIntegrityChecker, get_integrity_checker

if TYPE_CHECKING:
    from ..arbiter.arbiter import ArbiterDecision
    from ..decisions.decision_store import DecisionStore
    from ..epistemics.contradiction_detector import ContradictionDetector
    from ..memory.rag_store import MissionRAGStore

logger = logging.getLogger(__name__)


class ProofPacketBuilder:
    """
    Builds Proof Packets from outputs.
    
    Orchestrates:
    1. Claim extraction from output text
    2. Evidence binding from RAG store
    3. Contradiction detection
    4. Uncertainty estimation
    5. Decision trace collection
    6. Integrity checking
    
    Usage:
        builder = ProofPacketBuilder(
            evidence_store=mission_rag,
            decision_store=decision_store,
        )
        
        packet = builder.build_from_phase_output(
            output_text=phase_output,
            phase_name="research",
            mission_id=mission_id,
        )
    """
    
    def __init__(
        self,
        evidence_store: Optional["MissionRAGStore"] = None,
        decision_store: Optional["DecisionStore"] = None,
        contradiction_detector: Optional["ContradictionDetector"] = None,
        claim_extractor: Optional[ProofClaimExtractor] = None,
        evidence_binder: Optional[EvidenceBinder] = None,
        integrity_checker: Optional[ProofIntegrityChecker] = None,
    ):
        """
        Initialize the proof packet builder.
        
        Args:
            evidence_store: RAG store for evidence retrieval
            decision_store: Store for decision records
            contradiction_detector: Detector for contradictions
            claim_extractor: Custom claim extractor (uses default if None)
            evidence_binder: Custom evidence binder (uses default if None)
            integrity_checker: Custom integrity checker (uses default if None)
        """
        self._evidence_store = evidence_store
        self._decision_store = decision_store
        self._contradiction_detector = contradiction_detector
        
        # Initialize components
        self._claim_extractor = claim_extractor or get_proof_claim_extractor()
        self._evidence_binder = evidence_binder or get_evidence_binder(evidence_store)
        self._integrity_checker = integrity_checker or get_integrity_checker()
        
        # Update evidence binder with store
        if evidence_store is not None:
            self._evidence_binder.set_rag_store(evidence_store)
    
    def build_from_phase_output(
        self,
        output_text: str,
        phase_name: str,
        mission_id: str,
        model_name: str = "",
        prev_packet: Optional[ProofPacket] = None,
        is_compression: bool = False,
        depth_increased: bool = False,
        scorecard: Optional[Dict[str, float]] = None,
    ) -> ProofPacket:
        """
        Build a Proof Packet from a phase output.
        
        Args:
            output_text: Raw phase output text
            phase_name: Name of the phase
            mission_id: Mission identifier
            model_name: Name of model that generated output
            prev_packet: Previous packet for this phase (if any)
            is_compression: Whether this is a compression operation
            depth_increased: Whether depth/escalation increased
            scorecard: Optional scorecard metrics for divergence check
            
        Returns:
            Complete ProofPacket
        """
        logger.debug(f"[PROOF_BUILDER] Building packet for phase '{phase_name}'")
        
        # 1. Extract claims
        claims = self._claim_extractor.extract_claims(output_text)
        logger.debug(f"[PROOF_BUILDER] Extracted {len(claims)} claims")
        
        # 2. Bind evidence
        evidence_bindings = self._evidence_binder.bind_evidence(claims)
        
        # 3. Adjust claim confidence based on evidence
        claims = self._evidence_binder.adjust_claim_confidence(claims, evidence_bindings)
        
        # 4. Detect contradictions
        contradiction_entries = self._detect_contradictions(claims)
        
        # 5. Compute uncertainty declarations
        uncertainty_entries = self._compute_uncertainty(claims, evidence_bindings)
        
        # 6. Collect decision trace
        decision_trace = self._collect_decision_trace(mission_id, phase_name)
        
        # 7. Build metadata
        metadata = ProofPacketMetadata(
            proof_version="1.0",
            mission_id=mission_id,
            phase_id=phase_name,
            timestamp=datetime.utcnow(),
            generation_model_hash=hash_model_identifier(model_name),
            blinded_evaluation_ready=True,
        )
        
        # 8. Create packet (before integrity check)
        packet = ProofPacket(
            claim_set=claims,
            evidence_bindings=evidence_bindings,
            contradiction_status=contradiction_entries,
            uncertainty_declarations=uncertainty_entries,
            decision_trace=decision_trace,
            integrity_flags=IntegrityFlags(),  # Will be updated
            metadata=metadata,
        )
        
        # 9. Run integrity checks
        integrity_flags = self._integrity_checker.check_all(
            current=packet,
            prev=prev_packet,
            is_compression=is_compression,
            depth_increased=depth_increased,
            scorecard=scorecard,
        )
        packet.integrity_flags = integrity_flags
        
        logger.debug(
            f"[PROOF_BUILDER] Built packet {packet.packet_id}: "
            f"{len(claims)} claims, {packet.evidence_coverage_ratio:.1%} evidence coverage, "
            f"{packet.integrity_flags.violation_count} violations"
        )
        
        return packet
    
    def build_from_arbiter_decision(
        self,
        arbiter_decision: "ArbiterDecision",
        mission_id: str,
        council_packets: Optional[List[ProofPacket]] = None,
        model_name: str = "",
    ) -> ProofPacket:
        """
        Build a Proof Packet from an arbiter synthesis decision.
        
        Aggregates claims from council packets and adds synthesis claims.
        
        Args:
            arbiter_decision: The arbiter decision
            mission_id: Mission identifier
            council_packets: Proof packets from contributing councils
            model_name: Name of arbiter model
            
        Returns:
            Complete ProofPacket for the synthesis
        """
        logger.debug(f"[PROOF_BUILDER] Building synthesis packet for mission '{mission_id}'")
        
        # Extract synthesis output text
        output_text = str(arbiter_decision.final_output) if arbiter_decision.final_output else ""
        
        # 1. Extract claims from synthesis output
        synthesis_claims = self._claim_extractor.extract_claims(output_text)
        
        # 2. Aggregate claims from council packets
        all_claims = list(synthesis_claims)
        all_evidence_bindings: List[EvidenceBinding] = []
        all_contradictions: List[ContradictionEntry] = []
        all_uncertainties: List[UncertaintyEntry] = []
        
        if council_packets:
            for council_packet in council_packets:
                # Add claims (avoiding duplicates by ID)
                existing_ids = {c.claim_id for c in all_claims}
                for claim in council_packet.claim_set:
                    if claim.claim_id not in existing_ids:
                        all_claims.append(claim)
                        existing_ids.add(claim.claim_id)
                
                # Add evidence bindings
                all_evidence_bindings.extend(council_packet.evidence_bindings)
                
                # Add contradictions
                all_contradictions.extend(council_packet.contradiction_status)
                
                # Add uncertainties
                all_uncertainties.extend(council_packet.uncertainty_declarations)
        
        # 3. Bind evidence for synthesis claims
        synthesis_evidence = self._evidence_binder.bind_evidence(synthesis_claims)
        all_evidence_bindings.extend(synthesis_evidence)
        
        # 4. Adjust synthesis claim confidence
        synthesis_claims = self._evidence_binder.adjust_claim_confidence(
            synthesis_claims, synthesis_evidence
        )
        
        # 5. Detect contradictions in synthesis
        synthesis_contradictions = self._detect_contradictions(synthesis_claims)
        all_contradictions.extend(synthesis_contradictions)
        
        # 6. Compute synthesis uncertainty
        synthesis_uncertainty = self._compute_uncertainty(synthesis_claims, synthesis_evidence)
        all_uncertainties.extend(synthesis_uncertainty)
        
        # 7. Collect decision trace for entire mission
        decision_trace = self._collect_decision_trace(mission_id, "synthesis")
        
        # 8. Build metadata
        metadata = ProofPacketMetadata(
            proof_version="1.0",
            mission_id=mission_id,
            phase_id="synthesis",
            timestamp=datetime.utcnow(),
            generation_model_hash=hash_model_identifier(model_name),
            blinded_evaluation_ready=True,
        )
        
        # 9. Create packet
        packet = ProofPacket(
            claim_set=all_claims,
            evidence_bindings=all_evidence_bindings,
            contradiction_status=all_contradictions,
            uncertainty_declarations=all_uncertainties,
            decision_trace=decision_trace,
            integrity_flags=IntegrityFlags(),
            metadata=metadata,
        )
        
        # 10. Run integrity checks against last council packet
        last_packet = council_packets[-1] if council_packets else None
        integrity_flags = self._integrity_checker.check_all(
            current=packet,
            prev=last_packet,
            is_compression=False,
            depth_increased=False,
            scorecard=arbiter_decision.cost_context.get("final_scorecard") if arbiter_decision.cost_context else None,
        )
        packet.integrity_flags = integrity_flags
        
        logger.debug(
            f"[PROOF_BUILDER] Built synthesis packet {packet.packet_id}: "
            f"{len(all_claims)} total claims from {len(council_packets or [])} councils"
        )
        
        return packet
    
    def _detect_contradictions(
        self,
        claims: List[ClaimEntry],
    ) -> List[ContradictionEntry]:
        """
        Detect contradictions among claims.
        
        Args:
            claims: List of claims to check
            
        Returns:
            List of ContradictionEntry for each claim
        """
        entries = []
        
        # Build contradiction map
        contradiction_map: Dict[str, List[str]] = {c.claim_id: [] for c in claims}
        
        if self._contradiction_detector is not None and len(claims) >= 2:
            try:
                # Import Claim from epistemics for detector
                from ..epistemics.claim_validator import Claim, ClaimType
                
                # Convert ClaimEntry to Claim for detector
                raw_claims = []
                claim_id_map = {}
                
                for entry in claims:
                    claim = Claim(
                        text=entry.normalized_text,
                        claim_type=ClaimType.INFERENCE,  # Default type
                        confidence=entry.confidence_estimate,
                    )
                    raw_claims.append(claim)
                    claim_id_map[claim.id] = entry.claim_id
                
                # Detect contradictions
                results = self._contradiction_detector.detect_all(raw_claims, max_pairs=50)
                
                for result in results:
                    if result.is_contradiction:
                        claim1_id = claim_id_map.get(result.claim1.id, result.claim1.id)
                        claim2_id = claim_id_map.get(result.claim2.id, result.claim2.id)
                        
                        if claim1_id in contradiction_map:
                            contradiction_map[claim1_id].append(claim2_id)
                        if claim2_id in contradiction_map:
                            contradiction_map[claim2_id].append(claim1_id)
            
            except Exception as e:
                logger.debug(f"[PROOF_BUILDER] Contradiction detection failed: {e}")
        
        # Build entries
        for claim in claims:
            contradicting = contradiction_map.get(claim.claim_id, [])
            entries.append(ContradictionEntry(
                claim_id=claim.claim_id,
                contradiction_checked=self._contradiction_detector is not None,
                contradictions_found=len(contradicting),
                resolution_status=ResolutionStatus.UNRESOLVED if contradicting else ResolutionStatus.ACCEPTED,
                contradicting_claim_ids=contradicting,
            ))
        
        return entries
    
    def _compute_uncertainty(
        self,
        claims: List[ClaimEntry],
        evidence_bindings: List[EvidenceBinding],
    ) -> List[UncertaintyEntry]:
        """
        Compute uncertainty declarations for claims.
        
        Uncertainty is derived from:
        - Evidence coverage (low coverage = high uncertainty)
        - Claim type (assumptions have higher uncertainty)
        - Confidence (low confidence = high uncertainty)
        
        Args:
            claims: List of claims
            evidence_bindings: Evidence bindings for claims
            
        Returns:
            List of UncertaintyEntry for each claim
        """
        entries = []
        binding_map = {b.claim_id: b for b in evidence_bindings}
        
        for claim in claims:
            binding = binding_map.get(claim.claim_id)
            
            # Base uncertainty from inverse of confidence
            base_uncertainty = 1.0 - claim.confidence_estimate
            
            # Determine uncertainty source
            if binding is None or binding.coverage_score == 0:
                source = UncertaintySource.MISSING_EVIDENCE
                # Increase uncertainty for missing evidence
                epistemic_uncertainty = min(1.0, base_uncertainty + 0.2)
            elif claim.confidence_estimate < 0.5:
                source = UncertaintySource.MODEL_LIMITATION
                epistemic_uncertainty = base_uncertainty
            else:
                source = UncertaintySource.EXTRAPOLATION
                epistemic_uncertainty = base_uncertainty
            
            entries.append(UncertaintyEntry(
                claim_id=claim.claim_id,
                epistemic_uncertainty=min(1.0, max(0.0, epistemic_uncertainty)),
                source_of_uncertainty=source,
            ))
        
        return entries
    
    def _collect_decision_trace(
        self,
        mission_id: str,
        phase_name: str,
    ) -> DecisionTrace:
        """
        Collect decision IDs for the phase.
        
        Args:
            mission_id: Mission identifier
            phase_name: Phase name
            
        Returns:
            DecisionTrace with decision IDs
        """
        trace = DecisionTrace()
        
        if self._decision_store is None:
            return trace
        
        try:
            # Use the DecisionStore helper method
            if hasattr(self._decision_store, 'get_phase_decision_ids'):
                ids = self._decision_store.get_phase_decision_ids(mission_id, phase_name)
                trace.routing_decision_ids = ids.get("routing_decision_ids", [])
                trace.escalation_decision_ids = ids.get("escalation_decision_ids", [])
                trace.alignment_action_ids = ids.get("alignment_action_ids", [])
                trace.tool_event_ids = ids.get("tool_event_ids", [])
            else:
                # Fallback for older DecisionStore without the helper
                from ..decisions.decision_record import DecisionType
                
                records = self._decision_store.get_by_phase(mission_id, phase_name)
                
                for record in records:
                    if record.decision_type == DecisionType.ROUTING_DECISION:
                        trace.routing_decision_ids.append(record.decision_id)
                    elif record.decision_type == DecisionType.MODEL_SELECTION:
                        trace.routing_decision_ids.append(record.decision_id)
                    elif record.decision_type == DecisionType.RETRY_ESCALATION:
                        trace.escalation_decision_ids.append(record.decision_id)
                    elif record.decision_type == DecisionType.TOOL_USAGE:
                        trace.tool_event_ids.append(record.decision_id)
        
        except Exception as e:
            logger.debug(f"[PROOF_BUILDER] Decision trace collection failed: {e}")
        
        return trace
    
    def set_evidence_store(self, store: "MissionRAGStore") -> None:
        """Update the evidence store."""
        self._evidence_store = store
        self._evidence_binder.set_rag_store(store)
    
    def set_decision_store(self, store: "DecisionStore") -> None:
        """Update the decision store."""
        self._decision_store = store
    
    def set_contradiction_detector(self, detector: "ContradictionDetector") -> None:
        """Update the contradiction detector."""
        self._contradiction_detector = detector


# Global builder instance
_builder: Optional[ProofPacketBuilder] = None


def get_proof_packet_builder(
    evidence_store: Optional["MissionRAGStore"] = None,
    decision_store: Optional["DecisionStore"] = None,
    contradiction_detector: Optional["ContradictionDetector"] = None,
) -> ProofPacketBuilder:
    """Get the global proof packet builder instance."""
    global _builder
    if _builder is None:
        _builder = ProofPacketBuilder(
            evidence_store=evidence_store,
            decision_store=decision_store,
            contradiction_detector=contradiction_detector,
        )
    else:
        # Update stores if provided
        if evidence_store is not None:
            _builder.set_evidence_store(evidence_store)
        if decision_store is not None:
            _builder.set_decision_store(decision_store)
        if contradiction_detector is not None:
            _builder.set_contradiction_detector(contradiction_detector)
    
    return _builder

