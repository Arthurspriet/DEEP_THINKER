"""
Proof Store for Proof Packets.

Provides JSONL-based append-only persistence for Proof Packets.
Human-readable, deterministic, and compatible with existing kb/ structure.

Storage structure:
    kb/proofs/{mission_id}/
    ├── packets.jsonl          # All proof packets (append-only)
    ├── blinded_packets.jsonl  # Blinded views for evaluation
    └── integrity_audit.json   # Summary of integrity violations
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .proof_packet import ProofPacket, IntegrityFlags

logger = logging.getLogger(__name__)


class ProofStore:
    """
    Stores Proof Packets in JSONL format.
    
    Key properties:
    - Append-only: Never modifies existing packets
    - Human-readable: Plain JSONL format
    - Mission-scoped: Each mission has its own proof directory
    - Deterministic: Reproducible reads
    
    Usage:
        store = ProofStore()
        
        # Write a packet
        store.write(packet)
        
        # Read all packets
        packets = store.read_all(mission_id)
        
        # Get latest for a phase
        latest = store.get_latest(mission_id, "research")
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the proof store.
        
        Args:
            base_dir: Base directory for proofs (default: kb/proofs)
        """
        if base_dir is None:
            base_dir = Path("kb/proofs")
        
        self.base_dir = Path(base_dir)
    
    def _get_mission_dir(self, mission_id: str) -> Path:
        """Get the proof directory for a mission."""
        return self.base_dir / mission_id
    
    def _get_packets_path(self, mission_id: str) -> Path:
        """Get path to packets.jsonl for a mission."""
        return self._get_mission_dir(mission_id) / "packets.jsonl"
    
    def _get_blinded_path(self, mission_id: str) -> Path:
        """Get path to blinded_packets.jsonl for a mission."""
        return self._get_mission_dir(mission_id) / "blinded_packets.jsonl"
    
    def _get_audit_path(self, mission_id: str) -> Path:
        """Get path to integrity_audit.json for a mission."""
        return self._get_mission_dir(mission_id) / "integrity_audit.json"
    
    def _ensure_mission_dir(self, mission_id: str) -> bool:
        """
        Ensure mission directory exists.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            True if directory exists or was created
        """
        try:
            mission_dir = self._get_mission_dir(mission_id)
            mission_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to create mission dir: {e}")
            return False
    
    def write(self, packet: ProofPacket) -> bool:
        """
        Write a proof packet to the store.
        
        Appends to the mission's packets.jsonl file.
        
        Args:
            packet: ProofPacket to persist
            
        Returns:
            True if write succeeded
        """
        mission_id = packet.metadata.mission_id
        
        if not mission_id:
            logger.warning("[PROOF_STORE] Cannot write packet without mission_id")
            return False
        
        try:
            if not self._ensure_mission_dir(mission_id):
                return False
            
            packets_path = self._get_packets_path(mission_id)
            packet_dict = packet.to_dict()
            json_line = json.dumps(packet_dict, ensure_ascii=False)
            
            with open(packets_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            
            logger.debug(
                f"[PROOF_STORE] Wrote packet {packet.packet_id} "
                f"for phase {packet.metadata.phase_id}"
            )
            
            # Update integrity audit if there are violations
            if packet.integrity_flags.has_violations:
                self._update_integrity_audit(mission_id, packet)
            
            return True
            
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to write packet: {e}")
            return False
    
    def write_blinded(self, mission_id: str, blinded_data: Dict[str, Any]) -> bool:
        """
        Write a blinded packet view.
        
        Args:
            mission_id: Mission identifier
            blinded_data: Blinded packet data dictionary
            
        Returns:
            True if write succeeded
        """
        try:
            if not self._ensure_mission_dir(mission_id):
                return False
            
            blinded_path = self._get_blinded_path(mission_id)
            json_line = json.dumps(blinded_data, ensure_ascii=False)
            
            with open(blinded_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            
            return True
            
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to write blinded packet: {e}")
            return False
    
    def read_all(self, mission_id: str) -> List[ProofPacket]:
        """
        Read all proof packets for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            List of ProofPacket objects, sorted by timestamp
        """
        packets_path = self._get_packets_path(mission_id)
        
        if not packets_path.exists():
            return []
        
        packets = []
        try:
            with open(packets_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        packet = ProofPacket.from_dict(data)
                        packets.append(packet)
                    except Exception as e:
                        logger.warning(
                            f"[PROOF_STORE] Skipping malformed line {line_num}: {e}"
                        )
            
            # Sort by timestamp
            packets.sort(key=lambda p: p.metadata.timestamp)
            return packets
            
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to read packets: {e}")
            return []
    
    def iter_packets(self, mission_id: str) -> Iterator[ProofPacket]:
        """
        Iterate over packets without loading all into memory.
        
        Args:
            mission_id: Mission identifier
            
        Yields:
            ProofPacket objects in file order
        """
        packets_path = self._get_packets_path(mission_id)
        
        if not packets_path.exists():
            return
        
        try:
            with open(packets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield ProofPacket.from_dict(data)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to iterate packets: {e}")
    
    def get_latest(
        self,
        mission_id: str,
        phase_name: Optional[str] = None,
    ) -> Optional[ProofPacket]:
        """
        Get the most recent packet for a mission/phase.
        
        Args:
            mission_id: Mission identifier
            phase_name: Optional phase name to filter by
            
        Returns:
            Most recent ProofPacket, or None
        """
        packets = self.read_all(mission_id)
        
        if not packets:
            return None
        
        if phase_name:
            packets = [p for p in packets if p.metadata.phase_id == phase_name]
        
        if not packets:
            return None
        
        # Return latest (already sorted by timestamp)
        return packets[-1]
    
    def get_by_phase(self, mission_id: str, phase_name: str) -> List[ProofPacket]:
        """
        Get all packets for a specific phase.
        
        Args:
            mission_id: Mission identifier
            phase_name: Phase name to filter by
            
        Returns:
            List of matching ProofPacket objects
        """
        packets = self.read_all(mission_id)
        return [p for p in packets if p.metadata.phase_id == phase_name]
    
    def get_packet(self, mission_id: str, packet_id: str) -> Optional[ProofPacket]:
        """
        Get a specific packet by ID.
        
        Args:
            mission_id: Mission identifier
            packet_id: Packet ID to find
            
        Returns:
            ProofPacket if found, None otherwise
        """
        for packet in self.iter_packets(mission_id):
            if packet.packet_id == packet_id:
                return packet
        return None
    
    def _update_integrity_audit(self, mission_id: str, packet: ProofPacket) -> None:
        """
        Update integrity audit file with violation info.
        
        Args:
            mission_id: Mission identifier
            packet: Packet with violations
        """
        try:
            audit_path = self._get_audit_path(mission_id)
            
            # Load existing audit
            audit: Dict[str, Any] = {
                "mission_id": mission_id,
                "total_violations": 0,
                "violations_by_type": {},
                "packets_with_violations": [],
            }
            
            if audit_path.exists():
                with open(audit_path, "r", encoding="utf-8") as f:
                    audit = json.load(f)
            
            # Update with new violations
            flags = packet.integrity_flags
            
            if flags.evidence_conservation_violation:
                audit["violations_by_type"]["evidence_conservation"] = (
                    audit["violations_by_type"].get("evidence_conservation", 0) + 1
                )
                audit["total_violations"] += 1
            
            if flags.monotonic_uncertainty_violation:
                audit["violations_by_type"]["monotonic_uncertainty"] = (
                    audit["violations_by_type"].get("monotonic_uncertainty", 0) + 1
                )
                audit["total_violations"] += 1
            
            if flags.no_free_lunch_depth_violation:
                audit["violations_by_type"]["no_free_lunch"] = (
                    audit["violations_by_type"].get("no_free_lunch", 0) + 1
                )
                audit["total_violations"] += 1
            
            if flags.metric_divergence_flag:
                audit["violations_by_type"]["metric_divergence"] = (
                    audit["violations_by_type"].get("metric_divergence", 0) + 1
                )
                audit["total_violations"] += 1
            
            # Add packet to list
            audit["packets_with_violations"].append({
                "packet_id": packet.packet_id,
                "phase_id": packet.metadata.phase_id,
                "timestamp": packet.metadata.timestamp.isoformat(),
                "violation_count": flags.violation_count,
            })
            
            # Write updated audit
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.debug(f"[PROOF_STORE] Failed to update integrity audit: {e}")
    
    def get_integrity_audit(self, mission_id: str) -> Dict[str, Any]:
        """
        Get integrity audit summary for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            Audit summary dictionary
        """
        audit_path = self._get_audit_path(mission_id)
        
        if not audit_path.exists():
            return {
                "mission_id": mission_id,
                "total_violations": 0,
                "violations_by_type": {},
                "packets_with_violations": [],
            }
        
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to read integrity audit: {e}")
            return {}
    
    def compute_mission_summary(self, mission_id: str) -> Dict[str, Any]:
        """
        Compute summary statistics for a mission's proof packets.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            Summary dictionary with aggregated stats
        """
        packets = self.read_all(mission_id)
        
        if not packets:
            return {"mission_id": mission_id, "total_packets": 0}
        
        total_claims = sum(p.claim_count for p in packets)
        total_violations = sum(p.integrity_flags.violation_count for p in packets)
        
        phases_covered = set(p.metadata.phase_id for p in packets)
        
        avg_evidence_coverage = (
            sum(p.evidence_coverage_ratio for p in packets) / len(packets)
        )
        avg_confidence = (
            sum(p.average_confidence for p in packets) / len(packets)
        )
        avg_uncertainty = (
            sum(p.average_uncertainty for p in packets) / len(packets)
        )
        
        return {
            "mission_id": mission_id,
            "total_packets": len(packets),
            "total_claims": total_claims,
            "total_integrity_violations": total_violations,
            "phases_covered": list(phases_covered),
            "average_evidence_coverage": avg_evidence_coverage,
            "average_confidence": avg_confidence,
            "average_uncertainty": avg_uncertainty,
            "first_packet_at": packets[0].metadata.timestamp.isoformat(),
            "last_packet_at": packets[-1].metadata.timestamp.isoformat(),
        }
    
    def has_packets(self, mission_id: str) -> bool:
        """Check if a mission has any proof packets."""
        return self._get_packets_path(mission_id).exists()
    
    def delete_mission_proofs(self, mission_id: str) -> bool:
        """
        Delete all proof data for a mission.
        
        Use with caution - this is destructive.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            True if deletion succeeded
        """
        try:
            import shutil
            mission_dir = self._get_mission_dir(mission_id)
            if mission_dir.exists():
                shutil.rmtree(mission_dir)
            return True
        except Exception as e:
            logger.warning(f"[PROOF_STORE] Failed to delete mission proofs: {e}")
            return False


# Global store instance
_store: Optional[ProofStore] = None


def get_proof_store(base_dir: Optional[Path] = None) -> ProofStore:
    """Get the global proof store instance."""
    global _store
    if _store is None:
        _store = ProofStore(base_dir=base_dir)
    return _store


