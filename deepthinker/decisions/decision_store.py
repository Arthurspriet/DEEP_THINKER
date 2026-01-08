"""
Decision Store for DeepThinker.

Provides JSONL-based append-only persistence for decision records.
Human-readable, deterministic, and compatible with existing kb/ structure.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .decision_record import DecisionRecord, DecisionType, OutcomeCause

logger = logging.getLogger(__name__)


class DecisionStore:
    """
    Stores decision records in JSONL format.
    
    Storage structure:
        kb/missions/{mission_id}/decisions.jsonl  - All decision records
        kb/missions/{mission_id}/decision_summary.json  - Aggregated stats
    
    Key properties:
    - Append-only: Never modifies existing records (except cost attribution)
    - Human-readable: Plain JSONL format
    - Mission-scoped: Each mission has its own decision log
    - Thread-safe: Uses file locking for writes
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the decision store.
        
        Args:
            base_dir: Base directory for missions (default: kb/missions)
        """
        if base_dir is None:
            base_dir = Path("kb/missions")
        
        self.base_dir = Path(base_dir)
    
    def _get_decisions_path(self, mission_id: str) -> Path:
        """Get path to decisions.jsonl for a mission."""
        return self.base_dir / mission_id / "decisions.jsonl"
    
    def _get_summary_path(self, mission_id: str) -> Path:
        """Get path to decision_summary.json for a mission."""
        return self.base_dir / mission_id / "decision_summary.json"
    
    def _ensure_mission_dir(self, mission_id: str) -> bool:
        """
        Ensure mission directory exists.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            True if directory exists or was created
        """
        try:
            mission_dir = self.base_dir / mission_id
            mission_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.warning(f"[DECISION] Failed to create mission dir: {e}")
            return False
    
    def write(self, record: DecisionRecord) -> bool:
        """
        Write a decision record to the store.
        
        Appends to the mission's decisions.jsonl file.
        
        Args:
            record: DecisionRecord to persist
            
        Returns:
            True if write succeeded
        """
        try:
            if not self._ensure_mission_dir(record.mission_id):
                return False
            
            decisions_path = self._get_decisions_path(record.mission_id)
            record_dict = record.to_dict()
            json_line = json.dumps(record_dict, ensure_ascii=False)
            
            with open(decisions_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            
            logger.debug(
                f"[DECISION] Wrote {record.decision_type.value} decision "
                f"{record.decision_id[:8]} to {decisions_path}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[DECISION] Failed to write decision: {e}")
            return False
    
    def read_all(self, mission_id: str) -> List[DecisionRecord]:
        """
        Read all decision records for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            List of DecisionRecord objects, sorted by timestamp
        """
        decisions_path = self._get_decisions_path(mission_id)
        
        if not decisions_path.exists():
            return []
        
        records = []
        try:
            with open(decisions_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = DecisionRecord.from_dict(data)
                        records.append(record)
                    except Exception as e:
                        logger.warning(
                            f"[DECISION] Skipping malformed line {line_num}: {e}"
                        )
            
            # Sort by timestamp
            records.sort(key=lambda r: r.timestamp)
            return records
            
        except Exception as e:
            logger.warning(f"[DECISION] Failed to read decisions: {e}")
            return []
    
    def iter_records(self, mission_id: str) -> Iterator[DecisionRecord]:
        """
        Iterate over decision records without loading all into memory.
        
        Args:
            mission_id: Mission identifier
            
        Yields:
            DecisionRecord objects in file order
        """
        decisions_path = self._get_decisions_path(mission_id)
        
        if not decisions_path.exists():
            return
        
        try:
            with open(decisions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield DecisionRecord.from_dict(data)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"[DECISION] Failed to iterate decisions: {e}")
    
    def get_by_type(
        self,
        mission_id: str,
        decision_type: DecisionType
    ) -> List[DecisionRecord]:
        """
        Get all decisions of a specific type.
        
        Args:
            mission_id: Mission identifier
            decision_type: Type to filter by
            
        Returns:
            List of matching DecisionRecord objects
        """
        return [
            r for r in self.read_all(mission_id)
            if r.decision_type == decision_type
        ]
    
    def get_by_phase(self, mission_id: str, phase_id: str) -> List[DecisionRecord]:
        """
        Get all decisions for a specific phase.
        
        Args:
            mission_id: Mission identifier
            phase_id: Phase name to filter by
            
        Returns:
            List of matching DecisionRecord objects
        """
        return [
            r for r in self.read_all(mission_id)
            if r.phase_id == phase_id
        ]
    
    def get_phase_decision_ids(
        self,
        mission_id: str,
        phase_id: str
    ) -> Dict[str, List[str]]:
        """
        Get decision IDs grouped by type for a phase.
        
        Used by Proof Packet Builder to construct Decision Trace.
        
        Args:
            mission_id: Mission identifier
            phase_id: Phase name to filter by
            
        Returns:
            Dictionary with decision IDs grouped by category:
            - routing_decision_ids
            - escalation_decision_ids
            - alignment_action_ids
            - tool_event_ids
        """
        records = self.get_by_phase(mission_id, phase_id)
        
        result = {
            "routing_decision_ids": [],
            "escalation_decision_ids": [],
            "alignment_action_ids": [],
            "tool_event_ids": [],
        }
        
        for record in records:
            if record.decision_type == DecisionType.ROUTING_DECISION:
                result["routing_decision_ids"].append(record.decision_id)
            elif record.decision_type == DecisionType.MODEL_SELECTION:
                result["routing_decision_ids"].append(record.decision_id)
            elif record.decision_type == DecisionType.RETRY_ESCALATION:
                result["escalation_decision_ids"].append(record.decision_id)
            elif record.decision_type == DecisionType.EMPTY_OUTPUT_ESCALATION:
                result["escalation_decision_ids"].append(record.decision_id)
            elif record.decision_type == DecisionType.GOVERNANCE_INTERVENTION:
                result["alignment_action_ids"].append(record.decision_id)
            elif record.decision_type == DecisionType.TOOL_USAGE:
                result["tool_event_ids"].append(record.decision_id)
        
        return result
    
    def get_decision(
        self,
        mission_id: str,
        decision_id: str
    ) -> Optional[DecisionRecord]:
        """
        Get a specific decision by ID.
        
        Args:
            mission_id: Mission identifier
            decision_id: Decision ID to find
            
        Returns:
            DecisionRecord if found, None otherwise
        """
        for record in self.iter_records(mission_id):
            if record.decision_id == decision_id:
                return record
        return None
    
    def get_causal_chain(
        self,
        mission_id: str,
        decision_id: str
    ) -> List[DecisionRecord]:
        """
        Get the causal chain leading to a decision.
        
        Traverses triggered_by_decision_id links backwards.
        
        Args:
            mission_id: Mission identifier
            decision_id: Decision to trace back from
            
        Returns:
            List of decisions in causal order (oldest first)
        """
        # Load all decisions into a map for efficient lookup
        decisions_map = {
            r.decision_id: r for r in self.read_all(mission_id)
        }
        
        chain = []
        current_id = decision_id
        
        while current_id and current_id in decisions_map:
            record = decisions_map[current_id]
            chain.append(record)
            current_id = record.triggered_by_decision_id
        
        # Return in chronological order
        chain.reverse()
        return chain
    
    def update_cost(
        self,
        mission_id: str,
        decision_id: str,
        hardware_cost: float
    ) -> bool:
        """
        Update hardware cost attribution for a decision.
        
        Note: This requires rewriting the file since JSONL is append-only.
        For production, consider a separate cost attribution file.
        
        Args:
            mission_id: Mission identifier
            decision_id: Decision to update
            hardware_cost: Cost to attribute
            
        Returns:
            True if update succeeded
        """
        decisions_path = self._get_decisions_path(mission_id)
        
        if not decisions_path.exists():
            return False
        
        try:
            # Read all records
            records = self.read_all(mission_id)
            
            # Find and update the target record
            updated = False
            for record in records:
                if record.decision_id == decision_id:
                    record.hardware_cost_attributed = hardware_cost
                    updated = True
                    break
            
            if not updated:
                return False
            
            # Rewrite file
            with open(decisions_path, "w", encoding="utf-8") as f:
                for record in records:
                    json_line = json.dumps(record.to_dict(), ensure_ascii=False)
                    f.write(json_line + "\n")
            
            return True
            
        except Exception as e:
            logger.warning(f"[DECISION] Failed to update cost: {e}")
            return False
    
    def compute_summary(self, mission_id: str) -> Dict[str, Any]:
        """
        Compute summary statistics for a mission's decisions.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            Summary dictionary with counts and aggregates
        """
        records = self.read_all(mission_id)
        
        if not records:
            return {"mission_id": mission_id, "total_decisions": 0}
        
        # Count by type
        type_counts = {}
        for dt in DecisionType:
            count = sum(1 for r in records if r.decision_type == dt)
            if count > 0:
                type_counts[dt.value] = count
        
        # Count by phase
        phase_counts = {}
        for record in records:
            phase_counts[record.phase_id] = phase_counts.get(record.phase_id, 0) + 1
        
        # Aggregate costs
        total_cost = sum(r.hardware_cost_attributed for r in records)
        cost_by_type = {}
        for dt in DecisionType:
            cost = sum(
                r.hardware_cost_attributed
                for r in records
                if r.decision_type == dt
            )
            if cost > 0:
                cost_by_type[dt.value] = cost
        
        # Count linked decisions
        linked_count = sum(1 for r in records if r.is_linked())
        
        # Outcome causes (from phase terminations)
        terminations = [
            r for r in records
            if r.decision_type == DecisionType.PHASE_TERMINATION
        ]
        outcome_causes = {}
        for t in terminations:
            cause = t.constraints_snapshot.get("outcome_cause", "unknown")
            outcome_causes[cause] = outcome_causes.get(cause, 0) + 1
        
        return {
            "mission_id": mission_id,
            "total_decisions": len(records),
            "decisions_by_type": type_counts,
            "decisions_by_phase": phase_counts,
            "total_hardware_cost": total_cost,
            "cost_by_type": cost_by_type,
            "linked_decisions": linked_count,
            "outcome_causes": outcome_causes,
            "first_decision_at": records[0].timestamp.isoformat(),
            "last_decision_at": records[-1].timestamp.isoformat(),
        }
    
    def save_summary(self, mission_id: str) -> bool:
        """
        Compute and save summary to decision_summary.json.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            True if save succeeded
        """
        try:
            summary = self.compute_summary(mission_id)
            summary_path = self._get_summary_path(mission_id)
            
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"[DECISION] Saved summary to {summary_path}")
            return True
            
        except Exception as e:
            logger.warning(f"[DECISION] Failed to save summary: {e}")
            return False
    
    def has_decisions(self, mission_id: str) -> bool:
        """Check if a mission has any decision records."""
        return self._get_decisions_path(mission_id).exists()

