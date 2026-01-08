"""
Memory Retrieval Auditor - Logs all memory retrievals with traceability.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..schemas import RetrievalAuditLog

logger = logging.getLogger(__name__)


class MemoryRetrievalAuditor:
    """
    Logs all memory retrievals:
    - What was retrieved (claim IDs, evidence IDs)
    - Why it was selected (query, similarity score)
    - Whether it influenced output (trace to final answer)
    """
    
    def __init__(self):
        """Initialize retrieval auditor."""
        self._audit_logs: List[RetrievalAuditLog] = []
    
    def log_retrieval(
        self,
        mission_id: str,
        phase: str,
        query: str,
        retrieved_ids: List[str],
        similarity_scores: Optional[Dict[str, float]] = None,
        selection_reason: Optional[str] = None
    ) -> RetrievalAuditLog:
        """
        Log a memory retrieval operation.
        
        Args:
            mission_id: Mission ID
            phase: Phase name
            query: Query used for retrieval
            retrieved_ids: List of retrieved claim/evidence IDs
            similarity_scores: Optional dict mapping ID -> similarity score
            selection_reason: Optional reason for selection
            
        Returns:
            RetrievalAuditLog entry
        """
        log_entry = RetrievalAuditLog(
            timestamp=datetime.utcnow(),
            mission_id=mission_id,
            phase=phase,
            query=query,
            retrieved_ids=retrieved_ids,
            similarity_scores=similarity_scores or {},
            selection_reason=selection_reason or "Retrieved from memory",
            influenced_output=False,
            output_trace=None
        )
        
        self._audit_logs.append(log_entry)
        
        logger.debug(
            f"Logged retrieval: mission={mission_id}, phase={phase}, "
            f"retrieved={len(retrieved_ids)} items"
        )
        
        return log_entry
    
    def mark_influenced(
        self,
        log_entry: RetrievalAuditLog,
        output_trace: str
    ) -> None:
        """
        Mark that a retrieval influenced the final output.
        
        Args:
            log_entry: Retrieval log entry to update
            output_trace: Trace showing how it influenced output
        """
        log_entry.influenced_output = True
        log_entry.output_trace = output_trace
        
        logger.debug(
            f"Marked retrieval as influenced: "
            f"retrieved_ids={log_entry.retrieved_ids[:3]}..."
        )
    
    def get_audit_logs(
        self,
        mission_id: Optional[str] = None,
        phase: Optional[str] = None,
        influenced_only: bool = False
    ) -> List[RetrievalAuditLog]:
        """
        Get audit logs with optional filtering.
        
        Args:
            mission_id: Optional mission ID filter
            phase: Optional phase filter
            influenced_only: Only return logs that influenced output
            
        Returns:
            List of audit logs
        """
        logs = self._audit_logs
        
        if mission_id:
            logs = [log for log in logs if log.mission_id == mission_id]
        
        if phase:
            logs = [log for log in logs if log.phase == phase]
        
        if influenced_only:
            logs = [log for log in logs if log.influenced_output]
        
        return logs
    
    def get_retrieval_stats(
        self,
        mission_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about retrievals.
        
        Args:
            mission_id: Optional mission ID filter
            
        Returns:
            Dict with retrieval statistics
        """
        logs = self.get_audit_logs(mission_id=mission_id)
        
        total_retrievals = len(logs)
        total_items_retrieved = sum(len(log.retrieved_ids) for log in logs)
        influenced_count = sum(1 for log in logs if log.influenced_output)
        
        # Average similarity scores
        all_scores = []
        for log in logs:
            all_scores.extend(log.similarity_scores.values())
        avg_similarity = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "total_retrievals": total_retrievals,
            "total_items_retrieved": total_items_retrieved,
            "influenced_output_count": influenced_count,
            "influence_rate": influenced_count / total_retrievals if total_retrievals > 0 else 0.0,
            "avg_similarity_score": avg_similarity,
            "avg_items_per_retrieval": total_items_retrieved / total_retrievals if total_retrievals > 0 else 0.0
        }
    
    def clear_logs(self, mission_id: Optional[str] = None) -> None:
        """
        Clear audit logs.
        
        Args:
            mission_id: Optional mission ID - only clear logs for this mission
        """
        if mission_id:
            self._audit_logs = [
                log for log in self._audit_logs 
                if log.mission_id != mission_id
            ]
        else:
            self._audit_logs = []
        
        logger.debug(f"Cleared audit logs (mission_id={mission_id})")

