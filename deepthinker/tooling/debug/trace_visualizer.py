"""
Mission Trace Visualizer - Produces structured JSON trace of mission execution.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..schemas import MissionTrace

logger = logging.getLogger(__name__)


class MissionTraceVisualizer:
    """
    Produces structured JSON trace:
    - Phase durations
    - Searches executed vs skipped (with justifications)
    - Memory usage (retrievals, writes)
    - Hallucination flags (unverified claims)
    """
    
    def __init__(self):
        """Initialize trace visualizer."""
        self._phase_data: List[Dict[str, Any]] = []
        self._search_log: List[Dict[str, Any]] = []
        self._memory_log: List[Dict[str, Any]] = []
        self._hallucination_flags: List[str] = []
        self._unverified_claims: List[str] = []
        self._total_tokens = 0
        self._start_time: Optional[datetime] = None
    
    def start_mission(self, mission_id: str, objective: str) -> None:
        """
        Start tracking a mission.
        
        Args:
            mission_id: Mission ID
            objective: Mission objective
        """
        self.mission_id = mission_id
        self.objective = objective
        self._start_time = datetime.utcnow()
        self._phase_data = []
        self._search_log = []
        self._memory_log = []
        self._hallucination_flags = []
        self._unverified_claims = []
        self._total_tokens = 0
    
    def record_phase(
        self,
        phase_name: str,
        duration_seconds: float,
        tokens_spent: int = 0,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record phase execution data.
        
        Args:
            phase_name: Phase name
            duration_seconds: Phase duration in seconds
            tokens_spent: Tokens spent in phase
            metrics: Optional additional metrics
        """
        phase_entry = {
            "phase_name": phase_name,
            "duration_seconds": duration_seconds,
            "tokens_spent": tokens_spent,
            "metrics": metrics or {}
        }
        self._phase_data.append(phase_entry)
        self._total_tokens += tokens_spent
    
    def record_search(
        self,
        phase: str,
        executed: bool,
        justification: Optional[str] = None,
        queries: Optional[List[str]] = None,
        results_count: int = 0
    ) -> None:
        """
        Record search execution or skip.
        
        Args:
            phase: Phase where search occurred
            executed: Whether search was executed
            justification: Optional justification for search or skip
            queries: Optional list of queries
            results_count: Number of results if executed
        """
        search_entry = {
            "phase": phase,
            "executed": executed,
            "justification": justification,
            "queries": queries or [],
            "results_count": results_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._search_log.append(search_entry)
    
    def record_memory_operation(
        self,
        operation: str,  # "retrieval" or "write"
        phase: str,
        item_count: int = 0,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record memory operation.
        
        Args:
            operation: Operation type ("retrieval" or "write")
            phase: Phase where operation occurred
            item_count: Number of items retrieved/written
            details: Optional additional details
        """
        memory_entry = {
            "operation": operation,
            "phase": phase,
            "item_count": item_count,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self._memory_log.append(memory_entry)
    
    def flag_hallucination(
        self,
        claim_id: str,
        reason: str,
        phase: Optional[str] = None
    ) -> None:
        """
        Flag a potential hallucination.
        
        Args:
            claim_id: Claim ID that may be hallucinated
            reason: Reason for flagging
            phase: Optional phase where flagged
        """
        flag = f"{claim_id}: {reason}"
        if phase:
            flag = f"[{phase}] {flag}"
        self._hallucination_flags.append(flag)
    
    def record_unverified_claim(self, claim_id: str, claim_text: str) -> None:
        """
        Record an unverified claim.
        
        Args:
            claim_id: Claim ID
            claim_text: Claim text
        """
        self._unverified_claims.append(f"{claim_id}: {claim_text[:100]}...")
    
    def generate_trace(self) -> MissionTrace:
        """
        Generate structured trace.
        
        Returns:
            MissionTrace with all collected data
        """
        # Calculate total time
        total_time = 0.0
        if self._start_time:
            total_time = (datetime.utcnow() - self._start_time).total_seconds()
        
        # Count searches
        searches_executed = sum(1 for s in self._search_log if s["executed"])
        searches_skipped = sum(1 for s in self._search_log if not s["executed"])
        
        # Count memory operations
        memory_retrievals = sum(
            1 for m in self._memory_log 
            if m["operation"] == "retrieval"
        )
        memory_writes = sum(
            1 for m in self._memory_log 
            if m["operation"] == "write"
        )
        
        trace = MissionTrace(
            mission_id=self.mission_id,
            objective=self.objective,
            phases=self._phase_data,
            searches_executed=searches_executed,
            searches_skipped=searches_skipped,
            search_justifications=[
                {
                    "phase": s["phase"],
                    "executed": s["executed"],
                    "justification": s["justification"],
                    "queries": s["queries"]
                }
                for s in self._search_log
            ],
            memory_retrievals=memory_retrievals,
            memory_writes=memory_writes,
            hallucination_flags=self._hallucination_flags,
            unverified_claims=self._unverified_claims,
            total_tokens=self._total_tokens,
            total_time_seconds=total_time,
            created_at=datetime.utcnow()
        )
        
        logger.info(
            f"Generated trace for mission {self.mission_id}: "
            f"{len(self._phase_data)} phases, {searches_executed} searches executed, "
            f"{len(self._hallucination_flags)} hallucination flags"
        )
        
        return trace
    
    def export_json(self, filepath: Optional[str] = None) -> str:
        """
        Export trace as JSON.
        
        Args:
            filepath: Optional filepath to save to
            
        Returns:
            JSON string
        """
        trace = self.generate_trace()
        json_str = trace.json(indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Exported trace to {filepath}")
        
        return json_str

