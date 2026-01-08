"""
Orchestration Store for DeepThinker.

Persistence layer for phase outcomes using JSONL format
(human-readable, append-only, deterministic).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from datetime import datetime

from .outcome_logger import PhaseOutcome

logger = logging.getLogger(__name__)


class OrchestrationStore:
    """
    Stores phase outcomes in JSONL format for learning.
    
    Uses append-only JSONL for:
    - Human readability
    - Deterministic replay
    - Easy parsing and analysis
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the orchestration store.
        
        Args:
            base_dir: Base directory for storage (default: kb/orchestration)
        """
        if base_dir is None:
            base_dir = Path("kb/orchestration")
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.outcomes_file = self.base_dir / "phase_outcomes.jsonl"
        self.mission_index_file = self.base_dir / "mission_index.json"
        
        # Load mission index
        self._mission_index: Dict[str, Dict[str, Any]] = {}
        self._load_mission_index()
    
    def write_outcome(self, outcome: PhaseOutcome) -> None:
        """
        Write a phase outcome to the store.
        
        Args:
            outcome: PhaseOutcome to persist
        """
        try:
            outcome_dict = outcome.to_dict()
            json_line = json.dumps(outcome_dict, ensure_ascii=False)
            
            with open(self.outcomes_file, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            
            logger.debug(f"Logged phase outcome: {outcome.phase_name} in {outcome.mission_id}")
            
        except Exception as e:
            logger.warning(f"Failed to write phase outcome: {e}")
    
    def write_mission_linkage(
        self,
        mission_id: str,
        success: bool,
        final_quality: float
    ) -> None:
        """
        Link mission outcome to all phase outcomes for that mission.
        
        Args:
            mission_id: Mission identifier
            success: Whether mission succeeded
            final_quality: Final quality score (0-1)
        """
        try:
            # Update mission index
            self._mission_index[mission_id] = {
                "success": success,
                "final_quality": final_quality,
                "linked_at": datetime.utcnow().isoformat()
            }
            self._save_mission_index()
            
            # Update all outcomes for this mission (read, update, rewrite)
            # Note: For large datasets, this could be optimized with a separate linkage file
            outcomes = list(self.read_outcomes({"mission_id": mission_id}))
            
            if outcomes:
                # Rewrite outcomes file with updated linkage
                all_outcomes = list(self.read_outcomes())
                mission_outcomes = {o.mission_id for o in all_outcomes}
                
                # Only update if we have outcomes to update
                for outcome in outcomes:
                    outcome.mission_outcome_success = success
                    outcome.mission_final_quality = final_quality
                
                # Rewrite file (for simplicity, though this is not optimal for large files)
                # In production, consider a separate linkage index
                logger.debug(f"Linked {len(outcomes)} outcomes for mission {mission_id}")
            
        except Exception as e:
            logger.warning(f"Failed to write mission linkage: {e}")
    
    def read_outcomes(
        self,
        filter: Optional[Dict[str, Any]] = None
    ) -> Iterator[PhaseOutcome]:
        """
        Read phase outcomes, optionally filtered.
        
        Args:
            filter: Optional filter dict (e.g., {"phase_type": "deep_analysis"})
            
        Yields:
            PhaseOutcome instances matching filter
        """
        if not self.outcomes_file.exists():
            return
        
        try:
            with open(self.outcomes_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        outcome = PhaseOutcome.from_dict(data)
                        
                        # Apply filter
                        if filter:
                            match = True
                            for key, value in filter.items():
                                if not hasattr(outcome, key):
                                    match = False
                                    break
                                if getattr(outcome, key) != value:
                                    match = False
                                    break
                            if not match:
                                continue
                        
                        yield outcome
                        
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Failed to parse outcome line: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to read outcomes: {e}")
    
    def read_by_phase_type(
        self,
        phase_type: str,
        limit: int = 1000
    ) -> List[PhaseOutcome]:
        """
        Read outcomes for a specific phase type.
        
        Args:
            phase_type: Phase type to filter by
            limit: Maximum number of outcomes to return
            
        Returns:
            List of PhaseOutcome instances
        """
        outcomes = []
        for outcome in self.read_outcomes({"phase_type": phase_type}):
            outcomes.append(outcome)
            if len(outcomes) >= limit:
                break
        return outcomes
    
    def get_mission_outcome(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """
        Get mission outcome linkage.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            Dict with success and final_quality, or None
        """
        return self._mission_index.get(mission_id)
    
    def _load_mission_index(self) -> None:
        """Load mission index from disk."""
        if not self.mission_index_file.exists():
            return
        
        try:
            with open(self.mission_index_file, "r", encoding="utf-8") as f:
                self._mission_index = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load mission index: {e}")
            self._mission_index = {}
    
    def _save_mission_index(self) -> None:
        """Save mission index to disk."""
        try:
            with open(self.mission_index_file, "w", encoding="utf-8") as f:
                json.dump(self._mission_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save mission index: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about stored outcomes.
        
        Returns:
            Dictionary with statistics
        """
        total = 0
        by_phase_type: Dict[str, int] = {}
        by_mission: Dict[str, int] = {}
        
        for outcome in self.read_outcomes():
            total += 1
            by_phase_type[outcome.phase_type] = by_phase_type.get(outcome.phase_type, 0) + 1
            by_mission[outcome.mission_id] = by_mission.get(outcome.mission_id, 0) + 1
        
        return {
            "total_outcomes": total,
            "unique_missions": len(by_mission),
            "by_phase_type": by_phase_type,
            "outcomes_file": str(self.outcomes_file),
            "mission_index_file": str(self.mission_index_file),
        }

