"""
Alignment Control Layer - Persistence.

Handles JSON logging and persistence of alignment data.
Writes to kb/alignment_logs/{mission_id}.json

Also provides standard logging integration for [ALIGNMENT] log lines.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AlignmentConfig, get_alignment_config
from .models import (
    AlignmentAction,
    AlignmentAssessment,
    AlignmentPoint,
    AlignmentTrajectory,
    ControllerState,
    NorthStarGoal,
)

logger = logging.getLogger(__name__)

# Default log directory
DEFAULT_LOG_DIR = "kb/alignment_logs"


class AlignmentLogStore:
    """
    Persistent storage for alignment logs.
    
    Writes JSON files per mission containing:
    - Configuration used
    - NorthStarGoal
    - AlignmentTrajectory (all points)
    - AlignmentAssessments
    - Controller actions log
    - Summary statistics
    
    Usage:
        store = AlignmentLogStore(config)
        
        # Update throughout mission
        store.update(mission_id, trajectory, controller_state)
        
        # Save at end
        store.save(mission_id)
    """
    
    def __init__(
        self,
        config: Optional[AlignmentConfig] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the log store.
        
        Args:
            config: Alignment configuration
            log_dir: Directory for logs (default: kb/alignment_logs)
        """
        self.config = config or get_alignment_config()
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        
        # In-memory cache of log data per mission
        self._logs: Dict[str, Dict[str, Any]] = {}
    
    def _ensure_dir(self) -> bool:
        """Ensure log directory exists."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.warning(f"[ALIGNMENT] Failed to create log dir: {e}")
            return False
    
    def _get_log_path(self, mission_id: str) -> Path:
        """Get log file path for a mission."""
        # Sanitize mission_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in mission_id)
        return self.log_dir / f"{safe_id}.json"
    
    def initialize(
        self,
        mission_id: str,
        north_star: NorthStarGoal,
    ) -> None:
        """
        Initialize log for a new mission.
        
        Args:
            mission_id: Mission ID
            north_star: The north star goal
        """
        self._logs[mission_id] = {
            "mission_id": mission_id,
            "config": self.config.to_dict(),
            "north_star": north_star.to_dict(),
            "trajectory": [],
            "assessments": [],
            "actions": [],
            "summary": {
                "total_points": 0,
                "total_triggers": 0,
                "total_actions": 0,
                "final_alignment": None,
                "max_cusum": 0.0,
                "min_similarity": 1.0,
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        logger.debug(f"[ALIGNMENT] Initialized log for mission {mission_id}")
    
    def add_point(
        self,
        mission_id: str,
        point: AlignmentPoint,
    ) -> None:
        """
        Add an alignment point to the log.
        
        Args:
            mission_id: Mission ID
            point: AlignmentPoint to add
        """
        if mission_id not in self._logs:
            logger.warning(f"[ALIGNMENT] Log not initialized for {mission_id}")
            return
        
        log = self._logs[mission_id]
        log["trajectory"].append(point.to_dict())
        log["updated_at"] = datetime.utcnow().isoformat()
        
        # Update summary
        log["summary"]["total_points"] += 1
        if point.triggered:
            log["summary"]["total_triggers"] += 1
        log["summary"]["final_alignment"] = point.a_t
        log["summary"]["max_cusum"] = max(log["summary"]["max_cusum"], point.cusum_neg)
        log["summary"]["min_similarity"] = min(log["summary"]["min_similarity"], point.a_t)
    
    def add_assessment(
        self,
        mission_id: str,
        assessment: AlignmentAssessment,
    ) -> None:
        """
        Add an LLM assessment to the log.
        
        Args:
            mission_id: Mission ID
            assessment: AlignmentAssessment to add
        """
        if mission_id not in self._logs:
            logger.warning(f"[ALIGNMENT] Log not initialized for {mission_id}")
            return
        
        log = self._logs[mission_id]
        log["assessments"].append(assessment.to_dict())
        log["updated_at"] = datetime.utcnow().isoformat()
    
    def add_action(
        self,
        mission_id: str,
        action: AlignmentAction,
        point: AlignmentPoint,
        metadata: Optional[Dict[str, Any]] = None,
        injected_prompt: Optional[str] = None,
    ) -> None:
        """
        Add an action to the log.
        
        Args:
            mission_id: Mission ID
            action: AlignmentAction taken
            point: Point at which action was taken
            metadata: Optional additional metadata
            injected_prompt: Optional prompt that was injected (for audit)
        """
        if mission_id not in self._logs:
            logger.warning(f"[ALIGNMENT] Log not initialized for {mission_id}")
            return
        
        log = self._logs[mission_id]
        action_record = {
            "action": action.value,
            "t": point.t,
            "phase": point.phase_name,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "a_t": point.a_t,
                "d_t": point.d_t,
                "cusum_neg": point.cusum_neg,
            },
            "metadata": metadata or {},
        }
        
        # Add prompt injection audit trail
        if injected_prompt is not None:
            import hashlib
            action_record["prompt_injection"] = {
                "prompt_hash": hashlib.sha256(injected_prompt.encode()).hexdigest()[:16],
                "prompt_length": len(injected_prompt),
            }
            # Optionally include full prompt for debugging
            if self.config.log_full_prompts:
                action_record["prompt_injection"]["prompt_full"] = injected_prompt
        
        log["actions"].append(action_record)
        log["summary"]["total_actions"] += 1
        log["updated_at"] = datetime.utcnow().isoformat()
    
    def update_from_trajectory(
        self,
        mission_id: str,
        trajectory: AlignmentTrajectory,
        controller_state: Optional[ControllerState] = None,
    ) -> None:
        """
        Update log from a full trajectory.
        
        Convenience method to sync log with trajectory state.
        
        Args:
            mission_id: Mission ID
            trajectory: Full trajectory
            controller_state: Optional controller state
        """
        if mission_id not in self._logs:
            self.initialize(mission_id, trajectory.north_star)
        
        log = self._logs[mission_id]
        
        # Update trajectory
        log["trajectory"] = [p.to_dict() for p in trajectory.points]
        log["assessments"] = [a.to_dict() for a in trajectory.assessments]
        log["actions"] = trajectory.actions_taken
        
        # Update summary
        if trajectory.points:
            log["summary"]["total_points"] = len(trajectory.points)
            log["summary"]["total_triggers"] = trajectory.get_trigger_count()
            log["summary"]["final_alignment"] = trajectory.points[-1].a_t
            log["summary"]["max_cusum"] = max(p.cusum_neg for p in trajectory.points)
            log["summary"]["min_similarity"] = min(p.a_t for p in trajectory.points)
        
        log["summary"]["total_actions"] = len(trajectory.actions_taken)
        
        # Add controller state
        if controller_state:
            log["controller_state"] = controller_state.to_dict()
        
        log["updated_at"] = datetime.utcnow().isoformat()
    
    def save(self, mission_id: str) -> bool:
        """
        Save log to disk.
        
        Args:
            mission_id: Mission ID
            
        Returns:
            True if saved successfully
        """
        if not self.config.persist_logs:
            logger.debug("[ALIGNMENT] Persistence disabled, skipping save")
            return True
        
        if mission_id not in self._logs:
            logger.warning(f"[ALIGNMENT] No log data for {mission_id}")
            return False
        
        if not self._ensure_dir():
            return False
        
        try:
            log_path = self._get_log_path(mission_id)
            log_data = self._logs[mission_id]
            
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
            
            logger.info(f"[ALIGNMENT] Saved log to {log_path}")
            return True
            
        except Exception as e:
            logger.warning(f"[ALIGNMENT] Failed to save log: {e}")
            return False
    
    def load(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """
        Load log from disk.
        
        Args:
            mission_id: Mission ID
            
        Returns:
            Log data or None if not found
        """
        log_path = self._get_log_path(mission_id)
        
        if not log_path.exists():
            return None
        
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
            
            self._logs[mission_id] = data
            return data
            
        except Exception as e:
            logger.warning(f"[ALIGNMENT] Failed to load log: {e}")
            return None
    
    def get_log(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Get in-memory log for a mission."""
        return self._logs.get(mission_id)
    
    def clear(self, mission_id: str) -> None:
        """Clear in-memory log for a mission."""
        self._logs.pop(mission_id, None)


# Global log store instance (lazy-loaded)
_store: Optional[AlignmentLogStore] = None


def get_log_store(
    config: Optional[AlignmentConfig] = None,
    log_dir: Optional[str] = None,
) -> AlignmentLogStore:
    """
    Get the global log store instance.
    
    Args:
        config: Optional configuration override
        log_dir: Optional log directory override
        
    Returns:
        AlignmentLogStore instance
    """
    global _store
    
    if _store is None:
        _store = AlignmentLogStore(config, log_dir)
    
    return _store


def log_alignment_event(
    mission_id: str,
    event_type: str,
    data: Dict[str, Any],
    level: int = logging.INFO,
) -> None:
    """
    Log an alignment event with standard format.
    
    All alignment events are prefixed with [ALIGNMENT] for easy filtering.
    
    Args:
        mission_id: Mission ID
        event_type: Type of event (trigger, action, assessment, etc.)
        data: Event data
        level: Logging level
    """
    # Build log message
    data_str = ", ".join(f"{k}={v}" for k, v in data.items())
    message = f"[ALIGNMENT] mission={mission_id} event={event_type} {data_str}"
    
    logger.log(level, message)

