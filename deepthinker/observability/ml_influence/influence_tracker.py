"""
ML Influence Tracker.

Thread-safe, append-only event logging for ML predictor influence tracking.
Central entry point used by all predictors to record influence events.

Safety Guarantees:
- Thread-safe via threading.Lock
- Never throws exceptions (all errors logged and swallowed)
- Minimal overhead (synchronous append, no blocking I/O waits)
- System continues functioning if tracker fails
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .schemas import MLInfluenceEvent
from .config import (
    INFLUENCE_MONITORING_CONFIG,
    get_events_path,
    is_monitoring_enabled,
)

logger = logging.getLogger(__name__)

# Module-level singleton
_tracker_instance: Optional["MLInfluenceTracker"] = None
_tracker_lock = threading.Lock()


class MLInfluenceTracker:
    """
    Central tracker for ML influence events.
    
    Thread-safe singleton that handles event recording from all predictors.
    Uses append-only JSONL format for durability and easy analysis.
    
    Usage:
        tracker = get_influence_tracker()
        tracker.record_event(event)
    
    Or using the convenience method:
        tracker.record(
            mission_id="...",
            phase_name="...",
            predictor_name="cost_time",
            ...
        )
    """
    
    def __init__(self, events_path: Optional[Path] = None):
        """
        Initialize the tracker.
        
        Args:
            events_path: Path to events log file.
                        Defaults to kb/observability/ml_influence_events.jsonl
        """
        self._events_path = events_path or get_events_path()
        self._lock = threading.Lock()
        self._event_count = 0
        self._error_count = 0
        self._initialized = False
        
        # Ensure directory exists
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure the events directory exists."""
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialized = True
        except Exception as e:
            logger.warning(f"[MLInfluenceTracker] Failed to create events directory: {e}")
            self._initialized = False
    
    def record_event(self, event: MLInfluenceEvent) -> bool:
        """
        Record an influence event to the log.
        
        This is the primary method for recording events. It is thread-safe
        and never throws exceptions.
        
        Args:
            event: MLInfluenceEvent to record
            
        Returns:
            True if event was recorded successfully, False otherwise
        """
        # Check if monitoring is enabled
        if not is_monitoring_enabled():
            return False
        
        if not self._initialized:
            return False
        
        try:
            with self._lock:
                # Convert event to JSON
                event_dict = event.to_dict()
                json_line = json.dumps(event_dict, ensure_ascii=False)
                
                # Append to log file
                with open(self._events_path, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
                
                self._event_count += 1
                
                logger.debug(
                    f"[MLInfluenceTracker] Recorded event: "
                    f"{event.predictor_name} @ {event.phase_name}"
                )
                return True
                
        except Exception as e:
            self._error_count += 1
            logger.debug(f"[MLInfluenceTracker] Failed to record event: {e}")
            return False
    
    def record(
        self,
        mission_id: str,
        phase_name: str,
        phase_type: str,
        predictor_name: str,
        predictor_version: str,
        prediction_summary: dict,
        confidence: float,
        used_fallback: bool,
        predictor_mode: str = "shadow",
        planner_original_decision: Optional[dict] = None,
        planner_final_decision: Optional[dict] = None,
    ) -> bool:
        """
        Convenience method to record an event with individual parameters.
        
        Creates an MLInfluenceEvent and records it.
        
        Args:
            mission_id: ID of the current mission
            phase_name: Name of the phase
            phase_type: Type of phase (research, synthesis, etc.)
            predictor_name: Name of the predictor (cost_time, phase_risk, web_search)
            predictor_version: Version string of the predictor model
            prediction_summary: Dictionary of key predictions
            confidence: Prediction confidence (0-1)
            used_fallback: Whether fallback rules were used
            predictor_mode: Operating mode (shadow, advisory, active)
            planner_original_decision: Decision before prediction (if available)
            planner_final_decision: Decision after prediction (if available)
            
        Returns:
            True if recorded successfully
        """
        try:
            event = MLInfluenceEvent.create(
                mission_id=mission_id,
                phase_name=phase_name,
                phase_type=phase_type,
                predictor_name=predictor_name,
                predictor_version=predictor_version,
                prediction_summary=prediction_summary,
                confidence=confidence,
                used_fallback=used_fallback,
                predictor_mode=predictor_mode,
                planner_original_decision=planner_original_decision,
                planner_final_decision=planner_final_decision,
            )
            return self.record_event(event)
        except Exception as e:
            logger.debug(f"[MLInfluenceTracker] Failed to create event: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary with event_count, error_count, and path info
        """
        return {
            "initialized": self._initialized,
            "event_count": self._event_count,
            "error_count": self._error_count,
            "events_path": str(self._events_path),
            "monitoring_enabled": is_monitoring_enabled(),
        }
    
    def is_healthy(self) -> bool:
        """
        Check if the tracker is functioning properly.
        
        Returns:
            True if tracker is initialized and has low error rate
        """
        if not self._initialized:
            return False
        
        if self._event_count == 0:
            return True  # No events yet, assumed healthy
        
        error_rate = self._error_count / max(1, self._event_count + self._error_count)
        return error_rate < 0.1  # Less than 10% error rate


def get_influence_tracker() -> MLInfluenceTracker:
    """
    Get the singleton MLInfluenceTracker instance.
    
    Thread-safe singleton accessor for the global tracker.
    
    Returns:
        The singleton MLInfluenceTracker instance
    """
    global _tracker_instance
    
    if _tracker_instance is None:
        with _tracker_lock:
            # Double-check locking
            if _tracker_instance is None:
                _tracker_instance = MLInfluenceTracker()
    
    return _tracker_instance


def reset_tracker() -> None:
    """
    Reset the singleton tracker (for testing only).
    
    Warning: This should only be used in tests.
    """
    global _tracker_instance
    with _tracker_lock:
        _tracker_instance = None


