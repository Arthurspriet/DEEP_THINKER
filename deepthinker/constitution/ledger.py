"""
Constitution Ledger for DeepThinker.

Provides append-only JSONL storage for constitution events.
One file per mission: kb/constitution/{mission_id}.jsonl

Key properties:
- Append-only: Never modifies existing records
- Human-readable: Plain JSONL format
- Mission-scoped: Each mission has its own ledger
- Thread-safe: Uses file locking for writes
- Privacy-aware: Sensitive data is hashed

Follows the pattern established in DecisionStore.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .config import ConstitutionConfig, get_constitution_config
from .types import (
    BaseConstitutionEvent,
    ConstitutionEventType,
    EvidenceEvent,
    ScoreEvent,
    ContradictionEvent,
    DepthEvent,
    MemoryEvent,
    CompressionEvent,
    LearningUpdateEvent,
    ConstitutionViolationEvent,
    BaselineSnapshot,
)

logger = logging.getLogger(__name__)


class ConstitutionLedger:
    """
    Append-only ledger for constitution events.
    
    Storage structure:
        kb/constitution/{mission_id}.jsonl  - All constitution events
    
    Usage:
        ledger = ConstitutionLedger(mission_id="abc-123")
        
        ledger.write_event(EvidenceEvent(
            mission_id="abc-123",
            phase_id="research",
            count_added=3,
        ))
        
        events = ledger.read_all()
    """
    
    def __init__(
        self,
        mission_id: str,
        config: Optional[ConstitutionConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize the ledger for a mission.
        
        Args:
            mission_id: Mission identifier
            config: Optional ConstitutionConfig
            base_dir: Optional base directory override
        """
        self.mission_id = mission_id
        self.config = config or get_constitution_config()
        
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path(self.config.ledger_base_dir)
        
        self._lock = threading.Lock()
        self._initialized = False
        self._ensure_dir()
    
    @property
    def ledger_path(self) -> Path:
        """Get path to the mission's ledger file."""
        return self.base_dir / f"{self.mission_id}.jsonl"
    
    def _ensure_dir(self) -> bool:
        """Ensure the ledger directory exists."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"[CONSTITUTION] Failed to create ledger directory: {e}")
            self._initialized = False
            return False
    
    def write_event(self, event: BaseConstitutionEvent) -> bool:
        """
        Write an event to the ledger.
        
        Args:
            event: Constitution event to write
            
        Returns:
            True if write succeeded
        """
        if not self.config.ledger_enabled:
            return False
        
        if not self._initialized:
            if not self._ensure_dir():
                return False
        
        try:
            with self._lock:
                event_dict = event.to_dict()
                json_line = json.dumps(event_dict, ensure_ascii=False)
                
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
                
                logger.debug(
                    f"[CONSTITUTION] Wrote {event.event_type.value} event "
                    f"for phase {event.phase_id}"
                )
                return True
                
        except Exception as e:
            logger.warning(f"[CONSTITUTION] Failed to write event: {e}")
            return False
    
    def write_baseline(self, baseline: BaselineSnapshot) -> bool:
        """
        Write a baseline snapshot to the ledger.
        
        Args:
            baseline: Baseline snapshot to write
            
        Returns:
            True if write succeeded
        """
        if not self.config.ledger_enabled:
            return False
        
        try:
            with self._lock:
                event_dict = {
                    "event_type": ConstitutionEventType.BASELINE_SNAPSHOT.value,
                    **baseline.to_dict(),
                }
                json_line = json.dumps(event_dict, ensure_ascii=False)
                
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
                
                logger.debug(
                    f"[CONSTITUTION] Wrote baseline snapshot for phase {baseline.phase_id}"
                )
                return True
                
        except Exception as e:
            logger.warning(f"[CONSTITUTION] Failed to write baseline: {e}")
            return False
    
    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all events from the ledger.
        
        Returns:
            List of event dictionaries
        """
        events = []
        
        if not self.ledger_path.exists():
            return events
        
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"[CONSTITUTION] Invalid JSON line: {e}")
                            continue
        except Exception as e:
            logger.warning(f"[CONSTITUTION] Failed to read ledger: {e}")
        
        return events
    
    def read_events(
        self,
        event_type: Optional[ConstitutionEventType] = None,
        phase_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Read events with optional filtering.
        
        Args:
            event_type: Filter by event type
            phase_id: Filter by phase ID
            limit: Maximum number of events to return
            
        Yields:
            Event dictionaries matching filters
        """
        count = 0
        
        if not self.ledger_path.exists():
            return
        
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    if limit and count >= limit:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Apply filters
                    if event_type and event.get("event_type") != event_type.value:
                        continue
                    if phase_id and event.get("phase_id") != phase_id:
                        continue
                    
                    yield event
                    count += 1
                    
        except Exception as e:
            logger.warning(f"[CONSTITUTION] Failed to read events: {e}")
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get all violation events."""
        return list(self.read_events(event_type=ConstitutionEventType.VIOLATION))
    
    def get_baselines(self) -> List[BaselineSnapshot]:
        """Get all baseline snapshots."""
        baselines = []
        for event in self.read_events(event_type=ConstitutionEventType.BASELINE_SNAPSHOT):
            try:
                baselines.append(BaselineSnapshot.from_dict(event))
            except Exception:
                continue
        return baselines
    
    def get_latest_baseline(self, phase_id: str) -> Optional[BaselineSnapshot]:
        """Get the latest baseline for a phase."""
        latest = None
        for event in self.read_events(
            event_type=ConstitutionEventType.BASELINE_SNAPSHOT,
            phase_id=phase_id,
        ):
            try:
                baseline = BaselineSnapshot.from_dict(event)
                if latest is None or baseline.timestamp > latest.timestamp:
                    latest = baseline
            except Exception:
                continue
        return latest
    
    def count_events(
        self,
        event_type: Optional[ConstitutionEventType] = None,
    ) -> int:
        """Count events, optionally filtered by type."""
        count = 0
        for _ in self.read_events(event_type=event_type):
            count += 1
        return count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the ledger contents."""
        events = self.read_all()
        
        summary = {
            "mission_id": self.mission_id,
            "total_events": len(events),
            "event_counts": {},
            "violation_count": 0,
            "phases_covered": set(),
        }
        
        for event in events:
            event_type = event.get("event_type", "unknown")
            summary["event_counts"][event_type] = summary["event_counts"].get(event_type, 0) + 1
            
            if event_type == ConstitutionEventType.VIOLATION.value:
                summary["violation_count"] += 1
            
            phase_id = event.get("phase_id")
            if phase_id:
                summary["phases_covered"].add(phase_id)
        
        summary["phases_covered"] = list(summary["phases_covered"])
        
        return summary


# Global ledger cache
_ledgers: Dict[str, ConstitutionLedger] = {}
_ledgers_lock = threading.Lock()


def get_ledger(
    mission_id: str,
    config: Optional[ConstitutionConfig] = None,
) -> ConstitutionLedger:
    """
    Get or create a ledger for a mission.
    
    Args:
        mission_id: Mission identifier
        config: Optional configuration
        
    Returns:
        ConstitutionLedger instance
    """
    global _ledgers
    
    with _ledgers_lock:
        if mission_id not in _ledgers:
            _ledgers[mission_id] = ConstitutionLedger(
                mission_id=mission_id,
                config=config,
            )
        return _ledgers[mission_id]


def clear_ledger_cache() -> None:
    """Clear the global ledger cache (for testing)."""
    global _ledgers
    with _ledgers_lock:
        _ledgers.clear()


