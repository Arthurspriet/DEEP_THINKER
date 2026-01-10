"""
Review Queue for DeepThinker.

Async queue for decision review that NEVER blocks the mission loop.

Decisions are queued post-emit and reviewed asynchronously.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ReviewConfig, get_review_config

logger = logging.getLogger(__name__)


@dataclass
class QueuedDecision:
    """
    A decision queued for review.
    
    Attributes:
        queue_id: Unique queue identifier
        decision_id: Original decision ID
        decision_type: Type of decision
        mission_id: Mission identifier
        phase_id: Phase identifier
        decision_data: Full decision record
        queued_at: When queued
        status: pending | processing | completed | dropped
        priority: Higher = more urgent
    """
    queue_id: str
    decision_id: str
    decision_type: str
    mission_id: str
    phase_id: str
    decision_data: Dict[str, Any]
    queued_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "decision_data": self.decision_data,
            "queued_at": self.queued_at.isoformat(),
            "status": self.status,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedDecision":
        queued_at = data.get("queued_at")
        if isinstance(queued_at, str):
            queued_at = datetime.fromisoformat(queued_at)
        elif queued_at is None:
            queued_at = datetime.utcnow()
        
        return cls(
            queue_id=data.get("queue_id", ""),
            decision_id=data.get("decision_id", ""),
            decision_type=data.get("decision_type", ""),
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            decision_data=data.get("decision_data", {}),
            queued_at=queued_at,
            status=data.get("status", "pending"),
            priority=data.get("priority", 0),
        )


class ReviewQueue:
    """
    Async queue for decision review.
    
    CRITICAL: Never blocks mission loop.
    Decisions are queued post-emit and reviewed asynchronously.
    
    Design:
    - File-based queue for simplicity and durability
    - One JSON file per queued decision
    - Non-blocking enqueue (fire and forget)
    - Batch dequeue for async worker
    
    Usage:
        queue = ReviewQueue()
        
        # Non-blocking enqueue (in mission loop)
        queue_id = queue.enqueue(decision_record)
        
        # Batch dequeue (in async worker)
        decisions = queue.dequeue_batch(50)
        for decision in decisions:
            result = reviewer.review(decision)
            queue.mark_completed(decision.queue_id, result)
    """
    
    def __init__(self, config: Optional[ReviewConfig] = None):
        """
        Initialize the queue.
        
        Args:
            config: Optional ReviewConfig. Uses global if None.
        """
        self.config = config or get_review_config()
        self.queue_path = Path(self.config.queue_path)
        self._ensure_dirs()
    
    def _ensure_dirs(self) -> bool:
        """Ensure queue directories exist."""
        try:
            self.queue_path.mkdir(parents=True, exist_ok=True)
            (self.queue_path / "processing").mkdir(exist_ok=True)
            (self.queue_path / "completed").mkdir(exist_ok=True)
            return True
        except Exception as e:
            logger.warning(f"[REVIEW_QUEUE] Failed to create dirs: {e}")
            return False
    
    def enqueue(self, decision: Dict[str, Any]) -> Optional[str]:
        """
        Add decision to review queue.
        
        Non-blocking, returns queue_id immediately.
        
        Args:
            decision: Decision record dictionary
            
        Returns:
            Queue ID or None if enqueue failed
        """
        if not self.config.enabled:
            return None
        
        # Check queue size
        current_size = len(list(self.queue_path.glob("*.json")))
        if current_size >= self.config.max_queue_size:
            logger.warning(
                f"[REVIEW_QUEUE] Queue full ({current_size}), dropping decision"
            )
            return None
        
        try:
            # Generate queue ID
            decision_id = decision.get("decision_id", "unknown")
            timestamp = datetime.utcnow()
            queue_id = f"{decision_id}_{timestamp.timestamp():.0f}"
            
            # Determine priority based on decision type
            decision_type = decision.get("decision_type", "")
            priority = self._compute_priority(decision_type)
            
            # Create queued decision
            queued = QueuedDecision(
                queue_id=queue_id,
                decision_id=decision_id,
                decision_type=decision_type,
                mission_id=decision.get("mission_id", ""),
                phase_id=decision.get("phase_id", ""),
                decision_data=decision,
                queued_at=timestamp,
                priority=priority,
            )
            
            # Write to queue (non-blocking, fire and forget)
            queue_file = self.queue_path / f"{queue_id}.json"
            with open(queue_file, "w") as f:
                json.dump(queued.to_dict(), f)
            
            logger.debug(
                f"[REVIEW_QUEUE] Enqueued: {queue_id} "
                f"(type={decision_type}, priority={priority})"
            )
            
            return queue_id
            
        except Exception as e:
            # Non-blocking: log and return None
            logger.warning(f"[REVIEW_QUEUE] Enqueue failed: {e}")
            return None
    
    def _compute_priority(self, decision_type: str) -> int:
        """Compute priority based on decision type."""
        # Higher priority for more impactful decisions
        priority_map = {
            "final_synthesis": 10,
            "mission_completion": 10,
            "alignment_action": 8,
            "model_selection": 5,
            "council_selection": 5,
            "phase_termination": 4,
            "tool_selection": 2,
        }
        return priority_map.get(decision_type, 1)
    
    def dequeue_batch(self, batch_size: Optional[int] = None) -> List[QueuedDecision]:
        """
        Get batch of pending reviews (for async worker).
        
        Args:
            batch_size: Maximum decisions to return
            
        Returns:
            List of QueuedDecision objects
        """
        if not self.config.enabled:
            return []
        
        batch_size = batch_size or self.config.batch_review_size
        
        try:
            # Get all pending files
            pending_files = list(self.queue_path.glob("*.json"))
            
            # Load and sort by priority (descending) then timestamp
            decisions = []
            for file_path in pending_files:
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    decisions.append(QueuedDecision.from_dict(data))
                except Exception as e:
                    logger.warning(f"[REVIEW_QUEUE] Failed to load {file_path}: {e}")
            
            # Sort by priority (descending) then timestamp (ascending)
            decisions.sort(key=lambda d: (-d.priority, d.queued_at))
            
            # Take batch
            batch = decisions[:batch_size]
            
            # Move to processing
            for decision in batch:
                self._move_to_processing(decision.queue_id)
                decision.status = "processing"
            
            return batch
            
        except Exception as e:
            logger.warning(f"[REVIEW_QUEUE] Dequeue failed: {e}")
            return []
    
    def _move_to_processing(self, queue_id: str) -> bool:
        """Move decision from pending to processing."""
        try:
            src = self.queue_path / f"{queue_id}.json"
            dst = self.queue_path / "processing" / f"{queue_id}.json"
            
            if src.exists():
                src.rename(dst)
                return True
            return False
            
        except Exception as e:
            logger.warning(f"[REVIEW_QUEUE] Move to processing failed: {e}")
            return False
    
    def mark_completed(
        self,
        queue_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark decision as reviewed/completed.
        
        Args:
            queue_id: Queue identifier
            result: Review result to append
            
        Returns:
            True if marked successfully
        """
        try:
            processing_file = self.queue_path / "processing" / f"{queue_id}.json"
            completed_file = self.queue_path / "completed" / f"{queue_id}.json"
            
            if processing_file.exists():
                # Load and update
                with open(processing_file, "r") as f:
                    data = json.load(f)
                
                data["status"] = "completed"
                data["completed_at"] = datetime.utcnow().isoformat()
                if result:
                    data["review_result"] = result
                
                # Write to completed
                with open(completed_file, "w") as f:
                    json.dump(data, f)
                
                # Remove from processing
                processing_file.unlink()
                
                return True
            return False
            
        except Exception as e:
            logger.warning(f"[REVIEW_QUEUE] Mark completed failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            pending = len(list(self.queue_path.glob("*.json")))
            processing = len(list((self.queue_path / "processing").glob("*.json")))
            completed = len(list((self.queue_path / "completed").glob("*.json")))
            
            return {
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "max_size": self.config.max_queue_size,
                "batch_size": self.config.batch_review_size,
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def peek_pending(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Peek at pending decisions without dequeuing."""
        try:
            pending_files = list(self.queue_path.glob("*.json"))[:limit]
            
            summaries = []
            for file_path in pending_files:
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    summaries.append({
                        "queue_id": data.get("queue_id"),
                        "decision_type": data.get("decision_type"),
                        "priority": data.get("priority"),
                        "queued_at": data.get("queued_at"),
                    })
                except Exception:
                    continue
            
            return summaries
            
        except Exception as e:
            return []
    
    def clear_completed(self, older_than_hours: int = 24) -> int:
        """
        Clear old completed reviews.
        
        Args:
            older_than_hours: Remove reviews older than this
            
        Returns:
            Number of files removed
        """
        try:
            completed_dir = self.queue_path / "completed"
            cutoff = datetime.utcnow().timestamp() - (older_than_hours * 3600)
            
            removed = 0
            for file_path in completed_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff:
                    file_path.unlink()
                    removed += 1
            
            return removed
            
        except Exception as e:
            logger.warning(f"[REVIEW_QUEUE] Clear completed failed: {e}")
            return 0


# Global queue instance
_queue: Optional[ReviewQueue] = None


def get_review_queue(config: Optional[ReviewConfig] = None) -> ReviewQueue:
    """Get global review queue instance."""
    global _queue
    if _queue is None:
        _queue = ReviewQueue(config=config)
    return _queue


def reset_review_queue() -> None:
    """Reset global queue (mainly for testing)."""
    global _queue
    _queue = None


