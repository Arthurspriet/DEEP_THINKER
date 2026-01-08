"""
Decision Reviewer for DeepThinker.

Reviews queued decisions asynchronously.

Supports:
- AI reviewer (cheap model for fast review)
- Human-in-the-loop (API hooks for future integration)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ReviewConfig, get_review_config
from .review_queue import QueuedDecision, ReviewQueue, get_review_queue

logger = logging.getLogger(__name__)


class ReviewOutcome(str, Enum):
    """Possible review outcomes."""
    APPROVE = "approve"      # Decision is valid
    FLAG = "flag"            # Decision needs attention
    OVERRIDE = "override"    # Decision should be changed
    FEEDBACK = "feedback"    # Add feedback note


@dataclass
class ReviewResult:
    """
    Result of a decision review.
    
    Attributes:
        queue_id: Queue identifier
        decision_id: Original decision ID
        outcome: Review outcome
        reviewer: Who/what reviewed (ai_cheap, ai_strong, human)
        feedback_note: Optional feedback
        override_action: If overriding, what action to take
        confidence: Confidence in review
        reviewed_at: When review completed
    """
    queue_id: str
    decision_id: str
    outcome: ReviewOutcome
    reviewer: str = "ai_cheap"
    feedback_note: Optional[str] = None
    override_action: Optional[str] = None
    confidence: float = 0.8
    reviewed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "decision_id": self.decision_id,
            "outcome": self.outcome.value,
            "reviewer": self.reviewer,
            "feedback_note": self.feedback_note,
            "override_action": self.override_action,
            "confidence": self.confidence,
            "reviewed_at": self.reviewed_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewResult":
        reviewed_at = data.get("reviewed_at")
        if isinstance(reviewed_at, str):
            reviewed_at = datetime.fromisoformat(reviewed_at)
        elif reviewed_at is None:
            reviewed_at = datetime.utcnow()
        
        return cls(
            queue_id=data.get("queue_id", ""),
            decision_id=data.get("decision_id", ""),
            outcome=ReviewOutcome(data.get("outcome", "approve")),
            reviewer=data.get("reviewer", "ai_cheap"),
            feedback_note=data.get("feedback_note"),
            override_action=data.get("override_action"),
            confidence=data.get("confidence", 0.8),
            reviewed_at=reviewed_at,
        )


class DecisionReviewer:
    """
    Reviews queued decisions asynchronously.
    
    Current implementation uses rule-based review.
    Future: integrate with cheap model for AI review.
    
    Usage:
        reviewer = DecisionReviewer()
        
        # Review a single decision
        result = reviewer.review(queued_decision)
        
        # Process batch from queue
        results = reviewer.process_batch()
    """
    
    def __init__(
        self,
        config: Optional[ReviewConfig] = None,
        queue: Optional[ReviewQueue] = None,
    ):
        """
        Initialize the reviewer.
        
        Args:
            config: Optional ReviewConfig
            queue: Optional ReviewQueue
        """
        self.config = config or get_review_config()
        self.queue = queue or get_review_queue(config=self.config)
        self.store_path = Path(self.config.store_path)
    
    def review(self, decision: QueuedDecision) -> ReviewResult:
        """
        Review a single decision.
        
        Args:
            decision: QueuedDecision to review
            
        Returns:
            ReviewResult
        """
        # Currently: rule-based review
        # Future: call cheap model for AI review
        
        decision_type = decision.decision_type
        decision_data = decision.decision_data
        
        # Apply review rules
        outcome, feedback, confidence = self._apply_review_rules(
            decision_type, decision_data
        )
        
        result = ReviewResult(
            queue_id=decision.queue_id,
            decision_id=decision.decision_id,
            outcome=outcome,
            reviewer="rule_based",  # Will be "ai_cheap" when model integrated
            feedback_note=feedback,
            confidence=confidence,
        )
        
        # Store result
        self._store_result(result, decision)
        
        return result
    
    def _apply_review_rules(
        self,
        decision_type: str,
        decision_data: Dict[str, Any],
    ) -> tuple:
        """
        Apply review rules to decision.
        
        Returns:
            (outcome, feedback, confidence)
        """
        # Rule 1: Check for low confidence decisions
        confidence = decision_data.get("confidence", 1.0)
        if confidence < 0.5:
            return (
                ReviewOutcome.FLAG,
                f"Low confidence decision ({confidence:.2f})",
                0.9,
            )
        
        # Rule 2: Check for alignment-related decisions
        if decision_type == "alignment_action":
            action = decision_data.get("action", "")
            # Flag user-triggering actions
            if "user" in action.lower():
                return (
                    ReviewOutcome.FLAG,
                    "User-triggering alignment action should be reviewed",
                    0.85,
                )
        
        # Rule 3: Check for model escalation
        if decision_type == "model_selection":
            selected = decision_data.get("selected_model", "")
            if "large" in selected.lower() or "gpt-4" in selected.lower():
                return (
                    ReviewOutcome.FEEDBACK,
                    "Large model selected - verify necessity",
                    0.8,
                )
        
        # Rule 4: Check for phase termination
        if decision_type == "phase_termination":
            reason = decision_data.get("reason", "")
            score = decision_data.get("score", 1.0)
            if score < 0.6:
                return (
                    ReviewOutcome.FLAG,
                    f"Phase terminated with low score ({score:.2f}): {reason}",
                    0.85,
                )
        
        # Default: approve
        return (ReviewOutcome.APPROVE, None, 0.9)
    
    def _store_result(
        self,
        result: ReviewResult,
        decision: QueuedDecision,
    ) -> bool:
        """Store review result."""
        try:
            # Create mission-scoped directory
            mission_dir = self.store_path / decision.mission_id
            mission_dir.mkdir(parents=True, exist_ok=True)
            
            # Append to reviews file
            reviews_file = mission_dir / "reviews.jsonl"
            with open(reviews_file, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
            
            return True
            
        except Exception as e:
            logger.warning(f"[REVIEWER] Failed to store result: {e}")
            return False
    
    def process_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> List[ReviewResult]:
        """
        Process a batch of decisions from queue.
        
        Args:
            batch_size: Number of decisions to process
            
        Returns:
            List of ReviewResults
        """
        if not self.config.enabled:
            return []
        
        # Dequeue batch
        decisions = self.queue.dequeue_batch(batch_size)
        
        results = []
        for decision in decisions:
            try:
                result = self.review(decision)
                results.append(result)
                
                # Mark completed in queue
                self.queue.mark_completed(
                    decision.queue_id,
                    result.to_dict(),
                )
                
            except Exception as e:
                logger.warning(
                    f"[REVIEWER] Failed to review {decision.queue_id}: {e}"
                )
        
        logger.info(f"[REVIEWER] Processed {len(results)} decisions")
        
        return results
    
    def get_review_stats(self) -> Dict[str, Any]:
        """Get review statistics."""
        try:
            stats = {
                "queue_stats": self.queue.get_stats(),
                "outcomes": {},
            }
            
            # Count outcomes from stored reviews
            outcome_counts: Dict[str, int] = {}
            for mission_dir in self.store_path.glob("*/"):
                reviews_file = mission_dir / "reviews.jsonl"
                if reviews_file.exists():
                    with open(reviews_file, "r") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                outcome = data.get("outcome", "unknown")
                                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                            except Exception:
                                continue
            
            stats["outcomes"] = outcome_counts
            stats["total_reviews"] = sum(outcome_counts.values())
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_flagged_decisions(
        self,
        mission_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get flagged decisions for manual review.
        
        Args:
            mission_id: Filter by mission (optional)
            limit: Maximum to return
            
        Returns:
            List of flagged decisions
        """
        flagged = []
        
        try:
            search_path = self.store_path
            if mission_id:
                search_path = self.store_path / mission_id
            
            for reviews_file in search_path.glob("**/reviews.jsonl"):
                with open(reviews_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get("outcome") == "flag":
                                flagged.append(data)
                                if len(flagged) >= limit:
                                    return flagged
                        except Exception:
                            continue
            
            return flagged
            
        except Exception as e:
            logger.warning(f"[REVIEWER] Failed to get flagged: {e}")
            return []


# Human-in-the-loop API hooks (for future integration)
class HumanReviewAPI:
    """
    API hooks for human-in-the-loop review.
    
    This is a placeholder for future HTTP/WebSocket API.
    Currently provides local methods for manual review.
    """
    
    def __init__(self, reviewer: DecisionReviewer):
        self.reviewer = reviewer
    
    def get_pending_for_human(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get flagged decisions pending human review."""
        return self.reviewer.get_flagged_decisions(limit=limit)
    
    def submit_human_review(
        self,
        queue_id: str,
        outcome: str,
        feedback: Optional[str] = None,
        override_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit human review result.
        
        Args:
            queue_id: Queue identifier
            outcome: approve, flag, override, feedback
            feedback: Optional feedback note
            override_action: If overriding, what action
            
        Returns:
            Result status
        """
        try:
            result = ReviewResult(
                queue_id=queue_id,
                decision_id=queue_id.split("_")[0],
                outcome=ReviewOutcome(outcome),
                reviewer="human",
                feedback_note=feedback,
                override_action=override_action,
                confidence=1.0,  # Human review is definitive
            )
            
            # Store result
            # Note: would need to look up original decision for full storage
            
            return {
                "success": True,
                "result": result.to_dict(),
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# Global reviewer instance
_reviewer: Optional[DecisionReviewer] = None


def get_decision_reviewer(
    config: Optional[ReviewConfig] = None,
) -> DecisionReviewer:
    """Get global decision reviewer instance."""
    global _reviewer
    if _reviewer is None:
        _reviewer = DecisionReviewer(config=config)
    return _reviewer


def reset_decision_reviewer() -> None:
    """Reset global reviewer (mainly for testing)."""
    global _reviewer
    _reviewer = None

