"""
Review Module for DeepThinker.

Provides decision review capabilities:
- ReviewQueue: Async queue for pending reviews (never blocks)
- DecisionReviewer: AI-based decision review
- ReviewerStore: Persistence for completed reviews

Supports:
- AI reviewer (cheap model for fast review)
- Human-in-the-loop (API hooks for future integration)

Review outcomes:
- APPROVE: Decision is valid
- FLAG: Decision needs attention
- OVERRIDE: Decision should be changed
- FEEDBACK: Add feedback note

All features are gated behind ReviewConfig flags.
"""

from .config import ReviewConfig, get_review_config, reset_review_config
from .review_queue import (
    ReviewQueue,
    QueuedDecision,
    get_review_queue,
    reset_review_queue,
)
from .decision_reviewer import (
    DecisionReviewer,
    ReviewOutcome,
    ReviewResult,
    HumanReviewAPI,
    get_decision_reviewer,
    reset_decision_reviewer,
)

__all__ = [
    # Config
    "ReviewConfig",
    "get_review_config",
    "reset_review_config",
    # Queue
    "ReviewQueue",
    "QueuedDecision",
    "get_review_queue",
    "reset_review_queue",
    # Reviewer
    "DecisionReviewer",
    "ReviewOutcome",
    "ReviewResult",
    "HumanReviewAPI",
    "get_decision_reviewer",
    "reset_decision_reviewer",
]

