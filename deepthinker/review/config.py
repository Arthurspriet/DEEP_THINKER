"""
Review Configuration for DeepThinker.

Module-local configuration for the review subsystem.
All settings are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReviewConfig:
    """
    Configuration for the review subsystem.
    
    Attributes:
        enabled: Whether review is enabled
        queue_path: Directory for pending reviews (async queue)
        store_path: Directory for completed reviews
        
        # Review triggers
        review_routing_decisions: Review model/council routing decisions
        review_alignment_actions: Review alignment correction actions
        review_final_synthesis: Review final mission synthesis
        
        # Async settings
        max_queue_size: Maximum pending reviews before dropping
        batch_review_size: Batch size for async review processing
        
        # Reviewer settings
        reviewer_model: Model to use for AI review
    """
    
    enabled: bool = False
    queue_path: str = "kb/review_queue/"
    store_path: str = "kb/reviews/"
    
    # Review triggers
    review_routing_decisions: bool = True
    review_alignment_actions: bool = True
    review_final_synthesis: bool = True
    
    # Async settings
    max_queue_size: int = 1000
    batch_review_size: int = 50
    
    # Reviewer settings
    reviewer_model: str = "llama3.2:1b"
    
    @classmethod
    def from_env(cls) -> "ReviewConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            REVIEW_ENABLED: "true" to enable
            REVIEW_QUEUE_PATH: path string
            REVIEW_STORE_PATH: path string
            REVIEW_ROUTING_DECISIONS: "true" to review routing
            REVIEW_ALIGNMENT_ACTIONS: "true" to review alignment
            REVIEW_FINAL_SYNTHESIS: "true" to review synthesis
            REVIEW_MAX_QUEUE_SIZE: int
            REVIEW_BATCH_SIZE: int
            REVIEW_MODEL: model name
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        return cls(
            enabled=get_bool("REVIEW_ENABLED", False),
            queue_path=os.environ.get("REVIEW_QUEUE_PATH", "kb/review_queue/"),
            store_path=os.environ.get("REVIEW_STORE_PATH", "kb/reviews/"),
            review_routing_decisions=get_bool("REVIEW_ROUTING_DECISIONS", True),
            review_alignment_actions=get_bool("REVIEW_ALIGNMENT_ACTIONS", True),
            review_final_synthesis=get_bool("REVIEW_FINAL_SYNTHESIS", True),
            max_queue_size=get_int("REVIEW_MAX_QUEUE_SIZE", 1000),
            batch_review_size=get_int("REVIEW_BATCH_SIZE", 50),
            reviewer_model=os.environ.get("REVIEW_MODEL", "llama3.2:1b"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "queue_path": self.queue_path,
            "store_path": self.store_path,
            "review_routing_decisions": self.review_routing_decisions,
            "review_alignment_actions": self.review_alignment_actions,
            "review_final_synthesis": self.review_final_synthesis,
            "max_queue_size": self.max_queue_size,
            "batch_review_size": self.batch_review_size,
            "reviewer_model": self.reviewer_model,
        }


# Global config instance (lazy-loaded)
_config: Optional[ReviewConfig] = None


def get_review_config(force_reload: bool = False) -> ReviewConfig:
    """
    Get the global review configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        ReviewConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = ReviewConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[REVIEW] Enabled with queue_path={_config.queue_path}, "
                f"triggers=[routing={_config.review_routing_decisions}, "
                f"alignment={_config.review_alignment_actions}, "
                f"synthesis={_config.review_final_synthesis}]"
            )
    
    return _config


def reset_review_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None

