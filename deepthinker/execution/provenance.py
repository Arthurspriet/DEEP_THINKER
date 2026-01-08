"""
Code provenance tracking for trust boundaries and capability escalation.

Tracks the source, review status, and trust level of code to enable
controlled capability escalation based on code origin.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Dict, Any


@dataclass
class CodeProvenance:
    """
    Tracks the origin and trustworthiness of code.
    
    Attributes:
        source: Origin of the code (llm_generated, human_reviewed, predefined_tool)
        reviewer: Name/ID of human reviewer (if applicable)
        approval_level: Approval level 0-5 (0=unreviewed, 5=fully approved)
        trust_score: Trust score 0.0-1.0 (higher = more trusted)
        generated_at: When code was generated
        reviewed_at: When code was reviewed (if applicable)
    """
    
    source: Literal["llm_generated", "human_reviewed", "predefined_tool"]
    reviewer: Optional[str] = None
    approval_level: int = 0
    trust_score: float = 0.5
    generated_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate provenance data."""
        if not 0 <= self.approval_level <= 5:
            raise ValueError(f"approval_level must be 0-5, got {self.approval_level}")
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError(f"trust_score must be 0.0-1.0, got {self.trust_score}")
        if self.source == "human_reviewed" and not self.reviewer:
            raise ValueError("human_reviewed source requires reviewer")
        if self.reviewed_at and not self.reviewer:
            raise ValueError("reviewed_at requires reviewer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "reviewer": self.reviewer,
            "approval_level": self.approval_level,
            "trust_score": self.trust_score,
            "generated_at": self.generated_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeProvenance":
        """Create from dictionary."""
        if isinstance(data.get("generated_at"), str):
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        if isinstance(data.get("reviewed_at"), str):
            data["reviewed_at"] = datetime.fromisoformat(data["reviewed_at"])
        return cls(**data)
    
    def is_llm_generated(self) -> bool:
        """Check if code is LLM-generated."""
        return self.source == "llm_generated"
    
    def is_human_reviewed(self) -> bool:
        """Check if code has been human-reviewed."""
        return self.source == "human_reviewed" or self.reviewed_at is not None
    
    def can_escalate_automatically(self, min_trust: float = 0.6) -> bool:
        """Check if code can automatically escalate based on trust score."""
        return self.trust_score >= min_trust
    
    def requires_human_approval(self) -> bool:
        """Check if code requires human approval for escalation."""
        return not self.is_human_reviewed() and self.trust_score < 0.9

