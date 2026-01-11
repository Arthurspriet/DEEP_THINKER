"""
Event Types for Constitution Ledger.

Defines dataclasses for all constitution-relevant events:
- EvidenceEvent: New evidence added
- ScoreEvent: Score changes with deltas
- ContradictionEvent: Contradiction rate changes
- DepthEvent: Depth/round usage
- MemoryEvent: Memory injection
- CompressionEvent: Latent memory compression
- LearningUpdateEvent: Learning component updates
- ConstitutionViolationEvent: Invariant violations

All events are designed for append-only JSONL storage with privacy constraints.
"""

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstitutionEventType(str, Enum):
    """Types of constitution events."""
    EVIDENCE = "evidence"
    SCORE = "score"
    CONTRADICTION = "contradiction"
    DEPTH = "depth"
    MEMORY = "memory"
    COMPRESSION = "compression"
    LEARNING_UPDATE = "learning_update"
    VIOLATION = "violation"
    BASELINE_SNAPSHOT = "baseline_snapshot"
    PHASE_EVALUATION = "phase_evaluation"
    PRIOR_INFLUENCE = "prior_influence"  # Bio priors - explicitly non-evidence


@dataclass
class BaseConstitutionEvent:
    """Base class for all constitution events."""
    event_type: ConstitutionEventType
    mission_id: str = ""
    phase_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @staticmethod
    def _hash_sensitive(value: str, max_length: int = 16) -> str:
        """Hash sensitive data for privacy."""
        return hashlib.sha256(value.encode()).hexdigest()[:max_length]


@dataclass
class EvidenceEvent(BaseConstitutionEvent):
    """
    Event for evidence changes.
    
    Attributes:
        count_added: Number of new evidence objects added
        evidence_types: Types of evidence (web_search, code_output, etc.)
        sources_summary: Hashed summary of sources
        total_evidence_count: Total evidence count after addition
    """
    event_type: ConstitutionEventType = ConstitutionEventType.EVIDENCE
    count_added: int = 0
    evidence_types: List[str] = field(default_factory=list)
    sources_summary: str = ""  # Hashed for privacy
    total_evidence_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class ScoreEvent(BaseConstitutionEvent):
    """
    Event for score changes.
    
    Attributes:
        score_before: Score at phase start
        score_after: Score at phase end
        delta: Score change
        target_metrics: Primary metrics (e.g., overall)
        shadow_metrics: Shadow metrics for Goodhart detection
    """
    event_type: ConstitutionEventType = ConstitutionEventType.SCORE
    score_before: float = 0.0
    score_after: float = 0.0
    delta: float = 0.0
    target_metrics: Dict[str, float] = field(default_factory=dict)
    shadow_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class ContradictionEvent(BaseConstitutionEvent):
    """
    Event for contradiction rate changes.
    
    Attributes:
        rate_before: Contradiction rate at phase start
        rate_after: Contradiction rate at phase end
        top_k_conflicts: Summary of top-K conflicts (hashed claim IDs)
        consistency_score: Overall consistency score
    """
    event_type: ConstitutionEventType = ConstitutionEventType.CONTRADICTION
    rate_before: float = 0.0
    rate_after: float = 0.0
    top_k_conflicts: List[str] = field(default_factory=list)  # Hashed IDs
    consistency_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class DepthEvent(BaseConstitutionEvent):
    """
    Event for depth/round usage.
    
    Attributes:
        rounds: Number of rounds executed
        tools_used: List of tools invoked
        tool_time_ms: Total tool execution time
        evidence_gained: Evidence gained during rounds
        gain_achieved: Measurable gain (score delta, coverage, etc.)
    """
    event_type: ConstitutionEventType = ConstitutionEventType.DEPTH
    rounds: int = 0
    tools_used: List[str] = field(default_factory=list)
    tool_time_ms: float = 0.0
    evidence_gained: int = 0
    gain_achieved: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class MemoryEvent(BaseConstitutionEvent):
    """
    Event for memory injection.
    
    Attributes:
        injected_count: Number of memories injected
        token_budget: Token budget allocated
        tokens_used: Tokens actually used
        memory_types: Types of memories injected
        rejected_count: Number of memories rejected
    """
    event_type: ConstitutionEventType = ConstitutionEventType.MEMORY
    injected_count: int = 0
    token_budget: int = 0
    tokens_used: int = 0
    memory_types: List[str] = field(default_factory=list)
    rejected_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class CompressionEvent(BaseConstitutionEvent):
    """
    Event for latent memory compression.
    
    Attributes:
        method: Compression method used
        size_before: Size before compression (tokens/bytes)
        size_after: Size after compression
        uncertainty_before: Uncertainty measure before (if available)
        uncertainty_after: Uncertainty measure after
        validated: Whether uncertainty change was validated
    """
    event_type: ConstitutionEventType = ConstitutionEventType.COMPRESSION
    method: str = ""
    size_before: int = 0
    size_after: int = 0
    uncertainty_before: float = 0.0
    uncertainty_after: float = 0.0
    validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class LearningUpdateEvent(BaseConstitutionEvent):
    """
    Event for learning component updates.
    
    Attributes:
        component: Which component (bandit, router, predictor)
        allowed: Whether update was allowed
        blocked_reason: Reason if blocked
        reward: Reward value (if applicable)
        arm: Arm selected (if bandit)
    """
    event_type: ConstitutionEventType = ConstitutionEventType.LEARNING_UPDATE
    component: str = ""
    allowed: bool = True
    blocked_reason: str = ""
    reward: float = 0.0
    arm: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


@dataclass
class ConstitutionViolationEvent(BaseConstitutionEvent):
    """
    Event for constitution invariant violations.
    
    Attributes:
        invariant: Which invariant was violated
        severity: Severity level (0.0 - 1.0)
        message: Human-readable violation message
        suggested_action: Recommended corrective action
        details: Additional violation details
    """
    event_type: ConstitutionEventType = ConstitutionEventType.VIOLATION
    invariant: str = ""
    severity: float = 0.0
    message: str = ""
    suggested_action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        # Ensure severity is clamped
        result["severity"] = max(0.0, min(1.0, self.severity))
        return result


@dataclass
class PriorInfluenceEvent(BaseConstitutionEvent):
    """
    Event for bio prior influence on reasoning.
    
    This event is EXPLICITLY NON-EVIDENCE:
    - is_evidence is ALWAYS False
    - affects_confidence is ALWAYS False
    - Constitution invariants MUST ignore events where is_evidence == False
    
    Attributes:
        is_evidence: ALWAYS False - invariants must ignore this event
        affects_confidence: ALWAYS False - no confidence impact
        source: Source of the prior (e.g., "bio_priors")
        mode: Operating mode (off/advisory/shadow/soft)
        selected_patterns: List of selected pattern IDs
        signals_applied: Dictionary of applied signal values
        applied_fields: List of field names that were actually applied
        context_snapshot: Snapshot of context used for evaluation
    """
    event_type: ConstitutionEventType = ConstitutionEventType.PRIOR_INFLUENCE
    
    # Explicit non-evidence markers (future-proof)
    is_evidence: bool = False           # ALWAYS False - invariants must ignore
    affects_confidence: bool = False    # ALWAYS False - no confidence impact
    
    # Event payload
    source: str = "bio_priors"
    mode: str = ""                      # off/advisory/shadow/soft
    selected_patterns: List[str] = field(default_factory=list)
    signals_applied: Dict[str, Any] = field(default_factory=dict)
    applied_fields: List[str] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Ensure non-evidence markers are always False."""
        # Force these to False regardless of input
        object.__setattr__(self, 'is_evidence', False)
        object.__setattr__(self, 'affects_confidence', False)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        # Ensure non-evidence markers are always False in output
        result["is_evidence"] = False
        result["affects_confidence"] = False
        return result


def is_evidence_event(event: BaseConstitutionEvent) -> bool:
    """
    Check if an event should be treated as evidence for invariant checks.
    
    Events with is_evidence == False should be ignored by constitution
    invariants that check evidence-based confidence changes.
    
    Args:
        event: Constitution event to check
        
    Returns:
        True if event should be treated as evidence, False otherwise
    """
    # Check explicit is_evidence flag first (future-proof)
    if hasattr(event, 'is_evidence') and event.is_evidence is False:
        return False
    
    # Also check by event type for backward compatibility
    if event.event_type == ConstitutionEventType.PRIOR_INFLUENCE:
        return False
    
    # Default: events without is_evidence flag are evidence
    return True


@dataclass 
class BaselineSnapshot:
    """
    Snapshot of phase baseline for invariant checking.
    
    Captured at phase start, compared at phase end.
    """
    mission_id: str = ""
    phase_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Score baseline
    overall_score: float = 0.0
    goal_coverage: float = 0.0
    evidence_grounding: float = 0.0
    consistency: float = 0.0
    
    # Evidence baseline
    evidence_count: int = 0
    evidence_types: List[str] = field(default_factory=list)
    
    # Contradiction baseline
    contradiction_rate: float = 0.0
    consistency_score: float = 1.0
    
    # Depth baseline
    rounds_completed: int = 0
    tools_invoked: List[str] = field(default_factory=list)
    
    # Shadow metrics baseline
    judge_disagreement: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "goal_coverage": self.goal_coverage,
            "evidence_grounding": self.evidence_grounding,
            "consistency": self.consistency,
            "evidence_count": self.evidence_count,
            "evidence_types": self.evidence_types,
            "contradiction_rate": self.contradiction_rate,
            "consistency_score": self.consistency_score,
            "rounds_completed": self.rounds_completed,
            "tools_invoked": self.tools_invoked,
            "judge_disagreement": self.judge_disagreement,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineSnapshot":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            timestamp=timestamp,
            overall_score=data.get("overall_score", 0.0),
            goal_coverage=data.get("goal_coverage", 0.0),
            evidence_grounding=data.get("evidence_grounding", 0.0),
            consistency=data.get("consistency", 0.0),
            evidence_count=data.get("evidence_count", 0),
            evidence_types=data.get("evidence_types", []),
            contradiction_rate=data.get("contradiction_rate", 0.0),
            consistency_score=data.get("consistency_score", 1.0),
            rounds_completed=data.get("rounds_completed", 0),
            tools_invoked=data.get("tools_invoked", []),
            judge_disagreement=data.get("judge_disagreement", 0.0),
        )


