"""
Schemas and data models for the epistemic control tooling layer.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


# =============================================================================
# Phase 1: Epistemic Control
# =============================================================================

@dataclass
class Claim:
    """An atomic, verifiable claim extracted from council output."""
    id: str
    text: str
    context: str  # Surrounding context from original output
    claim_type: str  # "factual", "inference", "assumption", "uncertainty"
    source_council: Optional[str] = None
    source_phase: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConfidenceScore:
    """Confidence score for a claim with risk assessment."""
    score: float  # 0.0 to 1.0
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    signals: Dict[str, Any]  # Breakdown of confidence signals
    claim_id: str
    council_agreement: Optional[float] = None
    memory_presence: bool = False
    linguistic_uncertainty: bool = False


class UnverifiedClaimError(Exception):
    """Raised when an unverified claim attempts to pass citation gate."""
    def __init__(self, claim_id: str, claim_text: str, reason: str):
        self.claim_id = claim_id
        self.claim_text = claim_text
        self.reason = reason
        super().__init__(f"Unverified claim {claim_id}: {reason}")


# =============================================================================
# Phase 2: Web Search Enforcement
# =============================================================================

@dataclass
class SearchJustification:
    """Justification for web search with queries."""
    claim_id: str
    queries: List[str]
    rationale: str
    priority: Literal["low", "medium", "high"] = "medium"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchBudget:
    """Allocated search budget for a mission phase."""
    max_searches: int
    depth_per_query: int  # Max results per query
    priority_claims: List[str]  # Claim IDs that should be searched first
    time_remaining_minutes: float
    budget_used: int = 0


@dataclass
class CompressedEvidence:
    """Compressed evidence from web search for council consumption."""
    claim_id: str
    snippets: List[str]  # Quoted snippets with context
    sources: List[str]  # URLs or source identifiers
    reliability: float  # 0.0 to 1.0
    relevance_score: float = 0.0  # How relevant to the claim
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Phase 3: Memory Integrity
# =============================================================================

@dataclass
class RetrievalAuditLog:
    """Log entry for memory retrieval operation."""
    timestamp: datetime
    mission_id: str
    phase: str
    query: str
    retrieved_ids: List[str]  # Claim IDs or evidence IDs
    similarity_scores: Dict[str, float]  # ID -> score
    selection_reason: str
    influenced_output: bool = False
    output_trace: Optional[str] = None  # Trace to final answer if influenced


# =============================================================================
# Phase 4: Reasoning Depth Control
# =============================================================================

@dataclass
class DepthDecision:
    """Decision on whether to continue reasoning depth."""
    continue_reasoning: bool
    max_remaining: int  # Max iterations remaining
    reason: str
    diminishing_returns_detected: bool = False
    new_signal_detected: bool = True


@dataclass
class ROIMetrics:
    """Return on investment metrics for a phase."""
    phase_name: str
    tokens_spent: int
    facts_added: int  # New claims verified
    decisions_changed: int  # Plan revisions or goal changes
    roi_score: float  # (facts_added + decisions_changed) / tokens_spent
    should_abort: bool = False


# =============================================================================
# Phase 5: Resource Management
# =============================================================================

@dataclass
class RoutingDecision:
    """Decision on where to route a computational task."""
    task: str
    target: Literal["cpu", "gpu"]
    reason: str
    estimated_cost: Optional[float] = None  # Estimated resource cost


@dataclass
class EscalationDecision:
    """Decision on execution tier escalation."""
    allowed: bool
    target_tier: str
    justification: str
    claim_risk: Optional[float] = None
    expected_value: Optional[float] = None
    time_budget_remaining: Optional[float] = None


# =============================================================================
# Phase 6: Output Trust
# =============================================================================

@dataclass
class ConfidenceHeader:
    """Confidence metadata header for final output."""
    overall_confidence: float
    verified_claims_count: int
    unverified_claims_count: int
    assumptions_count: int
    uncertainty_markers: List[str] = field(default_factory=list)
    formatted_header: str = ""  # Human-readable header text


@dataclass
class CounterfactualResult:
    """Result of counterfactual fragility analysis."""
    fragile_claims: List[str]  # Claim IDs that are fragile
    risk_score: float  # 0.0 to 1.0
    failure_scenarios: List[str] = field(default_factory=list)


# =============================================================================
# Phase 7: Debugging
# =============================================================================

class MissionTrace(BaseModel):
    """Structured trace of mission execution."""
    mission_id: str
    objective: str
    phases: List[Dict[str, Any]] = Field(default_factory=list)  # Phase durations, metrics
    searches_executed: int = 0
    searches_skipped: int = 0
    search_justifications: List[Dict[str, Any]] = Field(default_factory=list)
    memory_retrievals: int = 0
    memory_writes: int = 0
    hallucination_flags: List[str] = Field(default_factory=list)
    unverified_claims: List[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


@dataclass
class DisagreementReport:
    """Report on council output disagreements."""
    divergence_score: float  # 0.0 (agreement) to 1.0 (complete disagreement)
    unresolved: List[str]  # List of unresolved claim IDs or topics
    agreement_areas: List[str] = field(default_factory=list)
    disagreement_areas: List[str] = field(default_factory=list)
    requires_arbitration: bool = True

