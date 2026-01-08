"""
Canonical Data Models for DeepThinker Memory System.

Pydantic schemas aligned with meta-cognition engine outputs from:
- meta/hypotheses.py (Hypothesis dataclass)
- meta/supervisor.py (PhaseMetrics, DepthContract)
- meta/meta_controller.py (reflection, debate, plan revision results)

These schemas provide type-safe, serializable representations for
persistent storage in the knowledge base.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Hypotheses & Evidence
# =============================================================================

class HypothesisSchema(BaseModel):
    """
    A reasoning hypothesis with evidence tracking.
    
    Aligns with meta/hypotheses.py Hypothesis dataclass.
    """
    id: str
    statement: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: Literal["active", "rejected", "confirmed"] = "active"
    evidence_ids: List[str] = Field(default_factory=list)
    contradiction_ids: List[str] = Field(default_factory=list)
    parent_ids: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class EvidenceSchema(BaseModel):
    """
    A piece of evidence supporting or contradicting hypotheses.
    
    Used for RAG storage and retrieval.
    
    Extended with provenance tracking:
    - origin: Where the evidence came from (web, inference, human)
    - decay_rate: Confidence decay per day
    - expiry_date: When evidence should be considered stale
    """
    id: str
    mission_id: str
    phase: str
    text: str
    artifact_type: str = "general"  # research, code, evaluation, simulation, debate
    hypothesis_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    
    # Provenance tracking fields
    origin: Literal["web", "inference", "human"] = "inference"
    decay_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # Confidence decay per day
    expiry_date: Optional[datetime] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


# =============================================================================
# Phase Outputs
# =============================================================================

class PhaseOutputSchema(BaseModel):
    """
    Structured output from a mission phase.
    """
    phase_name: str
    summary: Optional[str] = None
    final_output: Optional[str] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    council_used: Optional[str] = None
    models_used: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


# =============================================================================
# Supervisor Signals (aligned with ReasoningSupervisor)
# =============================================================================

class SupervisorSignalsSchema(BaseModel):
    """
    Signals collected from ReasoningSupervisor during mission execution.
    
    Aligns with meta/supervisor.py PhaseMetrics and MissionMetrics.
    """
    difficulty_scores: List[float] = Field(default_factory=list)
    uncertainty_scores: List[float] = Field(default_factory=list)
    progress_scores: List[float] = Field(default_factory=list)
    novelty_scores: List[float] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    depth_expectations: List[float] = Field(default_factory=list)
    model_tiers_used: List[str] = Field(default_factory=list)
    convergence_scores: List[float] = Field(default_factory=list)
    stagnation_count: int = 0
    loop_detected: bool = False
    
    def add_phase_metrics(
        self,
        difficulty: float,
        uncertainty: float,
        progress: float,
        novelty: float,
        confidence: float,
        model_tier: str = "medium"
    ) -> None:
        """Add metrics from a phase analysis."""
        self.difficulty_scores.append(difficulty)
        self.uncertainty_scores.append(uncertainty)
        self.progress_scores.append(progress)
        self.novelty_scores.append(novelty)
        self.confidence_scores.append(confidence)
        self.model_tiers_used.append(model_tier)
    
    @property
    def avg_difficulty(self) -> float:
        """Average difficulty across all phases."""
        return sum(self.difficulty_scores) / len(self.difficulty_scores) if self.difficulty_scores else 0.5
    
    @property
    def avg_uncertainty(self) -> float:
        """Average uncertainty across all phases."""
        return sum(self.uncertainty_scores) / len(self.uncertainty_scores) if self.uncertainty_scores else 0.5


# =============================================================================
# Meta-Layer Records (aligned with MetaController)
# =============================================================================

class ReflectionSchema(BaseModel):
    """
    Reflection analysis from a phase.
    
    Aligns with meta/reflection.py ReflectionEngine output.
    """
    phase_name: str
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class DebateSchema(BaseModel):
    """
    Internal debate results from a phase.
    
    Aligns with meta/debate.py DebateEngine output.
    """
    phase_name: str
    optimist_view: Optional[str] = None
    skeptic_view: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    contradictions_found: List[str] = Field(default_factory=list)
    confidence_adjustments: Dict[str, float] = Field(default_factory=dict)
    agreement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None


class PlanRevisionSchema(BaseModel):
    """
    Plan revision results from a phase.
    
    Aligns with meta/plan_revision.py PlanReviser output.
    """
    phase_name: str
    added_subgoals: List[str] = Field(default_factory=list)
    removed_subgoals: List[str] = Field(default_factory=list)
    priority_changes: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    invalidated_goals: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None


# =============================================================================
# Mission Summary (for long-term memory)
# =============================================================================

class MissionSummarySchema(BaseModel):
    """
    Compressed summary of a completed mission for long-term storage.
    
    Used for cross-mission intelligence and pattern recognition.
    """
    mission_id: str
    objective: str
    mission_type: Optional[str] = None
    domain: Optional[str] = None
    
    # Key outcomes
    key_insights: List[str] = Field(default_factory=list)
    resolved_hypotheses: List[str] = Field(default_factory=list)
    unresolved_hypotheses: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    
    # Quality metrics
    final_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    convergence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Execution metadata
    phases_completed: int = 0
    total_iterations: int = 0
    time_taken_minutes: Optional[float] = None
    models_used: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Tags for retrieval
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
    
    def to_search_text(self) -> str:
        """Generate text for semantic search indexing."""
        parts = [
            self.objective,
            self.mission_type or "",
            self.domain or "",
            " ".join(self.key_insights),
            " ".join(self.resolved_hypotheses),
            " ".join(self.tags),
            " ".join(self.keywords),
        ]
        return " ".join(filter(None, parts))


# =============================================================================
# Council Feedback (for structured storage)
# =============================================================================

class CouncilFeedbackSchema(BaseModel):
    """
    Feedback from a council execution.
    """
    council_name: str
    phase_name: str
    success: bool
    output_summary: Optional[str] = None
    models_used: List[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

