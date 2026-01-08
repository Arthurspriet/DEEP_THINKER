"""
Scorecard for DeepThinker Metrics.

Provides a structured quality score computed:
- per phase
- per mission final output

Scorecards are emitted via DecisionEmitter for persistence.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import get_metrics_config


@dataclass
class ScorecardCost:
    """
    Cost metrics for a scorecard.
    
    Attributes:
        tokens: Total tokens consumed
        usd: Estimated USD cost (if available)
        latency_ms: Total latency in milliseconds
    """
    tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "usd": self.usd,
            "latency_ms": self.latency_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScorecardCost":
        return cls(
            tokens=data.get("tokens", 0),
            usd=data.get("usd", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
        )


@dataclass
class ScorecardRuntime:
    """
    Runtime metrics for a scorecard.
    
    Attributes:
        wall_ms: Wall-clock time in milliseconds
        model_ms: Time spent in model calls
        tool_ms: Time spent in tool calls
    """
    wall_ms: float = 0.0
    model_ms: float = 0.0
    tool_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_ms": self.wall_ms,
            "model_ms": self.model_ms,
            "tool_ms": self.tool_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScorecardRuntime":
        return cls(
            wall_ms=data.get("wall_ms", 0.0),
            model_ms=data.get("model_ms", 0.0),
            tool_ms=data.get("tool_ms", 0.0),
        )


@dataclass
class ScorecardMetadata:
    """
    Metadata for a scorecard.
    
    Attributes:
        mission_id: Parent mission identifier
        phase_id: Phase name (if phase-level scorecard)
        timestamp: When the scorecard was computed
        models_used: List of model names used
        councils_used: List of council names used
        is_final: Whether this is the final mission scorecard
    """
    mission_id: str = ""
    phase_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    models_used: List[str] = field(default_factory=list)
    councils_used: List[str] = field(default_factory=list)
    is_final: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "phase_id": self.phase_id,
            "timestamp": self.timestamp.isoformat(),
            "models_used": self.models_used,
            "councils_used": self.councils_used,
            "is_final": self.is_final,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScorecardMetadata":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            mission_id=data.get("mission_id", ""),
            phase_id=data.get("phase_id", ""),
            timestamp=timestamp,
            models_used=data.get("models_used", []),
            councils_used=data.get("councils_used", []),
            is_final=data.get("is_final", False),
        )


@dataclass
class Scorecard:
    """
    Structured quality score for a phase or mission.
    
    Computed by JudgeEnsemble and used by ScorecardPolicy
    for stop/escalate decisions.
    
    All score fields are floats in range [0.0, 1.0].
    
    Attributes:
        goal_coverage: How well the output addresses the objective
        evidence_grounding: How well claims are supported by evidence
        actionability: How actionable/useful the output is
        consistency: Internal consistency of the output
        overall: Weighted combination of above scores
        cost: Cost metrics (tokens, USD, latency)
        runtime: Runtime metrics (wall, model, tool time)
        metadata: Context metadata
        judge_disagreement: Disagreement between judges (if ensemble)
        score_delta: Change from previous scorecard (if available)
    """
    # Quality scores (0.0 - 1.0)
    goal_coverage: float = 0.0
    evidence_grounding: float = 0.0
    actionability: float = 0.0
    consistency: float = 0.0
    overall: float = 0.0
    
    # Resource metrics
    cost: ScorecardCost = field(default_factory=ScorecardCost)
    runtime: ScorecardRuntime = field(default_factory=ScorecardRuntime)
    
    # Context
    metadata: ScorecardMetadata = field(default_factory=ScorecardMetadata)
    
    # Ensemble metrics
    judge_disagreement: float = 0.0
    
    # Delta from previous
    score_delta: Optional[float] = None
    
    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted overall score from component scores.
        
        Args:
            weights: Optional custom weights. Uses config weights if None.
            
        Returns:
            Weighted overall score (0.0 - 1.0)
        """
        if weights is None:
            config = get_metrics_config()
            weights = config.scorecard_weights
        
        total = (
            weights.get("goal_coverage", 0.3) * self.goal_coverage +
            weights.get("evidence_grounding", 0.25) * self.evidence_grounding +
            weights.get("actionability", 0.2) * self.actionability +
            weights.get("consistency", 0.25) * self.consistency
        )
        
        self.overall = min(1.0, max(0.0, total))
        return self.overall
    
    @classmethod
    def from_scores(
        cls,
        goal_coverage: float,
        evidence_grounding: float,
        actionability: float,
        consistency: float,
        metadata: Optional[ScorecardMetadata] = None,
        cost: Optional[ScorecardCost] = None,
        runtime: Optional[ScorecardRuntime] = None,
        judge_disagreement: float = 0.0,
        previous_overall: Optional[float] = None,
    ) -> "Scorecard":
        """
        Create a scorecard from individual scores.
        
        Args:
            goal_coverage: Goal coverage score (0-1)
            evidence_grounding: Evidence grounding score (0-1)
            actionability: Actionability score (0-1)
            consistency: Consistency score (0-1)
            metadata: Optional metadata
            cost: Optional cost metrics
            runtime: Optional runtime metrics
            judge_disagreement: Disagreement between judges
            previous_overall: Previous overall for delta computation
            
        Returns:
            Scorecard with overall computed
        """
        scorecard = cls(
            goal_coverage=min(1.0, max(0.0, goal_coverage)),
            evidence_grounding=min(1.0, max(0.0, evidence_grounding)),
            actionability=min(1.0, max(0.0, actionability)),
            consistency=min(1.0, max(0.0, consistency)),
            metadata=metadata or ScorecardMetadata(),
            cost=cost or ScorecardCost(),
            runtime=runtime or ScorecardRuntime(),
            judge_disagreement=judge_disagreement,
        )
        
        scorecard.compute_overall()
        
        if previous_overall is not None:
            scorecard.score_delta = scorecard.overall - previous_overall
        
        return scorecard
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal_coverage": self.goal_coverage,
            "evidence_grounding": self.evidence_grounding,
            "actionability": self.actionability,
            "consistency": self.consistency,
            "overall": self.overall,
            "cost": self.cost.to_dict(),
            "runtime": self.runtime.to_dict(),
            "metadata": self.metadata.to_dict(),
            "judge_disagreement": self.judge_disagreement,
            "score_delta": self.score_delta,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scorecard":
        """Create from dictionary."""
        return cls(
            goal_coverage=data.get("goal_coverage", 0.0),
            evidence_grounding=data.get("evidence_grounding", 0.0),
            actionability=data.get("actionability", 0.0),
            consistency=data.get("consistency", 0.0),
            overall=data.get("overall", 0.0),
            cost=ScorecardCost.from_dict(data.get("cost", {})),
            runtime=ScorecardRuntime.from_dict(data.get("runtime", {})),
            metadata=ScorecardMetadata.from_dict(data.get("metadata", {})),
            judge_disagreement=data.get("judge_disagreement", 0.0),
            score_delta=data.get("score_delta"),
        )
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if overall score meets quality threshold."""
        return self.overall >= threshold
    
    def needs_escalation(self, config: Optional[Any] = None) -> bool:
        """
        Check if scores indicate need for escalation.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
            
        Returns:
            True if escalation is recommended
        """
        if config is None:
            config = get_metrics_config()
        
        return (
            self.goal_coverage < config.escalate_goal_coverage_threshold or
            self.evidence_grounding < config.escalate_grounding_threshold
        )
    
    def can_stop(self, config: Optional[Any] = None) -> bool:
        """
        Check if scores indicate phase can stop.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
            
        Returns:
            True if stopping is recommended
        """
        if config is None:
            config = get_metrics_config()
        
        return (
            self.overall >= config.stop_overall_threshold and
            self.consistency >= config.stop_consistency_threshold
        )
    
    def __str__(self) -> str:
        return (
            f"Scorecard(overall={self.overall:.2f}, "
            f"goal={self.goal_coverage:.2f}, "
            f"evidence={self.evidence_grounding:.2f}, "
            f"action={self.actionability:.2f}, "
            f"consistency={self.consistency:.2f})"
        )

