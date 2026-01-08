"""
Schema definitions for ML Influence Monitoring.

Provides unified event schemas used by all predictors for consistent
tracking of ML influence across the system.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List


@dataclass
class MLInfluenceEvent:
    """
    Canonical event schema for ML predictor influence tracking.
    
    This schema is used by ALL predictors to ensure consistent
    cross-predictor analysis and governance reporting.
    
    Attributes:
        timestamp: ISO format timestamp of the prediction
        mission_id: ID of the mission being executed
        phase_name: Name of the phase (e.g., "Reconnaissance", "Deep Analysis")
        phase_type: Type classification (research, synthesis, deep_analysis, etc.)
        
        predictor_name: Identifier of the predictor (cost_time | phase_risk | web_search)
        predictor_version: Model version string (e.g., "v1", "v0-fallback")
        predictor_mode: Operating mode (shadow | advisory | active)
        
        prediction_summary: Key predictions from the predictor
        confidence: Prediction confidence score (0-1)
        used_fallback: Whether rule-based fallback was used instead of ML
        
        planner_original_decision: Decision before predictor influence (if any)
        planner_final_decision: Decision after predictor influence (if any)
        
        outcome_summary: Actual outcome after execution (populated post-hoc)
    """
    # Identification
    timestamp: str
    mission_id: str
    phase_name: str
    phase_type: str
    
    # Predictor identification
    predictor_name: str
    predictor_version: str
    predictor_mode: str  # shadow | advisory | active
    
    # Prediction data
    prediction_summary: Dict[str, Any]
    confidence: float
    used_fallback: bool
    
    # Decision tracking (for influence analysis)
    planner_original_decision: Optional[Dict[str, Any]] = None
    planner_final_decision: Optional[Dict[str, Any]] = None
    
    # Outcome tracking (populated after execution)
    outcome_summary: Optional[Dict[str, Any]] = None
    
    # Extension fields for future predictor types
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLInfluenceEvent":
        """Create from dictionary (e.g., from JSON)."""
        # Handle optional fields that may not exist in older records
        data.setdefault("planner_original_decision", None)
        data.setdefault("planner_final_decision", None)
        data.setdefault("outcome_summary", None)
        data.setdefault("extra_metadata", {})
        return cls(**data)
    
    @classmethod
    def create(
        cls,
        mission_id: str,
        phase_name: str,
        phase_type: str,
        predictor_name: str,
        predictor_version: str,
        prediction_summary: Dict[str, Any],
        confidence: float,
        used_fallback: bool,
        predictor_mode: str = "shadow",
        planner_original_decision: Optional[Dict[str, Any]] = None,
        planner_final_decision: Optional[Dict[str, Any]] = None,
    ) -> "MLInfluenceEvent":
        """
        Factory method to create an influence event with auto-generated timestamp.
        
        This is the recommended way to create events from predictors.
        """
        return cls(
            timestamp=datetime.utcnow().isoformat(),
            mission_id=mission_id,
            phase_name=phase_name,
            phase_type=phase_type,
            predictor_name=predictor_name,
            predictor_version=predictor_version,
            predictor_mode=predictor_mode,
            prediction_summary=prediction_summary,
            confidence=confidence,
            used_fallback=used_fallback,
            planner_original_decision=planner_original_decision,
            planner_final_decision=planner_final_decision,
        )


@dataclass
class MLDriftAlert:
    """
    Alert schema for detected drift or silent influence.
    
    Emitted when the drift detector identifies potentially concerning
    patterns in predictor behavior or planner response to predictions.
    
    Attributes:
        timestamp: When the alert was generated
        predictor_name: Which predictor triggered the alert
        drift_type: Classification of the drift
            - influence_leak: Shadow mode predictor appears to affect decisions
            - confidence_drift: Confidence distribution shifting unexpectedly
            - distribution_shift: Prediction distribution changing over time
            - correlation_spike: Sudden correlation between predictions and decisions
        severity: Alert severity (0-1, higher is more concerning)
        evidence: Supporting data for the alert
        window_start: Start of analysis window
        window_end: End of analysis window
        event_count: Number of events analyzed
        recommendations: Suggested actions
    """
    timestamp: str
    predictor_name: str
    drift_type: str
    severity: float
    evidence: Dict[str, Any]
    window_start: str
    window_end: str
    event_count: int
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLDriftAlert":
        """Create from dictionary (e.g., from JSON)."""
        data.setdefault("recommendations", [])
        return cls(**data)
    
    @classmethod
    def create(
        cls,
        predictor_name: str,
        drift_type: str,
        severity: float,
        evidence: Dict[str, Any],
        window_start: str,
        window_end: str,
        event_count: int,
        recommendations: Optional[List[str]] = None,
    ) -> "MLDriftAlert":
        """Factory method to create an alert with auto-generated timestamp."""
        return cls(
            timestamp=datetime.utcnow().isoformat(),
            predictor_name=predictor_name,
            drift_type=drift_type,
            severity=min(1.0, max(0.0, severity)),  # Clamp to [0, 1]
            evidence=evidence,
            window_start=window_start,
            window_end=window_end,
            event_count=event_count,
            recommendations=recommendations or [],
        )


@dataclass
class MLInfluenceMetrics:
    """
    Computed metrics snapshot for a predictor or the system.
    
    Stored periodically to track metric evolution over time.
    """
    timestamp: str
    period_start: str
    period_end: str
    
    # Scope
    predictor_name: Optional[str]  # None for system-wide metrics
    
    # Predictor-level metrics
    prediction_count: int = 0
    fallback_count: int = 0
    fallback_rate: float = 0.0
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # Influence-level metrics
    decision_divergence_count: int = 0
    decision_divergence_rate: float = 0.0
    
    # System-level metrics (when predictor_name is None)
    phases_with_predictions: int = 0
    phases_total: int = 0
    predictor_coverage_rate: float = 0.0
    
    # Raw data for further analysis
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLInfluenceMetrics":
        """Create from dictionary."""
        data.setdefault("extra_metrics", {})
        return cls(**data)


