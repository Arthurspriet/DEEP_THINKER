"""
Schema definitions for Phase Risk Predictor.

Typed dataclasses for prediction inputs and outputs.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class PhaseRiskContext:
    """
    Context information about the phase to be executed.
    
    Attributes:
        phase_name: Name of the phase (e.g., "Reconnaissance", "Deep Analysis")
        phase_type: Type classification (research, design, implementation, testing, synthesis, deep_analysis)
        effort_level: Execution effort level (minimal, standard, thorough)
        iteration_index: Current iteration/phase index in mission
        retry_count_so_far: Number of retries already attempted for this phase
        mission_time_remaining_seconds: Time remaining in mission at phase start
    """
    phase_name: str
    phase_type: str
    effort_level: str
    iteration_index: int
    retry_count_so_far: int
    mission_time_remaining_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PhaseRiskExecutionPlan:
    """
    Planned execution configuration for the phase.
    
    Attributes:
        model_tier: Model tier to use (small, medium, large, xlarge)
        model_names: List of model names to be invoked
        councils_invoked: List of council names to be invoked
        consensus_enabled: Whether consensus voting is enabled
        search_enabled: Whether web search is enabled
        max_iterations: Maximum iterations allowed
    """
    model_tier: str
    model_names: List[str]
    councils_invoked: List[str]
    consensus_enabled: bool
    search_enabled: bool
    max_iterations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PhaseRiskSystemState:
    """
    Current system resource state.
    
    Attributes:
        available_vram_mb: Available GPU VRAM in MB
        gpu_load_ratio: Current GPU utilization (0-1)
        memory_pressure_ratio: Current memory pressure (0-1, higher = more pressure)
    """
    available_vram_mb: int
    gpu_load_ratio: float
    memory_pressure_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PhaseRiskPrediction:
    """
    Prediction output for phase execution risk.
    
    Attributes:
        retry_probability: Probability of needing retry (0-1)
        expected_retries: Expected number of retries (≥0)
        dominant_failure_mode: Most likely failure mode (timeout, hallucination, low_quality, incoherent, unknown)
        confidence: Prediction confidence (0-1)
        model_version: Version of the prediction model used
        used_fallback: Whether fallback rules were used instead of ML
    """
    retry_probability: float
    expected_retries: float
    dominant_failure_mode: str
    confidence: float
    model_version: str
    used_fallback: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PhaseRiskEvaluation:
    """
    Evaluation record comparing prediction to actual execution.
    
    Used for shadow mode logging and model improvement.
    """
    timestamp: str
    mission_id: str
    phase_name: str
    phase_type: str
    
    # Prediction
    predicted_retry_probability: float
    predicted_expected_retries: float
    predicted_failure_mode: str
    prediction_confidence: float
    prediction_model_version: str
    prediction_used_fallback: bool
    
    # Actual
    actual_retry_count: int
    actual_failure_mode: str
    
    # Error metrics
    retry_error_abs: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_prediction_and_actual(
        cls,
        timestamp: str,
        mission_id: str,
        phase_name: str,
        phase_type: str,
        prediction: PhaseRiskPrediction,
        actual_retry_count: int,
        actual_failure_mode: str
    ) -> "PhaseRiskEvaluation":
        """Create evaluation from prediction and actual values."""
        # Compute retry error
        retry_error_abs = abs(prediction.expected_retries - actual_retry_count)
        
        return cls(
            timestamp=timestamp,
            mission_id=mission_id,
            phase_name=phase_name,
            phase_type=phase_type,
            predicted_retry_probability=prediction.retry_probability,
            predicted_expected_retries=prediction.expected_retries,
            predicted_failure_mode=prediction.dominant_failure_mode,
            prediction_confidence=prediction.confidence,
            prediction_model_version=prediction.model_version,
            prediction_used_fallback=prediction.used_fallback,
            actual_retry_count=actual_retry_count,
            actual_failure_mode=actual_failure_mode,
            retry_error_abs=retry_error_abs,
        )

