"""
Schema definitions for Cost & Time Predictor.

Typed dataclasses for prediction inputs and outputs.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class PhaseContext:
    """
    Context information about the phase to be executed.
    
    Attributes:
        phase_name: Name of the phase (e.g., "Reconnaissance", "Deep Analysis")
        phase_type: Type classification (research, design, implementation, testing, synthesis, deep_analysis)
        effort_level: Execution effort level (minimal, standard, thorough)
        mission_time_budget_seconds: Total mission time budget in seconds
        time_remaining_seconds: Time remaining in mission at phase start
        iteration_index: Current iteration/phase index in mission
    """
    phase_name: str
    phase_type: str
    effort_level: str
    mission_time_budget_seconds: float
    time_remaining_seconds: float
    iteration_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExecutionPlan:
    """
    Planned execution configuration for the phase.
    
    Attributes:
        model_tier: Model tier to use (small, medium, large, xlarge)
        model_names: List of model names to be invoked
        councils_invoked: List of council names to be invoked
        consensus_enabled: Whether consensus voting is enabled
        max_iterations: Maximum iterations allowed
        per_call_timeout_seconds: Timeout per model call in seconds
        search_enabled: Whether web search is enabled
    """
    model_tier: str
    model_names: List[str]
    councils_invoked: List[str]
    consensus_enabled: bool
    max_iterations: int
    per_call_timeout_seconds: float
    search_enabled: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SystemState:
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
class CostTimePrediction:
    """
    Prediction output for phase execution costs.
    
    Attributes:
        wall_time_seconds: Predicted wall-clock execution time
        gpu_seconds: Predicted GPU compute time
        vram_peak_mb: Predicted peak VRAM usage in MB
        confidence: Prediction confidence (0-1)
        model_version: Version of the prediction model used
        used_fallback: Whether fallback rules were used instead of ML
    """
    wall_time_seconds: float
    gpu_seconds: float
    vram_peak_mb: int
    confidence: float
    model_version: str
    used_fallback: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PredictionEvaluation:
    """
    Evaluation record comparing prediction to actual execution.
    
    Used for shadow mode logging and model improvement.
    """
    timestamp: str
    mission_id: str
    phase_name: str
    phase_type: str
    
    # Prediction
    predicted_wall_time_seconds: float
    predicted_gpu_seconds: float
    predicted_vram_peak_mb: int
    prediction_confidence: float
    prediction_model_version: str
    prediction_used_fallback: bool
    
    # Actual
    actual_wall_time_seconds: float
    actual_gpu_seconds: float
    actual_vram_peak_mb: int
    
    # Error metrics
    wall_time_error_abs: float
    wall_time_error_pct: float
    gpu_seconds_error_abs: float
    gpu_seconds_error_pct: float
    vram_error_abs: float
    vram_error_pct: float
    
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
        prediction: CostTimePrediction,
        actual_wall_time: float,
        actual_gpu_seconds: float,
        actual_vram_peak: int
    ) -> "PredictionEvaluation":
        """Create evaluation from prediction and actual values."""
        # Compute errors
        wall_time_error_abs = abs(prediction.wall_time_seconds - actual_wall_time)
        wall_time_error_pct = (wall_time_error_abs / actual_wall_time * 100) if actual_wall_time > 0 else 0.0
        
        gpu_seconds_error_abs = abs(prediction.gpu_seconds - actual_gpu_seconds)
        gpu_seconds_error_pct = (gpu_seconds_error_abs / actual_gpu_seconds * 100) if actual_gpu_seconds > 0 else 0.0
        
        vram_error_abs = abs(prediction.vram_peak_mb - actual_vram_peak)
        vram_error_pct = (vram_error_abs / actual_vram_peak * 100) if actual_vram_peak > 0 else 0.0
        
        return cls(
            timestamp=timestamp,
            mission_id=mission_id,
            phase_name=phase_name,
            phase_type=phase_type,
            predicted_wall_time_seconds=prediction.wall_time_seconds,
            predicted_gpu_seconds=prediction.gpu_seconds,
            predicted_vram_peak_mb=prediction.vram_peak_mb,
            prediction_confidence=prediction.confidence,
            prediction_model_version=prediction.model_version,
            prediction_used_fallback=prediction.used_fallback,
            actual_wall_time_seconds=actual_wall_time,
            actual_gpu_seconds=actual_gpu_seconds,
            actual_vram_peak_mb=actual_vram_peak,
            wall_time_error_abs=wall_time_error_abs,
            wall_time_error_pct=wall_time_error_pct,
            gpu_seconds_error_abs=gpu_seconds_error_abs,
            gpu_seconds_error_pct=gpu_seconds_error_pct,
            vram_error_abs=vram_error_abs,
            vram_error_pct=vram_error_pct,
        )

