"""
Schema definitions for Web Search Predictor.

Typed dataclasses for prediction inputs and outputs.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class WebSearchContext:
    """
    Context information about the phase to be executed.
    
    Attributes:
        phase_name: Name of the phase (e.g., "Reconnaissance", "Deep Analysis")
        phase_type: Type classification (research, design, implementation, testing, synthesis, deep_analysis)
        effort_level: Execution effort level (minimal, standard, thorough)
        iteration_index: Current iteration/phase index in mission
        prompt_token_count: Estimated token count of the prompt
        contains_dates: Whether prompt contains date references
        contains_named_entities: Whether prompt contains named entities (people, places, orgs)
        contains_factual_claims: Whether prompt contains factual claims requiring verification
    """
    phase_name: str
    phase_type: str
    effort_level: str
    iteration_index: int
    prompt_token_count: int
    contains_dates: bool
    contains_named_entities: bool
    contains_factual_claims: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class WebSearchExecutionPlan:
    """
    Planned execution configuration for the phase.
    
    Attributes:
        model_tier: Model tier to use (small, medium, large, xlarge)
        model_names: List of model names to be invoked
        councils_invoked: List of council names to be invoked
        consensus_enabled: Whether consensus voting is enabled
        search_enabled_by_planner: Whether planner enabled web search for this phase
        max_iterations: Maximum iterations allowed
    """
    model_tier: str
    model_names: List[str]
    councils_invoked: List[str]
    consensus_enabled: bool
    search_enabled_by_planner: bool
    max_iterations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class WebSearchSystemState:
    """
    Current system state relevant to web search decisions.
    
    Attributes:
        available_time_seconds: Time remaining in mission
    """
    available_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class WebSearchPrediction:
    """
    Prediction output for web search necessity.
    
    Attributes:
        search_required: Whether web search is predicted to be needed
        expected_queries: Expected number of search queries (≥0)
        hallucination_risk_without_search: Risk of hallucination if search not used (0-1)
        confidence: Prediction confidence (0-1)
        model_version: Version of the prediction model used
        used_fallback: Whether fallback rules were used instead of ML
    """
    search_required: bool
    expected_queries: int
    hallucination_risk_without_search: float
    confidence: float
    model_version: str
    used_fallback: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class WebSearchEvaluation:
    """
    Evaluation record comparing prediction to actual execution.
    
    Used for shadow mode logging and model improvement.
    """
    timestamp: str
    mission_id: str
    phase_name: str
    phase_type: str
    
    # Prediction
    predicted_search_required: bool
    predicted_expected_queries: int
    predicted_hallucination_risk: float
    prediction_confidence: float
    prediction_model_version: str
    prediction_used_fallback: bool
    
    # Actual
    actual_search_used: bool
    actual_num_queries: int
    actual_hallucination_detected: bool
    
    # Error metrics
    search_prediction_correct: bool
    query_count_error_abs: int
    
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
        prediction: WebSearchPrediction,
        actual_search_used: bool,
        actual_num_queries: int,
        actual_hallucination_detected: bool
    ) -> "WebSearchEvaluation":
        """Create evaluation from prediction and actual values."""
        # Compute metrics
        search_prediction_correct = prediction.search_required == actual_search_used
        query_count_error_abs = abs(prediction.expected_queries - actual_num_queries)
        
        return cls(
            timestamp=timestamp,
            mission_id=mission_id,
            phase_name=phase_name,
            phase_type=phase_type,
            predicted_search_required=prediction.search_required,
            predicted_expected_queries=prediction.expected_queries,
            predicted_hallucination_risk=prediction.hallucination_risk_without_search,
            prediction_confidence=prediction.confidence,
            prediction_model_version=prediction.model_version,
            prediction_used_fallback=prediction.used_fallback,
            actual_search_used=actual_search_used,
            actual_num_queries=actual_num_queries,
            actual_hallucination_detected=actual_hallucination_detected,
            search_prediction_correct=search_prediction_correct,
            query_count_error_abs=query_count_error_abs,
        )

