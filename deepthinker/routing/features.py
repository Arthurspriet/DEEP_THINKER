"""
Feature Extraction for DeepThinker Routing.

Extracts features for ML router and bandit decisions:
- Task type classification
- Input complexity metrics
- Phase context
- Time/budget constraints
- Historical performance trends
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoutingContext:
    """
    Context for routing decision.
    
    Provides all available information about the current state
    for feature extraction.
    
    Attributes:
        objective: Mission objective text
        phase_name: Current phase name
        input_text: Current input/context text
        time_remaining_minutes: Remaining time budget
        time_budget_minutes: Total time budget
        recent_scores: Recent scorecard overall scores
        alignment_drift_risk: Current drift risk (0-1)
        retry_count: Number of retries so far
        models_used: Models used so far
        councils_used: Councils used so far
        difficulty_estimate: Estimated difficulty (if available)
    """
    objective: str = ""
    phase_name: str = ""
    input_text: str = ""
    time_remaining_minutes: float = 60.0
    time_budget_minutes: float = 60.0
    recent_scores: List[float] = field(default_factory=list)
    alignment_drift_risk: float = 0.0
    retry_count: int = 0
    models_used: List[str] = field(default_factory=list)
    councils_used: List[str] = field(default_factory=list)
    difficulty_estimate: Optional[float] = None


def extract_routing_features(context: RoutingContext) -> Dict[str, float]:
    """
    Extract numerical features from routing context.
    
    Returns a dictionary of feature name -> float value suitable
    for ML model input.
    
    Args:
        context: RoutingContext with all available information
        
    Returns:
        Dictionary of feature names to float values
    """
    features: Dict[str, float] = {}
    
    # Task type features (binary indicators)
    task_type = _classify_task_type(context.objective)
    features["task_type_research"] = 1.0 if task_type == "research" else 0.0
    features["task_type_code"] = 1.0 if task_type == "code" else 0.0
    features["task_type_analysis"] = 1.0 if task_type == "analysis" else 0.0
    features["task_type_creative"] = 1.0 if task_type == "creative" else 0.0
    features["task_type_planning"] = 1.0 if task_type == "planning" else 0.0
    features["task_type_scholarly"] = 1.0 if task_type == "scholarly" else 0.0
    
    # Phase features (one-hot encoded common phases)
    phase = context.phase_name.lower()
    features["phase_research"] = 1.0 if "research" in phase else 0.0
    features["phase_plan"] = 1.0 if "plan" in phase else 0.0
    features["phase_code"] = 1.0 if "code" in phase else 0.0
    features["phase_eval"] = 1.0 if "eval" in phase else 0.0
    features["phase_synth"] = 1.0 if "synth" in phase else 0.0
    
    # Input complexity features
    input_len = len(context.input_text)
    features["input_length"] = min(input_len / 10000.0, 1.0)  # Normalize
    features["input_length_log"] = _safe_log(input_len)
    features["objective_length"] = min(len(context.objective) / 1000.0, 1.0)
    
    # Complexity indicators
    features["has_code_blocks"] = 1.0 if "```" in context.input_text else 0.0
    features["has_urls"] = 1.0 if re.search(r"https?://", context.input_text) else 0.0
    features["has_numbers"] = 1.0 if re.search(r"\d{3,}", context.input_text) else 0.0
    features["question_count"] = min(context.input_text.count("?") / 10.0, 1.0)
    
    # Time features
    features["time_remaining_ratio"] = (
        context.time_remaining_minutes / context.time_budget_minutes
        if context.time_budget_minutes > 0 else 1.0
    )
    features["time_remaining_minutes_log"] = _safe_log(context.time_remaining_minutes)
    features["time_pressure"] = max(0.0, 1.0 - features["time_remaining_ratio"])
    
    # Risk and difficulty features
    features["alignment_drift_risk"] = context.alignment_drift_risk
    features["retry_count"] = min(context.retry_count / 5.0, 1.0)
    
    if context.difficulty_estimate is not None:
        features["difficulty_estimate"] = context.difficulty_estimate
    else:
        # Estimate difficulty from heuristics
        features["difficulty_estimate"] = _estimate_difficulty(context)
    
    # Recent performance features
    if context.recent_scores:
        features["recent_score_mean"] = sum(context.recent_scores) / len(context.recent_scores)
        features["recent_score_min"] = min(context.recent_scores)
        features["recent_score_max"] = max(context.recent_scores)
        if len(context.recent_scores) >= 2:
            features["recent_score_trend"] = (
                context.recent_scores[-1] - context.recent_scores[0]
            ) / len(context.recent_scores)
        else:
            features["recent_score_trend"] = 0.0
    else:
        features["recent_score_mean"] = 0.5
        features["recent_score_min"] = 0.5
        features["recent_score_max"] = 0.5
        features["recent_score_trend"] = 0.0
    
    # Resource usage features
    features["models_used_count"] = min(len(context.models_used) / 10.0, 1.0)
    features["councils_used_count"] = min(len(context.councils_used) / 5.0, 1.0)
    
    return features


def _classify_task_type(objective: str) -> str:
    """
    Classify task type from objective text.
    
    Args:
        objective: Mission objective
        
    Returns:
        Task type string
    """
    objective_lower = objective.lower()
    
    # Scholarly/academic indicators (check first as more specific)
    scholarly_keywords = [
        "arxiv", "paper", "preprint", "citation", "cite",
        "literature review", "systematic review", "scholarly",
        "academic", "peer-reviewed", "journal", "conference paper",
    ]
    if any(kw in objective_lower for kw in scholarly_keywords):
        return "scholarly"
    
    # Research indicators
    research_keywords = ["research", "find", "search", "investigate", "explore", "learn"]
    if any(kw in objective_lower for kw in research_keywords):
        return "research"
    
    # Code indicators
    code_keywords = ["code", "implement", "program", "function", "class", "debug", "fix bug"]
    if any(kw in objective_lower for kw in code_keywords):
        return "code"
    
    # Analysis indicators
    analysis_keywords = ["analyze", "analyse", "evaluate", "compare", "assess", "review"]
    if any(kw in objective_lower for kw in analysis_keywords):
        return "analysis"
    
    # Creative indicators
    creative_keywords = ["create", "write", "design", "generate", "compose", "draft"]
    if any(kw in objective_lower for kw in creative_keywords):
        return "creative"
    
    # Planning indicators
    planning_keywords = ["plan", "strategy", "organize", "schedule", "roadmap"]
    if any(kw in objective_lower for kw in planning_keywords):
        return "planning"
    
    return "general"


def _safe_log(x: float) -> float:
    """Safe log with floor at 0."""
    import math
    if x <= 0:
        return 0.0
    return math.log(1 + x) / 10.0  # Normalize


def _estimate_difficulty(context: RoutingContext) -> float:
    """
    Estimate task difficulty from heuristics.
    
    Args:
        context: Routing context
        
    Returns:
        Difficulty estimate (0-1)
    """
    difficulty = 0.3  # Base difficulty
    
    # Objective length suggests complexity
    if len(context.objective) > 500:
        difficulty += 0.1
    if len(context.objective) > 1000:
        difficulty += 0.1
    
    # Input length suggests complexity
    if len(context.input_text) > 5000:
        difficulty += 0.1
    if len(context.input_text) > 20000:
        difficulty += 0.1
    
    # Retries suggest difficulty
    difficulty += min(context.retry_count * 0.1, 0.2)
    
    # Recent poor scores suggest difficulty
    if context.recent_scores and min(context.recent_scores) < 0.4:
        difficulty += 0.1
    
    return min(difficulty, 1.0)


def get_feature_names() -> List[str]:
    """Get list of all feature names."""
    return [
        # Task type
        "task_type_research",
        "task_type_code",
        "task_type_analysis",
        "task_type_creative",
        "task_type_planning",
        "task_type_scholarly",
        # Phase
        "phase_research",
        "phase_plan",
        "phase_code",
        "phase_eval",
        "phase_synth",
        # Input complexity
        "input_length",
        "input_length_log",
        "objective_length",
        "has_code_blocks",
        "has_urls",
        "has_numbers",
        "question_count",
        # Time
        "time_remaining_ratio",
        "time_remaining_minutes_log",
        "time_pressure",
        # Risk
        "alignment_drift_risk",
        "retry_count",
        "difficulty_estimate",
        # Performance
        "recent_score_mean",
        "recent_score_min",
        "recent_score_max",
        "recent_score_trend",
        # Resources
        "models_used_count",
        "councils_used_count",
    ]

