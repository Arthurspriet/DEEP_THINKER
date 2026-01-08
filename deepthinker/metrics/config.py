"""
Metrics Configuration for DeepThinker.

Centralized configuration for all metrics subsystems:
- Scorecard computation
- Judge ensemble
- Tool tracking
- Policy thresholds
- Routing/bandit settings

All features are gated behind flags (defaults OFF).
Supports environment variable overrides.
"""

import os
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """
    Centralized configuration for the metrics subsystem.
    
    All settings are configurable via environment variables.
    All feature flags default to OFF for backward compatibility.
    
    Attributes:
        scorecard_enabled: Enable scorecard computation at phase boundaries
        scorecard_policy_enabled: Enable stop/escalate policy based on scorecard
        learning_router_enabled: Enable ML router advisory
        bandit_enabled: Enable bandit for model tier selection
        claim_graph_enabled: Enable claim graph + contradiction detection
        
        judge_sample_rate: Rate at which to run judge scoring (1.0 = always)
        tool_track_sample_rate: Rate at which to record tool usage (1.0 = always)
        
        scorecard_weights: Weights for computing overall score
        cheap_judge_model: Model for cheap/fast judging
        strong_judge_model: Model for expensive/accurate judging
        use_strong_judge: Whether to use strong judge in ensemble
        
        stop_overall_threshold: Stop phase if overall >= this
        stop_consistency_threshold: Stop phase if consistency >= this
        escalate_goal_coverage_threshold: Escalate if goal_coverage < this
        
        bandit_lambda: Cost penalty weight for bandit reward
    """
    
    # Master switches (all default OFF)
    scorecard_enabled: bool = False
    scorecard_policy_enabled: bool = False
    learning_router_enabled: bool = False
    bandit_enabled: bool = False
    claim_graph_enabled: bool = False
    
    # Sampling rates (1.0 = always, 0.0 = never)
    judge_sample_rate: float = 1.0
    tool_track_sample_rate: float = 1.0
    
    # Scorecard weights (must sum to 1.0)
    scorecard_weights: Dict[str, float] = field(default_factory=lambda: {
        "goal_coverage": 0.30,
        "evidence_grounding": 0.25,
        "actionability": 0.20,
        "consistency": 0.25,
    })
    
    # Judge models
    cheap_judge_model: str = "llama3.2:1b"
    strong_judge_model: str = "gemma3:12b"
    use_strong_judge: bool = False
    ollama_base_url: str = "http://localhost:11434"
    
    # Policy thresholds
    stop_overall_threshold: float = 0.8
    stop_consistency_threshold: float = 0.7
    escalate_goal_coverage_threshold: float = 0.4
    escalate_grounding_threshold: float = 0.3
    
    # Time budget thresholds
    time_critical_threshold_minutes: float = 2.0
    
    # Bandit config
    bandit_lambda: float = 0.1  # cost penalty weight
    bandit_exploration_bonus: float = 1.0  # UCB exploration coefficient
    bandit_min_observations: int = 5  # minimum obs before exploitation
    
    # Claim graph config
    claim_graph_top_k: int = 20  # top-K claims for contradiction detection
    contradiction_threshold: float = 0.7  # NLI threshold for contradiction
    
    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            SCORECARD_ENABLED: "true" to enable scorecard
            SCORECARD_POLICY_ENABLED: "true" to enable policy
            LEARNING_ROUTER_ENABLED: "true" to enable ML router
            BANDIT_ENABLED: "true" to enable bandit
            CLAIM_GRAPH_ENABLED: "true" to enable claim graph
            
            JUDGE_SAMPLE_RATE: float 0.0-1.0
            TOOL_TRACK_SAMPLE_RATE: float 0.0-1.0
            
            SCORECARD_WEIGHT_GOAL_COVERAGE: float
            SCORECARD_WEIGHT_EVIDENCE_GROUNDING: float
            SCORECARD_WEIGHT_ACTIONABILITY: float
            SCORECARD_WEIGHT_CONSISTENCY: float
            
            CHEAP_JUDGE_MODEL: model name
            STRONG_JUDGE_MODEL: model name
            USE_STRONG_JUDGE: "true" to enable
            OLLAMA_BASE_URL: server URL
            
            STOP_OVERALL_THRESHOLD: float
            STOP_CONSISTENCY_THRESHOLD: float
            ESCALATE_GOAL_COVERAGE_THRESHOLD: float
            
            BANDIT_LAMBDA: float
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default
        
        def get_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        # Build scorecard weights from env
        weights = {
            "goal_coverage": get_float("SCORECARD_WEIGHT_GOAL_COVERAGE", 0.30),
            "evidence_grounding": get_float("SCORECARD_WEIGHT_EVIDENCE_GROUNDING", 0.25),
            "actionability": get_float("SCORECARD_WEIGHT_ACTIONABILITY", 0.20),
            "consistency": get_float("SCORECARD_WEIGHT_CONSISTENCY", 0.25),
        }
        
        return cls(
            # Master switches
            scorecard_enabled=get_bool("SCORECARD_ENABLED", False),
            scorecard_policy_enabled=get_bool("SCORECARD_POLICY_ENABLED", False),
            learning_router_enabled=get_bool("LEARNING_ROUTER_ENABLED", False),
            bandit_enabled=get_bool("BANDIT_ENABLED", False),
            claim_graph_enabled=get_bool("CLAIM_GRAPH_ENABLED", False),
            
            # Sampling
            judge_sample_rate=get_float("JUDGE_SAMPLE_RATE", 1.0),
            tool_track_sample_rate=get_float("TOOL_TRACK_SAMPLE_RATE", 1.0),
            
            # Weights
            scorecard_weights=weights,
            
            # Judge models
            cheap_judge_model=os.environ.get("CHEAP_JUDGE_MODEL", "llama3.2:1b"),
            strong_judge_model=os.environ.get("STRONG_JUDGE_MODEL", "gemma3:12b"),
            use_strong_judge=get_bool("USE_STRONG_JUDGE", False),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            
            # Policy thresholds
            stop_overall_threshold=get_float("STOP_OVERALL_THRESHOLD", 0.8),
            stop_consistency_threshold=get_float("STOP_CONSISTENCY_THRESHOLD", 0.7),
            escalate_goal_coverage_threshold=get_float("ESCALATE_GOAL_COVERAGE_THRESHOLD", 0.4),
            escalate_grounding_threshold=get_float("ESCALATE_GROUNDING_THRESHOLD", 0.3),
            
            # Time
            time_critical_threshold_minutes=get_float("TIME_CRITICAL_THRESHOLD_MINUTES", 2.0),
            
            # Bandit
            bandit_lambda=get_float("BANDIT_LAMBDA", 0.1),
            bandit_exploration_bonus=get_float("BANDIT_EXPLORATION_BONUS", 1.0),
            bandit_min_observations=get_int("BANDIT_MIN_OBSERVATIONS", 5),
            
            # Claim graph
            claim_graph_top_k=get_int("CLAIM_GRAPH_TOP_K", 20),
            contradiction_threshold=get_float("CONTRADICTION_THRESHOLD", 0.7),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scorecard_enabled": self.scorecard_enabled,
            "scorecard_policy_enabled": self.scorecard_policy_enabled,
            "learning_router_enabled": self.learning_router_enabled,
            "bandit_enabled": self.bandit_enabled,
            "claim_graph_enabled": self.claim_graph_enabled,
            "judge_sample_rate": self.judge_sample_rate,
            "tool_track_sample_rate": self.tool_track_sample_rate,
            "scorecard_weights": self.scorecard_weights,
            "cheap_judge_model": self.cheap_judge_model,
            "strong_judge_model": self.strong_judge_model,
            "use_strong_judge": self.use_strong_judge,
            "ollama_base_url": self.ollama_base_url,
            "stop_overall_threshold": self.stop_overall_threshold,
            "stop_consistency_threshold": self.stop_consistency_threshold,
            "escalate_goal_coverage_threshold": self.escalate_goal_coverage_threshold,
            "escalate_grounding_threshold": self.escalate_grounding_threshold,
            "time_critical_threshold_minutes": self.time_critical_threshold_minutes,
            "bandit_lambda": self.bandit_lambda,
            "bandit_exploration_bonus": self.bandit_exploration_bonus,
            "bandit_min_observations": self.bandit_min_observations,
            "claim_graph_top_k": self.claim_graph_top_k,
            "contradiction_threshold": self.contradiction_threshold,
        }
    
    def is_any_enabled(self) -> bool:
        """Check if any metrics feature is enabled."""
        return (
            self.scorecard_enabled or
            self.scorecard_policy_enabled or
            self.learning_router_enabled or
            self.bandit_enabled or
            self.claim_graph_enabled
        )


# Global config instance (lazy-loaded)
_config: Optional[MetricsConfig] = None


def get_metrics_config(force_reload: bool = False) -> MetricsConfig:
    """
    Get the global metrics configuration.
    
    Lazy-loads configuration from environment variables.
    
    Args:
        force_reload: Force reload from environment
        
    Returns:
        MetricsConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        _config = MetricsConfig.from_env()
        
        if _config.is_any_enabled():
            logger.info(
                f"[METRICS] Enabled features: "
                f"scorecard={_config.scorecard_enabled}, "
                f"policy={_config.scorecard_policy_enabled}, "
                f"router={_config.learning_router_enabled}, "
                f"bandit={_config.bandit_enabled}, "
                f"claims={_config.claim_graph_enabled}"
            )
    
    return _config


def reset_metrics_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None


def should_sample(rate: float) -> bool:
    """
    Determine if an operation should be sampled based on rate.
    
    Args:
        rate: Sampling rate 0.0-1.0 (1.0 = always sample)
        
    Returns:
        True if operation should proceed
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate

