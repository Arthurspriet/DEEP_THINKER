"""
Alignment Control Layer - Configuration.

Manages configuration for the alignment subsystem.
Supports loading from environment variables and mission constraints.

Configuration is opt-in: ALIGNMENT_ENABLED=False by default.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlignmentConfig:
    """
    Configuration for the Alignment Control Layer.
    
    All thresholds and settings are configurable via environment
    variables or explicit configuration.
    
    Attributes:
        enabled: Master switch for alignment control
        embedding_model: Ollama model for embeddings
        evaluator_model: Ollama model for LLM evaluation
        ollama_base_url: Ollama server URL
        
        min_similarity_soft: Trigger if a_t < this value
        delta_neg_soft: Trigger if d_t < this value
        cusum_k: CUSUM slack parameter
        cusum_h: CUSUM threshold for trigger
        min_events_before_trigger: Minimum events before allowing triggers
        
        warning_threshold: a_t below this triggers warning state (visibility only)
        correction_threshold: a_t below this triggers correction actions
        
        run_evaluator_on_trigger: Whether to run LLM evaluator on trigger
        persist_logs: Whether to persist alignment logs
        user_event_threshold: Consecutive triggers before user event
        
        reanchor_cooldown_phases: Minimum phases between re-anchors
        max_actions_per_phase: Maximum actions to apply per phase
        
        reanchor_prompt_template: Template for micro re-anchor prompt injection
        inject_reanchor_prompt: Whether to inject re-anchor prompts into councils
    """
    # Master switch
    enabled: bool = False
    
    # Model configuration
    embedding_model: str = "qwen3-embedding:4b"
    evaluator_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Drift detection thresholds
    min_similarity_soft: float = 0.4
    delta_neg_soft: float = -0.15
    cusum_k: float = 0.05
    cusum_h: float = 0.5
    min_events_before_trigger: int = 3
    
    # Two-tier threshold system (Horizon 1 Gap 1)
    # warning_threshold: a_t below this triggers warning state (visibility only)
    # correction_threshold: a_t below this triggers correction actions
    warning_threshold: float = 0.6
    correction_threshold: float = 0.4
    
    # Demo mode for testing (higher sensitivity)
    # When enabled, uses demo thresholds which trigger more easily
    demo_mode: bool = False
    demo_warning_threshold: float = 0.85
    demo_correction_threshold: float = 0.75
    demo_min_events_before_trigger: int = 1
    
    # Behavior settings
    run_evaluator_on_trigger: bool = True
    persist_logs: bool = True
    user_event_threshold: int = 5
    
    # Action limits
    reanchor_cooldown_phases: int = 2
    max_actions_per_phase: int = 2
    
    # Prompt injection for re-anchoring (Horizon 1 Gap 2)
    reanchor_prompt_template: str = "[ALIGNMENT REMINDER: Original mission objective: \"{objective}\". Ensure current work directly addresses this goal.]"
    inject_reanchor_prompt: bool = True
    
    # Prompt injection safety limits (prevents spam)
    max_injections_per_mission: int = 5
    min_phases_between_injections: int = 2
    
    # Audit logging for prompt injection
    log_full_prompts: bool = False  # If True, store full injected prompt in logs
    
    # Escalation ladder weights (how many consecutive triggers for each action)
    escalation_ladder: Dict[str, int] = field(default_factory=lambda: {
        "reanchor_internal": 1,
        "increase_skeptic_weight": 2,
        "switch_to_evidence": 3,
        "prune_focus_areas": 4,
        "user_event": 5,
    })
    
    @classmethod
    def from_env(cls) -> "AlignmentConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            ALIGNMENT_ENABLED: "true" to enable
            ALIGNMENT_EMBEDDING_MODEL: Embedding model name
            ALIGNMENT_EVALUATOR_MODEL: Evaluator model name
            ALIGNMENT_MIN_SIMILARITY: Minimum similarity threshold
            ALIGNMENT_CUSUM_H: CUSUM trigger threshold
            ALIGNMENT_PERSIST_LOGS: "true" to enable logging
            ALIGNMENT_USER_EVENT_THRESHOLD: Triggers before user event
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
        
        demo_mode = get_bool("ALIGNMENT_DEMO_MODE", False)
        
        # Use demo thresholds if demo mode is enabled
        if demo_mode:
            warning_threshold = get_float("ALIGNMENT_WARNING_THRESHOLD", 0.85)
            correction_threshold = get_float("ALIGNMENT_CORRECTION_THRESHOLD", 0.75)
            min_events = get_int("ALIGNMENT_MIN_EVENTS", 1)
        else:
            warning_threshold = get_float("ALIGNMENT_WARNING_THRESHOLD", 0.6)
            correction_threshold = get_float("ALIGNMENT_CORRECTION_THRESHOLD", 0.4)
            min_events = get_int("ALIGNMENT_MIN_EVENTS", 3)
        
        return cls(
            enabled=get_bool("ALIGNMENT_ENABLED", False),
            embedding_model=os.environ.get("ALIGNMENT_EMBEDDING_MODEL", "qwen3-embedding:4b"),
            evaluator_model=os.environ.get("ALIGNMENT_EVALUATOR_MODEL", "llama3.2:3b"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            min_similarity_soft=get_float("ALIGNMENT_MIN_SIMILARITY", 0.4),
            delta_neg_soft=get_float("ALIGNMENT_DELTA_NEG", -0.15),
            cusum_k=get_float("ALIGNMENT_CUSUM_K", 0.05),
            cusum_h=get_float("ALIGNMENT_CUSUM_H", 0.5),
            min_events_before_trigger=min_events,
            # Two-tier thresholds
            warning_threshold=warning_threshold,
            correction_threshold=correction_threshold,
            # Demo mode
            demo_mode=demo_mode,
            run_evaluator_on_trigger=get_bool("ALIGNMENT_RUN_EVALUATOR", True),
            persist_logs=get_bool("ALIGNMENT_PERSIST_LOGS", True),
            user_event_threshold=get_int("ALIGNMENT_USER_EVENT_THRESHOLD", 5),
            # Prompt injection
            inject_reanchor_prompt=get_bool("ALIGNMENT_INJECT_PROMPT", True),
            # Prompt injection safety limits
            max_injections_per_mission=get_int("ALIGNMENT_MAX_INJECTIONS", 5),
            min_phases_between_injections=get_int("ALIGNMENT_MIN_PHASES_BETWEEN_INJECTIONS", 2),
            log_full_prompts=get_bool("ALIGNMENT_LOG_FULL_PROMPTS", False),
        )
    
    @classmethod
    def from_constraints(
        cls,
        constraints: Any,
        env_fallback: bool = True,
    ) -> "AlignmentConfig":
        """
        Create configuration from mission constraints.
        
        Falls back to environment variables for unspecified values.
        
        Args:
            constraints: MissionConstraints object
            env_fallback: Whether to use env vars as fallback
            
        Returns:
            AlignmentConfig instance
        """
        # Start with env config if fallback enabled
        if env_fallback:
            config = cls.from_env()
        else:
            config = cls()
        
        # Override from constraints if present
        if constraints is None:
            return config
        
        # Check for alignment settings in constraints
        # Constraints may have a preferences object with alignment settings
        if hasattr(constraints, "alignment_enabled"):
            config.enabled = constraints.alignment_enabled
        
        # Check in preferences
        if hasattr(constraints, "preferences"):
            prefs = constraints.preferences
            if hasattr(prefs, "alignment_enabled"):
                config.enabled = prefs.alignment_enabled
        
        # Check in notes for explicit enablement
        if hasattr(constraints, "notes") and constraints.notes:
            notes_lower = constraints.notes.lower()
            if "alignment_enabled" in notes_lower or "enable_alignment" in notes_lower:
                config.enabled = True
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "embedding_model": self.embedding_model,
            "evaluator_model": self.evaluator_model,
            "ollama_base_url": self.ollama_base_url,
            "min_similarity_soft": self.min_similarity_soft,
            "delta_neg_soft": self.delta_neg_soft,
            "cusum_k": self.cusum_k,
            "cusum_h": self.cusum_h,
            "min_events_before_trigger": self.min_events_before_trigger,
            # Two-tier thresholds
            "warning_threshold": self.warning_threshold,
            "correction_threshold": self.correction_threshold,
            # Demo mode
            "demo_mode": self.demo_mode,
            "demo_warning_threshold": self.demo_warning_threshold,
            "demo_correction_threshold": self.demo_correction_threshold,
            "demo_min_events_before_trigger": self.demo_min_events_before_trigger,
            "run_evaluator_on_trigger": self.run_evaluator_on_trigger,
            "persist_logs": self.persist_logs,
            "user_event_threshold": self.user_event_threshold,
            "reanchor_cooldown_phases": self.reanchor_cooldown_phases,
            "max_actions_per_phase": self.max_actions_per_phase,
            "escalation_ladder": self.escalation_ladder,
            # Prompt injection
            "reanchor_prompt_template": self.reanchor_prompt_template,
            "inject_reanchor_prompt": self.inject_reanchor_prompt,
            # Prompt injection safety limits
            "max_injections_per_mission": self.max_injections_per_mission,
            "min_phases_between_injections": self.min_phases_between_injections,
            "log_full_prompts": self.log_full_prompts,
        }


# Global config instance (lazy-loaded)
_config: Optional[AlignmentConfig] = None


def get_alignment_config(
    constraints: Any = None,
    force_reload: bool = False,
) -> AlignmentConfig:
    """
    Get the global alignment configuration.
    
    Lazy-loads configuration from environment variables.
    Can be overridden with mission constraints.
    
    Args:
        constraints: Optional mission constraints to apply
        force_reload: Force reload from environment
        
    Returns:
        AlignmentConfig instance
    """
    global _config
    
    if _config is None or force_reload:
        if constraints is not None:
            _config = AlignmentConfig.from_constraints(constraints)
        else:
            _config = AlignmentConfig.from_env()
        
        if _config.enabled:
            logger.info(
                f"[ALIGNMENT] Enabled with config: "
                f"embedding={_config.embedding_model}, "
                f"evaluator={_config.evaluator_model}, "
                f"cusum_h={_config.cusum_h}"
            )
    
    return _config


def reset_alignment_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config
    _config = None

