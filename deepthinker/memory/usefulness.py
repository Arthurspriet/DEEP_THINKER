"""
Memory Usefulness Prediction for DeepThinker.

Provides:
- MemoryType: Explicit memory type taxonomy
- MemoryCandidate: Candidate memory for injection
- MemoryInjectionLog: Counterfactual logging for offline ROI
- MemoryUsefulnessPredictor: P(helpful) scoring and filtering
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .usefulness_config import MemoryUsefulnessConfig, get_memory_usefulness_config

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """
    Explicit memory type taxonomy.
    
    Used for categorizing memories and computing helpfulness.
    """
    MISSION_RAG = "mission_rag"           # Current mission's RAG store
    GLOBAL_RAG = "global_rag"             # Global RAG from past missions
    GENERAL_KNOWLEDGE = "general_knowledge"  # Static knowledge (CIA Factbook, etc.)
    PAST_INSIGHT = "past_insight"         # Insights from past missions
    HYPOTHESIS = "hypothesis"             # Active/resolved hypotheses
    EVIDENCE = "evidence"                 # Evidence artifacts
    SUMMARY = "summary"                   # Mission summaries
    REFLECTION = "reflection"             # Reflection outputs
    DEBATE = "debate"                     # Debate outputs
    UNKNOWN = "unknown"                   # Unclassified


# Type-specific helpfulness priors
MEMORY_TYPE_PRIORS = {
    MemoryType.MISSION_RAG: 0.7,
    MemoryType.GLOBAL_RAG: 0.5,
    MemoryType.GENERAL_KNOWLEDGE: 0.6,
    MemoryType.PAST_INSIGHT: 0.65,
    MemoryType.HYPOTHESIS: 0.6,
    MemoryType.EVIDENCE: 0.75,
    MemoryType.SUMMARY: 0.5,
    MemoryType.REFLECTION: 0.55,
    MemoryType.DEBATE: 0.5,
    MemoryType.UNKNOWN: 0.4,
}


@dataclass
class MemoryCandidate:
    """
    Candidate memory for injection.
    
    Contains all information needed to predict helpfulness
    and track counterfactuals.
    
    Attributes:
        memory_id: Unique identifier
        memory_type: Type from MemoryType taxonomy
        text_preview: First N chars of text (for logging)
        full_text: Complete text (for injection)
        similarity_score: Retrieval similarity score (0-1)
        age_phases: Number of phases since creation
        source: Source identifier (mission_id, etc.)
        prior_reward_history: Historical rewards when this memory was used
        predicted_helpfulness: P(helpful) from predictor
        was_injected: Whether this was actually injected
        token_estimate: Estimated tokens for this memory
    """
    memory_id: str = ""
    memory_type: MemoryType = MemoryType.UNKNOWN
    text_preview: str = ""
    full_text: str = ""
    similarity_score: float = 0.0
    age_phases: int = 0
    source: str = ""
    prior_reward_history: List[float] = field(default_factory=list)
    predicted_helpfulness: float = 0.0
    was_injected: bool = False
    token_estimate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "text_preview": self.text_preview[:200],  # Truncate for logs
            "similarity_score": self.similarity_score,
            "age_phases": self.age_phases,
            "source": self.source,
            "prior_reward_history": self.prior_reward_history,
            "predicted_helpfulness": self.predicted_helpfulness,
            "was_injected": self.was_injected,
            "token_estimate": self.token_estimate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryCandidate":
        memory_type_str = data.get("memory_type", "unknown")
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            memory_type = MemoryType.UNKNOWN
        
        return cls(
            memory_id=data.get("memory_id", ""),
            memory_type=memory_type,
            text_preview=data.get("text_preview", ""),
            full_text=data.get("full_text", ""),
            similarity_score=data.get("similarity_score", 0.0),
            age_phases=data.get("age_phases", 0),
            source=data.get("source", ""),
            prior_reward_history=data.get("prior_reward_history", []),
            predicted_helpfulness=data.get("predicted_helpfulness", 0.0),
            was_injected=data.get("was_injected", False),
            token_estimate=data.get("token_estimate", 0),
        )


@dataclass
class MemoryInjectionLog:
    """
    Counterfactual logging for offline ROI analysis.
    
    Records:
    - All candidates considered (retrieval_candidates)
    - What was actually injected (injected_memories)
    - What was NOT injected (rejected_memories)
    - Outcome (score_delta_observed, filled post-phase)
    
    This enables offline analysis of memory ROI by comparing
    outcomes with different injection policies.
    """
    phase_id: str = ""
    mission_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # All candidates considered
    retrieval_candidates: List[MemoryCandidate] = field(default_factory=list)
    
    # What was actually injected
    injected_memories: List[MemoryCandidate] = field(default_factory=list)
    
    # What was NOT injected (counterfactual)
    rejected_memories: List[MemoryCandidate] = field(default_factory=list)
    
    # Budget information
    token_budget: int = 0
    tokens_used: int = 0
    
    # Outcome (filled post-phase)
    score_delta_observed: Optional[float] = None
    phase_success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "mission_id": self.mission_id,
            "timestamp": self.timestamp.isoformat(),
            "retrieval_candidates": [c.to_dict() for c in self.retrieval_candidates],
            "injected_memories": [c.to_dict() for c in self.injected_memories],
            "rejected_memories": [c.to_dict() for c in self.rejected_memories],
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "score_delta_observed": self.score_delta_observed,
            "phase_success": self.phase_success,
            # Summary stats
            "summary": {
                "num_candidates": len(self.retrieval_candidates),
                "num_injected": len(self.injected_memories),
                "num_rejected": len(self.rejected_memories),
                "avg_helpfulness_injected": (
                    sum(m.predicted_helpfulness for m in self.injected_memories) /
                    max(1, len(self.injected_memories))
                ),
                "avg_helpfulness_rejected": (
                    sum(m.predicted_helpfulness for m in self.rejected_memories) /
                    max(1, len(self.rejected_memories))
                ) if self.rejected_memories else 0.0,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryInjectionLog":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            phase_id=data.get("phase_id", ""),
            mission_id=data.get("mission_id", ""),
            timestamp=timestamp,
            retrieval_candidates=[
                MemoryCandidate.from_dict(c)
                for c in data.get("retrieval_candidates", [])
            ],
            injected_memories=[
                MemoryCandidate.from_dict(c)
                for c in data.get("injected_memories", [])
            ],
            rejected_memories=[
                MemoryCandidate.from_dict(c)
                for c in data.get("rejected_memories", [])
            ],
            token_budget=data.get("token_budget", 0),
            tokens_used=data.get("tokens_used", 0),
            score_delta_observed=data.get("score_delta_observed"),
            phase_success=data.get("phase_success"),
        )


class MemoryUsefulnessPredictor:
    """
    Predicts P(helpful) for memory candidates.
    
    Uses a simple linear model with features:
    - memory_type prior
    - similarity_score
    - age_phases (decayed)
    - prior_reward_history (if available)
    
    The model weights can be loaded from JSON for tuning.
    
    Usage:
        predictor = MemoryUsefulnessPredictor()
        
        # Score candidates
        for candidate in candidates:
            candidate.predicted_helpfulness = predictor.predict(candidate)
        
        # Filter to top-K within budget
        injected, log = predictor.filter_memories(candidates, budget, phase)
        
        # Log for counterfactual analysis
        predictor.log_counterfactual(log)
    """
    
    # Default feature weights
    DEFAULT_WEIGHTS = {
        "type_prior": 0.3,
        "similarity": 0.4,
        "age_decay": 0.15,
        "prior_reward": 0.15,
    }
    
    def __init__(self, config: Optional[MemoryUsefulnessConfig] = None):
        """
        Initialize the predictor.
        
        Args:
            config: Optional config. Uses global if None.
        """
        self.config = config or get_memory_usefulness_config()
        self._weights = self._load_weights()
    
    def _load_weights(self) -> Dict[str, float]:
        """Load weights from file or use defaults."""
        try:
            if os.path.exists(self.config.weights_path):
                with open(self.config.weights_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(
                f"[MEMORY_USEFULNESS] Failed to load weights: {e}"
            )
        
        return self.DEFAULT_WEIGHTS.copy()
    
    def predict(self, candidate: MemoryCandidate) -> float:
        """
        Predict P(helpful) for a memory candidate.
        
        Args:
            candidate: MemoryCandidate to score
            
        Returns:
            Helpfulness probability in [0, 1]
        """
        if not self.config.enabled:
            # Return similarity score as fallback when disabled
            return candidate.similarity_score
        
        weights = self._weights
        
        # Feature 1: Type prior
        type_prior = MEMORY_TYPE_PRIORS.get(candidate.memory_type, 0.4)
        
        # Feature 2: Similarity score (already 0-1)
        similarity = candidate.similarity_score
        
        # Feature 3: Age decay
        age_factor = (
            self.config.age_decay_factor ** candidate.age_phases
        )
        
        # Feature 4: Prior reward history
        if candidate.prior_reward_history:
            # Use exponentially weighted recent rewards
            recent_rewards = candidate.prior_reward_history[-5:]
            weights_exp = [0.5 ** (len(recent_rewards) - i - 1) for i in range(len(recent_rewards))]
            weight_sum = sum(weights_exp)
            prior_reward = sum(r * w for r, w in zip(recent_rewards, weights_exp)) / weight_sum
            # Normalize from [-1, 1] to [0, 1]
            prior_reward = (prior_reward + 1) / 2
        else:
            prior_reward = 0.5  # Neutral prior
        
        # Weighted combination
        helpfulness = (
            weights.get("type_prior", 0.3) * type_prior +
            weights.get("similarity", 0.4) * similarity +
            weights.get("age_decay", 0.15) * age_factor +
            weights.get("prior_reward", 0.15) * prior_reward
        )
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, helpfulness))
    
    def filter_memories(
        self,
        candidates: List[MemoryCandidate],
        budget_tokens: int,
        phase: str,
        mission_id: str = "",
        constitution_ledger: Optional[Any] = None,
    ) -> Tuple[List[MemoryCandidate], MemoryInjectionLog]:
        """
        Filter to top-K helpful memories within budget.
        
        Args:
            candidates: List of candidates to filter
            budget_tokens: Token budget
            phase: Phase name
            mission_id: Mission identifier
            constitution_ledger: Optional ConstitutionLedger for event logging
            
        Returns:
            Tuple of (injected_memories, injection_log)
        """
        # Score all candidates
        for c in candidates:
            c.predicted_helpfulness = self.predict(c)
            # Estimate tokens (rough: 4 chars per token)
            if c.token_estimate == 0:
                c.token_estimate = len(c.full_text) // 4
        
        # Sort by helpfulness (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.predicted_helpfulness,
            reverse=True,
        )
        
        # Select within budget
        injected = []
        rejected = []
        current_tokens = 0
        
        for c in sorted_candidates:
            # Check helpfulness threshold
            if c.predicted_helpfulness < self.config.min_helpfulness_threshold:
                c.was_injected = False
                rejected.append(c)
                continue
            
            # Check token budget
            if current_tokens + c.token_estimate <= budget_tokens:
                # Check max memories per phase
                if len(injected) < self.config.max_memories_per_phase:
                    c.was_injected = True
                    injected.append(c)
                    current_tokens += c.token_estimate
                else:
                    c.was_injected = False
                    rejected.append(c)
            else:
                c.was_injected = False
                rejected.append(c)
        
        # Create counterfactual log
        log = MemoryInjectionLog(
            phase_id=phase,
            mission_id=mission_id,
            timestamp=datetime.utcnow(),
            retrieval_candidates=candidates,
            injected_memories=injected,
            rejected_memories=rejected,
            token_budget=budget_tokens,
            tokens_used=current_tokens,
        )
        
        logger.debug(
            f"[MEMORY_USEFULNESS] Filtered {len(candidates)} -> {len(injected)} memories "
            f"({current_tokens}/{budget_tokens} tokens)"
        )
        
        # Write to constitution ledger if provided
        if constitution_ledger is not None:
            try:
                from ..constitution.types import MemoryEvent
                constitution_ledger.write_event(MemoryEvent(
                    mission_id=mission_id,
                    phase_id=phase,
                    injected_count=len(injected),
                    token_budget=budget_tokens,
                    tokens_used=current_tokens,
                    memory_types=[m.memory_type.value for m in injected],
                    rejected_count=len(rejected),
                ))
            except Exception as e:
                logger.debug(f"[MEMORY_USEFULNESS] Constitution ledger write failed: {e}")
        
        return injected, log
    
    def log_counterfactual(self, log: MemoryInjectionLog) -> bool:
        """
        Persist counterfactual log for offline analysis.
        
        Args:
            log: MemoryInjectionLog to persist
            
        Returns:
            True if logged successfully
        """
        if not self.config.enabled:
            return False
        
        try:
            log_path = Path(self.config.counterfactual_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log.to_dict()) + "\n")
            
            return True
            
        except Exception as e:
            logger.warning(
                f"[MEMORY_USEFULNESS] Failed to log counterfactual: {e}"
            )
            return False
    
    def update_outcome(
        self,
        mission_id: str,
        phase_id: str,
        score_delta: float,
        success: bool,
    ) -> bool:
        """
        Update the most recent log for a phase with outcome.
        
        This is called post-phase to fill in score_delta_observed.
        
        Args:
            mission_id: Mission identifier
            phase_id: Phase identifier
            score_delta: Observed score change
            success: Whether phase was successful
            
        Returns:
            True if updated successfully
        """
        # This is a simplified implementation that appends an update record
        # A full implementation would update the original log entry
        if not self.config.enabled:
            return False
        
        try:
            log_path = Path(self.config.counterfactual_log_path)
            
            if not log_path.exists():
                return False
            
            # Append outcome record
            outcome = {
                "type": "outcome_update",
                "mission_id": mission_id,
                "phase_id": phase_id,
                "score_delta": score_delta,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome) + "\n")
            
            return True
            
        except Exception as e:
            logger.warning(
                f"[MEMORY_USEFULNESS] Failed to update outcome: {e}"
            )
            return False
    
    def save_weights(self, weights: Dict[str, float]) -> bool:
        """Save weights to file."""
        try:
            weights_path = Path(self.config.weights_path)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(weights_path, "w") as f:
                json.dump(weights, f, indent=2)
            
            self._weights = weights
            return True
            
        except Exception as e:
            logger.warning(
                f"[MEMORY_USEFULNESS] Failed to save weights: {e}"
            )
            return False


# Global predictor instance
_predictor: Optional[MemoryUsefulnessPredictor] = None


def get_memory_usefulness_predictor(
    config: Optional[MemoryUsefulnessConfig] = None,
) -> MemoryUsefulnessPredictor:
    """Get global memory usefulness predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MemoryUsefulnessPredictor(config=config)
    return _predictor


def reset_memory_usefulness_predictor() -> None:
    """Reset global predictor (mainly for testing)."""
    global _predictor
    _predictor = None

