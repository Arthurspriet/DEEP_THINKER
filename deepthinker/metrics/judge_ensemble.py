"""
Judge Ensemble for DeepThinker Metrics.

Provides multi-model quality evaluation:
- Cheap local judge (default: smallest tier model)
- Optional stronger judge (only if enabled and budget allows)
- Aggregation with disagreement measure

Supports:
- Judging text artifacts
- Judging phase outcomes
- Sampling for performance
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import MetricsConfig, get_metrics_config, should_sample
from .scorecard import Scorecard, ScorecardMetadata

# Constitution blinding integration
try:
    from ..constitution.blinding import sanitize_for_judge
    from ..constitution.config import get_constitution_config
    CONSTITUTION_BLINDING_AVAILABLE = True
except ImportError:
    CONSTITUTION_BLINDING_AVAILABLE = False
    sanitize_for_judge = None
    get_constitution_config = None

logger = logging.getLogger(__name__)


@dataclass
class JudgeScores:
    """
    Raw scores from a single judge.
    
    Attributes:
        goal_coverage: How well the output addresses the objective (0-1)
        evidence_grounding: How well claims are supported (0-1)
        actionability: How actionable/useful the output is (0-1)
        consistency: Internal consistency (0-1)
        raw_output: Raw judge output for debugging
        model_name: Model that produced these scores
    """
    goal_coverage: float = 0.0
    evidence_grounding: float = 0.0
    actionability: float = 0.0
    consistency: float = 0.0
    raw_output: str = ""
    model_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_coverage": self.goal_coverage,
            "evidence_grounding": self.evidence_grounding,
            "actionability": self.actionability,
            "consistency": self.consistency,
            "model_name": self.model_name,
        }


@dataclass
class JudgeResult:
    """
    Result from judge ensemble.
    
    Attributes:
        scorecard: Computed scorecard
        cheap_scores: Scores from cheap judge
        strong_scores: Scores from strong judge (if used)
        disagreement: Disagreement between judges (0-1)
        sampled: Whether this was sampled (vs skipped)
    """
    scorecard: Scorecard
    cheap_scores: Optional[JudgeScores] = None
    strong_scores: Optional[JudgeScores] = None
    disagreement: float = 0.0
    sampled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorecard": self.scorecard.to_dict(),
            "cheap_scores": self.cheap_scores.to_dict() if self.cheap_scores else None,
            "strong_scores": self.strong_scores.to_dict() if self.strong_scores else None,
            "disagreement": self.disagreement,
            "sampled": self.sampled,
        }


class JudgeEnsemble:
    """
    Multi-model judge ensemble for quality evaluation.
    
    Uses a cheap judge by default, optionally adding a stronger judge
    for ensemble averaging with disagreement tracking.
    
    Usage:
        ensemble = JudgeEnsemble()
        result = ensemble.score_artifact(text, objective)
        if result.scorecard.can_stop():
            # Phase quality is sufficient
    """
    
    # Prompt template for quality judgment
    JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an AI-generated output.

## OBJECTIVE
{objective}

## OUTPUT TO EVALUATE
{output}

## EVALUATION CRITERIA
Score each dimension from 0 to 10 (where 10 is best):

1. GOAL_COVERAGE: Does the output fully address the objective?
   - 10: Completely addresses all aspects
   - 5: Partially addresses the objective
   - 0: Does not address the objective at all

2. EVIDENCE_GROUNDING: Are claims supported by evidence/sources?
   - 10: All claims are well-grounded with citations
   - 5: Some claims are grounded, some are unsupported
   - 0: No evidence or citations provided

3. ACTIONABILITY: Is the output useful and actionable?
   - 10: Clear, actionable, immediately useful
   - 5: Somewhat useful but vague
   - 0: Not useful or actionable

4. CONSISTENCY: Is the output internally consistent?
   - 10: Fully consistent, no contradictions
   - 5: Minor inconsistencies
   - 0: Major contradictions

## YOUR EVALUATION
Provide scores in this exact format:
GOAL_COVERAGE: [0-10]
EVIDENCE_GROUNDING: [0-10]
ACTIONABILITY: [0-10]
CONSISTENCY: [0-10]
"""

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        model_caller: Optional[Any] = None,
    ):
        """
        Initialize the judge ensemble.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
            model_caller: Optional model caller. Uses default if None.
        """
        self.config = config or get_metrics_config()
        self._model_caller = model_caller
        self._llm = None
    
    def _get_llm(self, model_name: str) -> Any:
        """Get or create LLM instance for judging."""
        # Lazy import to avoid circular dependencies
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_name,
                base_url=self.config.ollama_base_url,
                temperature=0.1,  # Low temperature for consistent scoring
            )
        except ImportError:
            from langchain_community.llms import Ollama
            return Ollama(
                model=model_name,
                base_url=self.config.ollama_base_url,
                temperature=0.1,
            )
    
    def _call_judge(
        self,
        model_name: str,
        output: str,
        objective: str,
    ) -> JudgeScores:
        """
        Call a single judge model.
        
        Args:
            model_name: Model to use for judging
            output: Text to evaluate
            objective: Original objective
            
        Returns:
            JudgeScores from this model
        """
        # Truncate output if too long
        max_output_len = 4000
        if len(output) > max_output_len:
            output = output[:max_output_len] + "\n... [truncated]"
        
        prompt = self.JUDGE_PROMPT.format(
            objective=objective,
            output=output,
        )
        
        try:
            llm = self._get_llm(model_name)
            
            # Handle both ChatOllama and legacy Ollama
            if hasattr(llm, 'invoke'):
                response = llm.invoke(prompt)
                raw_output = response.content if hasattr(response, 'content') else str(response)
            else:
                raw_output = llm(prompt)
            
            return self._parse_scores(raw_output, model_name)
            
        except Exception as e:
            logger.warning(f"[JUDGE] Failed to call {model_name}: {e}")
            # Return neutral scores on failure
            return JudgeScores(
                goal_coverage=0.5,
                evidence_grounding=0.5,
                actionability=0.5,
                consistency=0.5,
                raw_output=f"Error: {e}",
                model_name=model_name,
            )
    
    def _parse_scores(self, raw_output: str, model_name: str) -> JudgeScores:
        """
        Parse scores from judge output.
        
        Args:
            raw_output: Raw text output from judge
            model_name: Name of the model
            
        Returns:
            Parsed JudgeScores
        """
        scores = JudgeScores(raw_output=raw_output, model_name=model_name)
        
        # Parse each score line
        patterns = {
            "goal_coverage": r"GOAL_COVERAGE:\s*(\d+(?:\.\d+)?)",
            "evidence_grounding": r"EVIDENCE_GROUNDING:\s*(\d+(?:\.\d+)?)",
            "actionability": r"ACTIONABILITY:\s*(\d+(?:\.\d+)?)",
            "consistency": r"CONSISTENCY:\s*(\d+(?:\.\d+)?)",
        }
        
        for field_name, pattern in patterns.items():
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Normalize from 0-10 to 0-1
                    normalized = min(1.0, max(0.0, value / 10.0))
                    setattr(scores, field_name, normalized)
                except ValueError:
                    pass
        
        return scores
    
    def _compute_disagreement(
        self,
        cheap_scores: JudgeScores,
        strong_scores: JudgeScores,
    ) -> float:
        """
        Compute disagreement between two judges.
        
        Uses mean absolute difference across all dimensions.
        
        Args:
            cheap_scores: Scores from cheap judge
            strong_scores: Scores from strong judge
            
        Returns:
            Disagreement score (0-1, where 0 is perfect agreement)
        """
        diffs = [
            abs(cheap_scores.goal_coverage - strong_scores.goal_coverage),
            abs(cheap_scores.evidence_grounding - strong_scores.evidence_grounding),
            abs(cheap_scores.actionability - strong_scores.actionability),
            abs(cheap_scores.consistency - strong_scores.consistency),
        ]
        return sum(diffs) / len(diffs)
    
    def _aggregate_scores(
        self,
        cheap_scores: JudgeScores,
        strong_scores: Optional[JudgeScores],
    ) -> Tuple[float, float, float, float]:
        """
        Aggregate scores from judges.
        
        If strong judge is available, uses weighted average.
        Otherwise, uses cheap judge scores directly.
        
        Args:
            cheap_scores: Scores from cheap judge
            strong_scores: Optional scores from strong judge
            
        Returns:
            Tuple of (goal_coverage, evidence_grounding, actionability, consistency)
        """
        if strong_scores is None:
            return (
                cheap_scores.goal_coverage,
                cheap_scores.evidence_grounding,
                cheap_scores.actionability,
                cheap_scores.consistency,
            )
        
        # Weighted average (strong judge weighted higher)
        w_cheap = 0.4
        w_strong = 0.6
        
        return (
            w_cheap * cheap_scores.goal_coverage + w_strong * strong_scores.goal_coverage,
            w_cheap * cheap_scores.evidence_grounding + w_strong * strong_scores.evidence_grounding,
            w_cheap * cheap_scores.actionability + w_strong * strong_scores.actionability,
            w_cheap * cheap_scores.consistency + w_strong * strong_scores.consistency,
        )
    
    def score_artifact(
        self,
        output: str,
        objective: str,
        metadata: Optional[ScorecardMetadata] = None,
        previous_overall: Optional[float] = None,
        force_sample: bool = False,
    ) -> JudgeResult:
        """
        Score a text artifact using the judge ensemble.
        
        Args:
            output: Text to evaluate
            objective: Original objective
            metadata: Optional metadata for scorecard
            previous_overall: Previous overall score for delta computation
            force_sample: Force sampling even if rate check fails
            
        Returns:
            JudgeResult with scorecard and judge details
        """
        # Check sampling
        if not force_sample and not should_sample(self.config.judge_sample_rate):
            # Return placeholder scorecard
            return JudgeResult(
                scorecard=Scorecard(
                    goal_coverage=0.5,
                    evidence_grounding=0.5,
                    actionability=0.5,
                    consistency=0.5,
                    overall=0.5,
                    metadata=metadata or ScorecardMetadata(),
                ),
                sampled=False,
            )
        
        # Apply blinding if constitution is enabled
        blinded_output = output
        blinded_objective = objective
        
        if CONSTITUTION_BLINDING_AVAILABLE:
            try:
                const_config = get_constitution_config()
                if const_config.is_enabled and const_config.blinding_enabled:
                    blinded_output = sanitize_for_judge(output, const_config)
                    blinded_objective = sanitize_for_judge(objective, const_config)
                    logger.debug("[JUDGE] Applied constitution blinding to judge inputs")
            except Exception as e:
                logger.debug(f"[JUDGE] Blinding check failed (continuing without): {e}")
        
        # Call cheap judge
        cheap_scores = self._call_judge(
            self.config.cheap_judge_model,
            blinded_output,
            blinded_objective,
        )
        
        # Optionally call strong judge
        strong_scores = None
        disagreement = 0.0
        
        if self.config.use_strong_judge:
            strong_scores = self._call_judge(
                self.config.strong_judge_model,
                blinded_output,
                blinded_objective,
            )
            disagreement = self._compute_disagreement(cheap_scores, strong_scores)
        
        # Aggregate scores
        goal, evidence, action, consistency = self._aggregate_scores(
            cheap_scores, strong_scores
        )
        
        # Create scorecard
        scorecard = Scorecard.from_scores(
            goal_coverage=goal,
            evidence_grounding=evidence,
            actionability=action,
            consistency=consistency,
            metadata=metadata or ScorecardMetadata(),
            judge_disagreement=disagreement,
            previous_overall=previous_overall,
        )
        
        logger.debug(
            f"[JUDGE] Scored artifact: {scorecard}, "
            f"disagreement={disagreement:.2f}"
        )
        
        return JudgeResult(
            scorecard=scorecard,
            cheap_scores=cheap_scores,
            strong_scores=strong_scores,
            disagreement=disagreement,
            sampled=True,
        )
    
    def score_phase_context(
        self,
        objective: str,
        phase_name: str,
        context: str,
        mission_id: str = "",
    ) -> JudgeResult:
        """
        Score the context at phase start (for before/after comparison).
        
        Args:
            objective: Mission objective
            phase_name: Current phase name
            context: Context/input for the phase
            mission_id: Mission identifier
            
        Returns:
            JudgeResult for the context
        """
        metadata = ScorecardMetadata(
            mission_id=mission_id,
            phase_id=phase_name,
        )
        
        return self.score_artifact(
            output=context,
            objective=f"Evaluate context quality for phase '{phase_name}': {objective}",
            metadata=metadata,
        )
    
    def score_phase_output(
        self,
        objective: str,
        phase_name: str,
        output: str,
        mission_id: str = "",
        previous_overall: Optional[float] = None,
        models_used: Optional[List[str]] = None,
        councils_used: Optional[List[str]] = None,
    ) -> JudgeResult:
        """
        Score the output at phase end.
        
        Args:
            objective: Mission objective
            phase_name: Current phase name
            output: Phase output to evaluate
            mission_id: Mission identifier
            previous_overall: Previous overall for delta
            models_used: List of models used in phase
            councils_used: List of councils used in phase
            
        Returns:
            JudgeResult for the phase output
        """
        metadata = ScorecardMetadata(
            mission_id=mission_id,
            phase_id=phase_name,
            models_used=models_used or [],
            councils_used=councils_used or [],
        )
        
        return self.score_artifact(
            output=output,
            objective=objective,
            metadata=metadata,
            previous_overall=previous_overall,
        )
    
    def score_final_synthesis(
        self,
        objective: str,
        final_output: str,
        mission_id: str = "",
        models_used: Optional[List[str]] = None,
        councils_used: Optional[List[str]] = None,
    ) -> JudgeResult:
        """
        Score the final mission synthesis.
        
        Args:
            objective: Mission objective
            final_output: Final synthesized output
            mission_id: Mission identifier
            models_used: All models used in mission
            councils_used: All councils used in mission
            
        Returns:
            JudgeResult for final synthesis
        """
        metadata = ScorecardMetadata(
            mission_id=mission_id,
            phase_id="final_synthesis",
            models_used=models_used or [],
            councils_used=councils_used or [],
            is_final=True,
        )
        
        return self.score_artifact(
            output=final_output,
            objective=objective,
            metadata=metadata,
            force_sample=True,  # Always score final synthesis
        )


# Global ensemble instance
_ensemble: Optional[JudgeEnsemble] = None


def get_judge_ensemble(config: Optional[MetricsConfig] = None) -> JudgeEnsemble:
    """Get global judge ensemble instance."""
    global _ensemble
    if _ensemble is None:
        _ensemble = JudgeEnsemble(config=config)
    return _ensemble

