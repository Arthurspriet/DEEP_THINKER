"""
Constitution Engine for DeepThinker.

Core engine that evaluates all four constitutional invariants:
1. Conservation of Evidence
2. Monotonic Uncertainty Under Compression
3. No-Free-Lunch Depth
4. Anti-Gaming Divergence (Goodhart Shield)

Provides runtime enforcement at phase boundaries.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .config import ConstitutionConfig, ConstitutionMode, get_constitution_config
from .constitution_spec import ConstitutionSpec, InvariantType, get_default_spec
from .enforcement import ConstitutionFlags, EnforcementAction
from .ledger import ConstitutionLedger, get_ledger
from .types import (
    BaselineSnapshot,
    ScoreEvent,
    EvidenceEvent,
    ContradictionEvent,
    DepthEvent,
    LearningUpdateEvent,
    ConstitutionViolationEvent,
    ConstitutionEventType,
)

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState, MissionPhase
    from ..metrics.scorecard import Scorecard
    from ..epistemics.claim_graph import ClaimGraph

logger = logging.getLogger(__name__)


@dataclass
class PhaseEvaluationContext:
    """
    Context for evaluating a phase against constitution.
    
    Captured at phase start, evaluated at phase end.
    """
    mission_id: str = ""
    phase_id: str = ""
    baseline: Optional[BaselineSnapshot] = None
    
    # Metrics at phase end
    score_after: float = 0.0
    evidence_added: int = 0
    contradiction_rate_after: float = 0.0
    consistency_after: float = 1.0
    rounds_used: int = 0
    tools_used: List[str] = field(default_factory=list)
    judge_disagreement: float = 0.0
    
    # Shadow metrics
    shadow_metrics_before: Dict[str, float] = field(default_factory=dict)
    shadow_metrics_after: Dict[str, float] = field(default_factory=dict)
    
    @property
    def score_delta(self) -> float:
        """Score change from baseline."""
        if self.baseline:
            return self.score_after - self.baseline.overall_score
        return 0.0
    
    @property
    def contradiction_delta(self) -> float:
        """Contradiction rate change from baseline."""
        if self.baseline:
            return self.contradiction_rate_after - self.baseline.contradiction_rate
        return 0.0
    
    @property
    def contradictions_reduced(self) -> int:
        """Number of contradictions reduced."""
        if self.baseline and self.contradiction_delta < 0:
            # Rough estimate - smaller rate = fewer contradictions
            return int(abs(self.contradiction_delta) * 10)
        return 0


class ConstitutionEngine:
    """
    Engine for evaluating constitutional invariants.
    
    Usage:
        engine = ConstitutionEngine(mission_id="abc-123")
        
        # At phase start
        ctx = engine.snapshot_baseline(state, phase)
        
        # At phase end
        flags = engine.evaluate_phase(
            ctx=ctx,
            scorecard=scorecard,
            claim_graph=claim_graph,
            evidence_added=5,
        )
        
        if flags.block_learning:
            # Block learning updates
        if flags.stop_deepening:
            # Stop additional rounds
    """
    
    def __init__(
        self,
        mission_id: str,
        config: Optional[ConstitutionConfig] = None,
        spec: Optional[ConstitutionSpec] = None,
    ):
        """
        Initialize the engine.
        
        Args:
            mission_id: Mission identifier
            config: Optional configuration
            spec: Optional constitution specification
        """
        self.mission_id = mission_id
        self.config = config or get_constitution_config()
        self.spec = spec or get_default_spec()
        
        # Initialize ledger
        self._ledger: Optional[ConstitutionLedger] = None
        if self.config.ledger_enabled:
            self._ledger = get_ledger(mission_id, self.config)
        
        # Track shadow metrics history
        self._shadow_history: List[Dict[str, float]] = []
    
    @property
    def is_enabled(self) -> bool:
        """Check if engine is enabled."""
        return self.config.is_enabled
    
    @property
    def is_enforcing(self) -> bool:
        """Check if engine is in enforce mode."""
        return self.config.is_enforcing
    
    def snapshot_baseline(
        self,
        state: "MissionState",
        phase: "MissionPhase",
        scorecard: Optional["Scorecard"] = None,
        claim_graph: Optional["ClaimGraph"] = None,
    ) -> PhaseEvaluationContext:
        """
        Snapshot baseline metrics at phase start.
        
        Args:
            state: Current mission state
            phase: Phase about to execute
            scorecard: Optional current scorecard
            claim_graph: Optional claim graph for contradiction rate
            
        Returns:
            PhaseEvaluationContext for later evaluation
        """
        if not self.is_enabled:
            return PhaseEvaluationContext(
                mission_id=self.mission_id,
                phase_id=phase.name,
            )
        
        # Build baseline snapshot
        baseline = BaselineSnapshot(
            mission_id=self.mission_id,
            phase_id=phase.name,
            timestamp=datetime.utcnow(),
        )
        
        # Extract scorecard metrics if available
        if scorecard:
            baseline.overall_score = scorecard.overall
            baseline.goal_coverage = scorecard.goal_coverage
            baseline.evidence_grounding = scorecard.evidence_grounding
            baseline.consistency = scorecard.consistency
            baseline.judge_disagreement = scorecard.judge_disagreement
        
        # Extract contradiction rate if claim graph available
        if claim_graph:
            try:
                baseline.contradiction_rate = 1.0 - claim_graph.compute_consistency_score()
                baseline.consistency_score = claim_graph.compute_consistency_score()
            except Exception:
                pass
        
        # Track phase depth
        baseline.rounds_completed = phase.deepening_rounds
        
        # Write to ledger
        if self._ledger:
            self._ledger.write_baseline(baseline)
        
        # Build shadow metrics baseline
        shadow_before = self._compute_shadow_metrics(
            scorecard=scorecard,
            claim_graph=claim_graph,
            evidence_count=0,  # Will be filled at phase end
        )
        
        ctx = PhaseEvaluationContext(
            mission_id=self.mission_id,
            phase_id=phase.name,
            baseline=baseline,
            shadow_metrics_before=shadow_before,
        )
        
        logger.debug(
            f"[CONSTITUTION] Baseline snapshot for phase {phase.name}: "
            f"score={baseline.overall_score:.3f}"
        )
        
        return ctx
    
    def evaluate_phase(
        self,
        ctx: PhaseEvaluationContext,
        scorecard: Optional["Scorecard"] = None,
        claim_graph: Optional["ClaimGraph"] = None,
        evidence_added: int = 0,
        rounds_used: int = 1,
        tools_used: Optional[List[str]] = None,
    ) -> ConstitutionFlags:
        """
        Evaluate constitutional invariants at phase end.
        
        Args:
            ctx: Context from phase start
            scorecard: Scorecard after phase execution
            claim_graph: Claim graph after phase execution
            evidence_added: Number of new evidence objects
            rounds_used: Number of rounds executed
            tools_used: Tools invoked during phase
            
        Returns:
            ConstitutionFlags with enforcement decisions
        """
        if not self.is_enabled:
            return ConstitutionFlags.all_ok()
        
        # Update context with phase-end metrics
        if scorecard:
            ctx.score_after = scorecard.overall
            ctx.judge_disagreement = scorecard.judge_disagreement
        
        if claim_graph:
            try:
                ctx.contradiction_rate_after = 1.0 - claim_graph.compute_consistency_score()
                ctx.consistency_after = claim_graph.compute_consistency_score()
            except Exception:
                pass
        
        ctx.evidence_added = evidence_added
        ctx.rounds_used = rounds_used
        ctx.tools_used = tools_used or []
        
        # Compute shadow metrics
        ctx.shadow_metrics_after = self._compute_shadow_metrics(
            scorecard=scorecard,
            claim_graph=claim_graph,
            evidence_count=evidence_added,
            score_delta=ctx.score_delta,
        )
        
        # Evaluate all invariants
        flags = ConstitutionFlags.all_ok()
        
        if self.config.evidence_conservation_enabled:
            evidence_flags = self._check_evidence_conservation(ctx)
            flags = flags.merge(evidence_flags)
        
        if self.config.no_free_lunch_enabled:
            depth_flags = self._check_no_free_lunch(ctx)
            flags = flags.merge(depth_flags)
        
        if self.config.goodhart_shield_enabled:
            goodhart_flags = self._check_goodhart_divergence(ctx)
            flags = flags.merge(goodhart_flags)
        
        # Log events
        self._log_phase_evaluation(ctx, flags)
        
        # Update shadow history
        self._shadow_history.append(ctx.shadow_metrics_after)
        if len(self._shadow_history) > self.config.shadow_window:
            self._shadow_history.pop(0)
        
        logger.info(
            f"[CONSTITUTION] Phase {ctx.phase_id} evaluation: "
            f"ok={flags.ok}, violations={len(flags.violations)}"
        )
        
        return flags
    
    def _check_evidence_conservation(
        self,
        ctx: PhaseEvaluationContext,
    ) -> ConstitutionFlags:
        """
        Check Conservation of Evidence invariant.
        
        Score cannot increase significantly without new evidence
        or reduced contradictions.
        """
        flags = ConstitutionFlags.all_ok()
        
        # Only check if score increased beyond threshold
        if ctx.score_delta <= self.config.evidence_threshold:
            return flags
        
        # Check if we have justification for the increase
        has_new_evidence = ctx.evidence_added >= self.config.min_evidence_for_score_increase
        has_reduced_contradictions = ctx.contradictions_reduced > 0
        
        if not has_new_evidence and not has_reduced_contradictions:
            message = (
                f"Score increased by {ctx.score_delta:.3f} without new evidence "
                f"(added={ctx.evidence_added}) or contradiction reduction"
            )
            
            flags.add_violation(message, EnforcementAction.BLOCK_LEARNING)
            
            # Log violation
            if self._ledger:
                self._ledger.write_event(ConstitutionViolationEvent(
                    mission_id=self.mission_id,
                    phase_id=ctx.phase_id,
                    invariant=InvariantType.EVIDENCE_CONSERVATION.value,
                    severity=min(1.0, ctx.score_delta * 10),  # Higher delta = higher severity
                    message=message,
                    suggested_action="Add evidence or validate score increase",
                    details={
                        "score_delta": ctx.score_delta,
                        "evidence_added": ctx.evidence_added,
                        "contradictions_reduced": ctx.contradictions_reduced,
                    },
                ))
        
        return flags
    
    def _check_no_free_lunch(
        self,
        ctx: PhaseEvaluationContext,
    ) -> ConstitutionFlags:
        """
        Check No-Free-Lunch Depth invariant.
        
        Additional rounds must produce measurable gain.
        """
        flags = ConstitutionFlags.all_ok()
        
        # Only check if we used multiple rounds
        if ctx.rounds_used <= 1:
            return flags
        
        # Check if we got measurable gain
        has_evidence = ctx.evidence_added > 0
        has_score_gain = ctx.score_delta >= self.config.min_gain_per_round
        has_contradiction_reduction = ctx.contradiction_delta < 0
        
        if not (has_evidence or has_score_gain or has_contradiction_reduction):
            message = (
                f"Depth {ctx.rounds_used} produced no measurable gain "
                f"(score_delta={ctx.score_delta:.3f}, evidence={ctx.evidence_added})"
            )
            
            flags.add_violation(message, EnforcementAction.STOP_DEEPENING)
            
            # Log violation
            if self._ledger:
                self._ledger.write_event(ConstitutionViolationEvent(
                    mission_id=self.mission_id,
                    phase_id=ctx.phase_id,
                    invariant=InvariantType.NO_FREE_LUNCH.value,
                    severity=0.7,
                    message=message,
                    suggested_action="Stop deepening or switch to evidence mode",
                    details={
                        "rounds_used": ctx.rounds_used,
                        "score_delta": ctx.score_delta,
                        "evidence_added": ctx.evidence_added,
                        "tools_used": ctx.tools_used,
                    },
                ))
        
        return flags
    
    def _check_goodhart_divergence(
        self,
        ctx: PhaseEvaluationContext,
    ) -> ConstitutionFlags:
        """
        Check Anti-Gaming Divergence (Goodhart Shield) invariant.
        
        If target metrics improve but shadow metrics don't,
        flag potential gaming.
        """
        flags = ConstitutionFlags.all_ok()
        
        # Check target improvement
        target_improved = ctx.score_delta >= self.config.target_improvement_threshold
        
        if not target_improved:
            return flags  # No target improvement to check
        
        # Check shadow improvement
        shadow_improved = self._shadow_metrics_improved(ctx)
        
        if not shadow_improved:
            message = (
                f"Potential Goodhart divergence: target improved by {ctx.score_delta:.3f} "
                f"but shadow metrics did not improve"
            )
            
            flags.add_violation(message, EnforcementAction.BLOCK_LEARNING)
            flags.force_evidence_mode = True
            
            # Log violation
            if self._ledger:
                self._ledger.write_event(ConstitutionViolationEvent(
                    mission_id=self.mission_id,
                    phase_id=ctx.phase_id,
                    invariant=InvariantType.GOODHART_SHIELD.value,
                    severity=0.9,
                    message=message,
                    suggested_action="Block learning updates; verify with evidence",
                    details={
                        "score_delta": ctx.score_delta,
                        "shadow_before": ctx.shadow_metrics_before,
                        "shadow_after": ctx.shadow_metrics_after,
                    },
                ))
        
        return flags
    
    def _shadow_metrics_improved(self, ctx: PhaseEvaluationContext) -> bool:
        """Check if any shadow metric improved."""
        before = ctx.shadow_metrics_before
        after = ctx.shadow_metrics_after
        
        # Check each shadow metric
        for key in after:
            if key not in before:
                continue
            
            delta = after[key] - before[key]
            
            # Different metrics have different "improvement" directions
            if key == "contradiction_rate":
                # Lower is better
                if delta < -self.config.shadow_improvement_threshold:
                    return True
            elif key == "judge_disagreement":
                # Lower is better
                if delta < -self.config.shadow_improvement_threshold:
                    return True
            elif key == "evidence_per_score":
                # Higher is better
                if delta > self.config.shadow_improvement_threshold:
                    return True
        
        return False
    
    def _compute_shadow_metrics(
        self,
        scorecard: Optional["Scorecard"] = None,
        claim_graph: Optional["ClaimGraph"] = None,
        evidence_count: int = 0,
        score_delta: float = 0.0,
    ) -> Dict[str, float]:
        """Compute shadow metrics from available data."""
        metrics = {}
        
        # Contradiction rate (from claim graph)
        if claim_graph:
            try:
                metrics["contradiction_rate"] = 1.0 - claim_graph.compute_consistency_score()
            except Exception:
                metrics["contradiction_rate"] = 0.0
        
        # Judge disagreement (from scorecard)
        if scorecard:
            metrics["judge_disagreement"] = scorecard.judge_disagreement
        
        # Evidence per score (efficiency metric)
        if score_delta > 0 and evidence_count > 0:
            metrics["evidence_per_score"] = evidence_count / score_delta
        else:
            metrics["evidence_per_score"] = 0.0
        
        return metrics
    
    def _log_phase_evaluation(
        self,
        ctx: PhaseEvaluationContext,
        flags: ConstitutionFlags,
    ) -> None:
        """Log phase evaluation events to ledger."""
        if not self._ledger:
            return
        
        # Log score event
        self._ledger.write_event(ScoreEvent(
            mission_id=self.mission_id,
            phase_id=ctx.phase_id,
            score_before=ctx.baseline.overall_score if ctx.baseline else 0.0,
            score_after=ctx.score_after,
            delta=ctx.score_delta,
            target_metrics={"overall": ctx.score_after},
            shadow_metrics=ctx.shadow_metrics_after,
        ))
        
        # Log evidence event if any added
        if ctx.evidence_added > 0:
            self._ledger.write_event(EvidenceEvent(
                mission_id=self.mission_id,
                phase_id=ctx.phase_id,
                count_added=ctx.evidence_added,
            ))
        
        # Log depth event
        if ctx.rounds_used > 0:
            self._ledger.write_event(DepthEvent(
                mission_id=self.mission_id,
                phase_id=ctx.phase_id,
                rounds=ctx.rounds_used,
                tools_used=ctx.tools_used,
                evidence_gained=ctx.evidence_added,
                gain_achieved=ctx.score_delta,
            ))
    
    def check_compression_uncertainty(
        self,
        uncertainty_before: float,
        uncertainty_after: float,
        validated: bool = False,
    ) -> ConstitutionFlags:
        """
        Check Monotonic Uncertainty Under Compression invariant.
        
        Args:
            uncertainty_before: Uncertainty before compression
            uncertainty_after: Uncertainty after compression
            validated: Whether the reduction was validated
            
        Returns:
            ConstitutionFlags
        """
        flags = ConstitutionFlags.all_ok()
        
        if not self.config.monotonic_uncertainty_enabled:
            return flags
        
        if not self.is_enabled:
            return flags
        
        delta = uncertainty_before - uncertainty_after
        
        # Check if uncertainty decreased beyond margin without validation
        if delta > self.config.compression_uncertainty_margin and not validated:
            message = (
                f"Compression reduced uncertainty by {delta:.3f} without validation"
            )
            
            flags.add_violation(message, EnforcementAction.WARN)
            
            # Log violation
            if self._ledger:
                self._ledger.write_event(ConstitutionViolationEvent(
                    mission_id=self.mission_id,
                    phase_id="compression",
                    invariant=InvariantType.MONOTONIC_UNCERTAINTY.value,
                    severity=0.5,
                    message=message,
                    suggested_action="Validate compression with contradiction check",
                    details={
                        "uncertainty_before": uncertainty_before,
                        "uncertainty_after": uncertainty_after,
                        "delta": delta,
                    },
                ))
        
        return flags
    
    def record_learning_update(
        self,
        component: str,
        allowed: bool,
        reason: str = "",
        reward: float = 0.0,
        arm: str = "",
    ) -> None:
        """
        Record a learning update attempt.
        
        Args:
            component: Component name (bandit, router, etc.)
            allowed: Whether update was allowed
            reason: Reason if blocked
            reward: Reward value
            arm: Arm selected (for bandits)
        """
        if not self._ledger:
            return
        
        self._ledger.write_event(LearningUpdateEvent(
            mission_id=self.mission_id,
            phase_id="",  # May not have phase context
            component=component,
            allowed=allowed,
            blocked_reason=reason,
            reward=reward,
            arm=arm,
        ))


# Global engine cache
_engines: Dict[str, ConstitutionEngine] = {}


def get_engine(
    mission_id: str,
    config: Optional[ConstitutionConfig] = None,
) -> ConstitutionEngine:
    """
    Get or create an engine for a mission.
    
    Args:
        mission_id: Mission identifier
        config: Optional configuration
        
    Returns:
        ConstitutionEngine instance
    """
    global _engines
    
    if mission_id not in _engines:
        _engines[mission_id] = ConstitutionEngine(
            mission_id=mission_id,
            config=config,
        )
    return _engines[mission_id]


def clear_engine_cache() -> None:
    """Clear the global engine cache (for testing)."""
    global _engines
    _engines.clear()

