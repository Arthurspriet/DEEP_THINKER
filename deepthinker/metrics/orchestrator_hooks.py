"""
Orchestrator Hooks for DeepThinker Metrics.

Provides integration hooks for MissionOrchestrator:
- Phase boundary scoring (before/after)
- Policy decision making
- Tool tracking at phase/step level
- Routing advisor integration
- Bandit updates

All operations are no-ops when features are disabled.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .config import MetricsConfig, get_metrics_config, should_sample
from .scorecard import Scorecard, ScorecardMetadata, ScorecardCost, ScorecardRuntime
from .judge_ensemble import JudgeEnsemble, JudgeResult, get_judge_ensemble
from .tool_tracker import ToolTracker, get_tool_tracker

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState, MissionPhase
    from ..decisions.decision_emitter import DecisionEmitter
    from ..decisions.decision_record import DecisionRecord, DecisionType

logger = logging.getLogger(__name__)


@dataclass
class PhaseMetricsContext:
    """
    Context for tracking metrics during phase execution.
    
    Holds before/after scorecards, tool usage, and routing decisions.
    """
    mission_id: str = ""
    phase_name: str = ""
    phase_type: str = ""
    objective: str = ""
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Scorecards
    score_before: Optional[JudgeResult] = None
    score_after: Optional[JudgeResult] = None
    
    # Models and councils used
    models_used: List[str] = field(default_factory=list)
    councils_used: List[str] = field(default_factory=list)
    
    # Tool tracking
    tools_invoked: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    @property
    def score_delta(self) -> Optional[float]:
        if self.score_before and self.score_after:
            return (
                self.score_after.scorecard.overall -
                self.score_before.scorecard.overall
            )
        return None


class MetricsOrchestrationHook:
    """
    Hook class for integrating metrics into MissionOrchestrator.
    
    Provides methods to be called at key points in phase execution:
    - on_phase_start: Score context before phase runs
    - on_phase_end: Score output after phase runs
    - make_policy_decision: Check stop/escalate conditions
    - get_routing_advice: Get ML router recommendation
    - update_bandit: Update bandit with outcome
    
    Usage:
        hook = MetricsOrchestrationHook()
        
        # In _run_phase:
        ctx = hook.on_phase_start(state, phase)
        
        # ... run phase ...
        
        decision = hook.on_phase_end(ctx, phase_output, decision_emitter)
        
        if decision.action == PolicyAction.STOP:
            # Phase can stop early
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize the hook."""
        self.config = config or get_metrics_config()
        self._judge_ensemble: Optional[JudgeEnsemble] = None
        self._tool_tracker: Optional[ToolTracker] = None
        self._policy = None
        self._router = None
        self._bandit = None
    
    def _get_judge_ensemble(self) -> JudgeEnsemble:
        """Lazy-load judge ensemble."""
        if self._judge_ensemble is None:
            self._judge_ensemble = get_judge_ensemble(self.config)
        return self._judge_ensemble
    
    def _get_tool_tracker(self) -> ToolTracker:
        """Lazy-load tool tracker."""
        if self._tool_tracker is None:
            self._tool_tracker = get_tool_tracker(self.config)
        return self._tool_tracker
    
    def _get_policy(self) -> Any:
        """Lazy-load policy."""
        if self._policy is None:
            from ..policy import get_scorecard_policy
            self._policy = get_scorecard_policy(self.config)
        return self._policy
    
    def _get_router(self) -> Any:
        """Lazy-load ML router."""
        if self._router is None:
            from ..routing import get_ml_router
            self._router = get_ml_router(self.config)
        return self._router
    
    def _get_bandit(self) -> Any:
        """Lazy-load bandit."""
        if self._bandit is None:
            from ..routing import get_model_tier_bandit
            self._bandit = get_model_tier_bandit(self.config)
        return self._bandit
    
    def on_phase_start(
        self,
        state: "MissionState",
        phase: "MissionPhase",
    ) -> PhaseMetricsContext:
        """
        Called at the start of phase execution.
        
        Scores the current context to establish a baseline.
        
        Args:
            state: Current mission state
            phase: Phase about to execute
            
        Returns:
            PhaseMetricsContext for tracking
        """
        ctx = PhaseMetricsContext(
            mission_id=state.mission_id,
            phase_name=phase.name,
            phase_type=self._classify_phase(phase),
            objective=state.objective,
            start_time=time.time(),
        )
        
        # Set tool tracker context
        if self.config.tool_track_sample_rate > 0:
            tracker = self._get_tool_tracker()
            tracker.set_context(
                mission_id=state.mission_id,
                phase_id=phase.name,
            )
        
        # Score context if enabled and should sample
        if (
            self.config.scorecard_enabled and
            should_sample(self.config.judge_sample_rate)
        ):
            try:
                # Get current context for scoring
                context_text = self._extract_context_for_scoring(state, phase)
                
                ctx.score_before = self._get_judge_ensemble().score_phase_context(
                    objective=state.objective,
                    phase_name=phase.name,
                    context=context_text,
                    mission_id=state.mission_id,
                )
                
                logger.debug(
                    f"[METRICS] Phase '{phase.name}' before score: "
                    f"{ctx.score_before.scorecard.overall:.2f}"
                )
            except Exception as e:
                logger.warning(f"[METRICS] Failed to score phase start: {e}")
        
        return ctx
    
    def on_phase_end(
        self,
        ctx: PhaseMetricsContext,
        phase: "MissionPhase",
        decision_emitter: Optional["DecisionEmitter"] = None,
    ) -> Tuple[Optional[Scorecard], Optional[Any]]:
        """
        Called at the end of phase execution.
        
        Scores the output, computes delta, makes policy decision.
        
        Args:
            ctx: Context from on_phase_start
            phase: Completed phase
            decision_emitter: Optional emitter for logging decisions
            
        Returns:
            Tuple of (scorecard, policy_decision)
        """
        ctx.end_time = time.time()
        
        scorecard = None
        policy_decision = None
        
        if not self.config.scorecard_enabled:
            return None, None
        
        try:
            # Get phase output for scoring
            output_text = self._extract_output_for_scoring(phase)
            
            # Get previous overall if we have a before score
            previous_overall = None
            if ctx.score_before is not None:
                previous_overall = ctx.score_before.scorecard.overall
            
            # Score the output
            ctx.score_after = self._get_judge_ensemble().score_phase_output(
                objective=ctx.objective,
                phase_name=phase.name,
                output=output_text,
                mission_id=ctx.mission_id,
                previous_overall=previous_overall,
                models_used=ctx.models_used,
                councils_used=ctx.councils_used,
            )
            
            scorecard = ctx.score_after.scorecard
            
            # Add cost and runtime info
            scorecard.runtime = ScorecardRuntime(
                wall_ms=ctx.duration_ms,
            )
            
            logger.info(
                f"[METRICS] Phase '{phase.name}' after score: "
                f"{scorecard.overall:.2f} (delta={ctx.score_delta or 0:.2f})"
            )
            
            # Emit scorecard decision
            if decision_emitter is not None:
                self._emit_scorecard_decision(
                    scorecard, ctx, decision_emitter
                )
            
            # Make policy decision if enabled
            if self.config.scorecard_policy_enabled:
                policy_decision = self._make_policy_decision(
                    scorecard, ctx, decision_emitter
                )
            
            # Attribute score delta to tools
            if ctx.score_delta is not None and self.config.tool_track_sample_rate > 0:
                tracker = self._get_tool_tracker()
                # Get all records for this phase and attribute
                for step_id in set(r.step_id for r in tracker.get_records(phase_id=phase.name)):
                    if step_id:
                        tracker.attribute_score_delta(
                            step_id=step_id,
                            score_delta=ctx.score_delta,
                            attribution_method="equal",
                        )
        
        except Exception as e:
            logger.warning(f"[METRICS] Failed to score phase end: {e}")
        
        return scorecard, policy_decision
    
    def get_routing_advice(
        self,
        state: "MissionState",
        phase: "MissionPhase",
        recent_scores: Optional[List[float]] = None,
    ) -> Optional[Any]:
        """
        Get ML router advice for phase execution.
        
        Args:
            state: Current mission state
            phase: Phase about to execute
            recent_scores: Recent scorecard overall values
            
        Returns:
            RoutingDecision or None
        """
        if not self.config.learning_router_enabled:
            return None
        
        try:
            from ..routing import RoutingContext
            
            context = RoutingContext(
                objective=state.objective,
                phase_name=phase.name,
                input_text=self._extract_context_for_scoring(state, phase),
                time_remaining_minutes=state.remaining_minutes(),
                time_budget_minutes=state.constraints.time_budget_minutes,
                recent_scores=recent_scores or [],
                retry_count=0,  # Could get from state
            )
            
            router = self._get_router()
            decision = router.advise(context)
            
            logger.debug(
                f"[METRICS] Router advice for '{phase.name}': "
                f"tier={decision.model_tier}, rounds={decision.num_rounds}, "
                f"confidence={decision.confidence:.2f}"
            )
            
            return decision
        
        except Exception as e:
            logger.warning(f"[METRICS] Router advice failed: {e}")
            return None
    
    def select_model_tier_with_bandit(
        self,
        phase: "MissionPhase",
    ) -> str:
        """
        Select model tier using bandit if enabled.
        
        Args:
            phase: Phase for context
            
        Returns:
            Selected tier name
        """
        if not self.config.bandit_enabled:
            return "MEDIUM"
        
        try:
            bandit = self._get_bandit()
            tier = bandit.select()
            logger.debug(f"[METRICS] Bandit selected tier: {tier}")
            return tier
        except Exception as e:
            logger.warning(f"[METRICS] Bandit selection failed: {e}")
            return "MEDIUM"
    
    def update_bandit(
        self,
        tier: str,
        score_delta: float,
        cost_delta: float,
    ) -> None:
        """
        Update bandit with observed outcome.
        
        Args:
            tier: Tier that was used
            score_delta: Score improvement
            cost_delta: Cost incurred
        """
        if not self.config.bandit_enabled:
            return
        
        try:
            bandit = self._get_bandit()
            reward = bandit.update(tier, score_delta, cost_delta)
            logger.debug(
                f"[METRICS] Bandit updated: tier={tier}, reward={reward:.3f}"
            )
        except Exception as e:
            logger.warning(f"[METRICS] Bandit update failed: {e}")
    
    def _classify_phase(self, phase: "MissionPhase") -> str:
        """Classify phase type from name."""
        name_lower = phase.name.lower()
        if "research" in name_lower or "recon" in name_lower:
            return "research"
        elif "plan" in name_lower or "design" in name_lower:
            return "plan"
        elif "code" in name_lower or "implement" in name_lower:
            return "code"
        elif "eval" in name_lower or "test" in name_lower:
            return "eval"
        elif "synth" in name_lower:
            return "synth"
        return "general"
    
    def _extract_context_for_scoring(
        self,
        state: "MissionState",
        phase: "MissionPhase",
    ) -> str:
        """Extract text context for scoring."""
        parts = [
            f"Objective: {state.objective}",
            f"Phase: {phase.name}",
        ]
        
        # Add recent outputs from previous phases
        for prev_phase in state.phases:
            if prev_phase.name == phase.name:
                break
            if prev_phase.artifacts:
                summary = str(prev_phase.artifacts)[:500]
                parts.append(f"From {prev_phase.name}: {summary}")
        
        return "\n".join(parts)
    
    def _extract_output_for_scoring(self, phase: "MissionPhase") -> str:
        """Extract phase output text for scoring."""
        if not phase.artifacts:
            return ""
        
        # Collect meaningful artifact content
        parts = []
        for key, value in phase.artifacts.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                parts.append(f"{key}: {value[:1000]}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {str(value)[:500]}")
            elif isinstance(value, list):
                parts.append(f"{key}: {len(value)} items")
        
        return "\n".join(parts)
    
    def _emit_scorecard_decision(
        self,
        scorecard: Scorecard,
        ctx: PhaseMetricsContext,
        decision_emitter: "DecisionEmitter",
    ) -> None:
        """Emit scorecard as a decision record."""
        try:
            from ..decisions.decision_record import DecisionType, DecisionRecord
            import uuid
            
            record = DecisionRecord(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.SCORECARD_STOP,
                timestamp=datetime.utcnow(),
                mission_id=ctx.mission_id,
                phase_id=ctx.phase_name,
                phase_type=ctx.phase_type,
                options_considered=["continue", "stop", "escalate"],
                selected_option="scored",
                rationale=str(scorecard),
                confidence=scorecard.overall,
                constraints_snapshot={
                    "goal_coverage": scorecard.goal_coverage,
                    "evidence_grounding": scorecard.evidence_grounding,
                    "actionability": scorecard.actionability,
                    "consistency": scorecard.consistency,
                    "overall": scorecard.overall,
                    "score_delta": ctx.score_delta,
                },
            )
            
            decision_emitter.emit(record)
        except Exception as e:
            logger.warning(f"[METRICS] Failed to emit scorecard decision: {e}")
    
    def _make_policy_decision(
        self,
        scorecard: Scorecard,
        ctx: PhaseMetricsContext,
        decision_emitter: Optional["DecisionEmitter"],
    ) -> Any:
        """Make and emit policy decision."""
        try:
            from ..policy import PolicyAction
            
            policy = self._get_policy()
            
            # Get time remaining (would need state access for real value)
            decision = policy.decide(
                scorecard=scorecard,
                time_remaining_minutes=10.0,  # Placeholder
            )
            
            logger.info(
                f"[METRICS] Policy decision for '{ctx.phase_name}': "
                f"{decision.action.value} - {decision.rationale}"
            )
            
            # Emit policy decision
            if decision_emitter is not None and decision.action != PolicyAction.CONTINUE:
                self._emit_policy_decision(decision, ctx, decision_emitter)
            
            return decision
        
        except Exception as e:
            logger.warning(f"[METRICS] Policy decision failed: {e}")
            return None
    
    def _emit_policy_decision(
        self,
        policy_decision: Any,
        ctx: PhaseMetricsContext,
        decision_emitter: "DecisionEmitter",
    ) -> None:
        """Emit policy decision as a decision record."""
        try:
            from ..decisions.decision_record import DecisionType, DecisionRecord
            from ..policy import PolicyAction
            import uuid
            
            # Map policy action to decision type
            if policy_decision.action == PolicyAction.STOP:
                decision_type = DecisionType.SCORECARD_STOP
            elif policy_decision.action == PolicyAction.ESCALATE:
                decision_type = DecisionType.SCORECARD_ESCALATE
            else:
                decision_type = DecisionType.ROUTING_DECISION
            
            record = DecisionRecord(
                decision_id=str(uuid.uuid4()),
                decision_type=decision_type,
                timestamp=datetime.utcnow(),
                mission_id=ctx.mission_id,
                phase_id=ctx.phase_name,
                phase_type=ctx.phase_type,
                options_considered=["continue", "stop", "escalate", "force_evidence"],
                selected_option=policy_decision.action.value,
                rationale=policy_decision.rationale,
                confidence=policy_decision.confidence,
                constraints_snapshot=policy_decision.constraints_snapshot,
            )
            
            decision_emitter.emit(record)
        except Exception as e:
            logger.warning(f"[METRICS] Failed to emit policy decision: {e}")


# Global hook instance
_metrics_hook: Optional[MetricsOrchestrationHook] = None


def get_metrics_hook(config: Optional[MetricsConfig] = None) -> MetricsOrchestrationHook:
    """Get global metrics orchestration hook."""
    global _metrics_hook
    if _metrics_hook is None:
        _metrics_hook = MetricsOrchestrationHook(config=config)
    return _metrics_hook

