"""
Bio Prior Engine.

Pure, deterministic engine that evaluates bio patterns against context
and produces bounded pressure signals.

This engine:
- Is PURE: No side effects, no state mutation
- Is DETERMINISTIC: Same input always produces same output
- Is NON-AUTHORITATIVE: Signals suggest, not mandate
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .config import BioPriorConfig
from .signals import PressureSignals
from .schema import BioPattern
from .metrics import BioPriorContext, RECENT_WINDOW_STEPS
from .loader import load_patterns
from .integration import compute_would_apply_diff

logger = logging.getLogger(__name__)


@dataclass
class BioPriorOutput:
    """
    Output from bio prior engine evaluation.
    
    Attributes:
        signals: Computed pressure signals
        selected_patterns: List of selected patterns with scores
        advisory_text: Human-readable advisory text
        trace: Trace information for debugging/auditing
        mode: Operating mode (off/advisory/shadow/soft)
        applied: True only if mode=="soft" AND fields were applied
        applied_fields: List of field names that were applied
    """
    signals: PressureSignals
    selected_patterns: List[Dict[str, Any]]
    advisory_text: str
    trace: Dict[str, Any]
    mode: str
    applied: bool
    applied_fields: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signals": self.signals.to_dict(),
            "selected_patterns": self.selected_patterns,
            "advisory_text": self.advisory_text,
            "trace": self.trace,
            "mode": self.mode,
            "applied": self.applied,
            "applied_fields": self.applied_fields,
        }


class BioPriorEngine:
    """
    Pure, deterministic bio prior evaluation engine.
    
    Evaluates bio patterns against context and produces bounded
    pressure signals for reasoning modulation.
    
    Usage:
        config = BioPriorConfig(enabled=True, mode="soft")
        engine = BioPriorEngine(config)
        
        ctx = BioPriorContext(phase="research", step_index=5, ...)
        output = engine.evaluate(ctx)
        
        if output.applied:
            # Signals were applied in soft mode
            print(f"Applied fields: {output.applied_fields}")
    """
    
    def __init__(
        self,
        config: Optional[BioPriorConfig] = None,
        patterns: Optional[List[BioPattern]] = None,
    ):
        """
        Initialize the engine.
        
        Args:
            config: Configuration (loads from env if not provided)
            patterns: Patterns to use (loads from directory if not provided)
        """
        from .config import get_bio_prior_config
        
        self.config = config or get_bio_prior_config()
        
        # Load patterns (cached)
        if patterns is not None:
            self._patterns = patterns
        else:
            self._patterns = load_patterns(validate=True) if self.config.is_active else []
        
        if self.config.is_active:
            logger.info(
                f"[BIO_PRIORS] Engine initialized with {len(self._patterns)} patterns, "
                f"mode={self.config.mode}"
            )
    
    @property
    def patterns(self) -> List[BioPattern]:
        """Get loaded patterns."""
        return self._patterns
    
    def evaluate(self, ctx: BioPriorContext) -> BioPriorOutput:
        """
        Evaluate context against patterns and produce signals.
        
        This method is PURE and DETERMINISTIC:
        - No side effects
        - Same input always produces same output
        
        Args:
            ctx: BioPriorContext with current state
            
        Returns:
            BioPriorOutput with signals and metadata
        """
        # Mode off: return empty output
        if not self.config.is_active:
            return self._empty_output("off")
        
        # Score all patterns against context
        scored_patterns = self._score_patterns(ctx)
        
        # Select top-K patterns
        selected = sorted(
            scored_patterns,
            key=lambda x: x["score"] * x["weight"],
            reverse=True,
        )[:self.config.topk]
        
        # Merge signals from selected patterns
        signals = self._merge_pattern_signals(selected)
        
        # Apply max pressure scaling
        if self.config.max_pressure != 1.0:
            signals = signals.scale(self.config.max_pressure)
        
        # Build advisory text
        advisory_text = self._build_advisory_text(selected, ctx)
        
        # Build trace
        trace = self._build_trace(ctx, scored_patterns, selected, signals)
        
        # Determine applied status based on mode
        applied = False
        applied_fields: List[str] = []
        
        if self.config.mode == "soft":
            # In soft mode, mark as applied (actual application happens in integration)
            # applied_fields will be populated by the caller after actual application
            applied = True
            # List fields that WOULD be applied (v1: only depth_budget_delta)
            if signals.depth_budget_delta != 0:
                applied_fields = ["depth_budget_delta"]
        
        return BioPriorOutput(
            signals=signals,
            selected_patterns=[
                {"id": p["id"], "score": p["score"], "weight": p["weight"]}
                for p in selected
            ],
            advisory_text=advisory_text,
            trace=trace,
            mode=self.config.mode,
            applied=applied,
            applied_fields=applied_fields,
        )
    
    def _score_patterns(
        self,
        ctx: BioPriorContext,
    ) -> List[Dict[str, Any]]:
        """
        Score all patterns against context.
        
        Deterministic scoring based on context signals.
        """
        scored = []
        
        for pattern in self._patterns:
            score = pattern.matches_context(
                phase=ctx.phase,
                has_stagnation=ctx.has_stagnation_signal,
                has_high_contradiction=ctx.has_high_contradiction,
                has_high_drift=ctx.has_high_drift,
                has_high_branching=ctx.has_high_branching,
                is_early_phase=ctx.is_early_phase,
                is_late_phase=ctx.is_late_phase,
            )
            
            # Apply heuristic boosts based on context
            score = self._apply_context_boosts(pattern, ctx, score)
            
            scored.append({
                "id": pattern.id,
                "name": pattern.name,
                "score": score,
                "weight": pattern.weight,
                "pattern": pattern,
            })
        
        return scored
    
    def _apply_context_boosts(
        self,
        pattern: BioPattern,
        ctx: BioPriorContext,
        base_score: float,
    ) -> float:
        """
        Apply deterministic context-based score boosts.
        
        Scoring logic (deterministic heuristic):
        - If evidence_new_count_recent == 0 => boost foraging/ant_trails/immune
        - If contradiction_rate high => boost redundancy/error_correction
        - If drift_score high => boost homeostasis/immune
        - If branching_factor high => boost metabolic_budget
        - If phase early => boost foraging/developmental
        - If phase late => boost redundancy/predator_prey
        """
        score = base_score
        pattern_id = pattern.id.upper()
        
        # Stagnation boost
        if ctx.evidence_new_count_recent == 0:
            if any(kw in pattern_id for kw in ["IMMUNE", "FORAGING", "ANT"]):
                score += 0.3
        
        # Contradiction boost
        if ctx.has_high_contradiction:
            if any(kw in pattern_id for kw in ["REDUNDANCY", "ERROR", "IMMUNE"]):
                score += 0.25
        
        # Drift boost
        if ctx.has_high_drift:
            if any(kw in pattern_id for kw in ["HOMEOSTASIS", "IMMUNE"]):
                score += 0.3
        
        # High branching boost
        if ctx.has_high_branching:
            if any(kw in pattern_id for kw in ["METABOLIC", "BUDGET"]):
                score += 0.25
        
        # Early phase boost
        if ctx.is_early_phase:
            if any(kw in pattern_id for kw in ["FORAGING", "DEVELOPMENTAL", "ANT"]):
                score += 0.2
        
        # Late phase boost
        if ctx.is_late_phase:
            if any(kw in pattern_id for kw in ["REDUNDANCY", "PREDATOR", "ERROR"]):
                score += 0.2
        
        return min(1.0, score)
    
    def _merge_pattern_signals(
        self,
        selected: List[Dict[str, Any]],
    ) -> PressureSignals:
        """
        Merge signals from selected patterns using weighted combination.
        """
        if not selected:
            return PressureSignals.zero()
        
        # Start with zero signals
        merged = PressureSignals.zero()
        
        for item in selected:
            pattern: BioPattern = item["pattern"]
            weight = item["weight"] * item["score"]
            
            pattern_signals = pattern.to_pressure_signals()
            merged = merged.merge(pattern_signals, weight_self=1.0, weight_other=weight)
        
        return merged.clamp()
    
    def _build_advisory_text(
        self,
        selected: List[Dict[str, Any]],
        ctx: BioPriorContext,
    ) -> str:
        """Build human-readable advisory text."""
        if not selected:
            return "No patterns selected for current context."
        
        lines = ["Bio Prior Advisory:"]
        
        for item in selected:
            pattern: BioPattern = item["pattern"]
            lines.append(
                f"- {pattern.name} (score={item['score']:.2f}, weight={item['weight']:.2f})"
            )
        
        # Add context summary
        lines.append("")
        lines.append("Context signals:")
        if ctx.has_stagnation_signal:
            lines.append("- Evidence stagnation detected")
        if ctx.has_high_contradiction:
            lines.append("- High contradiction rate")
        if ctx.has_high_drift:
            lines.append("- High drift from goal")
        if ctx.has_high_branching:
            lines.append("- High plan branching factor")
        if ctx.is_early_phase:
            lines.append("- Early phase (exploration appropriate)")
        if ctx.is_late_phase:
            lines.append("- Late phase (verification appropriate)")
        
        return "\n".join(lines)
    
    def _build_trace(
        self,
        ctx: BioPriorContext,
        all_scored: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        signals: PressureSignals,
    ) -> Dict[str, Any]:
        """Build trace for debugging and auditing."""
        trace = {
            "context_snapshot": ctx.to_dict(),
            "missing_metrics": ctx.get_missing_metrics(),
            "recent_window_steps": ctx.recent_window_steps,
            "patterns_scored": len(all_scored),
            "patterns_selected": len(selected),
            "topk": self.config.topk,
            "max_pressure": self.config.max_pressure,
            "all_scores": [
                {"id": p["id"], "score": p["score"], "weight": p["weight"]}
                for p in sorted(all_scored, key=lambda x: x["score"], reverse=True)
            ],
        }
        
        # Add would_apply_diff for shadow mode
        if self.config.mode in ("shadow", "soft"):
            # Create a mock DeepeningPlan-like object for diff computation
            trace["would_apply_diff"] = {
                "v1_applied": {},
                "v1_log_only": {},
            }
            
            # depth_budget_delta
            if signals.depth_budget_delta != 0:
                trace["would_apply_diff"]["v1_applied"]["depth_budget_delta"] = {
                    "value": signals.depth_budget_delta,
                    "bounded_range": "[-2, +2]",
                }
            
            # Log-only signals
            if signals.redundancy_check:
                trace["would_apply_diff"]["v1_log_only"]["redundancy_check"] = True
            if signals.force_falsification_step:
                trace["would_apply_diff"]["v1_log_only"]["force_falsification_step"] = True
            if signals.branch_pruning_suggested:
                trace["would_apply_diff"]["v1_log_only"]["branch_pruning_suggested"] = True
            if signals.retrieval_diversify:
                trace["would_apply_diff"]["v1_log_only"]["retrieval_diversify"] = True
            if signals.council_diversity_min != 1:
                trace["would_apply_diff"]["v1_log_only"]["council_diversity_min"] = signals.council_diversity_min
            if signals.confidence_penalty_delta != 0.0:
                trace["would_apply_diff"]["v1_log_only"]["confidence_penalty_delta"] = signals.confidence_penalty_delta
            if signals.exploration_bias_delta != 0.0:
                trace["would_apply_diff"]["v1_log_only"]["exploration_bias_delta"] = signals.exploration_bias_delta
        
        return trace
    
    def _empty_output(self, mode: str) -> BioPriorOutput:
        """Create empty output for disabled/off mode."""
        return BioPriorOutput(
            signals=PressureSignals.zero(),
            selected_patterns=[],
            advisory_text="Bio priors disabled or mode=off",
            trace={
                "context_snapshot": {},
                "missing_metrics": [],
                "recent_window_steps": RECENT_WINDOW_STEPS,
                "patterns_scored": 0,
                "patterns_selected": 0,
            },
            mode=mode,
            applied=False,
            applied_fields=[],
        )



