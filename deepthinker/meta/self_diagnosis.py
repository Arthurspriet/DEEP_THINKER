"""
Self-Diagnosis for DeepThinker.

Provides meta-analysis at mission end:
- What limited relevance?
- Which decisions hurt?
- What would I do differently next time?

Outputs:
- MissionPostMortem: Structured improvement suggestions
- Parameter tuning hints
- Candidate new features to log
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..trust.trust_metrics import TrustScore
from ..replay.counterfactual_engine import ReplayResult

logger = logging.getLogger(__name__)


@dataclass
class DecisionAnalysis:
    """
    Analysis of a single decision's impact.
    
    Attributes:
        decision_id: Decision identifier
        decision_type: Type of decision
        impact: positive, negative, or neutral
        explanation: Why this was impactful
        regret: Estimated regret (if available)
    """
    decision_id: str
    decision_type: str
    impact: str  # "positive" | "negative" | "neutral"
    explanation: str
    regret: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "impact": self.impact,
            "explanation": self.explanation,
            "regret": self.regret,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionAnalysis":
        return cls(
            decision_id=data.get("decision_id", ""),
            decision_type=data.get("decision_type", ""),
            impact=data.get("impact", "neutral"),
            explanation=data.get("explanation", ""),
            regret=data.get("regret", 0.0),
        )


@dataclass
class MissionPostMortem:
    """
    Post-mortem analysis for a mission.
    
    Answers:
    - What limited relevance?
    - Which decisions hurt?
    - What would I do differently next time?
    
    Attributes:
        mission_id: Mission identifier
        generated_at: When analysis was generated
        
        # Analysis sections
        what_limited_relevance: Factors that limited relevance
        which_decisions_hurt: Decisions with negative impact
        what_would_i_do_differently: Suggestions for next time
        
        # Tuning suggestions
        parameter_tuning_hints: Suggested parameter changes
        
        # Feature requests
        feature_requests: Capabilities that would have helped
        
        # Trust summary
        final_trust_score: Trust score at mission end
        
        # Replay summary (if available)
        estimated_total_regret: Total regret from replay
    """
    mission_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Analysis sections
    what_limited_relevance: List[str] = field(default_factory=list)
    which_decisions_hurt: List[DecisionAnalysis] = field(default_factory=list)
    what_would_i_do_differently: List[str] = field(default_factory=list)
    
    # Tuning suggestions
    parameter_tuning_hints: Dict[str, float] = field(default_factory=dict)
    
    # Feature requests
    feature_requests: List[str] = field(default_factory=list)
    
    # Summaries
    final_trust_score: float = 0.5
    estimated_total_regret: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "generated_at": self.generated_at.isoformat(),
            "what_limited_relevance": self.what_limited_relevance,
            "which_decisions_hurt": [d.to_dict() for d in self.which_decisions_hurt],
            "what_would_i_do_differently": self.what_would_i_do_differently,
            "parameter_tuning_hints": self.parameter_tuning_hints,
            "feature_requests": self.feature_requests,
            "final_trust_score": self.final_trust_score,
            "estimated_total_regret": self.estimated_total_regret,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionPostMortem":
        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at)
        elif generated_at is None:
            generated_at = datetime.utcnow()
        
        return cls(
            mission_id=data.get("mission_id", ""),
            generated_at=generated_at,
            what_limited_relevance=data.get("what_limited_relevance", []),
            which_decisions_hurt=[
                DecisionAnalysis.from_dict(d)
                for d in data.get("which_decisions_hurt", [])
            ],
            what_would_i_do_differently=data.get("what_would_i_do_differently", []),
            parameter_tuning_hints=data.get("parameter_tuning_hints", {}),
            feature_requests=data.get("feature_requests", []),
            final_trust_score=data.get("final_trust_score", 0.5),
            estimated_total_regret=data.get("estimated_total_regret", 0.0),
        )
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"# Mission Post-Mortem: {self.mission_id}",
            f"Generated: {self.generated_at.isoformat()}",
            "",
            f"## Trust Score: {self.final_trust_score:.2f}",
            f"## Estimated Regret: {self.estimated_total_regret:.2f}",
            "",
        ]
        
        if self.what_limited_relevance:
            lines.append("## What Limited Relevance?")
            for item in self.what_limited_relevance:
                lines.append(f"- {item}")
            lines.append("")
        
        if self.which_decisions_hurt:
            lines.append("## Which Decisions Hurt?")
            for d in self.which_decisions_hurt:
                lines.append(f"- [{d.decision_type}] {d.explanation}")
            lines.append("")
        
        if self.what_would_i_do_differently:
            lines.append("## What Would I Do Differently?")
            for item in self.what_would_i_do_differently:
                lines.append(f"- {item}")
            lines.append("")
        
        if self.parameter_tuning_hints:
            lines.append("## Parameter Tuning Hints")
            for param, value in self.parameter_tuning_hints.items():
                lines.append(f"- {param}: {value}")
            lines.append("")
        
        return "\n".join(lines)


class SelfDiagnosisEngine:
    """
    Generates post-mortem analysis for missions.
    
    Uses:
    - Trust metrics to identify uncertainty sources
    - Replay results to identify regretted decisions
    - Decision logs to trace causal chains
    
    Usage:
        engine = SelfDiagnosisEngine()
        
        postmortem = engine.analyze_mission(
            mission_id="abc-123",
            trust_score=trust_score,
            replay_result=replay_result,
        )
        
        print(postmortem.summary())
    """
    
    def __init__(self, output_dir: str = "kb/missions"):
        """
        Initialize the engine.
        
        Args:
            output_dir: Directory for storing post-mortems
        """
        self.output_dir = Path(output_dir)
    
    def analyze_mission(
        self,
        mission_id: str,
        trust_score: Optional[TrustScore] = None,
        replay_result: Optional[ReplayResult] = None,
        decisions: Optional[List[Dict[str, Any]]] = None,
        final_score: float = 0.5,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> MissionPostMortem:
        """
        Generate post-mortem analysis for a mission.
        
        Args:
            mission_id: Mission identifier
            trust_score: Trust metrics at mission end
            replay_result: Counterfactual replay results
            decisions: List of decision records
            final_score: Final mission score
            constraints: Mission constraints
            
        Returns:
            MissionPostMortem with analysis
        """
        postmortem = MissionPostMortem(mission_id=mission_id)
        
        # 1. Analyze what limited relevance
        postmortem.what_limited_relevance = self._analyze_relevance_limiters(
            trust_score, final_score, constraints
        )
        
        # 2. Identify decisions that hurt
        postmortem.which_decisions_hurt = self._identify_hurtful_decisions(
            replay_result, decisions
        )
        
        # 3. Generate improvement suggestions
        postmortem.what_would_i_do_differently = self._generate_suggestions(
            trust_score, replay_result, decisions
        )
        
        # 4. Suggest parameter tuning
        postmortem.parameter_tuning_hints = self._suggest_parameters(
            replay_result, final_score
        )
        
        # 5. Identify feature requests
        postmortem.feature_requests = self._identify_feature_requests(
            trust_score, decisions
        )
        
        # 6. Set summaries
        if trust_score:
            postmortem.final_trust_score = trust_score.overall_trust
        
        if replay_result:
            postmortem.estimated_total_regret = replay_result.estimated_regret
        
        # Store postmortem
        self._store_postmortem(postmortem)
        
        logger.info(
            f"[SELF_DIAGNOSIS] Generated postmortem for {mission_id}: "
            f"trust={postmortem.final_trust_score:.2f}, "
            f"regret={postmortem.estimated_total_regret:.2f}, "
            f"{len(postmortem.which_decisions_hurt)} hurtful decisions"
        )
        
        return postmortem
    
    def _analyze_relevance_limiters(
        self,
        trust_score: Optional[TrustScore],
        final_score: float,
        constraints: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Identify factors that limited relevance."""
        limiters = []
        
        if trust_score:
            # High epistemic uncertainty
            if trust_score.epistemic_uncertainty > 0.3:
                limiters.append(
                    f"High epistemic uncertainty ({trust_score.epistemic_uncertainty:.0%})"
                )
            
            # Low confidence calibration
            if trust_score.confidence_calibration < 0.6:
                limiters.append(
                    f"Poor judge agreement ({trust_score.confidence_calibration:.0%})"
                )
            
            # Evidence issues
            if trust_score.evidence_recency_score < 0.4:
                limiters.append("Outdated evidence sources")
            if trust_score.evidence_diversity_score < 0.4:
                limiters.append("Limited source diversity")
            
            # Reliance issues
            if trust_score.memory_reliance_ratio > 0.5:
                limiters.append(
                    f"High memory reliance ({trust_score.memory_reliance_ratio:.0%})"
                )
        
        if constraints:
            # Time pressure
            if constraints.get("time_pressure", False):
                limiters.append("Time pressure limited exploration")
            
            # Resource constraints
            if constraints.get("model_tier") == "SMALL":
                limiters.append("Limited to small models")
        
        if final_score < 0.5:
            limiters.append("Final score below acceptable threshold")
        
        return limiters
    
    def _identify_hurtful_decisions(
        self,
        replay_result: Optional[ReplayResult],
        decisions: Optional[List[Dict[str, Any]]],
    ) -> List[DecisionAnalysis]:
        """Identify decisions that had negative impact."""
        hurtful = []
        
        if replay_result and replay_result.decision_replays:
            for replay in replay_result.decision_replays:
                if replay.action_changed:
                    regret = max(0, replay.counterfactual_reward - replay.original_reward)
                    if regret > 0.1:
                        hurtful.append(DecisionAnalysis(
                            decision_id=replay.decision_id,
                            decision_type=replay.decision_type,
                            impact="negative",
                            explanation=(
                                f"Would have chosen {replay.counterfactual_action} "
                                f"instead of {replay.original_action}"
                            ),
                            regret=regret,
                        ))
        
        if decisions:
            for decision in decisions:
                confidence = decision.get("confidence", 1.0)
                
                # Low confidence decisions are potentially hurtful
                if confidence < 0.5:
                    hurtful.append(DecisionAnalysis(
                        decision_id=decision.get("decision_id", ""),
                        decision_type=decision.get("decision_type", ""),
                        impact="negative",
                        explanation=f"Low confidence decision ({confidence:.0%})",
                    ))
        
        # Sort by regret (highest first)
        hurtful.sort(key=lambda d: d.regret, reverse=True)
        
        return hurtful[:10]  # Top 10
    
    def _generate_suggestions(
        self,
        trust_score: Optional[TrustScore],
        replay_result: Optional[ReplayResult],
        decisions: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        """Generate improvement suggestions for next time."""
        suggestions = []
        
        if trust_score:
            if trust_score.evidence_recency_score < 0.5:
                suggestions.append("Prioritize recent sources in evidence gathering")
            
            if trust_score.evidence_diversity_score < 0.5:
                suggestions.append("Seek more diverse evidence sources")
            
            if trust_score.epistemic_uncertainty > 0.3:
                suggestions.append("Invest more in resolving contradictions")
        
        if replay_result:
            if replay_result.estimated_regret > 0.5:
                suggestions.append("Consider more exploration in routing decisions")
            
            by_type = replay_result.regret_by_type
            if by_type:
                worst_type = max(by_type, key=by_type.get)
                suggestions.append(f"Review {worst_type} decision policy")
        
        if decisions:
            # Analyze decision patterns
            low_conf_count = sum(1 for d in decisions if d.get("confidence", 1.0) < 0.6)
            if low_conf_count > 5:
                suggestions.append("Many low-confidence decisions - consider escalating sooner")
        
        return suggestions
    
    def _suggest_parameters(
        self,
        replay_result: Optional[ReplayResult],
        final_score: float,
    ) -> Dict[str, float]:
        """Suggest parameter adjustments."""
        hints = {}
        
        if replay_result:
            if replay_result.estimated_regret > 0.5:
                hints["bandit_exploration_bonus"] = 1.2  # Increase exploration
            
            if replay_result.decisions_changed > 0.3 * replay_result.total_decisions:
                hints["min_trials_before_exploit"] = 15  # Need more exploration
        
        if final_score < 0.6:
            hints["stop_threshold"] = 0.65  # Raise stop threshold
        
        return hints
    
    def _identify_feature_requests(
        self,
        trust_score: Optional[TrustScore],
        decisions: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        """Identify capabilities that would have helped."""
        requests = []
        
        if trust_score:
            if trust_score.epistemic_uncertainty > 0.4:
                requests.append("Automatic contradiction resolution")
            
            if trust_score.evidence_diversity_score < 0.3:
                requests.append("Multi-source evidence aggregation")
        
        if decisions:
            # Check for repeated escalations
            escalations = [d for d in decisions if "escalation" in d.get("decision_type", "").lower()]
            if len(escalations) > 3:
                requests.append("Smarter initial model selection")
        
        return requests
    
    def _store_postmortem(self, postmortem: MissionPostMortem) -> bool:
        """Store postmortem to disk."""
        try:
            mission_dir = self.output_dir / postmortem.mission_id
            mission_dir.mkdir(parents=True, exist_ok=True)
            
            postmortem_file = mission_dir / "postmortem.json"
            with open(postmortem_file, "w") as f:
                json.dump(postmortem.to_dict(), f, indent=2)
            
            return True
            
        except Exception as e:
            logger.warning(f"[SELF_DIAGNOSIS] Failed to store postmortem: {e}")
            return False
    
    def load_postmortem(self, mission_id: str) -> Optional[MissionPostMortem]:
        """Load postmortem from disk."""
        try:
            postmortem_file = self.output_dir / mission_id / "postmortem.json"
            
            if postmortem_file.exists():
                with open(postmortem_file, "r") as f:
                    data = json.load(f)
                return MissionPostMortem.from_dict(data)
            
            return None
            
        except Exception as e:
            logger.warning(f"[SELF_DIAGNOSIS] Failed to load postmortem: {e}")
            return None


# Global engine instance
_engine: Optional[SelfDiagnosisEngine] = None


def get_self_diagnosis_engine(output_dir: str = "kb/missions") -> SelfDiagnosisEngine:
    """Get global self-diagnosis engine instance."""
    global _engine
    if _engine is None:
        _engine = SelfDiagnosisEngine(output_dir=output_dir)
    return _engine


def reset_self_diagnosis_engine() -> None:
    """Reset global engine (mainly for testing)."""
    global _engine
    _engine = None




