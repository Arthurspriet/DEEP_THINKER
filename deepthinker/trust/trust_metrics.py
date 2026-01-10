"""
Trust Metrics for DeepThinker.

Provides trust and confidence metrics for missions:
- TrustScore: Aggregate trust score with explanation
- TrustCalculator: Computes trust from multiple signals
- TrustExplanation: Human-readable trust explanation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import TrustConfig, get_trust_config

logger = logging.getLogger(__name__)


@dataclass
class TrustExplanation:
    """
    Explanation for trust score.
    
    Provides human-readable factors affecting trust.
    """
    confidence_factors: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_factors": self.confidence_factors,
            "uncertainty_factors": self.uncertainty_factors,
            "recommendations": self.recommendations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustExplanation":
        return cls(
            confidence_factors=data.get("confidence_factors", []),
            uncertainty_factors=data.get("uncertainty_factors", []),
            recommendations=data.get("recommendations", []),
        )


@dataclass
class TrustScore:
    """
    Trust score with component breakdown.
    
    Attributes:
        # Core metrics
        confidence_calibration: Judge agreement vs contradiction rate
        epistemic_uncertainty: From ClaimGraph (contradictions)
        memory_reliance_ratio: Memory tokens / total context tokens
        tool_reliance_ratio: Tool output tokens / total output tokens
        
        # EvidenceObject signals (when web evidence exists)
        evidence_recency_score: How recent are sources?
        evidence_diversity_score: How diverse are sources?
        
        # Aggregate
        overall_trust: Final trust score [0, 1]
        explanation: Human-readable explanation
        
        # Metadata
        mission_id: Mission identifier
        computed_at: When computed
    """
    
    # Core metrics (all in [0, 1])
    confidence_calibration: float = 0.5
    epistemic_uncertainty: float = 0.5
    memory_reliance_ratio: float = 0.0
    tool_reliance_ratio: float = 0.0
    
    # EvidenceObject signals
    evidence_recency_score: float = 0.5
    evidence_diversity_score: float = 0.5
    
    # Aggregate
    overall_trust: float = 0.5
    explanation: TrustExplanation = field(default_factory=TrustExplanation)
    
    # Metadata
    mission_id: str = ""
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    def compute_overall(self, config: TrustConfig) -> float:
        """
        Compute overall trust as weighted combination.
        
        Higher values for positive signals, invert uncertainties.
        """
        # Positive contributions (higher = more trust)
        positive = (
            config.confidence_calibration_weight * self.confidence_calibration +
            config.evidence_recency_weight * self.evidence_recency_score +
            config.evidence_diversity_weight * self.evidence_diversity_score
        )
        
        # Negative contributions (higher uncertainty = less trust)
        negative = (
            config.epistemic_uncertainty_weight * self.epistemic_uncertainty
        )
        
        # Neutral contributions (informational, not penalized)
        # Memory and tool reliance are tracked but don't directly affect trust
        
        # Combine (normalize to [0, 1])
        raw = positive - negative
        
        # Add small contribution from reliance (excessive reliance slightly lowers trust)
        if self.memory_reliance_ratio > 0.5:
            raw -= config.memory_reliance_weight * (self.memory_reliance_ratio - 0.5)
        if self.tool_reliance_ratio > 0.7:
            raw -= config.tool_reliance_weight * (self.tool_reliance_ratio - 0.7)
        
        self.overall_trust = max(0.0, min(1.0, raw))
        return self.overall_trust
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_calibration": self.confidence_calibration,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "memory_reliance_ratio": self.memory_reliance_ratio,
            "tool_reliance_ratio": self.tool_reliance_ratio,
            "evidence_recency_score": self.evidence_recency_score,
            "evidence_diversity_score": self.evidence_diversity_score,
            "overall_trust": self.overall_trust,
            "explanation": self.explanation.to_dict(),
            "mission_id": self.mission_id,
            "computed_at": self.computed_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustScore":
        computed_at = data.get("computed_at")
        if isinstance(computed_at, str):
            computed_at = datetime.fromisoformat(computed_at)
        elif computed_at is None:
            computed_at = datetime.utcnow()
        
        explanation_data = data.get("explanation", {})
        explanation = TrustExplanation.from_dict(explanation_data)
        
        return cls(
            confidence_calibration=data.get("confidence_calibration", 0.5),
            epistemic_uncertainty=data.get("epistemic_uncertainty", 0.5),
            memory_reliance_ratio=data.get("memory_reliance_ratio", 0.0),
            tool_reliance_ratio=data.get("tool_reliance_ratio", 0.0),
            evidence_recency_score=data.get("evidence_recency_score", 0.5),
            evidence_diversity_score=data.get("evidence_diversity_score", 0.5),
            overall_trust=data.get("overall_trust", 0.5),
            explanation=explanation,
            mission_id=data.get("mission_id", ""),
            computed_at=computed_at,
        )


class TrustCalculator:
    """
    Computes trust score from multiple signals.
    
    Incorporates:
    - JudgeEnsemble (disagreement)
    - ClaimGraph (contradictions)
    - EvidenceObject (recency, diversity)
    - Memory injection logs
    - Tool usage
    
    Usage:
        calculator = TrustCalculator()
        
        trust = calculator.compute(
            judge_result=judge_result,
            claim_graph=claim_graph,
            evidence_objects=evidence_objects,
        )
        
        print(f"Trust: {trust.overall_trust:.2f}")
        print(trust.explanation)
    """
    
    def __init__(self, config: Optional[TrustConfig] = None):
        """
        Initialize the calculator.
        
        Args:
            config: Optional TrustConfig
        """
        self.config = config or get_trust_config()
    
    def compute(
        self,
        mission_id: str = "",
        # JudgeEnsemble signals
        judge_disagreement: float = 0.0,
        judge_scores: Optional[List[float]] = None,
        # ClaimGraph signals
        total_claims: int = 0,
        contradictions: int = 0,
        ungrounded_claims: int = 0,
        # EvidenceObject signals
        evidence_objects: Optional[List[Dict[str, Any]]] = None,
        # Memory signals
        memory_tokens_injected: int = 0,
        total_context_tokens: int = 1,
        # Tool signals
        tool_output_tokens: int = 0,
        total_output_tokens: int = 1,
    ) -> TrustScore:
        """
        Compute trust score from all available signals.
        
        Args:
            mission_id: Mission identifier
            judge_disagreement: Disagreement rate from judge ensemble
            judge_scores: Individual judge scores
            total_claims: Total claims in ClaimGraph
            contradictions: Number of contradictions
            ungrounded_claims: Number of ungrounded claims
            evidence_objects: List of EvidenceObject dicts
            memory_tokens_injected: Tokens from memory injection
            total_context_tokens: Total context tokens
            tool_output_tokens: Tokens from tool outputs
            total_output_tokens: Total output tokens
            
        Returns:
            TrustScore with computed metrics
        """
        if not self.config.enabled:
            return TrustScore(mission_id=mission_id)
        
        score = TrustScore(mission_id=mission_id)
        explanation = TrustExplanation()
        
        # 1. Confidence calibration (from judge agreement)
        score.confidence_calibration = 1.0 - min(1.0, judge_disagreement)
        
        if judge_scores and len(judge_scores) > 1:
            # Additional calibration from score variance
            mean_score = sum(judge_scores) / len(judge_scores)
            variance = sum((s - mean_score) ** 2 for s in judge_scores) / len(judge_scores)
            if variance < 0.01:
                explanation.confidence_factors.append("Judges highly consistent")
            elif variance > 0.1:
                explanation.uncertainty_factors.append("High judge variance")
        
        # 2. Epistemic uncertainty (from ClaimGraph)
        if total_claims > 0:
            contradiction_rate = contradictions / total_claims
            ungrounded_rate = ungrounded_claims / total_claims
            score.epistemic_uncertainty = min(1.0, contradiction_rate + 0.5 * ungrounded_rate)
            
            if contradiction_rate > 0.1:
                explanation.uncertainty_factors.append(
                    f"High contradiction rate ({contradiction_rate:.0%})"
                )
                explanation.recommendations.append("Review contradictory claims")
            
            if ungrounded_rate > 0.2:
                explanation.uncertainty_factors.append(
                    f"Many ungrounded claims ({ungrounded_rate:.0%})"
                )
                explanation.recommendations.append("Seek additional evidence")
        
        # 3. Evidence signals (from EvidenceObject)
        if evidence_objects:
            recency = self._compute_recency(evidence_objects)
            diversity = self._compute_diversity(evidence_objects)
            
            score.evidence_recency_score = recency
            score.evidence_diversity_score = diversity
            
            if recency > 0.7:
                explanation.confidence_factors.append("Evidence is recent")
            elif recency < 0.3:
                explanation.uncertainty_factors.append("Evidence may be outdated")
                explanation.recommendations.append("Check for recent updates")
            
            if diversity > 0.7:
                explanation.confidence_factors.append("Diverse evidence sources")
            elif diversity < 0.3:
                explanation.uncertainty_factors.append("Limited source diversity")
                explanation.recommendations.append("Seek additional sources")
        
        # 4. Memory reliance
        if total_context_tokens > 0:
            score.memory_reliance_ratio = memory_tokens_injected / total_context_tokens
            
            if score.memory_reliance_ratio > 0.5:
                explanation.uncertainty_factors.append("High memory reliance")
        
        # 5. Tool reliance
        if total_output_tokens > 0:
            score.tool_reliance_ratio = tool_output_tokens / total_output_tokens
        
        # Compute overall and finalize
        score.explanation = explanation
        score.compute_overall(self.config)
        
        # Log
        logger.debug(
            f"[TRUST] Computed for {mission_id}: "
            f"overall={score.overall_trust:.2f}, "
            f"confidence={score.confidence_calibration:.2f}, "
            f"uncertainty={score.epistemic_uncertainty:.2f}"
        )
        
        return score
    
    def _compute_recency(self, evidence_objects: List[Dict[str, Any]]) -> float:
        """
        Compute recency score from evidence objects.
        
        More recent evidence = higher score.
        """
        if not evidence_objects:
            return 0.5
        
        now = datetime.utcnow()
        recency_days = self.config.evidence_recency_days
        
        recency_scores = []
        for ev in evidence_objects:
            # Try to extract timestamp
            timestamp = ev.get("timestamp") or ev.get("date") or ev.get("retrieved_at")
            
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except Exception:
                        timestamp = None
                
                if isinstance(timestamp, datetime):
                    age_days = (now - timestamp).days
                    recency = max(0.0, 1.0 - (age_days / recency_days))
                    recency_scores.append(recency)
        
        if recency_scores:
            return sum(recency_scores) / len(recency_scores)
        
        return 0.5  # Unknown recency
    
    def _compute_diversity(self, evidence_objects: List[Dict[str, Any]]) -> float:
        """
        Compute diversity score from evidence objects.
        
        More unique sources/domains = higher score.
        """
        if not evidence_objects:
            return 0.5
        
        # Extract unique sources
        sources = set()
        domains = set()
        
        for ev in evidence_objects:
            source = ev.get("source") or ev.get("origin")
            url = ev.get("url") or ev.get("uri")
            
            if source:
                sources.add(source)
            
            if url:
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    if parsed.netloc:
                        domains.add(parsed.netloc)
                except Exception:
                    pass
        
        # Diversity based on unique sources
        unique_count = len(sources | domains)
        total_count = len(evidence_objects)
        
        # Perfect diversity if all unique
        diversity = unique_count / max(1, total_count)
        
        # Bonus for having multiple sources
        if unique_count > 3:
            diversity = min(1.0, diversity + 0.1)
        
        return diversity


# Global calculator instance
_calculator: Optional[TrustCalculator] = None


def get_trust_calculator(config: Optional[TrustConfig] = None) -> TrustCalculator:
    """Get global trust calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = TrustCalculator(config=config)
    return _calculator


def reset_trust_calculator() -> None:
    """Reset global calculator (mainly for testing)."""
    global _calculator
    _calculator = None


