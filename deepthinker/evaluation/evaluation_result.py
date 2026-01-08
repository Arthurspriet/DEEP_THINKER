"""
Data structures for evaluation results.

Enhanced with epistemic risk tracking for DeepThinker 2.0 epistemic hardening.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpistemicRiskScore:
    """
    Quantifies epistemic risk in evaluated content.
    
    Higher scores indicate higher risk of hallucination/ungrounded content.
    Used to cap confidence and penalize scores.
    
    Attributes:
        claim_to_source_ratio: Claims per source (higher = more risk)
        repetition_penalty: Penalty for repeated assertions (0-1)
        confidence_vs_evidence_delta: Gap between confidence and evidence (0-1)
        speculative_density: Fraction of speculative content (0-1)
        overall_risk: Combined risk score (0-1, higher = worse)
        ungrounded_claim_count: Number of ungrounded claims
        source_quality_avg: Average quality of sources used
    """
    claim_to_source_ratio: float = 0.0
    repetition_penalty: float = 0.0
    confidence_vs_evidence_delta: float = 0.0
    speculative_density: float = 0.0
    overall_risk: float = 0.0
    ungrounded_claim_count: int = 0
    source_quality_avg: float = 0.5
    
    def compute_overall_risk(self) -> float:
        """Compute overall risk score from components."""
        weights = {
            "claim_source": 0.30,
            "repetition": 0.15,
            "confidence_gap": 0.25,
            "speculation": 0.30,
        }
        
        # Normalize claim_to_source_ratio (optimal is 1-2 claims per source)
        source_risk = min(1.0, max(0.0, (self.claim_to_source_ratio - 2) / 8))
        
        self.overall_risk = (
            weights["claim_source"] * source_risk +
            weights["repetition"] * self.repetition_penalty +
            weights["confidence_gap"] * self.confidence_vs_evidence_delta +
            weights["speculation"] * self.speculative_density
        )
        
        # Penalty for low source quality
        if self.source_quality_avg < 0.5:
            self.overall_risk = min(1.0, self.overall_risk + 0.1)
        
        return self.overall_risk
    
    def is_high_risk(self) -> bool:
        """Check if epistemic risk is high."""
        return self.overall_risk > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "claim_to_source_ratio": self.claim_to_source_ratio,
            "repetition_penalty": self.repetition_penalty,
            "confidence_vs_evidence_delta": self.confidence_vs_evidence_delta,
            "speculative_density": self.speculative_density,
            "overall_risk": self.overall_risk,
            "ungrounded_claim_count": self.ungrounded_claim_count,
            "source_quality_avg": self.source_quality_avg,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpistemicRiskScore":
        """Create from dictionary."""
        return cls(
            claim_to_source_ratio=data.get("claim_to_source_ratio", 0.0),
            repetition_penalty=data.get("repetition_penalty", 0.0),
            confidence_vs_evidence_delta=data.get("confidence_vs_evidence_delta", 0.0),
            speculative_density=data.get("speculative_density", 0.0),
            overall_risk=data.get("overall_risk", 0.0),
            ungrounded_claim_count=data.get("ungrounded_claim_count", 0),
            source_quality_avg=data.get("source_quality_avg", 0.5),
        )


@dataclass
class IssueItem:
    """Represents a single issue found in code evaluation."""
    
    severity: str  # "critical", "major", or "minor"
    description: str
    
    def __post_init__(self):
        """Validate severity level."""
        valid_severities = ["critical", "major", "minor"]
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity '{self.severity}'. Must be one of: {valid_severities}"
            )


@dataclass
class EvaluationResult:
    """
    Structured result from code evaluation.
    
    Attributes:
        quality_score: Score from 0-10 indicating code quality
        passed: Whether the code meets minimum quality standards
        issues: List of identified issues categorized by severity
        recommendations: Specific suggestions for improvement
        strengths: Positive aspects of the code
        raw_output: Original LLM response text
        metrics: Optional metric results from execution
        missing_info: Information that is missing to complete evaluation
        questions: Unresolved questions that need investigation
        data_needs: Data or evidence needed to improve confidence
        confidence_score: Confidence in the evaluation (0-1)
        critical_missing: Whether critical information is missing
    """
    
    quality_score: float
    passed: bool
    issues: List[IssueItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    raw_output: str = ""
    metrics: Optional["MetricResult"] = None
    # New fields for iteration control
    missing_info: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    data_needs: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    critical_missing: bool = False
    # Epistemic risk tracking (DeepThinker 2.0)
    epistemic_risk: Optional[EpistemicRiskScore] = None
    grounded_claim_ratio: float = 1.0
    phase_contamination_score: float = 0.0
    
    def __post_init__(self):
        """Validate and clamp quality score."""
        # Clamp score to valid range instead of raising error
        if self.quality_score > 10:
            self.quality_score = 10.0
        elif self.quality_score < 0:
            self.quality_score = 0.0
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.severity == "critical" for issue in self.issues)
    
    def summary(self) -> str:
        """Generate a human-readable summary of the evaluation."""
        lines = [
            f"Quality Score: {self.quality_score}/10",
            f"Confidence: {self.confidence_score:.0%}",
            f"Status: {'PASSED' if self.passed else 'NEEDS IMPROVEMENT'}",
            f"Issues: {len(self.issues)} total "
            f"({sum(1 for i in self.issues if i.severity == 'critical')} critical, "
            f"{sum(1 for i in self.issues if i.severity == 'major')} major, "
            f"{sum(1 for i in self.issues if i.severity == 'minor')} minor)",
        ]
        
        # Epistemic quality metrics
        if self.epistemic_risk is not None:
            lines.append(f"\n--- Epistemic Quality ---")
            lines.append(f"Epistemic Risk: {self.epistemic_risk.overall_risk:.2f}")
            lines.append(f"Grounded Claims: {self.grounded_claim_ratio:.0%}")
            if self.epistemic_risk.is_high_risk():
                lines.append("⚠️ HIGH EPISTEMIC RISK: Claims may be poorly grounded")
            if self.epistemic_risk.ungrounded_claim_count > 0:
                lines.append(f"Ungrounded Facts: {self.epistemic_risk.ungrounded_claim_count}")
            if self.epistemic_risk.speculative_density > 0.3:
                lines.append(f"Speculative Content: {self.epistemic_risk.speculative_density:.0%}")
        
        if self.phase_contamination_score > 0.3:
            lines.append(f"\n⚠️ Phase Contamination: {self.phase_contamination_score:.2f}")
        
        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.description}")
        
        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        
        if self.strengths:
            lines.append("\nStrengths:")
            for strength in self.strengths:
                lines.append(f"  + {strength}")
        
        if self.missing_info:
            lines.append("\nMissing Information:")
            for info in self.missing_info:
                lines.append(f"  ? {info}")
        
        if self.questions:
            lines.append("\nOutstanding Questions:")
            for q in self.questions:
                lines.append(f"  ? {q}")
        
        if self.data_needs:
            lines.append("\nData Needs:")
            for need in self.data_needs:
                lines.append(f"  > {need}")
        
        if self.critical_missing:
            lines.append("\n⚠️ CRITICAL: Missing information prevents full evaluation")
        
        return "\n".join(lines)
    
    def apply_epistemic_risk_penalty(self) -> None:
        """
        Apply epistemic risk penalty to quality score and confidence.
        
        High epistemic risk caps confidence and reduces quality score.
        """
        if self.epistemic_risk is None:
            return
        
        risk = self.epistemic_risk.overall_risk
        
        # Cap confidence based on risk
        if risk > 0.7:
            self.confidence_score = min(self.confidence_score, 0.5)
        elif risk > 0.5:
            self.confidence_score = min(self.confidence_score, 0.7)
        elif risk > 0.3:
            self.confidence_score = min(self.confidence_score, 0.85)
        
        # Apply quality score penalty (max 2 points)
        penalty = min(2.0, risk * 3.0)
        self.quality_score = max(0.0, self.quality_score - penalty)
        
        # Re-evaluate pass status
        self.passed = self.quality_score >= 7.0 and self.confidence_score >= 0.5
    
    def has_unresolved_issues(self) -> bool:
        """Check if there are unresolved questions or missing info."""
        return bool(self.missing_info or self.questions or self.critical_missing)
    
    def iteration_should_continue(self) -> bool:
        """
        Check if another iteration is warranted based on evaluation state.
        
        Returns True if:
        - Critical information is missing
        - Low confidence (< 0.7)
        - Unresolved questions exist
        """
        if self.critical_missing:
            return True
        if self.confidence_score < 0.7:
            return True
        if self.questions:
            return True
        return False


@dataclass
class CombinedEvaluationResult(EvaluationResult):
    """
    Extended evaluation result that combines LLM quality assessment with metric performance.
    
    Attributes:
        All attributes from EvaluationResult, plus:
        metric_weight: Weight given to metrics in combined score (0-1)
        combined_score: Weighted combination of quality_score and metric performance
    """
    
    metric_weight: float = 0.5
    combined_score: Optional[float] = None
    
    def compute_combined_score(self) -> float:
        """
        Compute combined score from LLM quality and metrics.
        
        Formula: combined = (1 - weight) * llm_score + weight * metric_score
        
        Returns:
            Combined score on 0-10 scale
        """
        if self.metrics is None or not hasattr(self.metrics, "normalized_score"):
            # Fall back to LLM score if no metrics available
            return self.quality_score
        
        metric_score = self.metrics.normalized_score()
        self.combined_score = (
            (1 - self.metric_weight) * self.quality_score +
            self.metric_weight * metric_score
        )
        return self.combined_score
    
    def summary(self) -> str:
        """Generate summary including combined score."""
        base_summary = super().summary()
        
        if self.metrics is not None:
            lines = [base_summary]
            lines.append(f"\nMetric Performance: {self.metrics.normalized_score():.1f}/10")
            
            if self.combined_score is not None:
                lines.append(
                    f"Combined Score (weight={self.metric_weight}): "
                    f"{self.combined_score:.1f}/10"
                )
            
            return "\n".join(lines)
        
        return base_summary

