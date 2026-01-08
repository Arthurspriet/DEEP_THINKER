"""
Split Reconnaissance Phases for Epistemic Hardening.

Splits the single Recon phase into two explicit phases:
1. ReconExploration: Speculation and hypothesis generation (no epistemic gating)
2. ReconGrounding: Evidence collection and grounding (strict epistemic gating)

This separation ensures that exploration is allowed to be speculative,
while grounding requires evidence before proceeding.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..epistemics.claim_validator import Claim, Source

logger = logging.getLogger(__name__)


class ReconPhaseType(str, Enum):
    """Types of reconnaissance sub-phases."""
    EXPLORATION = "recon_exploration"
    GROUNDING = "recon_grounding"


@dataclass
class ReconExplorationOutput:
    """
    Output from the exploration sub-phase.
    
    This phase is allowed to speculate and generate hypotheses without
    requiring sources or evidence. The output will be filtered before
    passing to the grounding phase.
    
    Attributes:
        candidate_focus_areas: Potential areas to investigate
        research_questions: Questions to answer
        hypotheses: Hypotheses to test
        landscape_summary: High-level summary of the problem space
        assumptions: Explicit assumptions being made
        data_needs: Data that would help validate hypotheses
    """
    candidate_focus_areas: List[str] = field(default_factory=list)
    research_questions: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    landscape_summary: str = ""
    assumptions: List[str] = field(default_factory=list)
    data_needs: List[str] = field(default_factory=list)
    
    # Maximum allowed items to prevent sprawl
    MAX_FOCUS_AREAS = 10
    MAX_QUESTIONS = 15
    MAX_HYPOTHESES = 10
    
    def validate_limits(self) -> List[str]:
        """
        Check if outputs exceed allowed limits.
        
        Returns:
            List of violation messages
        """
        violations = []
        
        if len(self.candidate_focus_areas) > self.MAX_FOCUS_AREAS:
            violations.append(
                f"Too many focus areas: {len(self.candidate_focus_areas)} > {self.MAX_FOCUS_AREAS}"
            )
        
        if len(self.research_questions) > self.MAX_QUESTIONS:
            violations.append(
                f"Too many questions: {len(self.research_questions)} > {self.MAX_QUESTIONS}"
            )
        
        if len(self.hypotheses) > self.MAX_HYPOTHESES:
            violations.append(
                f"Too many hypotheses: {len(self.hypotheses)} > {self.MAX_HYPOTHESES}"
            )
        
        return violations
    
    def truncate_to_limits(self) -> "ReconExplorationOutput":
        """
        Truncate outputs to respect limits.
        
        Returns:
            New output with truncated lists
        """
        return ReconExplorationOutput(
            candidate_focus_areas=self.candidate_focus_areas[:self.MAX_FOCUS_AREAS],
            research_questions=self.research_questions[:self.MAX_QUESTIONS],
            hypotheses=self.hypotheses[:self.MAX_HYPOTHESES],
            landscape_summary=self.landscape_summary,
            assumptions=self.assumptions,
            data_needs=self.data_needs,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_focus_areas": self.candidate_focus_areas,
            "research_questions": self.research_questions,
            "hypotheses": self.hypotheses,
            "landscape_summary": self.landscape_summary,
            "assumptions": self.assumptions,
            "data_needs": self.data_needs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconExplorationOutput":
        """Create from dictionary."""
        return cls(
            candidate_focus_areas=data.get("candidate_focus_areas", []),
            research_questions=data.get("research_questions", []),
            hypotheses=data.get("hypotheses", []),
            landscape_summary=data.get("landscape_summary", ""),
            assumptions=data.get("assumptions", []),
            data_needs=data.get("data_needs", []),
        )


@dataclass
class ReconGroundingOutput:
    """
    Output from the grounding sub-phase.
    
    This phase requires evidence for all claims. Epistemic gates are
    enforced here. Only grounded claims are promoted to downstream phases.
    
    Attributes:
        grounded_claims: Claims with proper source backing
        approved_focus_areas: Focus areas that passed grounding (max 5)
        sources: Sources used to back claims
        validated_hypotheses: Hypotheses with supporting evidence
        rejected_hypotheses: Hypotheses without evidence (for transparency)
        source_quality_score: Average quality of sources
    """
    grounded_claims: List["Claim"] = field(default_factory=list)
    approved_focus_areas: List[str] = field(default_factory=list)
    sources: List["Source"] = field(default_factory=list)
    validated_hypotheses: List[str] = field(default_factory=list)
    rejected_hypotheses: List[str] = field(default_factory=list)
    source_quality_score: float = 0.0
    
    # Hard limit on approved focus areas
    MAX_APPROVED_FOCUS_AREAS = 5
    
    def enforce_focus_area_limit(self) -> List[str]:
        """
        Enforce the focus area limit.
        
        Returns:
            List of focus areas that were dropped
        """
        if len(self.approved_focus_areas) <= self.MAX_APPROVED_FOCUS_AREAS:
            return []
        
        dropped = self.approved_focus_areas[self.MAX_APPROVED_FOCUS_AREAS:]
        self.approved_focus_areas = self.approved_focus_areas[:self.MAX_APPROVED_FOCUS_AREAS]
        
        logger.warning(
            f"[RECON GROUNDING] Dropped {len(dropped)} focus areas to enforce limit: {dropped}"
        )
        
        return dropped
    
    def get_grounding_ratio(self) -> float:
        """
        Calculate the grounding ratio.
        
        Returns:
            Ratio of grounded to total claims
        """
        if not self.grounded_claims:
            return 0.0
        
        grounded_count = sum(1 for c in self.grounded_claims if c.is_grounded())
        return grounded_count / len(self.grounded_claims)
    
    def passes_epistemic_gate(self, min_sources: int = 3, min_grounding_ratio: float = 0.6) -> tuple:
        """
        Check if output passes epistemic gates.
        
        Args:
            min_sources: Minimum number of sources required
            min_grounding_ratio: Minimum ratio of grounded claims
            
        Returns:
            Tuple of (passes: bool, reason: str)
        """
        if len(self.sources) < min_sources:
            return False, f"Insufficient sources: {len(self.sources)} < {min_sources}"
        
        ratio = self.get_grounding_ratio()
        if ratio < min_grounding_ratio:
            return False, f"Low grounding ratio: {ratio:.2f} < {min_grounding_ratio}"
        
        if not self.approved_focus_areas:
            return False, "No approved focus areas"
        
        return True, "Epistemic gates passed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "grounded_claims": [
                c.to_dict() if hasattr(c, "to_dict") else str(c) 
                for c in self.grounded_claims
            ],
            "approved_focus_areas": self.approved_focus_areas,
            "sources": [
                s.to_dict() if hasattr(s, "to_dict") else str(s)
                for s in self.sources
            ],
            "validated_hypotheses": self.validated_hypotheses,
            "rejected_hypotheses": self.rejected_hypotheses,
            "source_quality_score": self.source_quality_score,
        }


@dataclass
class SplitReconConfig:
    """
    Configuration for split reconnaissance phases.
    
    Attributes:
        exploration_time_ratio: Fraction of recon time for exploration (0-1)
        min_sources_for_grounding: Minimum sources required
        min_grounding_ratio: Minimum ratio of grounded claims
        max_focus_areas: Maximum approved focus areas
        allow_hypothesis_passthrough: Allow unvalidated hypotheses as questions
    """
    exploration_time_ratio: float = 0.3
    min_sources_for_grounding: int = 5
    min_grounding_ratio: float = 0.6
    max_focus_areas: int = 5
    allow_hypothesis_passthrough: bool = True
    
    def get_exploration_budget(self, total_recon_seconds: float) -> float:
        """Get time budget for exploration phase."""
        return total_recon_seconds * self.exploration_time_ratio
    
    def get_grounding_budget(self, total_recon_seconds: float) -> float:
        """Get time budget for grounding phase."""
        return total_recon_seconds * (1 - self.exploration_time_ratio)


def create_exploration_phase_from_recon(
    original_phase_name: str,
    original_description: str,
    time_budget_seconds: Optional[float] = None,
    config: Optional[SplitReconConfig] = None
) -> Dict[str, Any]:
    """
    Create exploration sub-phase definition from original recon phase.
    
    Args:
        original_phase_name: Name of the original recon phase
        original_description: Description of the original phase
        time_budget_seconds: Total time budget for recon
        config: Split recon configuration
        
    Returns:
        Phase definition dictionary
    """
    config = config or SplitReconConfig()
    
    exploration_budget = None
    if time_budget_seconds:
        exploration_budget = config.get_exploration_budget(time_budget_seconds)
    
    return {
        "name": f"{original_phase_name}_exploration",
        "phase_type": ReconPhaseType.EXPLORATION.value,
        "description": (
            f"Exploration sub-phase of {original_phase_name}. "
            f"Goal: Generate hypotheses, identify focus areas, and map the problem space. "
            f"Speculation is allowed. No sources required."
        ),
        "original_phase": original_phase_name,
        "time_budget_seconds": exploration_budget,
        "requires_sources": False,
        "epistemic_gating": False,
    }


def create_grounding_phase_from_recon(
    original_phase_name: str,
    original_description: str,
    exploration_output: Optional[ReconExplorationOutput] = None,
    time_budget_seconds: Optional[float] = None,
    config: Optional[SplitReconConfig] = None
) -> Dict[str, Any]:
    """
    Create grounding sub-phase definition from original recon phase.
    
    Args:
        original_phase_name: Name of the original recon phase
        original_description: Description of the original phase
        exploration_output: Output from exploration phase
        time_budget_seconds: Total time budget for recon
        config: Split recon configuration
        
    Returns:
        Phase definition dictionary
    """
    config = config or SplitReconConfig()
    
    grounding_budget = None
    if time_budget_seconds:
        grounding_budget = config.get_grounding_budget(time_budget_seconds)
    
    # Extract focus areas and questions from exploration output
    focus_areas_to_ground = []
    questions_to_answer = []
    
    if exploration_output:
        focus_areas_to_ground = exploration_output.candidate_focus_areas[:config.max_focus_areas]
        questions_to_answer = exploration_output.research_questions
    
    return {
        "name": f"{original_phase_name}_grounding",
        "phase_type": ReconPhaseType.GROUNDING.value,
        "description": (
            f"Grounding sub-phase of {original_phase_name}. "
            f"Goal: Gather evidence to support or refute hypotheses. "
            f"All claims must be grounded with sources."
        ),
        "original_phase": original_phase_name,
        "time_budget_seconds": grounding_budget,
        "requires_sources": True,
        "epistemic_gating": True,
        "focus_areas_to_ground": focus_areas_to_ground,
        "questions_to_answer": questions_to_answer,
        "min_sources": config.min_sources_for_grounding,
        "min_grounding_ratio": config.min_grounding_ratio,
    }


def should_split_recon_phase(phase_name: str) -> bool:
    """
    Determine if a phase should be split into exploration + grounding.
    
    Args:
        phase_name: Name of the phase
        
    Returns:
        True if phase should be split
    """
    phase_lower = phase_name.lower()
    
    # Patterns that indicate a recon phase
    recon_patterns = [
        "recon",
        "reconnaissance",
        "research",
        "explore",
        "discovery",
        "landscape",
    ]
    
    # Exclude patterns - these are already specialized
    exclude_patterns = [
        "grounding",
        "exploration",
        "deep",
        "analysis",
        "synthesis",
    ]
    
    # Check for exclude patterns first
    if any(p in phase_lower for p in exclude_patterns):
        return False
    
    # Check for recon patterns
    return any(p in phase_lower for p in recon_patterns)

