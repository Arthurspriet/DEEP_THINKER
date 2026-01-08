"""
Phase Contracts for Governance Layer.

Defines what each phase can and cannot produce from a governance perspective.
Extends the existing phase_contracts from deepthinker/phases/ with additional
governance-specific constraints.

These contracts enforce:
- Allowed/forbidden output types per phase
- Minimum source requirements
- Maximum speculation ratios
- Structural requirements (e.g., scenario counts)
- Evidence budgets (Epistemic Hardening Phase 3)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvidenceBudget:
    """
    Evidence budget requirements for a phase.
    
    Defines minimum evidence thresholds that must be met before
    a phase can be considered complete.
    
    Attributes:
        min_sources: Minimum number of sources required
        min_grounded_claims: Minimum number of grounded claims required
        enforce: Whether to enforce this budget (block phase completion)
    """
    min_sources: int = 0
    min_grounded_claims: int = 0
    enforce: bool = True
    
    def is_met(self, sources_count: int, grounded_claims_count: int) -> Tuple[bool, str]:
        """
        Check if evidence budget is met.
        
        Args:
            sources_count: Number of sources collected
            grounded_claims_count: Number of grounded claims
            
        Returns:
            Tuple of (is_met: bool, reason: str)
        """
        if sources_count < self.min_sources:
            return False, f"Insufficient sources: {sources_count} < {self.min_sources}"
        
        if grounded_claims_count < self.min_grounded_claims:
            return False, f"Insufficient grounded claims: {grounded_claims_count} < {self.min_grounded_claims}"
        
        return True, "Evidence budget met"
    
    def get_shortfall(self, sources_count: int, grounded_claims_count: int) -> Dict[str, int]:
        """
        Calculate shortfall from budget targets.
        
        Args:
            sources_count: Number of sources collected
            grounded_claims_count: Number of grounded claims
            
        Returns:
            Dictionary with shortfall amounts (0 if met)
        """
        return {
            "sources_needed": max(0, self.min_sources - sources_count),
            "claims_needed": max(0, self.min_grounded_claims - grounded_claims_count),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "min_sources": self.min_sources,
            "min_grounded_claims": self.min_grounded_claims,
            "enforce": self.enforce,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBudget":
        """Create from dictionary."""
        return cls(
            min_sources=data.get("min_sources", 0),
            min_grounded_claims=data.get("min_grounded_claims", 0),
            enforce=data.get("enforce", True),
        )


# Default evidence budgets per phase type
DEFAULT_EVIDENCE_BUDGETS: Dict[str, EvidenceBudget] = {
    "recon_grounding": EvidenceBudget(min_sources=5, min_grounded_claims=0, enforce=True),
    "reconnaissance": EvidenceBudget(min_sources=3, min_grounded_claims=0, enforce=True),
    "research": EvidenceBudget(min_sources=5, min_grounded_claims=0, enforce=True),
    "analysis": EvidenceBudget(min_sources=0, min_grounded_claims=12, enforce=True),
    "deep_analysis": EvidenceBudget(min_sources=0, min_grounded_claims=0, enforce=False),
    "synthesis": EvidenceBudget(min_sources=0, min_grounded_claims=5, enforce=True),
    "implementation": EvidenceBudget(min_sources=0, min_grounded_claims=0, enforce=False),
    "recon_exploration": EvidenceBudget(min_sources=0, min_grounded_claims=0, enforce=False),
}


def get_evidence_budget(phase_name: str) -> EvidenceBudget:
    """
    Get evidence budget for a phase.
    
    Args:
        phase_name: Name of the phase
        
    Returns:
        EvidenceBudget for the phase
    """
    phase_lower = phase_name.lower()
    
    # Direct match
    if phase_lower in DEFAULT_EVIDENCE_BUDGETS:
        return DEFAULT_EVIDENCE_BUDGETS[phase_lower]
    
    # Fuzzy match
    for key, budget in DEFAULT_EVIDENCE_BUDGETS.items():
        if key in phase_lower or phase_lower in key:
            return budget
    
    # Default: no enforcement
    return EvidenceBudget(min_sources=0, min_grounded_claims=0, enforce=False)


@dataclass
class GovernancePhaseContract:
    """
    Governance contract for a single phase.
    
    Attributes:
        phase_name: Name of the phase
        allowed: Output types allowed in this phase
        forbidden: Output types forbidden in this phase
        min_sources: Minimum number of sources required
        max_speculation_ratio: Maximum fraction of speculative content (0-1)
        required_scenario_count: Number of scenarios required (synthesis only)
        min_content_length: Minimum output content length in characters
        requires_web_search: Whether web search is mandatory for facts
        strictness_weight: Base strictness multiplier for this phase
        evidence_budget: Evidence budget requirements (Epistemic Hardening Phase 3)
    """
    
    phase_name: str
    allowed: Set[str] = field(default_factory=set)
    forbidden: Set[str] = field(default_factory=set)
    min_sources: int = 0
    max_speculation_ratio: float = 0.5
    required_scenario_count: Optional[int] = None
    min_content_length: int = 100
    requires_web_search: bool = False
    strictness_weight: float = 1.0
    evidence_budget: Optional[EvidenceBudget] = None
    
    def __post_init__(self):
        """Initialize evidence budget from defaults if not provided."""
        if self.evidence_budget is None:
            self.evidence_budget = get_evidence_budget(self.phase_name)
    
    def check_evidence_budget(
        self,
        sources_count: int,
        grounded_claims_count: int
    ) -> Tuple[bool, str]:
        """
        Check if evidence budget is met.
        
        Args:
            sources_count: Number of sources collected
            grounded_claims_count: Number of grounded claims
            
        Returns:
            Tuple of (is_met: bool, reason: str)
        """
        if self.evidence_budget is None:
            return True, "No evidence budget defined"
        
        if not self.evidence_budget.enforce:
            return True, "Evidence budget not enforced"
        
        return self.evidence_budget.is_met(sources_count, grounded_claims_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase_name": self.phase_name,
            "allowed": list(self.allowed),
            "forbidden": list(self.forbidden),
            "min_sources": self.min_sources,
            "max_speculation_ratio": self.max_speculation_ratio,
            "required_scenario_count": self.required_scenario_count,
            "min_content_length": self.min_content_length,
            "requires_web_search": self.requires_web_search,
            "strictness_weight": self.strictness_weight,
            "evidence_budget": self.evidence_budget.to_dict() if self.evidence_budget else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernancePhaseContract":
        """Create from dictionary."""
        evidence_budget = None
        if data.get("evidence_budget"):
            evidence_budget = EvidenceBudget.from_dict(data["evidence_budget"])
        
        return cls(
            phase_name=data.get("phase_name", ""),
            allowed=set(data.get("allowed", [])),
            forbidden=set(data.get("forbidden", [])),
            min_sources=data.get("min_sources", 0),
            max_speculation_ratio=data.get("max_speculation_ratio", 0.5),
            required_scenario_count=data.get("required_scenario_count"),
            min_content_length=data.get("min_content_length", 100),
            requires_web_search=data.get("requires_web_search", False),
            strictness_weight=data.get("strictness_weight", 1.0),
            evidence_budget=evidence_budget,
        )


# Governance phase contracts with epistemic and structural requirements
GOVERNANCE_PHASE_CONTRACTS: Dict[str, GovernancePhaseContract] = {
    "reconnaissance": GovernancePhaseContract(
        phase_name="reconnaissance",
        allowed={
            "sources", "facts", "unknowns", "questions",
            "landscape", "hypotheses", "data_needs", "context"
        },
        forbidden={
            "recommendations", "conclusions", "synthesis",
            "action_items", "final_report", "decisions"
        },
        min_sources=2,
        max_speculation_ratio=0.3,
        requires_web_search=True,
        strictness_weight=0.8,
    ),
    
    # Split recon phases (Epistemic Hardening Phase 2)
    "recon_exploration": GovernancePhaseContract(
        phase_name="recon_exploration",
        allowed={
            "hypotheses", "questions", "focus_areas", "landscape",
            "assumptions", "data_needs", "speculation", "brainstorming"
        },
        forbidden={
            "recommendations", "conclusions", "synthesis",
            "action_items", "final_report", "decisions",
            "grounded_claims"  # Claims require grounding phase
        },
        min_sources=0,  # No sources required for exploration
        max_speculation_ratio=0.8,  # High speculation allowed
        requires_web_search=False,
        strictness_weight=0.3,  # Permissive for exploration
    ),
    
    "recon_grounding": GovernancePhaseContract(
        phase_name="recon_grounding",
        allowed={
            "sources", "facts", "grounded_claims", "evidence",
            "validated_hypotheses", "approved_focus_areas"
        },
        forbidden={
            "recommendations", "conclusions", "synthesis",
            "action_items", "final_report", "decisions",
            "ungrounded_speculation"  # Must ground claims
        },
        min_sources=5,  # HARD REQUIREMENT for grounding
        max_speculation_ratio=0.2,  # Low speculation allowed
        requires_web_search=True,
        strictness_weight=1.0,  # Strict epistemic gating
    ),
    
    "research": GovernancePhaseContract(
        phase_name="research",
        allowed={
            "sources", "facts", "unknowns", "questions",
            "landscape", "hypotheses", "data_needs", "evidence"
        },
        forbidden={
            "recommendations", "conclusions", "synthesis",
            "action_items", "final_report", "decisions"
        },
        min_sources=3,
        max_speculation_ratio=0.3,
        requires_web_search=True,
        strictness_weight=0.8,
    ),
    
    "analysis": GovernancePhaseContract(
        phase_name="analysis",
        allowed={
            "causal_links", "synthesis", "validated_facts",
            "analysis", "findings", "trade_offs", "patterns",
            "correlations", "insights"
        },
        forbidden={
            "new_sources", "recommendations", "final_report",
            "raw_research", "action_items", "decisions"
        },
        min_sources=0,  # Can work with previously gathered sources
        max_speculation_ratio=0.4,
        strictness_weight=0.7,
    ),
    
    "deep_analysis": GovernancePhaseContract(
        phase_name="deep_analysis",
        allowed={
            "scenarios", "risk_analysis", "trade_offs",
            "validated_hypotheses", "stress_tests", "edge_cases",
            "failure_modes", "sensitivity_analysis"
        },
        forbidden={
            "final_report", "implementation", "code",
            "action_items"
        },
        min_sources=0,
        max_speculation_ratio=0.5,
        strictness_weight=0.9,
    ),
    
    "synthesis": GovernancePhaseContract(
        phase_name="synthesis",
        allowed={
            "scenarios", "recommendations", "conclusions",
            "action_items", "final_report", "strategy",
            "decisions", "summary"
        },
        forbidden={
            "raw_research", "new_facts", "new_sources",
            "data_gathering", "hypothesis_generation"
        },
        min_sources=0,
        max_speculation_ratio=0.3,
        required_scenario_count=3,  # HARD REQUIREMENT
        min_content_length=500,
        strictness_weight=1.0,  # Highest strictness
    ),
    
    "implementation": GovernancePhaseContract(
        phase_name="implementation",
        allowed={
            "code", "implementation", "tests", "documentation",
            "configuration", "deployment"
        },
        forbidden={
            "recommendations", "synthesis", "analysis",
            "research", "hypothesis"
        },
        min_sources=0,
        max_speculation_ratio=0.1,
        strictness_weight=0.9,
    ),
}


def get_governance_contract(phase_name: str) -> GovernancePhaseContract:
    """
    Get governance contract for a phase.
    
    Performs case-insensitive matching and fuzzy matching for phase names.
    
    Args:
        phase_name: Name of the phase
        
    Returns:
        GovernancePhaseContract for the phase
    """
    phase_lower = phase_name.lower()
    
    # Direct match
    if phase_lower in GOVERNANCE_PHASE_CONTRACTS:
        return GOVERNANCE_PHASE_CONTRACTS[phase_lower]
    
    # Fuzzy match - check if phase name contains contract key or vice versa
    for key, contract in GOVERNANCE_PHASE_CONTRACTS.items():
        if key in phase_lower or phase_lower in key:
            return contract
    
    # Map common phase name patterns (order matters - check specific patterns first)
    phase_mappings = [
        # Split recon phases (check first - more specific)
        ("recon_exploration", "recon_exploration"),
        ("recon_grounding", "recon_grounding"),
        ("exploration", "recon_exploration"),
        ("grounding", "recon_grounding"),
        # General mappings
        ("recon", "reconnaissance"),
        ("explore", "reconnaissance"),
        ("design", "analysis"),
        ("plan", "analysis"),
        ("deep", "deep_analysis"),
        ("final", "synthesis"),
        ("conclude", "synthesis"),
        ("implement", "implementation"),
        ("code", "implementation"),
        ("build", "implementation"),
    ]
    
    for pattern, mapped in phase_mappings:
        if pattern in phase_lower:
            return GOVERNANCE_PHASE_CONTRACTS[mapped]
    
    # Default permissive contract for unknown phases
    return GovernancePhaseContract(
        phase_name=phase_name,
        allowed=set(),
        forbidden=set(),
        strictness_weight=0.5,
    )


def is_output_allowed(
    output_type: str,
    phase_name: str,
    contract: Optional[GovernancePhaseContract] = None
) -> bool:
    """
    Check if an output type is allowed in a phase.
    
    Args:
        output_type: Type of output to check
        phase_name: Name of the phase
        contract: Optional pre-fetched contract
        
    Returns:
        True if output type is allowed
    """
    if contract is None:
        contract = get_governance_contract(phase_name)
    
    output_lower = output_type.lower()
    
    # Explicitly forbidden
    if any(f.lower() in output_lower or output_lower in f.lower() for f in contract.forbidden):
        return False
    
    # If allowed set is empty, default to allowing
    if not contract.allowed:
        return True
    
    # Check if explicitly allowed
    return any(a.lower() in output_lower or output_lower in a.lower() for a in contract.allowed)


def is_output_forbidden(
    output_type: str,
    phase_name: str,
    contract: Optional[GovernancePhaseContract] = None
) -> bool:
    """
    Check if an output type is forbidden in a phase.
    
    Args:
        output_type: Type of output to check
        phase_name: Name of the phase
        contract: Optional pre-fetched contract
        
    Returns:
        True if output type is forbidden
    """
    if contract is None:
        contract = get_governance_contract(phase_name)
    
    output_lower = output_type.lower()
    
    return any(f.lower() in output_lower or output_lower in f.lower() for f in contract.forbidden)

