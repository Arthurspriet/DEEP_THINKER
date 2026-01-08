"""
Phase Specification for DeepThinker 2.0.

Defines hard contracts for mission phases including:
- Allowed/forbidden councils
- Allowed/forbidden artifact types
- Token budgets
- Memory write policies

These contracts are enforced by PhaseValidator to prevent:
- Reconnaissance producing recommendations
- Early synthesis triggering
- Uncontrolled memory growth
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set
from enum import Enum


class MemoryWritePolicy(str, Enum):
    """Memory write policy for a phase."""
    NONE = "none"           # No memory writes allowed
    EPHEMERAL = "ephemeral" # Ephemeral only, cleared at phase boundary
    STABLE = "stable"       # Stable memory writes allowed


@dataclass
class PhaseSpec:
    """
    Hard contract specifying what a phase can and cannot do.
    
    Attributes:
        name: Phase identifier (e.g., "reconnaissance", "deep_analysis")
        allowed_councils: List of council names that can run in this phase
        forbidden_councils: List of council names explicitly blocked
        allowed_artifacts: Artifact types this phase can produce
        forbidden_artifacts: Artifact types this phase must NOT produce
        max_tokens: Maximum token budget for this phase
        memory_write_policy: What type of memory writes are allowed
        can_trigger_arbiter: Whether this phase can trigger Arbiter
        can_run_synthesis: Whether this phase can produce synthesis output
        max_iterations: Maximum iterations within this phase
        time_budget_ratio: Fraction of total mission time allocated (0-1)
    """
    name: str
    allowed_councils: List[str] = field(default_factory=list)
    forbidden_councils: List[str] = field(default_factory=list)
    allowed_artifacts: List[str] = field(default_factory=list)
    forbidden_artifacts: List[str] = field(default_factory=list)
    max_tokens: int = 5000
    memory_write_policy: MemoryWritePolicy = MemoryWritePolicy.EPHEMERAL
    can_trigger_arbiter: bool = False
    can_run_synthesis: bool = False
    max_iterations: int = 3
    time_budget_ratio: float = 0.2
    
    def is_council_allowed(self, council_name: str) -> bool:
        """Check if a council is allowed in this phase."""
        council_lower = council_name.lower()
        
        # Check forbidden first
        if any(f.lower() in council_lower for f in self.forbidden_councils):
            return False
        
        # If allowed list is empty, allow all (except forbidden)
        if not self.allowed_councils:
            return True
        
        # Check allowed list
        return any(a.lower() in council_lower for a in self.allowed_councils)
    
    def is_artifact_allowed(self, artifact_type: str) -> bool:
        """Check if an artifact type is allowed in this phase."""
        artifact_lower = artifact_type.lower()
        
        # Check forbidden first
        if any(f.lower() in artifact_lower for f in self.forbidden_artifacts):
            return False
        
        # If allowed list is empty, allow all (except forbidden)
        if not self.allowed_artifacts:
            return True
        
        # Check allowed list
        return any(a.lower() in artifact_lower for a in self.allowed_artifacts)
    
    def can_write_memory(self, write_type: str) -> bool:
        """Check if a memory write type is allowed."""
        if self.memory_write_policy == MemoryWritePolicy.NONE:
            return False
        if self.memory_write_policy == MemoryWritePolicy.EPHEMERAL:
            return write_type.lower() in ("ephemeral", "delta")
        # STABLE allows all
        return True


# =============================================================================
# Predefined Phase Specifications
# =============================================================================

RECONNAISSANCE_PHASE = PhaseSpec(
    name="reconnaissance",
    allowed_councils=["ExplorerCouncil", "explorer"],
    forbidden_councils=["SynthesisCouncil", "ArbiterCouncil"],
    allowed_artifacts=["known_facts", "unknowns", "hypotheses", "questions", "landscape"],
    forbidden_artifacts=["recommendations", "synthesis", "final_report", "conclusion", "action_items"],
    max_tokens=3000,
    memory_write_policy=MemoryWritePolicy.EPHEMERAL,
    can_trigger_arbiter=False,
    can_run_synthesis=False,
    max_iterations=2,
    time_budget_ratio=0.15,
)

ANALYSIS_PHASE = PhaseSpec(
    name="analysis",
    allowed_councils=["EvidenceCouncil", "PlannerCouncil", "evidence", "planner"],
    forbidden_councils=["SynthesisCouncil"],
    allowed_artifacts=["evidence", "plan", "analysis", "findings", "questions"],
    forbidden_artifacts=["final_report", "conclusion"],
    max_tokens=5000,
    memory_write_policy=MemoryWritePolicy.EPHEMERAL,
    can_trigger_arbiter=False,
    can_run_synthesis=False,
    max_iterations=3,
    time_budget_ratio=0.25,
)

DEEP_ANALYSIS_PHASE = PhaseSpec(
    name="deep_analysis",
    allowed_councils=["EvidenceCouncil", "EvaluatorCouncil", "evidence", "evaluator"],
    forbidden_councils=[],
    allowed_artifacts=["evidence", "evaluation", "deep_findings", "validated_hypotheses"],
    forbidden_artifacts=["final_report"],
    max_tokens=8000,
    memory_write_policy=MemoryWritePolicy.STABLE,
    can_trigger_arbiter=False,
    can_run_synthesis=False,
    max_iterations=4,
    time_budget_ratio=0.30,
)

SYNTHESIS_PHASE = PhaseSpec(
    name="synthesis",
    allowed_councils=["SynthesisCouncil", "PlannerCouncil", "synthesis", "planner"],
    forbidden_councils=[],
    allowed_artifacts=["synthesis", "final_report", "conclusion", "recommendations", "action_items"],
    forbidden_artifacts=[],
    max_tokens=10000,
    memory_write_policy=MemoryWritePolicy.STABLE,
    can_trigger_arbiter=True,
    can_run_synthesis=True,
    max_iterations=2,
    time_budget_ratio=0.20,
)

IMPLEMENTATION_PHASE = PhaseSpec(
    name="implementation",
    allowed_councils=["CoderCouncil", "coder"],
    forbidden_councils=["SynthesisCouncil"],
    allowed_artifacts=["code", "implementation", "tests", "documentation"],
    forbidden_artifacts=["final_report"],
    max_tokens=15000,
    memory_write_policy=MemoryWritePolicy.STABLE,
    can_trigger_arbiter=False,
    can_run_synthesis=False,
    max_iterations=5,
    time_budget_ratio=0.30,
)

SIMULATION_PHASE = PhaseSpec(
    name="simulation",
    allowed_councils=["SimulationCouncil", "simulation"],
    forbidden_councils=["SynthesisCouncil"],
    allowed_artifacts=["simulation_results", "scenarios", "outcomes", "risk_analysis"],
    forbidden_artifacts=["final_report", "code"],
    max_tokens=6000,
    memory_write_policy=MemoryWritePolicy.EPHEMERAL,
    can_trigger_arbiter=False,
    can_run_synthesis=False,
    max_iterations=3,
    time_budget_ratio=0.15,
)

# Default permissive phase for backward compatibility
DEFAULT_PHASE = PhaseSpec(
    name="default",
    allowed_councils=[],  # Empty = all allowed
    forbidden_councils=[],
    allowed_artifacts=[],  # Empty = all allowed
    forbidden_artifacts=[],
    max_tokens=10000,
    memory_write_policy=MemoryWritePolicy.STABLE,
    can_trigger_arbiter=True,
    can_run_synthesis=True,
    max_iterations=5,
    time_budget_ratio=0.25,
)


# Phase registry mapping phase names to specs
PHASE_REGISTRY: Dict[str, PhaseSpec] = {
    "reconnaissance": RECONNAISSANCE_PHASE,
    "recon": RECONNAISSANCE_PHASE,
    "context_gathering": RECONNAISSANCE_PHASE,
    "analysis": ANALYSIS_PHASE,
    "analysis_planning": ANALYSIS_PHASE,
    "deep_analysis": DEEP_ANALYSIS_PHASE,
    "deep": DEEP_ANALYSIS_PHASE,
    "synthesis": SYNTHESIS_PHASE,
    "report": SYNTHESIS_PHASE,
    "final": SYNTHESIS_PHASE,
    "implementation": IMPLEMENTATION_PHASE,
    "coding": IMPLEMENTATION_PHASE,
    "code": IMPLEMENTATION_PHASE,
    "simulation": SIMULATION_PHASE,
    "scenario": SIMULATION_PHASE,
    "default": DEFAULT_PHASE,
}


def get_phase_spec(phase_name: str, strict: bool = False) -> PhaseSpec:
    """
    Get the PhaseSpec for a given phase name.
    
    Args:
        phase_name: Name of the phase
        strict: If True, raise error for unknown phases. If False, return DEFAULT_PHASE.
        
    Returns:
        PhaseSpec for the phase
        
    Raises:
        ValueError: If strict=True and phase not found
    """
    name_lower = phase_name.lower()
    
    # Direct lookup
    if name_lower in PHASE_REGISTRY:
        return PHASE_REGISTRY[name_lower]
    
    # Fuzzy match on keywords
    for key, spec in PHASE_REGISTRY.items():
        if key in name_lower:
            return spec
    
    if strict:
        raise ValueError(f"Unknown phase: {phase_name}")
    
    return DEFAULT_PHASE


def infer_phase_type(phase_name: str) -> str:
    """
    Infer the canonical phase type from a phase name.
    
    Args:
        phase_name: Name of the phase
        
    Returns:
        Canonical phase type string
    """
    name_lower = phase_name.lower()
    
    if any(kw in name_lower for kw in ["recon", "gather", "context", "initial"]):
        return "reconnaissance"
    elif any(kw in name_lower for kw in ["deep", "thorough"]):
        return "deep_analysis"
    elif any(kw in name_lower for kw in ["analy", "plan", "design"]):
        return "analysis"
    elif any(kw in name_lower for kw in ["synth", "report", "final", "conclusion"]):
        return "synthesis"
    elif any(kw in name_lower for kw in ["impl", "code", "build"]):
        return "implementation"
    elif any(kw in name_lower for kw in ["simul", "scenario"]):
        return "simulation"
    else:
        return "default"

