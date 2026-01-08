"""
Decision Record Data Types for DeepThinker.

Defines the core data structures for decision accountability:
- DecisionType: Categories of decisions
- OutcomeCause: Causes of phase outcomes
- DecisionRecord: First-class artifact for system choices
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DecisionType(str, Enum):
    """
    Categories of system decisions.
    
    Each type represents a meaningful choice point in mission execution.
    """
    MODEL_SELECTION = "model_selection"
    """Supervisor chose models for phase execution."""
    
    RETRY_ESCALATION = "retry_escalation"
    """Governance triggered retry with model escalation."""
    
    GOVERNANCE_INTERVENTION = "governance_veto"
    """BLOCK/WARN verdict with recommended action."""
    
    PHASE_TERMINATION = "phase_termination"
    """Phase ended with success/partial/fail status."""
    
    EMPTY_OUTPUT_ESCALATION = "empty_escalation"
    """Council escalated due to empty structured output."""
    
    # Sprint 1-2: Scorecard and Metrics Decision Types
    SCORECARD_STOP = "scorecard_stop"
    """Policy decided to stop phase based on scorecard thresholds."""
    
    SCORECARD_ESCALATE = "scorecard_escalate"
    """Policy decided to escalate based on low scorecard scores."""
    
    ROUTING_DECISION = "routing_decision"
    """ML router or bandit made a routing decision."""
    
    TOOL_USAGE = "tool_usage"
    """Tool usage metrics recorded for a step."""
    
    # Cognitive Constitution v1: Constitution Decision Types
    CONSTITUTION_VIOLATION = "constitution_violation"
    """Constitution invariant violated."""
    
    CONSTITUTION_FLAG = "constitution_flag"
    """Constitution advisory flag emitted."""
    
    CONSTITUTION_LEARNING_BLOCKED = "constitution_learning_blocked"
    """Learning update blocked by constitution."""


class OutcomeCause(str, Enum):
    """
    Primary causes of phase outcomes.
    
    Used for attribution to understand why phases terminated.
    """
    SUCCESSFUL_CONVERGENCE = "convergence"
    """Quality threshold met, phase completed successfully."""
    
    MODEL_UNDERPOWERED = "model_underpowered"
    """Output quality insufficient for the task."""
    
    GOVERNANCE_VETO = "governance_veto"
    """BLOCK verdict from governance layer."""
    
    TIME_EXHAUSTION = "time_exhaustion"
    """Mission time budget ran out."""
    
    INPUT_AMBIGUITY = "input_ambiguity"
    """Objective unclear or insufficiently specified."""
    
    EXECUTION_ERROR = "execution_error"
    """Exception during execution."""
    
    RETRY_EXHAUSTION = "retry_exhaustion"
    """Maximum retries exceeded."""
    
    DEPTH_TARGET_REACHED = "depth_target_reached"
    """Depth target met, no further enrichment needed."""
    
    ENRICHMENT_LIMIT_REACHED = "enrichment_limit_reached"
    """Maximum enrichment passes performed."""


@dataclass
class DecisionRecord:
    """
    First-class artifact for any meaningful system choice.
    
    Decision Records are:
    - Immutable after creation
    - Linkable via triggered_by_decision_id
    - Cost-attributable post-execution
    - Human-reviewable without chain-of-thought exposure
    
    Attributes:
        decision_id: Unique identifier (UUID)
        decision_type: Category of decision
        timestamp: When the decision was made
        mission_id: Parent mission identifier
        phase_id: Phase name where decision occurred
        phase_type: Type of phase (reconnaissance, synthesis, etc.)
        options_considered: Abstract labels of alternatives (not prompts)
        selected_option: The choice that was made
        rationale: Brief, system-generated justification (NOT chain-of-thought)
        confidence: Decision confidence score (0.0-1.0)
        constraints_snapshot: State at decision time
        triggered_by_decision_id: Causal link to prior decision
        hardware_cost_attributed: Cost filled post-execution
    """
    
    # Identity
    decision_id: str
    decision_type: DecisionType
    timestamp: datetime
    
    # Context
    mission_id: str
    phase_id: str
    phase_type: str
    
    # Decision content
    options_considered: List[str] = field(default_factory=list)
    selected_option: str = ""
    rationale: str = ""
    confidence: float = 0.8
    
    # Constraints at decision time
    constraints_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Optional linkage
    triggered_by_decision_id: Optional[str] = None
    hardware_cost_attributed: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary with all fields, datetime as ISO string,
            enums as their string values.
        """
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["decision_type"] = self.decision_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        """
        Create DecisionRecord from dictionary.
        
        Args:
            data: Dictionary with decision data
            
        Returns:
            DecisionRecord instance
        """
        # Convert ISO string back to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Convert string back to enum
        if isinstance(data.get("decision_type"), str):
            data["decision_type"] = DecisionType(data["decision_type"])
        
        return cls(**data)
    
    def is_linked(self) -> bool:
        """Check if this decision was triggered by another."""
        return self.triggered_by_decision_id is not None
    
    def has_cost(self) -> bool:
        """Check if hardware cost has been attributed."""
        return self.hardware_cost_attributed > 0.0

