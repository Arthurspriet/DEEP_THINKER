"""
Council Specification for DeepThinker 2.0.

Defines stateless council contracts with:
- Explicit input schemas
- Explicit output schemas
- Required/optional/forbidden fields
- Strict validation mode

These specs enforce councils as pure functions: (input_context) -> output
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CouncilSpec:
    """
    Stateless contract for a council's input/output behavior.
    
    Councils must behave as pure functions with explicit schemas.
    No reliance on internal mutable state or implicit context fields.
    
    Attributes:
        council_name: Canonical name of the council
        input_schema: Type of the expected input context (dataclass)
        output_schema: Type of the expected output (dataclass)
        required_fields: Fields that MUST be present and non-empty
        optional_fields: Fields with default values if missing
        forbidden_fields: Fields that must NOT be present (reject if found)
        max_input_chars: Maximum total characters in input context
        max_output_chars: Maximum characters in output
    """
    council_name: str
    input_schema: Optional[Type] = None
    output_schema: Optional[Type] = None
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Dict[str, Any] = field(default_factory=dict)
    forbidden_fields: Set[str] = field(default_factory=set)
    max_input_chars: int = 50000
    max_output_chars: int = 20000
    
    def validate_input(self, context: Any) -> Tuple[bool, List[str], Any]:
        """
        Validate input context against this spec.
        
        Args:
            context: Input context object or dict
            
        Returns:
            Tuple of (is_valid, errors, corrected_context)
        """
        errors = []
        corrected = context
        
        # Convert to dict for easier manipulation
        if hasattr(context, '__dict__'):
            context_dict = dict(vars(context))
        elif isinstance(context, dict):
            context_dict = dict(context)
        else:
            return False, ["Context must be dict or dataclass"], context
        
        # Check required fields
        for field_name in self.required_fields:
            value = context_dict.get(field_name)
            if value is None or value == "" or value == []:
                errors.append(f"Required field '{field_name}' is missing or empty")
        
        # Check forbidden fields - REJECT if present
        for field_name in self.forbidden_fields:
            if field_name in context_dict and context_dict[field_name] is not None:
                errors.append(f"Forbidden field '{field_name}' is present")
        
        # Fill optional fields with defaults
        for field_name, default_value in self.optional_fields.items():
            if field_name not in context_dict or context_dict[field_name] is None:
                context_dict[field_name] = default_value
        
        # Check size limits
        total_chars = sum(len(str(v)) for v in context_dict.values())
        if total_chars > self.max_input_chars:
            errors.append(f"Input too large: {total_chars} chars > {self.max_input_chars} max")
        
        # Reconstruct context if we made changes
        if isinstance(context, dict):
            corrected = context_dict
        else:
            # Try to update the original object
            for key, value in context_dict.items():
                if hasattr(context, key):
                    try:
                        setattr(context, key, value)
                    except AttributeError:
                        pass
            corrected = context
        
        is_valid = len(errors) == 0
        return is_valid, errors, corrected
    
    def validate_output(self, output: Any) -> Tuple[bool, List[str]]:
        """
        Validate output against this spec.
        
        Args:
            output: Output object from council
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check size limits
        output_str = str(output)
        if len(output_str) > self.max_output_chars:
            errors.append(f"Output too large: {len(output_str)} chars > {self.max_output_chars} max")
        
        # If we have an output schema, validate against it
        if self.output_schema is not None:
            if not isinstance(output, self.output_schema):
                # Check if it has the expected fields
                if hasattr(self.output_schema, '__dataclass_fields__'):
                    expected_fields = set(self.output_schema.__dataclass_fields__.keys())
                    if hasattr(output, '__dict__'):
                        actual_fields = set(vars(output).keys())
                    elif isinstance(output, dict):
                        actual_fields = set(output.keys())
                    else:
                        actual_fields = set()
                    
                    missing = expected_fields - actual_fields
                    if missing:
                        errors.append(f"Output missing fields: {missing}")
        
        is_valid = len(errors) == 0
        return is_valid, errors


# =============================================================================
# Council Specifications Registry
# =============================================================================

EXPLORER_COUNCIL_SPEC = CouncilSpec(
    council_name="explorer_council",
    required_fields={"objective"},
    optional_fields={
        "focus_areas": [],
        "time_budget_seconds": 60,
        "max_depth": 1,
    },
    forbidden_fields={
        "recommendations",
        "synthesis",
        "conclusions",
        "action_items",
        "final_report",
    },
    max_input_chars=10000,
    max_output_chars=5000,
)

EVIDENCE_COUNCIL_SPEC = CouncilSpec(
    council_name="evidence_council",
    required_fields={"objective", "questions"},
    optional_fields={
        "prior_evidence": [],
        "allow_web_search": True,
        "max_sources": 5,
    },
    forbidden_fields=set(),
    max_input_chars=20000,
    max_output_chars=15000,
)

RESEARCHER_COUNCIL_SPEC = CouncilSpec(
    council_name="researcher_council",
    required_fields={"objective"},
    optional_fields={
        "focus_areas": [],
        "prior_knowledge": None,
        "constraints": None,
        "allow_internet": True,
        "data_needs": [],
        "unresolved_questions": [],
    },
    forbidden_fields=set(),
    max_input_chars=30000,
    max_output_chars=20000,
)

PLANNER_COUNCIL_SPEC = CouncilSpec(
    council_name="planner_council",
    required_fields={"objective"},
    optional_fields={
        "context": None,
        "available_agents": ["researcher", "coder", "evaluator", "simulator"],
        "max_iterations": 3,
        "quality_threshold": 7.0,
    },
    forbidden_fields=set(),
    max_input_chars=30000,
    max_output_chars=15000,
)

EVALUATOR_COUNCIL_SPEC = CouncilSpec(
    council_name="evaluator_council",
    required_fields={"objective", "content_to_evaluate"},
    optional_fields={
        "evaluation_criteria": [],
        "prior_evaluations": None,
        "quality_threshold": 7.0,
    },
    forbidden_fields=set(),
    max_input_chars=40000,
    max_output_chars=10000,
)

CODER_COUNCIL_SPEC = CouncilSpec(
    council_name="coder_council",
    required_fields={"objective"},
    optional_fields={
        "existing_code": None,
        "requirements": [],
        "language": "python",
        "test_requirements": True,
    },
    forbidden_fields=set(),
    max_input_chars=50000,
    max_output_chars=30000,
)

SIMULATION_COUNCIL_SPEC = CouncilSpec(
    council_name="simulation_council",
    required_fields={"objective", "scenario_description"},
    optional_fields={
        "variables": {},
        "constraints": [],
        "success_criteria": [],
        "max_iterations": 100,
    },
    forbidden_fields=set(),
    max_input_chars=20000,
    max_output_chars=15000,
)

SYNTHESIS_COUNCIL_SPEC = CouncilSpec(
    council_name="synthesis_council",
    required_fields={"objective", "prior_findings"},
    optional_fields={
        "unresolved_issues": [],
        "structural_gaps": [],
        "evaluator_feedback": [],
        "iteration": 1,
    },
    forbidden_fields=set(),
    max_input_chars=50000,
    max_output_chars=30000,
)


# Council spec registry
COUNCIL_SPECS: Dict[str, CouncilSpec] = {
    "explorer_council": EXPLORER_COUNCIL_SPEC,
    "explorer": EXPLORER_COUNCIL_SPEC,
    "evidence_council": EVIDENCE_COUNCIL_SPEC,
    "evidence": EVIDENCE_COUNCIL_SPEC,
    "researcher_council": RESEARCHER_COUNCIL_SPEC,
    "researcher": RESEARCHER_COUNCIL_SPEC,
    "research": RESEARCHER_COUNCIL_SPEC,
    "planner_council": PLANNER_COUNCIL_SPEC,
    "planner": PLANNER_COUNCIL_SPEC,
    "evaluator_council": EVALUATOR_COUNCIL_SPEC,
    "evaluator": EVALUATOR_COUNCIL_SPEC,
    "coder_council": CODER_COUNCIL_SPEC,
    "coder": CODER_COUNCIL_SPEC,
    "simulation_council": SIMULATION_COUNCIL_SPEC,
    "simulation": SIMULATION_COUNCIL_SPEC,
    "synthesis_council": SYNTHESIS_COUNCIL_SPEC,
    "synthesis": SYNTHESIS_COUNCIL_SPEC,
}


def get_council_spec(council_name: str) -> Optional[CouncilSpec]:
    """
    Get the CouncilSpec for a given council name.
    
    Args:
        council_name: Name of the council
        
    Returns:
        CouncilSpec or None if not found
    """
    return COUNCIL_SPECS.get(council_name.lower())


def validate_council_input(
    council_name: str,
    context: Any,
    strict: bool = True
) -> Tuple[bool, List[str], Any]:
    """
    Validate input for a council.
    
    Args:
        council_name: Name of the council
        context: Input context
        strict: If True, reject unknown fields
        
    Returns:
        Tuple of (is_valid, errors, corrected_context)
    """
    spec = get_council_spec(council_name)
    if spec is None:
        if strict:
            return False, [f"Unknown council: {council_name}"], context
        return True, [], context
    
    return spec.validate_input(context)

