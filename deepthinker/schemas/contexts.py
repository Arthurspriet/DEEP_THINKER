"""
Canonical Context Schemas for DeepThinker 2.0.

Defines versioned schemas for all council context types with:
- Mandatory field declarations
- Default values for optional fields
- Version tracking for schema evolution
- Validation utilities
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional, Type, Set
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Schema Versioning
# =============================================================================

SCHEMA_VERSIONS: Dict[str, int] = {
    "ResearchContext": 1,
    "PlannerContext": 1,
    "EvaluatorContext": 1,
    "CoderContext": 1,
    "SimulationContext": 1,
    "SynthesisContext": 1,
    "OptimistContext": 1,
    "SkepticContext": 1,
}


@dataclass
class SchemaBase:
    """
    Base class for all context schemas.
    
    Provides version tracking and validation state.
    """
    _version: int = field(default=1, repr=False)
    _validated: bool = field(default=False, repr=False)
    
    def mark_validated(self) -> None:
        """Mark this context as validated by CognitiveSpine."""
        object.__setattr__(self, '_validated', True)
    
    @property
    def is_validated(self) -> bool:
        """Check if context has been validated."""
        return self._validated


# =============================================================================
# Context Schemas with Mandatory/Optional Fields
# =============================================================================

@dataclass
class ResearchContextSchema(SchemaBase):
    """
    Canonical schema for ResearcherCouncil context.
    
    Mandatory: objective
    Optional: focus_areas, prior_knowledge, constraints, etc.
    """
    # Mandatory
    objective: str = ""
    
    # Optional with defaults
    focus_areas: List[str] = field(default_factory=list)
    prior_knowledge: Optional[str] = None
    constraints: Optional[str] = None
    planner_requirements: Optional[str] = None
    allow_internet: bool = True
    data_needs: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    requires_evidence: bool = False
    subgoals: List[str] = field(default_factory=list)
    # RAG knowledge context (Sprint 35)
    knowledge_context: Optional[str] = None
    # Phase context (prevents phase confusion in multi-phase missions)
    current_phase: Optional[str] = None


@dataclass
class PlannerContextSchema(SchemaBase):
    """
    Canonical schema for PlannerCouncil context.
    
    Mandatory: objective
    Optional: context, available_agents, etc.
    """
    # Mandatory
    objective: str = ""
    
    # Optional with defaults
    context: Optional[Dict[str, Any]] = None
    available_agents: List[str] = field(default_factory=lambda: [
        "researcher", "coder", "evaluator", "simulator", "executor"
    ])
    max_iterations: int = 3
    quality_threshold: float = 7.0
    data_config: Optional[Any] = None
    simulation_config: Optional[Any] = None
    # RAG knowledge context (Sprint 35)
    knowledge_context: Optional[str] = None


@dataclass
class SynthesisContextSchema(SchemaBase):
    """
    Canonical schema for synthesis phase context.
    
    Mandatory: objective, prior_findings
    Optional: unresolved_issues, structural_gaps, etc.
    """
    # Mandatory
    objective: str = ""
    prior_findings: str = ""
    
    # Optional with defaults
    unresolved_issues: List[str] = field(default_factory=list)
    structural_gaps: List[str] = field(default_factory=list)
    recommended_sections: List[str] = field(default_factory=list)
    evaluator_feedback: List[str] = field(default_factory=list)
    iteration: int = 1
    max_iterations: int = 3
    prior_synthesis_summary: Optional[str] = None
    addressed_issues: List[str] = field(default_factory=list)
    # RAG knowledge context (Sprint 35)
    knowledge_context: Optional[str] = None


@dataclass
class EvaluatorContextSchema(SchemaBase):
    """
    Canonical schema for EvaluatorCouncil context.
    
    Mandatory: objective, content_to_evaluate
    Optional: evaluation_criteria, prior_evaluations, etc.
    """
    # Mandatory
    objective: str = ""
    content_to_evaluate: str = ""
    
    # Optional with defaults
    evaluation_criteria: List[str] = field(default_factory=list)
    prior_evaluations: Optional[str] = None
    quality_threshold: float = 7.0
    focus_areas: List[str] = field(default_factory=list)
    # RAG knowledge context (Sprint 35)
    knowledge_context: Optional[str] = None
    # Additional context fields used by EvaluatorCouncil (prevents stripping)
    task_type: str = "auto"  # "code", "research", "document", "analysis"
    prior_analysis: Optional[str] = None
    previous_evaluation: Optional[Any] = None
    iteration: int = 1
    allow_internet: bool = False
    web_searches_performed: bool = False
    research_findings: Optional[str] = None
    metric_result: Optional[Any] = None


@dataclass
class CoderContextSchema(SchemaBase):
    """
    Canonical schema for CoderCouncil context.
    
    Mandatory: objective
    Optional: existing_code, requirements, language, etc.
    """
    # Mandatory
    objective: str = ""
    
    # Optional with defaults
    existing_code: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    language: str = "python"
    style_guidelines: Optional[str] = None
    test_requirements: bool = True
    # RAG knowledge context (Sprint 35)
    knowledge_context: Optional[str] = None


@dataclass
class SimulationContextSchema(SchemaBase):
    """
    Canonical schema for SimulationCouncil context.
    
    Mandatory: objective, scenario_description
    Optional: variables, constraints, etc.
    """
    # Mandatory
    objective: str = ""
    scenario_description: str = ""
    
    # Optional with defaults
    variables: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    max_iterations: int = 100
    # RAG knowledge context (Sprint 35)
    knowledge_context: Optional[str] = None


# =============================================================================
# Schema Registry
# =============================================================================

CONTEXT_SCHEMAS: Dict[str, Type[SchemaBase]] = {
    "researcher_council": ResearchContextSchema,
    "planner_council": PlannerContextSchema,
    "evaluator_council": EvaluatorContextSchema,
    "coder_council": CoderContextSchema,
    "simulation_council": SimulationContextSchema,
    "synthesis": SynthesisContextSchema,
    # Aliases
    "research": ResearchContextSchema,
    "planner": PlannerContextSchema,
    "evaluator": EvaluatorContextSchema,
    "coder": CoderContextSchema,
    "simulation": SimulationContextSchema,
}

# Mandatory fields per schema (fields without default that aren't _version/_validated)
MANDATORY_FIELDS: Dict[str, Set[str]] = {
    "ResearchContextSchema": {"objective"},
    "PlannerContextSchema": {"objective"},
    "SynthesisContextSchema": {"objective", "prior_findings"},
    "EvaluatorContextSchema": {"objective", "content_to_evaluate"},
    "CoderContextSchema": {"objective"},
    "SimulationContextSchema": {"objective", "scenario_description"},
}


def get_schema_for_council(council_name: str) -> Optional[Type[SchemaBase]]:
    """
    Get the canonical schema class for a council.
    
    Args:
        council_name: Name of the council (e.g., "researcher_council")
        
    Returns:
        Schema class or None if not found
    """
    return CONTEXT_SCHEMAS.get(council_name.lower())


def get_mandatory_fields(schema_class: Type[SchemaBase]) -> Set[str]:
    """
    Get mandatory fields for a schema class.
    
    Args:
        schema_class: The schema class
        
    Returns:
        Set of mandatory field names
    """
    class_name = schema_class.__name__
    return MANDATORY_FIELDS.get(class_name, set())


def get_default_values(schema_class: Type[SchemaBase]) -> Dict[str, Any]:
    """
    Get default values for all optional fields in a schema.
    
    Args:
        schema_class: The schema class
        
    Returns:
        Dictionary of field_name -> default_value
    """
    defaults = {}
    
    for f in dataclass_fields(schema_class):
        if f.name.startswith('_'):
            continue
        
        if f.default is not field:
            defaults[f.name] = f.default
        elif f.default_factory is not field:
            defaults[f.name] = f.default_factory()
    
    return defaults


def get_known_fields(schema_class: Type[SchemaBase]) -> Set[str]:
    """
    Get all known field names for a schema.
    
    Args:
        schema_class: The schema class
        
    Returns:
        Set of all field names (including private ones)
    """
    return {f.name for f in dataclass_fields(schema_class)}


def validate_context_fields(
    context: Any,
    schema_class: Type[SchemaBase]
) -> tuple:
    """
    Validate context fields against schema.
    
    Args:
        context: Context object or dict to validate
        schema_class: Expected schema class
        
    Returns:
        Tuple of (is_valid, missing_mandatory, unknown_fields)
    """
    known_fields = get_known_fields(schema_class)
    mandatory = get_mandatory_fields(schema_class)
    
    # Get context fields
    if hasattr(context, '__dataclass_fields__'):
        context_fields = set(context.__dataclass_fields__.keys())
        context_values = {k: getattr(context, k, None) for k in context_fields}
    elif isinstance(context, dict):
        context_fields = set(context.keys())
        context_values = context
    else:
        # Try to extract from object attributes
        context_fields = set(k for k in dir(context) if not k.startswith('_'))
        context_values = {k: getattr(context, k, None) for k in context_fields}
    
    # Check for missing mandatory fields
    missing_mandatory = []
    for field_name in mandatory:
        if field_name not in context_fields:
            missing_mandatory.append(field_name)
        elif context_values.get(field_name) in (None, "", []):
            missing_mandatory.append(field_name)
    
    # Check for unknown fields
    unknown_fields = [f for f in context_fields 
                     if f not in known_fields and not f.startswith('_')]
    
    is_valid = len(missing_mandatory) == 0
    
    return is_valid, missing_mandatory, unknown_fields

