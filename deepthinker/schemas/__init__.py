"""
Canonical Schemas for DeepThinker 2.0 Cognitive Spine.

Provides versioned context schemas and output contracts that all councils
must adhere to for consistent, predictable behavior.

Also includes:
- PhaseSpec: Hard contracts for mission phases
- CouncilSpec: Stateless council input/output specifications
"""

from .contexts import (
    SchemaBase,
    ResearchContextSchema,
    PlannerContextSchema,
    EvaluatorContextSchema,
    CoderContextSchema,
    SimulationContextSchema,
    SynthesisContextSchema,
    CONTEXT_SCHEMAS,
    SCHEMA_VERSIONS,
    get_schema_for_council,
    get_mandatory_fields,
    get_default_values,
)

from .outputs import (
    CouncilOutputContract,
    normalize_output_to_contract,
    validate_output_contract,
)

from .phase_spec import (
    PhaseSpec,
    MemoryWritePolicy,
    PHASE_REGISTRY,
    RECONNAISSANCE_PHASE,
    ANALYSIS_PHASE,
    DEEP_ANALYSIS_PHASE,
    SYNTHESIS_PHASE,
    IMPLEMENTATION_PHASE,
    SIMULATION_PHASE,
    DEFAULT_PHASE,
    get_phase_spec,
    infer_phase_type,
)

from .council_spec import (
    CouncilSpec,
    COUNCIL_SPECS,
    get_council_spec,
    validate_council_input,
    EXPLORER_COUNCIL_SPEC,
    EVIDENCE_COUNCIL_SPEC,
    RESEARCHER_COUNCIL_SPEC,
    PLANNER_COUNCIL_SPEC,
    EVALUATOR_COUNCIL_SPEC,
    CODER_COUNCIL_SPEC,
    SIMULATION_COUNCIL_SPEC,
    SYNTHESIS_COUNCIL_SPEC,
)

__all__ = [
    # Context schemas
    "SchemaBase",
    "ResearchContextSchema",
    "PlannerContextSchema",
    "EvaluatorContextSchema",
    "CoderContextSchema",
    "SimulationContextSchema",
    "SynthesisContextSchema",
    "CONTEXT_SCHEMAS",
    "SCHEMA_VERSIONS",
    "get_schema_for_council",
    "get_mandatory_fields",
    "get_default_values",
    # Output contracts
    "CouncilOutputContract",
    "normalize_output_to_contract",
    "validate_output_contract",
    # Phase specs
    "PhaseSpec",
    "MemoryWritePolicy",
    "PHASE_REGISTRY",
    "RECONNAISSANCE_PHASE",
    "ANALYSIS_PHASE",
    "DEEP_ANALYSIS_PHASE",
    "SYNTHESIS_PHASE",
    "IMPLEMENTATION_PHASE",
    "SIMULATION_PHASE",
    "DEFAULT_PHASE",
    "get_phase_spec",
    "infer_phase_type",
    # Council specs
    "CouncilSpec",
    "COUNCIL_SPECS",
    "get_council_spec",
    "validate_council_input",
    "EXPLORER_COUNCIL_SPEC",
    "EVIDENCE_COUNCIL_SPEC",
    "RESEARCHER_COUNCIL_SPEC",
    "PLANNER_COUNCIL_SPEC",
    "EVALUATOR_COUNCIL_SPEC",
    "CODER_COUNCIL_SPEC",
    "SIMULATION_COUNCIL_SPEC",
    "SYNTHESIS_COUNCIL_SPEC",
]

