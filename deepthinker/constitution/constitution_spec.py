"""
Constitution Specification for DeepThinker.

Defines the formal specification of constitutional invariants:
1. Conservation of Evidence
2. Monotonic Uncertainty Under Compression
3. No-Free-Lunch Depth
4. Anti-Gaming Divergence (Goodhart Shield)

Provides JSON schema for validation and documentation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class InvariantType(str, Enum):
    """Types of constitutional invariants."""
    EVIDENCE_CONSERVATION = "evidence_conservation"
    """Confidence cannot increase without new evidence."""
    
    MONOTONIC_UNCERTAINTY = "monotonic_uncertainty"
    """Compression cannot reduce uncertainty without validation."""
    
    NO_FREE_LUNCH = "no_free_lunch"
    """Deeper rounds must produce measurable gain."""
    
    GOODHART_SHIELD = "goodhart_shield"
    """Target/shadow metric divergence detection."""


@dataclass
class InvariantSpec:
    """
    Specification for a single invariant.
    
    Attributes:
        invariant_type: Type of invariant
        description: Human-readable description
        formula: Formal expression of the invariant
        parameters: Configurable parameters
        severity_weight: Weight for violation severity (0-1)
        enforcement_action: Default action on violation
    """
    invariant_type: InvariantType
    description: str
    formula: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity_weight: float = 1.0
    enforcement_action: str = "warn"  # warn | block_learning | stop_deepening
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_type": self.invariant_type.value,
            "description": self.description,
            "formula": self.formula,
            "parameters": self.parameters,
            "severity_weight": self.severity_weight,
            "enforcement_action": self.enforcement_action,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvariantSpec":
        return cls(
            invariant_type=InvariantType(data["invariant_type"]),
            description=data.get("description", ""),
            formula=data.get("formula", ""),
            parameters=data.get("parameters", {}),
            severity_weight=data.get("severity_weight", 1.0),
            enforcement_action=data.get("enforcement_action", "warn"),
        )


# Default invariant specifications
DEFAULT_INVARIANTS: Dict[InvariantType, InvariantSpec] = {
    InvariantType.EVIDENCE_CONSERVATION: InvariantSpec(
        invariant_type=InvariantType.EVIDENCE_CONSERVATION,
        description=(
            "Confidence (or score) cannot increase unless new evidence objects "
            "are introduced OR contradictions decrease OR explicit uncertainty "
            "decreases with justification."
        ),
        formula=(
            "IF score_delta > threshold THEN "
            "(evidence_added > 0 OR contradictions_reduced > 0 OR uncertainty_justified)"
        ),
        parameters={
            "threshold": 0.01,
            "min_evidence": 1,
        },
        severity_weight=1.0,
        enforcement_action="block_learning",
    ),
    
    InvariantType.MONOTONIC_UNCERTAINTY: InvariantSpec(
        invariant_type=InvariantType.MONOTONIC_UNCERTAINTY,
        description=(
            "Any compression/distillation/latent memory operation cannot reduce "
            "uncertainty unless validated by contradiction checks or evidence."
        ),
        formula=(
            "IF uncertainty_after < uncertainty_before - margin THEN "
            "(validated_by_contradiction_check OR validated_by_evidence)"
        ),
        parameters={
            "margin": 0.05,
        },
        severity_weight=0.8,
        enforcement_action="warn",
    ),
    
    InvariantType.NO_FREE_LUNCH: InvariantSpec(
        invariant_type=InvariantType.NO_FREE_LUNCH,
        description=(
            "Increasing depth/rounds/tools must produce measurable gain: "
            "new EvidenceObjects, reduced contradiction rate, or increased goal coverage. "
            "Otherwise penalize and stop deepening."
        ),
        formula=(
            "IF rounds > 1 THEN "
            "(evidence_added > 0 OR contradiction_rate_reduced OR goal_coverage_increased)"
        ),
        parameters={
            "max_unproductive_rounds": 2,
            "min_gain_per_round": 0.02,
        },
        severity_weight=0.9,
        enforcement_action="stop_deepening",
    ),
    
    InvariantType.GOODHART_SHIELD: InvariantSpec(
        invariant_type=InvariantType.GOODHART_SHIELD,
        description=(
            "Maintain target metrics (e.g., Scorecard.overall) and shadow metrics "
            "(harder to game). If target rises but shadow doesn't, flag 'false progress' "
            "and block learning updates."
        ),
        formula=(
            "IF target_delta > target_threshold AND shadow_delta < shadow_threshold THEN "
            "flag_divergence AND block_learning"
        ),
        parameters={
            "target_threshold": 0.03,
            "shadow_threshold": 0.01,
            "shadow_metrics": [
                "contradiction_rate",
                "judge_disagreement",
                "evidence_per_score",
            ],
        },
        severity_weight=1.0,
        enforcement_action="block_learning",
    ),
}


@dataclass
class ConstitutionSpec:
    """
    Complete specification of the Cognitive Constitution.
    
    Contains all invariant specs plus metadata.
    
    Attributes:
        version: Constitution version string
        created_at: When this spec was created
        north_star: The guiding principle
        invariants: Dictionary of invariant specifications
        blinding_rules: Rules for judge input sanitization
    """
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    north_star: str = (
        "No cognition without a ledger. "
        "No confidence without evidence. "
        "No learning without blinding."
    )
    invariants: Dict[InvariantType, InvariantSpec] = field(
        default_factory=lambda: DEFAULT_INVARIANTS.copy()
    )
    blinding_rules: Dict[str, Any] = field(default_factory=lambda: {
        "remove_model_identifiers": True,
        "remove_routing_identifiers": True,
        "remove_policy_identifiers": True,
        "remove_council_identifiers": True,
        "remove_bandit_arm_identifiers": True,
        "allowed_fields": [
            "objective",
            "artifact_text",
            "evidence_objects",
            "phase_name",
        ],
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "north_star": self.north_star,
            "invariants": {
                k.value: v.to_dict() for k, v in self.invariants.items()
            },
            "blinding_rules": self.blinding_rules,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstitutionSpec":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        invariants = {}
        for k, v in data.get("invariants", {}).items():
            inv_type = InvariantType(k)
            invariants[inv_type] = InvariantSpec.from_dict(v)
        
        # Fill in missing invariants from defaults
        for inv_type, default_spec in DEFAULT_INVARIANTS.items():
            if inv_type not in invariants:
                invariants[inv_type] = default_spec
        
        return cls(
            version=data.get("version", "1.0.0"),
            created_at=created_at,
            north_star=data.get("north_star", cls.north_star),
            invariants=invariants,
            blinding_rules=data.get("blinding_rules", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConstitutionSpec":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_invariant(self, inv_type: InvariantType) -> InvariantSpec:
        """Get spec for a specific invariant."""
        return self.invariants.get(inv_type, DEFAULT_INVARIANTS[inv_type])


# JSON Schema for validation
CONSTITUTION_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ConstitutionSpec",
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "north_star": {"type": "string"},
        "invariants": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "invariant_type": {
                        "type": "string",
                        "enum": [t.value for t in InvariantType],
                    },
                    "description": {"type": "string"},
                    "formula": {"type": "string"},
                    "parameters": {"type": "object"},
                    "severity_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "enforcement_action": {
                        "type": "string",
                        "enum": ["warn", "block_learning", "stop_deepening"],
                    },
                },
                "required": ["invariant_type", "description"],
            },
        },
        "blinding_rules": {
            "type": "object",
            "properties": {
                "remove_model_identifiers": {"type": "boolean"},
                "remove_routing_identifiers": {"type": "boolean"},
                "remove_policy_identifiers": {"type": "boolean"},
                "remove_council_identifiers": {"type": "boolean"},
                "remove_bandit_arm_identifiers": {"type": "boolean"},
                "allowed_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "required": ["version", "north_star", "invariants"],
}


def get_default_spec() -> ConstitutionSpec:
    """Get the default constitution specification."""
    return ConstitutionSpec()


