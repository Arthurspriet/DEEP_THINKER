"""
Rule Engine for Normative Control Layer.

Provides deterministic rule evaluation without LLM calls.
Evaluates rules across categories:
- Epistemic admissibility
- Phase purity
- Structural completeness
- Confidence governance

Rules are loaded from governance_config.yaml and applied consistently.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import yaml

from .violation import Violation, ViolationType, create_violation
from .phase_contracts import (
    GOVERNANCE_PHASE_CONTRACTS,
    GovernancePhaseContract,
    get_governance_contract,
)

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState

logger = logging.getLogger(__name__)


@dataclass
class GovernanceConfig:
    """
    Configuration for governance rules.
    
    Loaded from governance_config.yaml.
    """
    
    # Violation severities
    violation_severity: Dict[str, float] = field(default_factory=dict)
    
    # Hard rules that trigger BLOCK
    hard_rules: List[str] = field(default_factory=list)
    
    # Thresholds
    warn_threshold: float = 0.3
    block_threshold: float = 0.7
    confidence_clamp_threshold: float = 0.2
    min_grounded_ratio: float = 0.5
    max_speculation_density: float = 0.4
    
    # Phase strictness
    phase_strictness: Dict[str, float] = field(default_factory=dict)
    
    # Resource modifiers
    resource_modifiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Escalation policy
    escalation: Dict[str, Any] = field(default_factory=dict)
    
    # Scenario requirements
    scenario_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Source requirements
    source_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Recommended actions mapping
    recommended_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Reporting settings
    reporting: Dict[str, bool] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GovernanceConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        thresholds = data.get("thresholds", {})
        
        return cls(
            violation_severity=data.get("violation_severity", {}),
            hard_rules=data.get("hard_rules", []),
            warn_threshold=thresholds.get("warn_threshold", 0.3),
            block_threshold=thresholds.get("block_threshold", 0.7),
            confidence_clamp_threshold=thresholds.get("confidence_clamp_threshold", 0.2),
            min_grounded_ratio=thresholds.get("min_grounded_ratio", 0.5),
            max_speculation_density=thresholds.get("max_speculation_density", 0.4),
            phase_strictness=data.get("phase_strictness", {}),
            resource_modifiers=data.get("resource_modifiers", {}),
            escalation=data.get("escalation", {}),
            scenario_requirements=data.get("scenario_requirements", {}),
            source_requirements=data.get("source_requirements", {}),
            recommended_actions=data.get("recommended_actions", {}),
            reporting=data.get("reporting", {}),
        )
    
    def get_severity(self, violation_type: str) -> float:
        """Get severity for a violation type."""
        return self.violation_severity.get(violation_type, 0.5)
    
    def is_hard_rule(self, violation_type: str) -> bool:
        """Check if a violation type is a hard rule."""
        return violation_type in self.hard_rules
    
    def get_phase_strictness(self, phase_name: str) -> float:
        """Get strictness multiplier for a phase."""
        phase_lower = phase_name.lower()
        
        # Direct match
        if phase_lower in self.phase_strictness:
            return self.phase_strictness[phase_lower]
        
        # Fuzzy match
        for key, value in self.phase_strictness.items():
            if key in phase_lower or phase_lower in key:
                return value
        
        return self.phase_strictness.get("default", 0.6)
    
    def get_resource_modifier(self, pressure: str) -> Dict[str, Any]:
        """Get resource modifier for current GPU pressure."""
        key = f"gpu_pressure_{pressure}"
        return self.resource_modifiers.get(key, {"strictness_multiplier": 1.0, "max_retries": 2})


def load_governance_config(config_path: Optional[str] = None) -> GovernanceConfig:
    """
    Load governance configuration.
    
    Args:
        config_path: Optional path to config file. If None, uses default.
        
    Returns:
        Loaded GovernanceConfig
    """
    if config_path is None:
        # Default to config file in same directory
        config_path = os.path.join(
            os.path.dirname(__file__),
            "governance_config.yaml"
        )
    
    if os.path.exists(config_path):
        return GovernanceConfig.from_yaml(config_path)
    
    logger.warning(f"Governance config not found at {config_path}, using defaults")
    return GovernanceConfig()


class RuleEngine:
    """
    Deterministic rule engine for governance evaluation.
    
    Evaluates rules without LLM calls, producing typed violations
    with severity scores.
    """
    
    def __init__(self, config: Optional[GovernanceConfig] = None):
        """
        Initialize rule engine.
        
        Args:
            config: Governance configuration. Loads default if None.
        """
        self.config = config or load_governance_config()
        self._violation_history: Dict[str, List[Violation]] = {}
    
    def evaluate_rules(
        self,
        phase_name: str,
        phase_output: Dict[str, Any],
        mission_state: Optional["MissionState"] = None,
        gpu_pressure: str = "low",
    ) -> List[Violation]:
        """
        Evaluate all governance rules for a phase output.
        
        Args:
            phase_name: Name of the current phase
            phase_output: Phase artifacts/output dictionary
            mission_state: Optional mission state for context
            gpu_pressure: Current GPU pressure level
            
        Returns:
            List of violations detected
        """
        violations: List[Violation] = []
        
        # Get phase contract
        contract = get_governance_contract(phase_name)
        
        # Collect text content from output
        output_text = self._extract_text_content(phase_output)
        
        # === 1. Epistemic Rules ===
        violations.extend(self._check_epistemic_rules(
            phase_name, phase_output, output_text, contract, mission_state
        ))
        
        # === 2. Phase Purity Rules ===
        violations.extend(self._check_phase_purity_rules(
            phase_name, output_text, contract
        ))
        
        # === 3. Structural Rules ===
        violations.extend(self._check_structural_rules(
            phase_name, phase_output, output_text, contract
        ))
        
        # === 4. Confidence Rules ===
        violations.extend(self._check_confidence_rules(
            phase_name, phase_output, mission_state
        ))
        
        # Apply resource-aware strictness
        violations = self._apply_resource_modifiers(violations, gpu_pressure)
        
        # Track for escalation
        self._track_violations(phase_name, violations)
        
        return violations
    
    def _extract_text_content(self, phase_output: Dict[str, Any]) -> str:
        """Extract text content from phase output for analysis."""
        if not phase_output:
            return ""
        
        text_parts = []
        for key, value in phase_output.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, dict):
                text_parts.append(str(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        text_parts.append(str(item))
        
        return "\n\n".join(text_parts)
    
    def _check_epistemic_rules(
        self,
        phase_name: str,
        phase_output: Dict[str, Any],
        output_text: str,
        contract: GovernancePhaseContract,
        mission_state: Optional["MissionState"],
    ) -> List[Violation]:
        """Check epistemic admissibility rules."""
        violations = []
        
        # === Source count check ===
        if contract.min_sources > 0:
            sources_found = self._count_sources(phase_output, output_text)
            if sources_found < contract.min_sources:
                violations.append(create_violation(
                    ViolationType.EPISTEMIC_LOW_SOURCE_COUNT,
                    f"Found {sources_found} sources, minimum required is {contract.min_sources}",
                    phase_name,
                    details={"found": sources_found, "required": contract.min_sources},
                ))
        
        # === Speculation ratio check ===
        speculation_ratio = self._compute_speculation_ratio(output_text)
        if speculation_ratio > contract.max_speculation_ratio:
            violations.append(create_violation(
                ViolationType.EPISTEMIC_HIGH_SPECULATION,
                f"Speculation ratio {speculation_ratio:.2f} exceeds maximum {contract.max_speculation_ratio}",
                phase_name,
                severity=0.4 + (speculation_ratio - contract.max_speculation_ratio),
                details={"ratio": speculation_ratio, "max": contract.max_speculation_ratio},
            ))
        
        # === Grounded claim ratio check (from mission state) ===
        if mission_state is not None:
            telemetry = getattr(mission_state, "epistemic_telemetry", {})
            grounded_ratio = telemetry.get("grounded_claim_ratio", 1.0)
            
            if grounded_ratio < self.config.min_grounded_ratio:
                violations.append(create_violation(
                    ViolationType.EPISTEMIC_UNGROUNDED_CLAIM,
                    f"Grounded claim ratio {grounded_ratio:.2f} below minimum {self.config.min_grounded_ratio}",
                    phase_name,
                    severity=0.6,
                    details={"ratio": grounded_ratio, "min": self.config.min_grounded_ratio},
                ))
        
        # === Web search requirement ===
        if contract.requires_web_search:
            has_web_sources = self._has_web_sources(phase_output, output_text)
            if not has_web_sources:
                violations.append(create_violation(
                    ViolationType.EPISTEMIC_MISSING_CITATIONS,
                    f"Phase '{phase_name}' requires web search for factual grounding",
                    phase_name,
                    severity=0.4,
                ))
        
        return violations
    
    def _check_phase_purity_rules(
        self,
        phase_name: str,
        output_text: str,
        contract: GovernancePhaseContract,
    ) -> List[Violation]:
        """Check phase purity (contamination) rules."""
        violations = []
        phase_lower = phase_name.lower()
        
        # Use existing PhaseGuard if available
        try:
            from ..phases.phase_contracts import PhaseGuard
            guard = PhaseGuard()
            contamination = guard.inspect_output(output_text, phase_name)
            
            if not contamination.is_clean:
                violations.append(create_violation(
                    ViolationType.PHASE_CONTAMINATION,
                    f"Phase contamination detected: {len(contamination.violations)} issues",
                    phase_name,
                    severity=contamination.contamination_score,
                    details={
                        "contamination_score": contamination.contamination_score,
                        "violation_count": len(contamination.violations),
                    },
                ))
        except ImportError:
            # Fallback to simple pattern-based detection
            pass
        
        # Additional pattern-based checks for forbidden content
        forbidden_patterns = self._get_forbidden_patterns(phase_lower)
        
        for pattern_name, pattern in forbidden_patterns.items():
            if re.search(pattern, output_text, re.IGNORECASE):
                violation_type = self._pattern_to_violation_type(pattern_name)
                violations.append(create_violation(
                    violation_type,
                    f"Forbidden content '{pattern_name}' found in {phase_name} phase",
                    phase_name,
                    severity=self.config.get_severity(violation_type.value),
                ))
        
        return violations
    
    def _check_structural_rules(
        self,
        phase_name: str,
        phase_output: Dict[str, Any],
        output_text: str,
        contract: GovernancePhaseContract,
    ) -> List[Violation]:
        """Check structural completeness rules."""
        violations = []
        
        # === Content length check ===
        if len(output_text) < contract.min_content_length:
            violations.append(create_violation(
                ViolationType.STRUCTURAL_INSUFFICIENT_CONTENT,
                f"Output length {len(output_text)} below minimum {contract.min_content_length}",
                phase_name,
                severity=0.6,
                details={"length": len(output_text), "min": contract.min_content_length},
            ))
        
        # === Scenario requirement (synthesis phase) ===
        if contract.required_scenario_count is not None:
            scenarios = self._extract_scenarios(phase_output, output_text)
            required = contract.required_scenario_count
            
            # Check count
            if len(scenarios) != required:
                violations.append(create_violation(
                    ViolationType.STRUCTURAL_MISSING_SCENARIOS,
                    f"Expected exactly {required} scenarios, found {len(scenarios)}",
                    phase_name,
                    is_hard=True,  # This is a hard requirement
                    details={"found": len(scenarios), "required": required},
                ))
            
            # Check scenario quality
            scenario_reqs = self.config.scenario_requirements
            for i, scenario in enumerate(scenarios):
                # Check name
                if scenario_reqs.get("require_name", True):
                    if not scenario.get("name"):
                        violations.append(create_violation(
                            ViolationType.STRUCTURAL_MALFORMED_SCENARIO,
                            f"Scenario {i + 1} missing name",
                            phase_name,
                            is_hard=True,
                            details={"scenario_index": i},
                        ))
                
                # Check description length
                min_desc = scenario_reqs.get("min_description_length", 50)
                desc = scenario.get("description", "")
                if len(desc) < min_desc:
                    violations.append(create_violation(
                        ViolationType.STRUCTURAL_MALFORMED_SCENARIO,
                        f"Scenario {i + 1} description too short ({len(desc)} < {min_desc})",
                        phase_name,
                        severity=0.5,
                        details={"scenario_index": i, "length": len(desc), "min": min_desc},
                    ))
            
            # Check distinctness
            if len(scenarios) >= 2:
                distinctness = self._compute_scenario_distinctness(scenarios)
                min_distinctness = scenario_reqs.get("min_distinctness", 0.5)
                if distinctness < min_distinctness:
                    violations.append(create_violation(
                        ViolationType.STRUCTURAL_SCENARIO_NOT_DISTINCT,
                        f"Scenarios not distinct enough (score={distinctness:.2f}, min={min_distinctness})",
                        phase_name,
                        severity=0.5,
                        details={"distinctness": distinctness, "min": min_distinctness},
                    ))
        
        return violations
    
    def _check_confidence_rules(
        self,
        phase_name: str,
        phase_output: Dict[str, Any],
        mission_state: Optional["MissionState"],
    ) -> List[Violation]:
        """Check confidence governance rules."""
        violations = []
        
        if mission_state is None:
            return violations
        
        telemetry = getattr(mission_state, "epistemic_telemetry", {})
        epistemic_risk = telemetry.get("epistemic_risk", 0.0)
        
        # Get stated confidence from various sources
        stated_confidence = self._extract_stated_confidence(phase_output)
        
        if stated_confidence is not None:
            # Check if confidence exceeds grounding
            max_allowed = 1.0 - epistemic_risk
            
            if stated_confidence > max_allowed + 0.1:  # 10% tolerance
                violations.append(create_violation(
                    ViolationType.CONFIDENCE_EXCEEDS_GROUNDING,
                    f"Stated confidence {stated_confidence:.2f} exceeds epistemic grounding {max_allowed:.2f}",
                    phase_name,
                    severity=0.5,
                    details={
                        "stated": stated_confidence,
                        "max_allowed": max_allowed,
                        "epistemic_risk": epistemic_risk,
                    },
                ))
        
        return violations
    
    def _apply_resource_modifiers(
        self,
        violations: List[Violation],
        gpu_pressure: str,
    ) -> List[Violation]:
        """Apply resource-aware strictness modifiers."""
        modifier = self.config.get_resource_modifier(gpu_pressure)
        multiplier = modifier.get("strictness_multiplier", 1.0)
        
        if multiplier == 1.0:
            return violations
        
        # Increase severity under resource pressure
        for v in violations:
            v.severity = min(1.0, v.severity * multiplier)
        
        return violations
    
    def _track_violations(self, phase_name: str, violations: List[Violation]) -> None:
        """Track violations for escalation."""
        if phase_name not in self._violation_history:
            self._violation_history[phase_name] = []
        
        self._violation_history[phase_name].extend(violations)
    
    def get_violation_history(self, phase_name: Optional[str] = None) -> List[Violation]:
        """Get violation history, optionally filtered by phase."""
        if phase_name is not None:
            return self._violation_history.get(phase_name, [])
        
        all_violations = []
        for violations in self._violation_history.values():
            all_violations.extend(violations)
        return all_violations
    
    def clear_history(self) -> None:
        """Clear violation history."""
        self._violation_history.clear()
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _count_sources(self, phase_output: Dict[str, Any], output_text: str) -> int:
        """Count sources in output."""
        count = 0
        
        # Check artifacts for source lists
        for key, value in phase_output.items():
            key_lower = key.lower()
            if "source" in key_lower or "url" in key_lower or "reference" in key_lower:
                if isinstance(value, list):
                    count += len(value)
                elif isinstance(value, str) and value.strip():
                    count += 1
        
        # Count URL patterns in text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = set(re.findall(url_pattern, output_text))
        count += len(urls)
        
        # Count citation patterns
        citation_pattern = r'\[(?:\d+|[a-zA-Z]+\d{4})\]'
        citations = set(re.findall(citation_pattern, output_text))
        count += len(citations)
        
        return count
    
    def _compute_speculation_ratio(self, text: str) -> float:
        """Compute ratio of speculative language in text."""
        if not text:
            return 0.0
        
        speculative_patterns = [
            r'\b(?:might|may|could|possibly|perhaps|probably|likely)\b',
            r'\b(?:appears to|seems to|suggests that|indicates that)\b',
            r'\b(?:it is possible|there is a chance|uncertain|unclear)\b',
            r'\b(?:estimated|approximately|roughly|around)\b',
            r'\b(?:in theory|hypothetically|potentially)\b',
        ]
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        speculative_matches = 0
        for pattern in speculative_patterns:
            speculative_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize by word count (speculative phrases per 100 words)
        ratio = (speculative_matches / total_words) * 10
        return min(1.0, ratio)
    
    def _has_web_sources(self, phase_output: Dict[str, Any], output_text: str) -> bool:
        """Check if output has web sources."""
        # Check for URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        if re.search(url_pattern, output_text):
            return True
        
        # Check artifacts
        for key, value in phase_output.items():
            if "source" in key.lower() or "url" in key.lower():
                if value:
                    return True
        
        return False
    
    def _get_forbidden_patterns(self, phase_name: str) -> Dict[str, str]:
        """Get forbidden content patterns for a phase."""
        patterns = {}
        
        if "recon" in phase_name or "research" in phase_name:
            patterns["recommendation"] = r'\b(?:recommend|should|must|advise)\b.{0,50}\b(?:implement|adopt|use|action)\b'
            patterns["conclusion"] = r'\b(?:in conclusion|therefore we recommend|final assessment|we conclude)\b'
        
        if "synthesis" in phase_name:
            patterns["new_research"] = r'\b(?:new research|new source|additional study|just found|just discovered)\b'
            patterns["late_sourcing"] = r'\baccording to (?:new|recent|additional|fresh)\b'
        
        return patterns
    
    def _pattern_to_violation_type(self, pattern_name: str) -> ViolationType:
        """Map pattern name to violation type."""
        mapping = {
            "recommendation": ViolationType.PHASE_PREMATURE_RECOMMENDATION,
            "conclusion": ViolationType.PHASE_PREMATURE_CONCLUSION,
            "new_research": ViolationType.PHASE_RAW_RESEARCH_IN_SYNTHESIS,
            "late_sourcing": ViolationType.PHASE_LATE_SOURCING,
        }
        return mapping.get(pattern_name, ViolationType.PHASE_CONTAMINATION)
    
    def _extract_scenarios(
        self,
        phase_output: Dict[str, Any],
        output_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract scenarios from phase output."""
        scenarios = []
        
        # Check for structured scenario data
        for key, value in phase_output.items():
            if "scenario" in key.lower():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            scenarios.append(item)
                        elif isinstance(item, str):
                            scenarios.append({"name": item, "description": item})
                elif isinstance(value, dict):
                    if "scenarios" in value:
                        scenarios.extend(value["scenarios"])
                    else:
                        scenarios.append(value)
        
        # Try to use ScenarioFactory if available
        if not scenarios:
            try:
                from ..scenarios.scenario_model import ScenarioFactory
                factory = ScenarioFactory()
                scenario_set = factory.parse_from_output(output_text)
                for s in scenario_set.scenarios:
                    scenarios.append(s.to_dict())
            except (ImportError, Exception):
                pass
        
        # Fallback: parse from text
        if not scenarios:
            scenarios = self._parse_scenarios_from_text(output_text)
        
        return scenarios
    
    def _parse_scenarios_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse scenarios from text using simple heuristics."""
        scenarios = []
        
        # Look for numbered scenarios
        pattern = r'(?:^|\n)(?:##?\s*)?(?:scenario\s*)?(\d+)[.:\s]+([^\n]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for num, title in matches:
            scenarios.append({
                "name": title.strip(),
                "description": title.strip(),
            })
        
        # Look for named scenarios
        named_pattern = r'(?:^|\n)##?\s*(?:scenario[:\s]*)?([^:\n]+?)(?:\n|$)'
        named_matches = re.findall(named_pattern, text, re.IGNORECASE)
        
        for name in named_matches:
            name = name.strip()
            if name and "scenario" not in name.lower() and len(name) > 5:
                # Avoid duplicates
                if not any(s.get("name", "").lower() == name.lower() for s in scenarios):
                    scenarios.append({
                        "name": name,
                        "description": name,
                    })
        
        return scenarios[:5]  # Limit to prevent noise
    
    def _compute_scenario_distinctness(self, scenarios: List[Dict[str, Any]]) -> float:
        """Compute distinctness score between scenarios."""
        if len(scenarios) < 2:
            return 1.0
        
        # Extract key terms from each scenario
        term_sets = []
        for s in scenarios:
            text = f"{s.get('name', '')} {s.get('description', '')}"
            terms = set(w.lower() for w in text.split() if len(w) > 4)
            term_sets.append(terms)
        
        # Compute pairwise Jaccard distance
        overlaps = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                if term_sets[i] and term_sets[j]:
                    intersection = len(term_sets[i] & term_sets[j])
                    union = len(term_sets[i] | term_sets[j])
                    overlap = intersection / union if union > 0 else 0
                    overlaps.append(overlap)
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        return 1.0 - avg_overlap
    
    def _extract_stated_confidence(self, phase_output: Dict[str, Any]) -> Optional[float]:
        """Extract stated confidence from phase output."""
        # Check various keys
        for key in ["confidence", "confidence_score", "stated_confidence"]:
            if key in phase_output:
                value = phase_output[key]
                if isinstance(value, (int, float)):
                    return float(value)
        
        # Check nested structures
        for key, value in phase_output.items():
            if isinstance(value, dict):
                for subkey in ["confidence", "score"]:
                    if subkey in value:
                        subvalue = value[subkey]
                        if isinstance(subvalue, (int, float)):
                            return float(subvalue)
        
        return None

