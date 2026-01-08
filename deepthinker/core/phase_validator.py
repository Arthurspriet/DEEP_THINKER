"""
Phase Validator for DeepThinker 2.0.

Enforces phase contracts by validating:
- Council usage against phase specs
- Artifact types against allowed/forbidden lists
- Memory writes against phase policies
- Output content for forbidden patterns

Integration points:
- MissionOrchestrator._run_phase() calls validate before/after
- CognitiveSpine uses for phase boundary validation
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schemas.phase_spec import (
    PhaseSpec,
    MemoryWritePolicy,
    get_phase_spec,
    infer_phase_type,
    DEFAULT_PHASE,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of a phase validation check.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of validation errors
        warnings: List of non-blocking warnings
        blocked_items: Items that were blocked/stripped
        phase_spec: The spec used for validation
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    blocked_items: List[str]
    phase_spec: Optional[PhaseSpec] = None
    
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "blocked_items": self.blocked_items,
            "phase_name": self.phase_spec.name if self.phase_spec else None,
        }


class PhaseValidator:
    """
    Validates phase contracts and enforces restrictions.
    
    Responsibilities:
    - Validate council usage for a phase
    - Validate artifact types for a phase
    - Validate memory write operations
    - Strip/block forbidden output content
    - Log all validation decisions
    """
    
    # Patterns that indicate forbidden content types
    RECOMMENDATION_PATTERNS = [
        r'\brecommend\w*\b',
        r'\bshould\s+(consider|implement|use|adopt)\b',
        r'\baction\s+items?\b',
        r'\bnext\s+steps?\b',
        r'\bwe\s+suggest\b',
    ]
    
    SYNTHESIS_PATTERNS = [
        r'\bin\s+conclusion\b',
        r'\bfinal\s+(report|summary|analysis)\b',
        r'\bto\s+summarize\b',
        r'\boverall[,]?\s+(we|the)\b',
        r'\bsynthesis\b',
    ]
    
    CONCLUSION_PATTERNS = [
        r'\bconclusion\b',
        r'\bfinal\s+assessment\b',
        r'\bour\s+verdict\b',
        r'\bdecision\s*:\b',
    ]
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self._validation_log: List[Dict] = []
    
    def validate_council_for_phase(
        self,
        phase_spec: PhaseSpec,
        council_name: str
    ) -> ValidationResult:
        """
        Validate if a council can run in a phase.
        
        Args:
            phase_spec: The phase specification
            council_name: Name of the council to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        blocked = []
        
        if not phase_spec.is_council_allowed(council_name):
            errors.append(
                f"Council '{council_name}' is not allowed in phase '{phase_spec.name}'. "
                f"Allowed: {phase_spec.allowed_councils}, Forbidden: {phase_spec.forbidden_councils}"
            )
            blocked.append(council_name)
        
        # Log the validation
        self._log_validation(
            "council",
            phase_spec.name,
            council_name,
            len(errors) == 0,
            errors
        )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            blocked_items=blocked,
            phase_spec=phase_spec,
        )
    
    def validate_artifact_for_phase(
        self,
        phase_spec: PhaseSpec,
        artifact_type: str
    ) -> ValidationResult:
        """
        Validate if an artifact type can be produced in a phase.
        
        Args:
            phase_spec: The phase specification
            artifact_type: Type of artifact to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        blocked = []
        
        if not phase_spec.is_artifact_allowed(artifact_type):
            if self.strict_mode:
                errors.append(
                    f"Artifact type '{artifact_type}' is forbidden in phase '{phase_spec.name}'"
                )
            else:
                warnings.append(
                    f"Artifact type '{artifact_type}' is not ideal for phase '{phase_spec.name}'"
                )
            blocked.append(artifact_type)
        
        self._log_validation(
            "artifact",
            phase_spec.name,
            artifact_type,
            len(errors) == 0,
            errors + warnings
        )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            blocked_items=blocked,
            phase_spec=phase_spec,
        )
    
    def validate_memory_write(
        self,
        phase_spec: PhaseSpec,
        write_type: str,
        content_size: int = 0
    ) -> ValidationResult:
        """
        Validate if a memory write is allowed in a phase.
        
        Args:
            phase_spec: The phase specification
            write_type: Type of write (stable, ephemeral, delta)
            content_size: Size of content to write
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        blocked = []
        
        if not phase_spec.can_write_memory(write_type):
            errors.append(
                f"Memory write type '{write_type}' is not allowed in phase '{phase_spec.name}' "
                f"(policy: {phase_spec.memory_write_policy.value})"
            )
            blocked.append(f"memory:{write_type}")
        
        # Check token budget (rough estimate: 4 chars per token)
        estimated_tokens = content_size // 4
        if estimated_tokens > phase_spec.max_tokens:
            warnings.append(
                f"Content size ({estimated_tokens} est. tokens) exceeds phase budget ({phase_spec.max_tokens})"
            )
        
        self._log_validation(
            "memory",
            phase_spec.name,
            write_type,
            len(errors) == 0,
            errors + warnings
        )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            blocked_items=blocked,
            phase_spec=phase_spec,
        )
    
    def validate_output_content(
        self,
        phase_spec: PhaseSpec,
        output: Any
    ) -> ValidationResult:
        """
        Validate output content against phase restrictions.
        
        Checks for forbidden content patterns like recommendations
        in reconnaissance phase.
        
        Args:
            phase_spec: The phase specification
            output: Output content to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        blocked = []
        
        output_str = str(output).lower()
        
        # Check for forbidden artifact types in content
        for forbidden in phase_spec.forbidden_artifacts:
            forbidden_lower = forbidden.lower()
            if forbidden_lower in output_str:
                if self.strict_mode:
                    errors.append(f"Output contains forbidden content: '{forbidden}'")
                else:
                    warnings.append(f"Output may contain forbidden content: '{forbidden}'")
                blocked.append(forbidden)
        
        # Check specific patterns if synthesis is forbidden
        if "synthesis" in [f.lower() for f in phase_spec.forbidden_artifacts]:
            for pattern in self.SYNTHESIS_PATTERNS:
                if re.search(pattern, output_str, re.IGNORECASE):
                    warnings.append(f"Output contains synthesis-like content")
                    break
        
        # Check specific patterns if recommendations are forbidden
        if "recommendations" in [f.lower() for f in phase_spec.forbidden_artifacts]:
            for pattern in self.RECOMMENDATION_PATTERNS:
                if re.search(pattern, output_str, re.IGNORECASE):
                    warnings.append(f"Output contains recommendation-like content")
                    break
        
        # Check if trying to trigger arbiter when not allowed
        if not phase_spec.can_trigger_arbiter:
            if "arbiter" in output_str or "final decision" in output_str:
                warnings.append("Output references arbiter but phase cannot trigger it")
        
        # Check synthesis when not allowed
        if not phase_spec.can_run_synthesis:
            for pattern in self.CONCLUSION_PATTERNS:
                if re.search(pattern, output_str, re.IGNORECASE):
                    warnings.append("Output contains conclusion-like content but synthesis not allowed")
                    break
        
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        self._log_validation(
            "content",
            phase_spec.name,
            f"output_length={len(output_str)}",
            is_valid,
            errors + warnings
        )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            blocked_items=blocked,
            phase_spec=phase_spec,
        )
    
    def block_forbidden_output(
        self,
        phase_spec: PhaseSpec,
        output: Any
    ) -> Tuple[Any, List[str]]:
        """
        Strip or block forbidden content from output.
        
        Args:
            phase_spec: The phase specification
            output: Output to sanitize
            
        Returns:
            Tuple of (sanitized_output, list of blocked items)
        """
        blocked = []
        
        if output is None:
            return output, blocked
        
        # Handle dict outputs
        if isinstance(output, dict):
            sanitized = {}
            for key, value in output.items():
                if not phase_spec.is_artifact_allowed(key):
                    blocked.append(key)
                    logger.info(f"Blocked artifact '{key}' in phase '{phase_spec.name}'")
                else:
                    sanitized[key] = value
            return sanitized, blocked
        
        # Handle object outputs with attributes
        if hasattr(output, '__dict__'):
            for attr in list(vars(output).keys()):
                if not phase_spec.is_artifact_allowed(attr):
                    blocked.append(attr)
                    try:
                        delattr(output, attr)
                        logger.info(f"Removed attribute '{attr}' in phase '{phase_spec.name}'")
                    except AttributeError:
                        pass
        
        return output, blocked
    
    def validate_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        state: Any = None
    ) -> ValidationResult:
        """
        Validate a phase transition is allowed.
        
        Args:
            from_phase: Current phase name
            to_phase: Target phase name
            state: Optional mission state for context
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        from_spec = get_phase_spec(from_phase)
        to_spec = get_phase_spec(to_phase)
        
        # Check for premature synthesis
        if to_spec.can_run_synthesis and not from_spec.can_run_synthesis:
            # Synthesis should only follow deep analysis or similar
            from_type = infer_phase_type(from_phase)
            if from_type in ["reconnaissance"]:
                warnings.append(
                    f"Transitioning from '{from_phase}' directly to synthesis phase '{to_phase}' "
                    "may skip important analysis"
                )
        
        # Log transition
        logger.info(f"Phase transition: {from_phase} -> {to_phase}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            blocked_items=[],
            phase_spec=to_spec,
        )
    
    def get_phase_spec_for(self, phase_name: str, strict: bool = False) -> PhaseSpec:
        """
        Get the appropriate PhaseSpec for a phase name.
        
        Args:
            phase_name: Name of the phase
            strict: If True, raise error for unknown phases
            
        Returns:
            PhaseSpec for the phase
        """
        return get_phase_spec(phase_name, strict=strict)
    
    def _log_validation(
        self,
        validation_type: str,
        phase_name: str,
        target: str,
        passed: bool,
        messages: List[str]
    ) -> None:
        """Log a validation event."""
        entry = {
            "type": validation_type,
            "phase": phase_name,
            "target": target,
            "passed": passed,
            "messages": messages,
        }
        self._validation_log.append(entry)
        
        if not passed:
            logger.warning(
                f"Phase validation failed [{validation_type}] in '{phase_name}': "
                f"{target} - {messages}"
            )
        elif messages:
            logger.debug(
                f"Phase validation [{validation_type}] in '{phase_name}': "
                f"{target} - warnings: {messages}"
            )
    
    def get_validation_log(self) -> List[Dict]:
        """Get the validation log."""
        return self._validation_log.copy()
    
    def clear_validation_log(self) -> None:
        """Clear the validation log."""
        self._validation_log.clear()


# Global validator instance
_validator: Optional[PhaseValidator] = None


def get_phase_validator(strict_mode: bool = False) -> PhaseValidator:
    """Get the global phase validator instance."""
    global _validator
    if _validator is None:
        _validator = PhaseValidator(strict_mode=strict_mode)
    return _validator

