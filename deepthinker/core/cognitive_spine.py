"""
Cognitive Spine for DeepThinker 2.0.

The central unifying layer that enforces:
- Schema coherence across all councils and phases
- Predictable output structures via contracts
- Resource allocation discipline (tokens, depth, iterations)
- Phase boundary validation
- Consensus engine availability
- Memory compression between phases

This module integrates with:
- All councils (via injection)
- ReasoningSupervisor (for validation and fallbacks)
- MissionOrchestrator (as the central coordinator)
- VerboseLogger (for decision logging)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..consensus import (
        MajorityVoteConsensus,
        WeightedBlendConsensus,
        CritiqueConsensus,
        SemanticDistanceConsensus,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

class SchemaVersion(Enum):
    """Schema version identifiers."""
    V1 = 1
    V2 = 2


class OutputLimits:
    """
    Single source of truth for output length limits.
    
    Phase 5.2: Centralized constants to replace magic number truncations.
    """
    MAX_ARTIFACT_CHARS = 10000  # Increased from 2000
    MAX_CONTEXT_CHARS = 8000    # Increased from 4000
    MAX_HISTORY_CHARS = 10000   # Increased from 5000
    MAX_INTERNAL_REASONING = 50000  # Keep high for internal use
    MAX_USER_FACING_CHARS = 10000  # For final artifacts


@dataclass
class ResourceBudget:
    """
    Resource budget for a phase or council execution.
    
    Tracks and limits resource consumption to prevent runaway expansions.
    """
    max_tokens: int = 50000
    max_output_chars: int = OutputLimits.MAX_INTERNAL_REASONING  # For internal reasoning
    max_user_facing_chars: int = OutputLimits.MAX_USER_FACING_CHARS  # For final artifacts
    max_iterations: int = 5
    max_depth: int = 3
    time_budget_seconds: float = 300.0
    
    # Tracking
    tokens_used: int = 0
    output_chars_used: int = 0
    iterations_used: int = 0
    current_depth: int = 0
    time_elapsed_seconds: float = 0.0
    
    def is_exceeded(self) -> bool:
        """Check if any budget limit is exceeded."""
        return (
            self.tokens_used >= self.max_tokens or
            self.output_chars_used >= self.max_output_chars or
            self.iterations_used >= self.max_iterations or
            self.current_depth >= self.max_depth
        )
    
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.tokens_used)
    
    def remaining_output_chars(self) -> int:
        """Get remaining output character budget."""
        return max(0, self.max_output_chars - self.output_chars_used)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "max_output_chars": self.max_output_chars,
            "max_iterations": self.max_iterations,
            "tokens_used": self.tokens_used,
            "output_chars_used": self.output_chars_used,
            "iterations_used": self.iterations_used,
            "is_exceeded": self.is_exceeded(),
        }


@dataclass
class ValidationResult:
    """
    Result of context validation.
    
    Contains validation status and any corrections made.
    """
    is_valid: bool
    context: Any  # The validated/corrected context
    missing_fields: List[str] = field(default_factory=list)
    unknown_fields: List[str] = field(default_factory=list)
    fields_filled: Dict[str, Any] = field(default_factory=dict)
    fields_stripped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def has_corrections(self) -> bool:
        """Check if any corrections were made."""
        return bool(self.fields_filled or self.fields_stripped)


@dataclass
class MemorySlot:
    """
    Memory slot for phase context management.
    
    Three types of slots:
    - stable: Persistent across all phases
    - ephemeral: Cleared after each phase
    - delta: Accumulated changes from iterations
    """
    stable: str = ""
    ephemeral: str = ""
    delta: str = ""
    max_stable_chars: int = 5000
    max_ephemeral_chars: int = 3000
    max_delta_chars: int = 2000
    
    def total_chars(self) -> int:
        """Get total character count across all slots."""
        return len(self.stable) + len(self.ephemeral) + len(self.delta)
    
    def compress_to_stable(self) -> None:
        """
        Compress ephemeral and delta into stable memory.
        
        Called at phase boundaries to consolidate learnings.
        """
        # Combine ephemeral and delta
        combined = f"{self.ephemeral}\n\n{self.delta}".strip()
        
        if not combined:
            return
        
        # Add to stable (truncating if needed)
        if self.stable:
            self.stable = f"{self.stable}\n\n---\n\n{combined}"
        else:
            self.stable = combined
        
        # Truncate stable to limit
        if len(self.stable) > self.max_stable_chars:
            self.stable = self.stable[-self.max_stable_chars:]
            # Find clean break point
            break_point = self.stable.find('\n\n')
            if break_point > 0 and break_point < 500:
                self.stable = self.stable[break_point + 2:]
        
        # Clear ephemeral and delta
        self.ephemeral = ""
        self.delta = ""
    
    def clear_ephemeral(self) -> None:
        """Clear ephemeral slot."""
        self.ephemeral = ""
    
    def add_to_delta(self, content: str) -> None:
        """Add content to delta slot."""
        if self.delta:
            self.delta = f"{self.delta}\n{content}"
        else:
            self.delta = content
        
        # Truncate if needed
        if len(self.delta) > self.max_delta_chars:
            self.delta = self.delta[-self.max_delta_chars:]


# =============================================================================
# Cognitive Spine Core
# =============================================================================

class CognitiveSpine:
    """
    Central unifying layer for DeepThinker mission execution.
    
    Responsibilities:
    1. Schema validation for all council contexts
    2. Consensus engine provisioning
    3. Resource budget tracking
    4. Memory compression between phases
    5. Phase boundary validation
    6. Output contract enforcement
    
    Usage:
        spine = CognitiveSpine(ollama_base_url="http://localhost:11434")
        
        # Validate context before council call
        result = spine.validate_context(context, "researcher_council")
        if not result.is_valid:
            context = spine.create_minimal_context("researcher_council", objective)
        
        # Get consensus engine for council
        consensus = spine.get_consensus_engine("voting", "researcher_council")
        
        # Track resources
        spine.track_output(output, "researcher_council")
        if spine.is_budget_exceeded("researcher_council"):
            spine.enter_contraction_mode()
    """
    
    # Default resource limits
    DEFAULT_MAX_OUTPUT_CHARS = 50000  # Increased from 10000 for complex research
    DEFAULT_MAX_TOKENS = 50000
    DEFAULT_MAX_ITERATIONS = 5
    
    # Contraction mode thresholds
    CONTRACTION_TIME_THRESHOLD = 0.20  # 20% time remaining
    CONTRACTION_OUTPUT_THRESHOLD = 20000  # chars - aligned with RESEARCHER_COUNCIL_SPEC
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        max_output_chars: int = 50000,  # Increased from 10000 for complex research
        max_tokens: int = 50000,
        enable_compression: bool = True,
        verbose_logger: Optional[Any] = None
    ):
        """
        Initialize the Cognitive Spine.
        
        Args:
            ollama_base_url: URL for Ollama server (used by consensus engines)
            max_output_chars: Maximum output characters before triggering compression
            max_tokens: Maximum token budget per phase
            enable_compression: Whether to enable automatic memory compression
            verbose_logger: Optional VerboseLogger instance for logging decisions
        """
        self.ollama_base_url = ollama_base_url
        self.max_output_chars = max_output_chars
        self.max_tokens = max_tokens
        self.enable_compression = enable_compression
        self._verbose_logger = verbose_logger
        
        # Consensus engine cache
        self._consensus_engines: Dict[str, Any] = {}
        
        # Resource budgets per council
        self._budgets: Dict[str, ResourceBudget] = {}
        
        # Memory slots per phase
        self._memory_slots: Dict[str, MemorySlot] = {}
        
        # Contraction mode state
        self._contraction_mode: bool = False
        
        # Import schemas lazily to avoid circular imports
        self._schemas_loaded = False
        self._schema_registry: Dict[str, type] = {}
    
    def _ensure_schemas_loaded(self) -> None:
        """Lazily load schema registry."""
        if self._schemas_loaded:
            return
        
        try:
            from ..schemas.contexts import CONTEXT_SCHEMAS
            self._schema_registry = CONTEXT_SCHEMAS
            self._schemas_loaded = True
        except ImportError as e:
            logger.warning(f"Could not load schemas: {e}")
            self._schema_registry = {}
            self._schemas_loaded = True
    
    # =========================================================================
    # Schema Validation
    # =========================================================================
    
    def validate_context(
        self,
        context: Any,
        council_name: str
    ) -> ValidationResult:
        """
        Validate and sanitize context for a council.
        
        Performs:
        1. Check against canonical schema
        2. Strip unknown fields (with warning)
        3. Fill missing optional fields with defaults
        4. Check mandatory fields are present
        
        Args:
            context: Context object to validate
            council_name: Name of the target council
            
        Returns:
            ValidationResult with validated context and corrections
        """
        self._ensure_schemas_loaded()
        
        result = ValidationResult(is_valid=True, context=context)
        
        # Get schema for council
        schema_class = self._schema_registry.get(council_name.lower())
        if schema_class is None:
            # No schema defined - pass through
            result.warnings.append(f"No schema defined for {council_name}")
            return result
        
        # Import validation utilities
        try:
            from ..schemas.contexts import (
                validate_context_fields,
                get_default_values,
                get_known_fields,
            )
        except ImportError:
            result.warnings.append("Could not import schema validation utilities")
            return result
        
        # Validate fields
        is_valid, missing_mandatory, unknown_fields = validate_context_fields(
            context, schema_class
        )
        
        result.missing_fields = missing_mandatory
        result.unknown_fields = unknown_fields
        result.is_valid = is_valid
        
        # Handle unknown fields
        if unknown_fields:
            result.fields_stripped = unknown_fields
            context = self._strip_unknown_fields(context, schema_class)
            result.context = context
            
            # Log the correction
            self._log_schema_correction(
                council_name, 
                unknown_fields, 
                "stripped unknown fields"
            )
        
        # Handle missing optional fields
        if not is_valid:
            defaults = get_default_values(schema_class)
            context, filled = self._fill_defaults(context, defaults, missing_mandatory)
            result.context = context
            result.fields_filled = filled
            
            # Re-validate after filling
            still_missing = [f for f in missing_mandatory if f not in filled]
            result.is_valid = len(still_missing) == 0
            result.missing_fields = still_missing
        
        # Log the validation result
        self.log_context_validation(council_name, result)
        
        return result
    
    def _strip_unknown_fields(self, context: Any, schema_class: type) -> Any:
        """Strip unknown fields from context."""
        from ..schemas.contexts import get_known_fields
        
        known = get_known_fields(schema_class)
        
        if isinstance(context, dict):
            return {k: v for k, v in context.items() if k in known or k.startswith('_')}
        
        # For dataclass/object, we can't easily strip - return as is
        return context
    
    def _fill_defaults(
        self,
        context: Any,
        defaults: Dict[str, Any],
        missing: List[str]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Fill missing fields with defaults."""
        filled = {}
        
        if isinstance(context, dict):
            for field_name in missing:
                if field_name in defaults:
                    context[field_name] = defaults[field_name]
                    filled[field_name] = defaults[field_name]
            return context, filled
        
        # For dataclass/object
        for field_name in missing:
            if field_name in defaults and hasattr(context, field_name):
                try:
                    setattr(context, field_name, defaults[field_name])
                    filled[field_name] = defaults[field_name]
                except AttributeError:
                    pass
        
        return context, filled
    
    def create_minimal_context(
        self,
        council_name: str,
        objective: str
    ) -> Dict[str, Any]:
        """
        Create a minimal valid context for a council.
        
        Used as fallback when context validation fails completely.
        
        Args:
            council_name: Name of the target council
            objective: The primary objective
            
        Returns:
            Minimal valid context dictionary
        """
        self._ensure_schemas_loaded()
        
        schema_class = self._schema_registry.get(council_name.lower())
        
        if schema_class is None:
            return {"objective": objective}
        
        try:
            from ..schemas.contexts import get_default_values
            defaults = get_default_values(schema_class)
        except ImportError:
            defaults = {}
        
        # Start with defaults
        context = dict(defaults)
        
        # Always set objective
        context["objective"] = objective
        
        # Set other common mandatory fields
        if "content_to_evaluate" in context:
            context["content_to_evaluate"] = objective
        if "prior_findings" in context:
            context["prior_findings"] = ""
        if "scenario_description" in context:
            context["scenario_description"] = objective
        
        return context
    
    # =========================================================================
    # Consensus Engine Provisioning
    # =========================================================================
    
    def get_consensus_engine(
        self,
        algorithm: str = "voting",
        council_name: str = ""
    ) -> Any:
        """
        Get a consensus engine instance.
        
        Caches engines for reuse across councils.
        
        Args:
            algorithm: Consensus algorithm name
                      ("voting", "weighted_blend", "critique_exchange", "semantic_distance")
            council_name: Name of the requesting council (for logging)
            
        Returns:
            Consensus engine instance
        """
        cache_key = f"{algorithm}_{self.ollama_base_url}"
        
        if cache_key in self._consensus_engines:
            return self._consensus_engines[cache_key]
        
        # Import and create engine
        try:
            from ..consensus import get_consensus_engine as factory
            engine = factory(algorithm)
            
            # Configure with our base URL if applicable
            if hasattr(engine, 'ollama_base_url'):
                engine.ollama_base_url = self.ollama_base_url
            
            self._consensus_engines[cache_key] = engine
            
            logger.debug(f"Created consensus engine '{algorithm}' for {council_name}")
            return engine
            
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not create consensus engine '{algorithm}': {e}")
            # Return a simple passthrough
            return self._create_fallback_consensus()
    
    def _create_fallback_consensus(self) -> Any:
        """Create a simple fallback consensus that picks the first output."""
        
        class FallbackConsensus:
            """Simple fallback that returns first successful output."""
            
            def apply(self, outputs: Dict[str, Any]) -> Any:
                for name, output in outputs.items():
                    if hasattr(output, 'success') and output.success:
                        return output
                    elif isinstance(output, str) and output:
                        return output
                return next(iter(outputs.values())) if outputs else None
        
        return FallbackConsensus()
    
    # =========================================================================
    # Resource Budget Management
    # =========================================================================
    
    def get_budget(self, council_name: str) -> ResourceBudget:
        """
        Get or create resource budget for a council.
        
        Args:
            council_name: Name of the council
            
        Returns:
            ResourceBudget for the council
        """
        if council_name not in self._budgets:
            self._budgets[council_name] = ResourceBudget(
                max_tokens=self.max_tokens,
                max_output_chars=self.max_output_chars,
                max_iterations=self.DEFAULT_MAX_ITERATIONS,
            )
        return self._budgets[council_name]
    
    def track_output(
        self,
        output: Any,
        council_name: str,
        log_status: bool = False
    ) -> None:
        """
        Track output size against resource budget.
        
        Args:
            output: The output to track
            council_name: Name of the producing council
            log_status: Whether to log budget status after tracking
        """
        budget = self.get_budget(council_name)
        
        # Estimate output size
        if hasattr(output, 'raw_output'):
            chars = len(str(output.raw_output))
        elif isinstance(output, str):
            chars = len(output)
        elif isinstance(output, dict):
            chars = len(str(output))
        else:
            chars = len(str(output))
        
        budget.output_chars_used += chars
        budget.iterations_used += 1
        
        # Log tracking
        logger.debug(
            f"[SPINE] Tracked output [{council_name}]: +{chars} chars, "
            f"iteration {budget.iterations_used}/{budget.max_iterations}"
        )
        
        # Check if contraction mode should be triggered
        if chars > self.CONTRACTION_OUTPUT_THRESHOLD:
            self._log_spine_decision(
                "output_size_warning",
                {"council": council_name, "chars": chars, "threshold": self.CONTRACTION_OUTPUT_THRESHOLD}
            )
        
        # Log budget status if requested or near limits
        if log_status or budget.is_exceeded():
            self.log_budget_status(council_name)
    
    def is_budget_exceeded(self, council_name: str) -> bool:
        """
        Check if resource budget is exceeded for a council.
        
        Args:
            council_name: Name of the council
            
        Returns:
            True if budget exceeded
        """
        budget = self.get_budget(council_name)
        return budget.is_exceeded()
    
    def reset_budget(self, council_name: str) -> None:
        """
        Reset resource budget for a council.
        
        Args:
            council_name: Name of the council
        """
        if council_name in self._budgets:
            del self._budgets[council_name]
    
    # =========================================================================
    # Memory Compression
    # =========================================================================
    
    def get_memory_slot(self, phase_name: str) -> MemorySlot:
        """
        Get or create memory slot for a phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            MemorySlot for the phase
        """
        if phase_name not in self._memory_slots:
            self._memory_slots[phase_name] = MemorySlot()
        return self._memory_slots[phase_name]
    
    def _intelligent_summarize(
        self,
        text: str,
        max_chars: int,
        preserve_headers: bool = True,
        max_bullets_per_section: int = 5
    ) -> str:
        """
        Intelligently summarize text while preserving structure.
        
        Phase 5.3: Replaces simple truncation with structured summarization.
        - Preserves all headers (##, ###, **)
        - Preserves first N bullet points per section
        - Summarizes content paragraphs (first sentence + key phrases)
        
        Args:
            text: Text to summarize
            max_chars: Maximum characters for output
            preserve_headers: Whether to preserve all headers
            max_bullets_per_section: Maximum bullets to preserve per section
            
        Returns:
            Summarized text with footer indicating original length
        """
        if len(text) <= max_chars:
            return text
        
        original_length = len(text)
        lines = text.split('\n')
        result_lines = []
        current_chars = 0
        current_section_bullets = 0
        in_section = False
        section_header = None
        
        # Reserve space for footer
        footer = f"\n\n[... content summarized, original length: {original_length:,} chars ...]"
        footer_size = len(footer)
        available_chars = max_chars - footer_size
        
        for i, line in enumerate(lines):
            line_chars = len(line) + 1  # +1 for newline
            line_stripped = line.strip()
            
            # Detect section headers
            is_header = (line_stripped.startswith('#') or 
                        line_stripped.startswith('**') and line_stripped.endswith('**'))
            
            # Detect bullet points
            is_bullet = line_stripped.startswith(('-', '*', '•')) and not is_header
            
            if is_header:
                # New section - reset bullet counter
                current_section_bullets = 0
                in_section = True
                section_header = line
                
                # Always keep headers if we have space
                if preserve_headers and current_chars + line_chars <= available_chars:
                    result_lines.append(line)
                    current_chars += line_chars
                elif preserve_headers:
                    # Header won't fit - truncate content to make room
                    break
            elif is_bullet and in_section:
                # Preserve first N bullets per section
                if current_section_bullets < max_bullets_per_section:
                    if current_chars + line_chars <= available_chars - 100:  # Reserve some space
                        result_lines.append(line)
                        current_chars += line_chars
                        current_section_bullets += 1
                    else:
                        result_lines.append("  ... (more bullets truncated)")
                        break
            else:
                # Regular content - keep first sentence if it fits
                if current_chars + 100 > available_chars:  # Approaching limit
                    break
                
                # Try to preserve first sentence of paragraphs
                sentences = line_stripped.split('. ')
                if sentences and sentences[0]:
                    first_sentence = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
                    if len(first_sentence) + current_chars + 2 <= available_chars:
                        result_lines.append(first_sentence)
                        current_chars += len(first_sentence) + 2
                    # Skip if too long
                # Skip long paragraphs entirely to save space
        
        # Add footer
        result = '\n'.join(result_lines) + footer
        return result
    
    def compress_text(
        self,
        text: str,
        max_chars: int = OutputLimits.MAX_ARTIFACT_CHARS,
        preserve_structure: bool = True
    ) -> str:
        """
        Compress text to fit within character limit using intelligent summarization.
        
        Phase 5.3: Uses _intelligent_summarize() instead of simple truncation.
        
        Args:
            text: Text to compress
            max_chars: Maximum characters (defaults to OutputLimits.MAX_ARTIFACT_CHARS)
            preserve_structure: Try to preserve section headers and bullets
            
        Returns:
            Compressed text with preserved structure
        """
        if len(text) <= max_chars:
            return text
        
        if not preserve_structure:
            # Fallback to simple truncation if structure preservation disabled
            return text[:max_chars - 3] + "..."
        
        # Use intelligent summarization
        return self._intelligent_summarize(text, max_chars, preserve_headers=True)
    
    def compress_context(
        self,
        context: Dict[str, Any],
        max_total_chars: int = 5000
    ) -> Dict[str, Any]:
        """
        Compress a context dictionary to fit within limits.
        
        Prioritizes:
        1. Keep objective/mandatory fields
        2. Compress prior_knowledge
        3. Truncate lists
        
        Args:
            context: Context dictionary to compress
            max_total_chars: Maximum total characters
            
        Returns:
            Compressed context dictionary
        """
        result = dict(context)
        
        # Calculate current size
        current_size = sum(len(str(v)) for v in result.values())
        
        if current_size <= max_total_chars:
            return result
        
        # Compress prior_knowledge first (usually largest)
        if 'prior_knowledge' in result and result['prior_knowledge']:
            pk = result['prior_knowledge']
            if len(pk) > max_total_chars // 3:
                result['prior_knowledge'] = self.compress_text(pk, max_total_chars // 3)
        
        # Truncate lists
        list_fields = [
            'focus_areas', 'data_needs', 'unresolved_questions',
            'subgoals', 'key_points', 'recommendations'
        ]
        for field_name in list_fields:
            if field_name in result and isinstance(result[field_name], list):
                result[field_name] = result[field_name][:5]
        
        # Recalculate size
        current_size = sum(len(str(v)) for v in result.values())
        
        # If still too large, more aggressive compression
        if current_size > max_total_chars:
            if 'prior_knowledge' in result:
                result['prior_knowledge'] = self.compress_text(
                    result['prior_knowledge'], 
                    max_total_chars // 4
                )
        
        return result
    
    def compress_phase_memory(self, phase_name: str) -> None:
        """
        Compress memory at phase boundary.
        
        Moves ephemeral and delta to stable, clearing them.
        
        Args:
            phase_name: Name of the completed phase
        """
        slot = self.get_memory_slot(phase_name)
        
        # Track before sizes for logging
        before_ephemeral = len(slot.ephemeral)
        before_delta = len(slot.delta)
        
        slot.compress_to_stable()
        
        self._log_spine_decision(
            "memory_compression",
            {
                "phase": phase_name, 
                "stable_chars": len(slot.stable),
                "compressed_from": before_ephemeral + before_delta,
            }
        )
        
        logger.info(
            f"[SPINE] Phase '{phase_name}' memory compressed: "
            f"ephemeral({before_ephemeral}) + delta({before_delta}) -> "
            f"stable({len(slot.stable)} chars)"
        )
    
    # =========================================================================
    # Contraction Mode
    # =========================================================================
    
    def enter_contraction_mode(self) -> None:
        """
        Enter contraction mode.
        
        In contraction mode:
        - Skip non-essential phases
        - Force synthesis with compressed memory
        - Use smaller models
        """
        if not self._contraction_mode:
            self._contraction_mode = True
            self._log_spine_decision("contraction_mode_entered", {})
            logger.warning("CognitiveSpine entering contraction mode")
    
    def exit_contraction_mode(self) -> None:
        """Exit contraction mode."""
        if self._contraction_mode:
            self._contraction_mode = False
            self._log_spine_decision("contraction_mode_exited", {})
    
    def is_contraction_mode(self) -> bool:
        """Check if in contraction mode."""
        return self._contraction_mode
    
    def should_enter_contraction(
        self,
        time_remaining: float,
        total_time: float,
        output_chars: int = 0
    ) -> bool:
        """
        Check if contraction mode should be triggered.
        
        Args:
            time_remaining: Remaining time in seconds
            total_time: Total time budget in seconds
            output_chars: Current output size in characters
            
        Returns:
            True if contraction mode should be entered
        """
        if self._contraction_mode:
            return True
        
        # Check time threshold
        if total_time > 0:
            time_ratio = time_remaining / total_time
            if time_ratio < self.CONTRACTION_TIME_THRESHOLD:
                return True
        
        # Check output size
        if output_chars > self.CONTRACTION_OUTPUT_THRESHOLD:
            return True
        
        return False
    
    # =========================================================================
    # Phase Boundary Validation
    # =========================================================================
    
    def validate_phase_boundary(
        self,
        phase_name: str,
        context: Any,
        previous_output: Optional[Any] = None
    ) -> Tuple[bool, Any, List[str]]:
        """
        Validate before starting a phase.
        
        Checks:
        1. Context schema valid
        2. Previous output not too large
        3. Resource budget not exceeded
        4. Memory compressed if needed
        
        Args:
            phase_name: Name of the phase to start
            context: Context for the phase
            previous_output: Output from previous phase (optional)
            
        Returns:
            Tuple of (is_valid, corrected_context, warnings)
        """
        warnings = []
        corrected_context = context
        
        # Determine council name from phase
        council_name = self._infer_council_from_phase(phase_name)
        
        # Validate context
        validation = self.validate_context(context, council_name)
        if not validation.is_valid:
            warnings.append(f"Context validation failed: {validation.missing_fields}")
            # Try to create minimal context
            objective = ""
            if hasattr(context, 'objective'):
                objective = context.objective
            elif isinstance(context, dict):
                objective = context.get('objective', '')
            corrected_context = self.create_minimal_context(council_name, objective)
        elif validation.has_corrections():
            corrected_context = validation.context
            warnings.extend(validation.warnings)
        
        # Check previous output size
        if previous_output is not None:
            output_size = len(str(previous_output))
            if output_size > self.max_output_chars:
                warnings.append(f"Previous output large ({output_size} chars), compressing")
                self.compress_phase_memory(phase_name)
        
        # Check budget
        if self.is_budget_exceeded(council_name):
            warnings.append(f"Resource budget exceeded for {council_name}")
        
        # Compress if in contraction mode
        if self._contraction_mode and isinstance(corrected_context, dict):
            corrected_context = self.compress_context(corrected_context)
            warnings.append("Context compressed (contraction mode)")
        
        is_valid = len([w for w in warnings if "failed" in w.lower()]) == 0
        
        return is_valid, corrected_context, warnings
    
    def _infer_council_from_phase(self, phase_name: str) -> str:
        """Infer council name from phase name."""
        name_lower = phase_name.lower()
        
        if any(kw in name_lower for kw in ["research", "recon", "gather", "context"]):
            return "researcher_council"
        elif any(kw in name_lower for kw in ["plan", "design", "strategy"]):
            return "planner_council"
        elif any(kw in name_lower for kw in ["eval", "test", "validation"]):
            return "evaluator_council"
        elif any(kw in name_lower for kw in ["code", "implement", "build"]):
            return "coder_council"
        elif any(kw in name_lower for kw in ["simul", "scenario"]):
            return "simulation_council"
        elif any(kw in name_lower for kw in ["synth", "report", "final"]):
            return "planner_council"  # Synthesis uses planner
        else:
            return "researcher_council"  # Default
    
    # =========================================================================
    # Output Contract Enforcement
    # =========================================================================
    
    def enforce_output_contract(
        self,
        output: Any,
        council_name: str,
        phase_name: str = "",
        iteration: int = 1
    ) -> Any:
        """
        Normalize output to standard contract.
        
        Args:
            output: Raw council output
            council_name: Name of the producing council
            phase_name: Name of the phase
            iteration: Current iteration number
            
        Returns:
            Normalized CouncilOutputContract
        """
        try:
            from ..schemas.outputs import normalize_output_to_contract
            return normalize_output_to_contract(
                output, council_name, phase_name, iteration
            )
        except ImportError:
            # Fallback - return as is
            return output
    
    # =========================================================================
    # Logging Integration - Verbose Visibility
    # =========================================================================
    
    def _log_schema_correction(
        self,
        council_name: str,
        fields: List[str],
        action: str
    ) -> None:
        """Log schema correction via verbose logger with visible output."""
        msg = f"[SPINE] Schema correction [{council_name}]: {action} - {fields}"
        
        if self._verbose_logger and hasattr(self._verbose_logger, 'log_schema_correction'):
            for field_name in fields:
                self._verbose_logger.log_schema_correction(council_name, field_name, action)
        
        # Always log at INFO level for visibility
        logger.info(msg)
    
    def _log_spine_decision(
        self,
        decision_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Log spine decision via verbose logger with visible output."""
        # Build readable message
        if decision_type == "context_validation":
            msg = f"[SPINE] Context validated: {details.get('council', 'unknown')}"
            if details.get('corrections'):
                msg += f" (corrections: {details['corrections']})"
        elif decision_type == "memory_compression":
            msg = f"[SPINE] Memory compressed: {details.get('phase', 'unknown')} -> {details.get('stable_chars', 0)} chars"
        elif decision_type == "budget_warning":
            msg = f"[SPINE] Budget warning [{details.get('council', 'unknown')}]: {details.get('metric', '')} at {details.get('usage', 0)}/{details.get('limit', 0)}"
        elif decision_type == "contraction_mode_entered":
            msg = "[SPINE] Entering CONTRACTION MODE - time/resources exhausted"
        elif decision_type == "contraction_mode_exited":
            msg = "[SPINE] Exiting contraction mode"
        elif decision_type == "output_size_warning":
            msg = f"[SPINE] Large output warning: {details.get('chars', 0)} chars from {details.get('council', 'unknown')}"
        else:
            msg = f"[SPINE] {decision_type}: {details}"
        
        if self._verbose_logger and hasattr(self._verbose_logger, 'log_spine_decision'):
            self._verbose_logger.log_spine_decision(decision_type, details)
        
        # Always log at INFO level for visibility
        logger.info(msg)
    
    def log_context_validation(
        self,
        council_name: str,
        result: "ValidationResult"
    ) -> None:
        """
        Log context validation result with visibility.
        
        Args:
            council_name: Council being validated for
            result: ValidationResult from validate_context()
        """
        details = {
            "council": council_name,
            "is_valid": result.is_valid,
            "corrections": {
                "stripped": result.fields_stripped,
                "filled": list(result.fields_filled.keys()),
            } if result.has_corrections() else None,
            "warnings": result.warnings,
        }
        
        if result.is_valid and not result.has_corrections():
            logger.info(f"[SPINE] Context valid for {council_name}")
        elif result.is_valid:
            self._log_spine_decision("context_validation", details)
        else:
            logger.warning(f"[SPINE] Context invalid for {council_name}: missing={result.missing_fields}")
    
    def log_phase_boundary(
        self,
        phase_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log phase boundary event.
        
        Args:
            phase_name: Name of the phase
            action: "entering", "completed", "skipped"
            details: Optional additional details
        """
        if action == "entering":
            msg = f"[SPINE] Phase boundary -> entering '{phase_name}'"
        elif action == "completed":
            msg = f"[SPINE] Phase boundary -> completed '{phase_name}'"
        elif action == "skipped":
            msg = f"[SPINE] Phase boundary -> SKIPPED '{phase_name}' (contraction mode)"
        else:
            msg = f"[SPINE] Phase boundary [{action}]: {phase_name}"
        
        if details:
            msg += f" | {details}"
        
        logger.info(msg)
    
    def log_budget_status(self, council_name: str) -> None:
        """
        Log current budget status for a council.
        
        Args:
            council_name: Council to check
        """
        budget = self.get_budget(council_name)
        
        # Calculate percentages
        tokens_pct = (budget.tokens_used / budget.max_tokens * 100) if budget.max_tokens > 0 else 0
        chars_pct = (budget.output_chars_used / budget.max_output_chars * 100) if budget.max_output_chars > 0 else 0
        iters_pct = (budget.iterations_used / budget.max_iterations * 100) if budget.max_iterations > 0 else 0
        
        logger.info(
            f"[SPINE] Budget [{council_name}]: "
            f"tokens={budget.tokens_used}/{budget.max_tokens} ({tokens_pct:.0f}%), "
            f"chars={budget.output_chars_used}/{budget.max_output_chars} ({chars_pct:.0f}%), "
            f"iterations={budget.iterations_used}/{budget.max_iterations} ({iters_pct:.0f}%)"
        )
        
        # Warn if near limits
        if tokens_pct > 80:
            self._log_spine_decision("budget_warning", {
                "council": council_name,
                "metric": "tokens",
                "usage": budget.tokens_used,
                "limit": budget.max_tokens,
            })
        if chars_pct > 80:
            self._log_spine_decision("budget_warning", {
                "council": council_name,
                "metric": "output_chars",
                "usage": budget.output_chars_used,
                "limit": budget.max_output_chars,
            })
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def reset(self) -> None:
        """Reset all spine state for a new mission."""
        self._budgets.clear()
        self._memory_slots.clear()
        self._contraction_mode = False
        logger.debug("CognitiveSpine reset")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current spine state."""
        return {
            "contraction_mode": self._contraction_mode,
            "budgets": {k: v.to_dict() for k, v in self._budgets.items()},
            "memory_slots": {
                k: {"stable_chars": len(v.stable), "total_chars": v.total_chars()}
                for k, v in self._memory_slots.items()
            },
            "consensus_engines_cached": list(self._consensus_engines.keys()),
        }

