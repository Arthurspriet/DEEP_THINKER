"""
Memory Policy for DeepThinker 2.0.

Defines memory discipline rules including:
- Per-phase memory caps
- Importance scoring
- Decay/compression rules
- Storage restrictions

Prevents uncontrolled memory growth and ensures quality over quantity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of content for importance scoring."""
    UNRESOLVED_QUESTION = "unresolved_question"
    VALIDATED_EVIDENCE = "validated_evidence"
    HYPOTHESIS = "hypothesis"
    SYNTHESIS = "synthesis"
    RAW_OUTPUT = "raw_output"
    FINDING = "finding"
    CITATION = "citation"
    ERROR = "error"
    UNKNOWN = "unknown"


# Importance weights by content type
IMPORTANCE_WEIGHTS: Dict[ContentType, float] = {
    ContentType.UNRESOLVED_QUESTION: 0.9,
    ContentType.VALIDATED_EVIDENCE: 0.85,
    ContentType.SYNTHESIS: 0.8,
    ContentType.CITATION: 0.75,
    ContentType.HYPOTHESIS: 0.7,
    ContentType.FINDING: 0.6,
    ContentType.ERROR: 0.5,
    ContentType.RAW_OUTPUT: 0.3,
    ContentType.UNKNOWN: 0.4,
}


@dataclass
class MemoryPolicy:
    """
    Policy governing memory storage discipline.
    
    Attributes:
        max_stable_chars_per_phase: Maximum stable memory per phase
        max_ephemeral_chars: Maximum ephemeral memory at any time
        max_delta_per_iteration: Maximum memory delta per iteration
        importance_threshold: Minimum importance score to store
        decay_rate: Decay factor applied per phase (0-1)
        compression_trigger_chars: Trigger compression at this size
        forbidden_raw_storage: Block verbatim large outputs
        max_item_chars: Maximum characters per memory item
        max_items_per_phase: Maximum discrete items per phase
    """
    max_stable_chars_per_phase: int = 2000
    max_ephemeral_chars: int = 3000
    max_delta_per_iteration: int = 1000
    importance_threshold: float = 0.5
    decay_rate: float = 0.1
    compression_trigger_chars: int = 5000
    forbidden_raw_storage: bool = True
    max_item_chars: int = 500
    max_items_per_phase: int = 20
    
    def should_store(
        self,
        content: str,
        content_type: ContentType,
        current_chars: int,
        is_stable: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if content should be stored.
        
        Args:
            content: Content to potentially store
            content_type: Type of the content
            current_chars: Current memory size in chars
            is_stable: Whether this is stable memory
            
        Returns:
            Tuple of (should_store, reason)
        """
        importance = self.score_importance(content, content_type)
        
        # Check importance threshold
        if importance < self.importance_threshold:
            return False, f"importance {importance:.2f} < threshold {self.importance_threshold}"
        
        # Check raw storage restriction
        if self.forbidden_raw_storage and content_type == ContentType.RAW_OUTPUT:
            if len(content) > self.max_item_chars:
                return False, "raw storage forbidden for large outputs"
        
        # Check size limits
        content_len = len(content)
        if is_stable:
            if current_chars + content_len > self.max_stable_chars_per_phase:
                return False, f"would exceed stable limit ({current_chars + content_len} > {self.max_stable_chars_per_phase})"
        else:
            if current_chars + content_len > self.max_ephemeral_chars:
                return False, f"would exceed ephemeral limit ({current_chars + content_len} > {self.max_ephemeral_chars})"
        
        return True, "approved"
    
    def score_importance(
        self,
        content: str,
        content_type: Optional[ContentType] = None
    ) -> float:
        """
        Score the importance of content for storage priority.
        
        Args:
            content: Content to score
            content_type: Optional explicit content type
            
        Returns:
            Importance score (0-1)
        """
        # Get base weight from content type
        if content_type is None:
            content_type = self.infer_content_type(content)
        
        base_weight = IMPORTANCE_WEIGHTS.get(content_type, 0.4)
        
        # Adjust based on content characteristics
        adjustments = 0.0
        
        # Questions are important
        if '?' in content:
            adjustments += 0.1
        
        # Citations/sources are important
        if any(kw in content.lower() for kw in ['source:', 'citation:', 'reference:', 'http']):
            adjustments += 0.1
        
        # Specific findings are important
        if any(kw in content.lower() for kw in ['found that', 'evidence shows', 'confirmed']):
            adjustments += 0.1
        
        # Very short content is less important
        if len(content) < 50:
            adjustments -= 0.1
        
        # Very long content without structure is less important
        if len(content) > 1000 and '\n' not in content:
            adjustments -= 0.15
        
        return min(1.0, max(0.0, base_weight + adjustments))
    
    def infer_content_type(self, content: str) -> ContentType:
        """
        Infer content type from content text.
        
        Args:
            content: Content to classify
            
        Returns:
            Inferred ContentType
        """
        content_lower = content.lower()
        
        # Check for questions
        if '?' in content and len(content) < 500:
            if any(kw in content_lower for kw in ['unresolved', 'unclear', 'unknown', 'need to']):
                return ContentType.UNRESOLVED_QUESTION
        
        # Check for evidence
        if any(kw in content_lower for kw in ['evidence:', 'found that', 'data shows', 'verified']):
            return ContentType.VALIDATED_EVIDENCE
        
        # Check for citations
        if any(kw in content_lower for kw in ['source:', 'citation:', 'reference:', '[1]', '[2]']):
            return ContentType.CITATION
        
        # Check for hypothesis
        if any(kw in content_lower for kw in ['hypothesis:', 'we hypothesize', 'likely that', 'possibly']):
            return ContentType.HYPOTHESIS
        
        # Check for synthesis
        if any(kw in content_lower for kw in ['synthesis:', 'conclusion:', 'summary:', 'in summary']):
            return ContentType.SYNTHESIS
        
        # Check for findings
        if any(kw in content_lower for kw in ['finding:', 'discovered', 'observed', 'noted']):
            return ContentType.FINDING
        
        # Check for errors
        if any(kw in content_lower for kw in ['error:', 'failed', 'exception', 'warning:']):
            return ContentType.ERROR
        
        # Large unstructured content is likely raw output
        if len(content) > 500:
            return ContentType.RAW_OUTPUT
        
        return ContentType.UNKNOWN
    
    def compute_decay(self, age_in_phases: int) -> float:
        """
        Compute decay factor based on age.
        
        Args:
            age_in_phases: Number of phases since creation
            
        Returns:
            Decay multiplier (0-1)
        """
        return max(0.0, 1.0 - (self.decay_rate * age_in_phases))
    
    def compress_content(
        self,
        content: str,
        target_chars: int,
        preserve_questions: bool = True
    ) -> str:
        """
        Compress content to fit within target size.
        
        Args:
            content: Content to compress
            target_chars: Target character count
            preserve_questions: Keep question lines intact
            
        Returns:
            Compressed content
        """
        if len(content) <= target_chars:
            return content
        
        lines = content.split('\n')
        preserved = []
        other = []
        
        for line in lines:
            # Preserve questions and key markers
            if preserve_questions and '?' in line:
                preserved.append(line)
            elif any(kw in line.lower() for kw in ['hypothesis', 'evidence', 'finding', 'conclusion']):
                preserved.append(line)
            else:
                other.append(line)
        
        # Start with preserved content
        result_lines = preserved.copy()
        current_len = sum(len(l) + 1 for l in result_lines)
        
        # Add other content until limit
        for line in other:
            line_len = len(line) + 1
            if current_len + line_len <= target_chars:
                result_lines.append(line)
                current_len += line_len
            elif current_len < target_chars - 20:
                # Truncate last line
                remaining = target_chars - current_len - 4
                if remaining > 20:
                    result_lines.append(line[:remaining] + "...")
                break
        
        result = '\n'.join(result_lines)
        
        # Final truncation if still too long
        if len(result) > target_chars:
            result = result[:target_chars - 3] + "..."
        
        return result
    
    def needs_compression(self, current_chars: int) -> bool:
        """Check if compression is needed."""
        return current_chars > self.compression_trigger_chars
    
    def get_limits_for_phase(self, phase_name: str) -> Dict[str, int]:
        """
        Get memory limits adjusted for phase type.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Dict with limit values
        """
        phase_lower = phase_name.lower()
        
        # Reconnaissance gets stricter limits
        if any(kw in phase_lower for kw in ['recon', 'gather', 'initial']):
            return {
                "stable": 0,  # No stable writes
                "ephemeral": self.max_ephemeral_chars,
                "delta": self.max_delta_per_iteration // 2,
            }
        
        # Synthesis gets larger limits
        if any(kw in phase_lower for kw in ['synth', 'final', 'report']):
            return {
                "stable": self.max_stable_chars_per_phase * 2,
                "ephemeral": self.max_ephemeral_chars,
                "delta": self.max_delta_per_iteration * 2,
            }
        
        # Default limits
        return {
            "stable": self.max_stable_chars_per_phase,
            "ephemeral": self.max_ephemeral_chars,
            "delta": self.max_delta_per_iteration,
        }


# Default policy instance
DEFAULT_MEMORY_POLICY = MemoryPolicy()

# Strict policy for resource-constrained execution
STRICT_MEMORY_POLICY = MemoryPolicy(
    max_stable_chars_per_phase=1000,
    max_ephemeral_chars=2000,
    max_delta_per_iteration=500,
    importance_threshold=0.6,
    decay_rate=0.15,
    compression_trigger_chars=3000,
    forbidden_raw_storage=True,
    max_item_chars=300,
    max_items_per_phase=10,
)

# Permissive policy for long-running missions
PERMISSIVE_MEMORY_POLICY = MemoryPolicy(
    max_stable_chars_per_phase=5000,
    max_ephemeral_chars=8000,
    max_delta_per_iteration=2000,
    importance_threshold=0.3,
    decay_rate=0.05,
    compression_trigger_chars=10000,
    forbidden_raw_storage=False,
    max_item_chars=1000,
    max_items_per_phase=50,
)

