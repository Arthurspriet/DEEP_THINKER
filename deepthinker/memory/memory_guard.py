"""
Memory Guard for DeepThinker 2.0.

Enforces memory discipline by:
- Scoring content importance
- Enforcing per-phase memory caps
- Applying decay to old content
- Compressing when needed
- Blocking forbidden writes

Integration:
- MemoryManager calls guard before all writes
- CognitiveSpine uses for phase boundary compression
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .memory_policy import (
    MemoryPolicy,
    ContentType,
    DEFAULT_MEMORY_POLICY,
    STRICT_MEMORY_POLICY,
)
from ..schemas.phase_spec import PhaseSpec, MemoryWritePolicy, get_phase_spec

logger = logging.getLogger(__name__)


@dataclass
class MemoryWriteRequest:
    """
    A request to write to memory.
    
    Attributes:
        content: Content to write
        phase_name: Current phase
        write_type: Type of write (stable, ephemeral, delta)
        content_type: Type of content
        source: Source of the content (council name, etc.)
        importance_override: Optional manual importance score
    """
    content: str
    phase_name: str
    write_type: str = "ephemeral"
    content_type: Optional[ContentType] = None
    source: str = ""
    importance_override: Optional[float] = None


@dataclass
class MemoryWriteResult:
    """
    Result of a memory write attempt.
    
    Attributes:
        allowed: Whether the write was allowed
        content: The content (possibly compressed)
        reason: Reason for the decision
        importance_score: Calculated importance
        original_size: Original content size
        final_size: Final content size after compression
    """
    allowed: bool
    content: str
    reason: str
    importance_score: float
    original_size: int
    final_size: int
    
    def was_compressed(self) -> bool:
        return self.final_size < self.original_size


class MemoryGuard:
    """
    Guards memory writes to enforce discipline.
    
    Responsibilities:
    - Score content importance
    - Check phase memory policies
    - Enforce per-phase caps
    - Apply compression when needed
    - Track memory usage per phase
    - Apply decay to aging content
    """
    
    def __init__(
        self,
        policy: Optional[MemoryPolicy] = None,
        strict_mode: bool = False
    ):
        """
        Initialize the memory guard.
        
        Args:
            policy: Memory policy to enforce
            strict_mode: If True, use stricter limits
        """
        if policy is None:
            policy = STRICT_MEMORY_POLICY if strict_mode else DEFAULT_MEMORY_POLICY
        
        self.policy = policy
        self.strict_mode = strict_mode
        
        # Track usage per phase
        self._phase_usage: Dict[str, Dict[str, int]] = {}
        self._phase_item_counts: Dict[str, int] = {}
        self._write_log: List[Dict] = []
    
    def check_write(
        self,
        request: MemoryWriteRequest,
        current_stable_chars: int = 0,
        current_ephemeral_chars: int = 0
    ) -> MemoryWriteResult:
        """
        Check if a memory write should be allowed.
        
        Args:
            request: The write request
            current_stable_chars: Current stable memory size
            current_ephemeral_chars: Current ephemeral memory size
            
        Returns:
            MemoryWriteResult with decision
        """
        original_size = len(request.content)
        
        # Get phase spec and limits
        phase_spec = get_phase_spec(request.phase_name)
        limits = self.policy.get_limits_for_phase(request.phase_name)
        
        # Check phase memory policy
        if phase_spec.memory_write_policy == MemoryWritePolicy.NONE:
            return MemoryWriteResult(
                allowed=False,
                content=request.content,
                reason=f"Phase '{request.phase_name}' does not allow memory writes",
                importance_score=0.0,
                original_size=original_size,
                final_size=original_size,
            )
        
        if phase_spec.memory_write_policy == MemoryWritePolicy.EPHEMERAL:
            if request.write_type.lower() == "stable":
                return MemoryWriteResult(
                    allowed=False,
                    content=request.content,
                    reason=f"Phase '{request.phase_name}' only allows ephemeral writes",
                    importance_score=0.0,
                    original_size=original_size,
                    final_size=original_size,
                )
        
        # Score importance
        content_type = request.content_type or self.policy.infer_content_type(request.content)
        importance = request.importance_override or self.policy.score_importance(
            request.content, content_type
        )
        
        # Check importance threshold
        if importance < self.policy.importance_threshold:
            return MemoryWriteResult(
                allowed=False,
                content=request.content,
                reason=f"Content importance {importance:.2f} below threshold {self.policy.importance_threshold}",
                importance_score=importance,
                original_size=original_size,
                final_size=original_size,
            )
        
        # Check size limits
        is_stable = request.write_type.lower() == "stable"
        current = current_stable_chars if is_stable else current_ephemeral_chars
        limit = limits["stable"] if is_stable else limits["ephemeral"]
        
        if current + original_size > limit:
            # Try compression
            compressed = self.policy.compress_content(
                request.content,
                target_chars=limit - current,
                preserve_questions=True
            )
            
            if len(compressed) < original_size * 0.5:
                # Too much compression needed - reject
                return MemoryWriteResult(
                    allowed=False,
                    content=request.content,
                    reason=f"Would exceed {request.write_type} limit ({current + original_size} > {limit})",
                    importance_score=importance,
                    original_size=original_size,
                    final_size=original_size,
                )
            
            # Compression succeeded
            self._log_write(request, True, importance, "compressed")
            return MemoryWriteResult(
                allowed=True,
                content=compressed,
                reason=f"Compressed to fit ({original_size} -> {len(compressed)} chars)",
                importance_score=importance,
                original_size=original_size,
                final_size=len(compressed),
            )
        
        # Check item count limit
        phase_items = self._phase_item_counts.get(request.phase_name, 0)
        if phase_items >= self.policy.max_items_per_phase:
            return MemoryWriteResult(
                allowed=False,
                content=request.content,
                reason=f"Phase item limit reached ({phase_items} >= {self.policy.max_items_per_phase})",
                importance_score=importance,
                original_size=original_size,
                final_size=original_size,
            )
        
        # Check raw storage restriction
        if self.policy.forbidden_raw_storage and content_type == ContentType.RAW_OUTPUT:
            if original_size > self.policy.max_item_chars:
                # Compress raw output
                compressed = self.policy.compress_content(
                    request.content,
                    target_chars=self.policy.max_item_chars,
                    preserve_questions=True
                )
                
                self._log_write(request, True, importance, "raw_compressed")
                return MemoryWriteResult(
                    allowed=True,
                    content=compressed,
                    reason=f"Raw output compressed ({original_size} -> {len(compressed)} chars)",
                    importance_score=importance,
                    original_size=original_size,
                    final_size=len(compressed),
                )
        
        # All checks passed
        self._log_write(request, True, importance, "approved")
        self._update_usage(request.phase_name, request.write_type, original_size)
        
        return MemoryWriteResult(
            allowed=True,
            content=request.content,
            reason="Approved",
            importance_score=importance,
            original_size=original_size,
            final_size=original_size,
        )
    
    def apply_phase_decay(
        self,
        content: str,
        age_in_phases: int
    ) -> str:
        """
        Apply decay to content based on age.
        
        Args:
            content: Content to potentially decay
            age_in_phases: Number of phases since creation
            
        Returns:
            Decayed/compressed content
        """
        decay_factor = self.policy.compute_decay(age_in_phases)
        
        if decay_factor >= 0.9:
            return content
        
        # Compress based on decay
        target_chars = int(len(content) * decay_factor)
        if target_chars < 100:
            return ""  # Too decayed
        
        return self.policy.compress_content(content, target_chars)
    
    def get_phase_usage(self, phase_name: str) -> Dict[str, int]:
        """Get memory usage for a phase."""
        return self._phase_usage.get(phase_name, {"stable": 0, "ephemeral": 0, "delta": 0})
    
    def get_phase_limits(self, phase_name: str) -> Dict[str, int]:
        """Get memory limits for a phase."""
        return self.policy.get_limits_for_phase(phase_name)
    
    def clear_phase_ephemeral(self, phase_name: str) -> None:
        """Clear ephemeral memory tracking for a phase (at phase boundary)."""
        if phase_name in self._phase_usage:
            self._phase_usage[phase_name]["ephemeral"] = 0
            self._phase_usage[phase_name]["delta"] = 0
        
        logger.debug(f"Cleared ephemeral memory for phase '{phase_name}'")
    
    def should_compress(self, phase_name: str) -> bool:
        """Check if compression is needed for a phase."""
        usage = self.get_phase_usage(phase_name)
        limits = self.get_phase_limits(phase_name)
        
        total_usage = sum(usage.values())
        total_limit = sum(limits.values())
        
        return total_usage > self.policy.compression_trigger_chars or \
               total_usage > total_limit * 0.8
    
    def get_recommendations(self, phase_name: str) -> List[str]:
        """Get memory management recommendations for a phase."""
        recommendations = []
        usage = self.get_phase_usage(phase_name)
        limits = self.get_phase_limits(phase_name)
        
        if usage.get("stable", 0) > limits.get("stable", 0) * 0.8:
            recommendations.append("Stable memory nearly full - prioritize high-importance items")
        
        if usage.get("ephemeral", 0) > limits.get("ephemeral", 0) * 0.9:
            recommendations.append("Ephemeral memory nearly full - consider phase transition")
        
        items = self._phase_item_counts.get(phase_name, 0)
        if items > self.policy.max_items_per_phase * 0.8:
            recommendations.append("Approaching item limit - consolidate related items")
        
        return recommendations
    
    def _update_usage(
        self,
        phase_name: str,
        write_type: str,
        size: int
    ) -> None:
        """Update usage tracking."""
        if phase_name not in self._phase_usage:
            self._phase_usage[phase_name] = {"stable": 0, "ephemeral": 0, "delta": 0}
        
        write_type_lower = write_type.lower()
        if write_type_lower in self._phase_usage[phase_name]:
            self._phase_usage[phase_name][write_type_lower] += size
        
        # Update item count
        self._phase_item_counts[phase_name] = self._phase_item_counts.get(phase_name, 0) + 1
    
    def _log_write(
        self,
        request: MemoryWriteRequest,
        allowed: bool,
        importance: float,
        reason: str
    ) -> None:
        """Log a write decision."""
        entry = {
            "phase": request.phase_name,
            "write_type": request.write_type,
            "source": request.source,
            "size": len(request.content),
            "allowed": allowed,
            "importance": importance,
            "reason": reason,
        }
        self._write_log.append(entry)
        
        if not allowed:
            logger.debug(
                f"Memory write blocked [{request.phase_name}]: "
                f"{request.write_type} from {request.source} - {reason}"
            )
    
    def get_write_log(self) -> List[Dict]:
        """Get the write decision log."""
        return self._write_log.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory guard statistics."""
        total_writes = len(self._write_log)
        allowed = sum(1 for w in self._write_log if w["allowed"])
        blocked = total_writes - allowed
        
        return {
            "total_writes": total_writes,
            "allowed": allowed,
            "blocked": blocked,
            "block_rate": blocked / total_writes if total_writes > 0 else 0,
            "phase_usage": dict(self._phase_usage),
            "phase_item_counts": dict(self._phase_item_counts),
        }
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self._phase_usage.clear()
        self._phase_item_counts.clear()
        self._write_log.clear()


# Global guard instance
_guard: Optional[MemoryGuard] = None


def get_memory_guard(
    policy: Optional[MemoryPolicy] = None,
    strict_mode: bool = False
) -> MemoryGuard:
    """Get the global memory guard instance."""
    global _guard
    if _guard is None:
        _guard = MemoryGuard(policy=policy, strict_mode=strict_mode)
    return _guard

