"""
Phase Artifact Promotion Firewall for Epistemic Hardening.

Prevents contamination of downstream phases by filtering artifacts
from phases that fail epistemic gates.

Key behaviors:
- If a phase passes gates: all artifacts promoted
- If a phase fails gates: only objective + grounded claims promoted
- Speculative content is filtered out
- All filtering decisions are logged for auditability

Usage:
    firewall = ArtifactFirewall()
    
    # After phase completion
    promoted = firewall.promote_phase_artifacts(
        phase=phase,
        verdict=normative_verdict,
        claim_registry=registry
    )
    
    # Use promoted artifacts for next phase
    next_phase_context = promoted
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from datetime import datetime
import logging
import copy

if TYPE_CHECKING:
    from .mission_types import MissionPhase, MissionState
    from ..governance.normative_layer import NormativeVerdict, VerdictStatus
    from ..epistemics.claim_registry import ClaimRegistry
    from ..epistemics.claim_validator import Claim, ClaimStatus

logger = logging.getLogger(__name__)


@dataclass
class FilteredArtifact:
    """
    Record of an artifact that was filtered.
    
    Attributes:
        key: Artifact key/name
        reason: Why it was filtered
        phase_name: Phase it came from
        content_preview: Preview of filtered content
        timestamp: When filtering occurred
    """
    key: str
    reason: str
    phase_name: str
    content_preview: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "reason": self.reason,
            "phase_name": self.phase_name,
            "content_preview": self.content_preview[:200] if self.content_preview else "",
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass 
class PromotionResult:
    """
    Result of artifact promotion.
    
    Attributes:
        promoted_artifacts: Artifacts that passed filtering
        filtered_artifacts: Artifacts that were blocked
        grounded_claims: Grounded claims that were promoted
        promotion_ratio: Ratio of promoted to total artifacts
        phase_name: Source phase
        verdict_status: Original verdict status
    """
    promoted_artifacts: Dict[str, Any] = field(default_factory=dict)
    filtered_artifacts: List[FilteredArtifact] = field(default_factory=list)
    grounded_claims: List["Claim"] = field(default_factory=list)
    promotion_ratio: float = 1.0
    phase_name: str = ""
    verdict_status: str = "ALLOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "promoted_keys": list(self.promoted_artifacts.keys()),
            "filtered_count": len(self.filtered_artifacts),
            "filtered_keys": [f.key for f in self.filtered_artifacts],
            "grounded_claims_count": len(self.grounded_claims),
            "promotion_ratio": self.promotion_ratio,
            "phase_name": self.phase_name,
            "verdict_status": self.verdict_status,
        }


class ArtifactFirewall:
    """
    Filters phase artifacts to prevent contamination of downstream phases.
    
    When epistemic gates fail, this firewall ensures that only
    objective-level content and grounded claims are passed forward,
    preventing speculative or ungrounded content from polluting
    later phases.
    """
    
    # Artifact keys that are always promoted (objective-level)
    ALWAYS_PROMOTE = {
        "objective",
        "mission_objective",
        "constraints",
        "mission_constraints",
        "phase_name",
        "phase_objective",
        "time_budget",
        "error",
        "skip_reason",
    }
    
    # Artifact keys that contain speculative content
    SPECULATIVE_PATTERNS = [
        "speculation",
        "hypothesis",
        "hypotheses",
        "ungrounded",
        "unvalidated",
        "proposed",
        "candidate",
        "tentative",
    ]
    
    # Artifact keys that require grounding check
    GROUNDING_REQUIRED = {
        "claims",
        "findings",
        "facts",
        "assertions",
        "conclusions",
        "insights",
        "analysis",
        "research",
    }
    
    def __init__(
        self,
        strict_mode: bool = False,
        log_filtered: bool = True
    ):
        """
        Initialize the artifact firewall.
        
        Args:
            strict_mode: If True, filter more aggressively
            log_filtered: Whether to log filtered artifacts
        """
        self.strict_mode = strict_mode
        self.log_filtered = log_filtered
        self._promotion_history: List[PromotionResult] = []
    
    def promote_phase_artifacts(
        self,
        phase: "MissionPhase",
        verdict: "NormativeVerdict",
        claim_registry: Optional["ClaimRegistry"] = None,
        mission_state: Optional["MissionState"] = None,
    ) -> PromotionResult:
        """
        Promote artifacts from a completed phase.
        
        If the verdict allows (ALLOW status), all artifacts are promoted.
        If the verdict blocks (BLOCK/WARN), only objective + grounded claims promoted.
        
        Args:
            phase: The completed phase
            verdict: Normative verdict from governance layer
            claim_registry: Optional claim registry for grounded claim lookup
            mission_state: Optional mission state for context
            
        Returns:
            PromotionResult with promoted and filtered artifacts
        """
        from ..governance.normative_layer import VerdictStatus
        
        phase_name = phase.name
        artifacts = phase.artifacts or {}
        
        result = PromotionResult(
            phase_name=phase_name,
            verdict_status=verdict.status.value,
        )
        
        # If verdict allows, promote everything
        if verdict.status == VerdictStatus.ALLOW:
            result.promoted_artifacts = copy.deepcopy(artifacts)
            result.promotion_ratio = 1.0
            
            # Still extract grounded claims if registry available
            if claim_registry:
                result.grounded_claims = claim_registry.get_grounded_claims()
            
            logger.info(
                f"[ARTIFACT FIREWALL] Phase '{phase_name}' passed gates, "
                f"promoted all {len(artifacts)} artifacts"
            )
            
            self._promotion_history.append(result)
            return result
        
        # Verdict is WARN or BLOCK - filter artifacts
        logger.warning(
            f"[ARTIFACT FIREWALL] Phase '{phase_name}' failed gates ({verdict.status.value}), "
            f"filtering artifacts"
        )
        
        promoted = {}
        filtered = []
        
        for key, value in artifacts.items():
            # Check if always promoted
            if self._is_always_promoted(key):
                promoted[key] = value
                continue
            
            # Check if speculative
            if self._is_speculative(key, value):
                filtered.append(FilteredArtifact(
                    key=key,
                    reason="Speculative content",
                    phase_name=phase_name,
                    content_preview=self._get_preview(value),
                ))
                continue
            
            # Check if requires grounding
            if self._requires_grounding(key):
                # Check if content is grounded
                if self._is_grounded(key, value, claim_registry):
                    promoted[key] = value
                else:
                    filtered.append(FilteredArtifact(
                        key=key,
                        reason="Ungrounded content requiring evidence",
                        phase_name=phase_name,
                        content_preview=self._get_preview(value),
                    ))
                continue
            
            # In strict mode, filter everything else
            if self.strict_mode:
                filtered.append(FilteredArtifact(
                    key=key,
                    reason="Strict mode - only explicit promotes allowed",
                    phase_name=phase_name,
                    content_preview=self._get_preview(value),
                ))
            else:
                # In normal mode, allow other artifacts through
                promoted[key] = value
        
        # Add grounded claims from registry
        if claim_registry:
            grounded_claims = claim_registry.get_grounded_claims()
            result.grounded_claims = grounded_claims
            
            # Add grounded claims summary to promoted artifacts
            if grounded_claims:
                promoted["grounded_claims_summary"] = [
                    {
                        "id": c.id,
                        "text": c.text[:200] if c.text else "",
                        "type": c.claim_type.value,
                        "focus_area": c.focus_area,
                    }
                    for c in grounded_claims
                ]
        
        result.promoted_artifacts = promoted
        result.filtered_artifacts = filtered
        
        total = len(artifacts)
        result.promotion_ratio = len(promoted) / total if total > 0 else 1.0
        
        # Log filtering
        if self.log_filtered and filtered:
            for f in filtered:
                logger.debug(
                    f"[ARTIFACT FIREWALL] Filtered '{f.key}' from '{phase_name}': {f.reason}"
                )
        
        logger.info(
            f"[ARTIFACT FIREWALL] Phase '{phase_name}': "
            f"promoted {len(promoted)}/{total} artifacts, "
            f"filtered {len(filtered)}, "
            f"grounded claims: {len(result.grounded_claims)}"
        )
        
        self._promotion_history.append(result)
        return result
    
    def _is_always_promoted(self, key: str) -> bool:
        """Check if artifact key is always promoted."""
        key_lower = key.lower()
        return key_lower in self.ALWAYS_PROMOTE or key.startswith("_")
    
    def _is_speculative(self, key: str, value: Any) -> bool:
        """Check if artifact is speculative."""
        key_lower = key.lower()
        
        # Check key for speculative patterns
        for pattern in self.SPECULATIVE_PATTERNS:
            if pattern in key_lower:
                return True
        
        # Check value content for speculation markers
        if isinstance(value, str):
            value_lower = value.lower()
            speculation_markers = [
                "might be",
                "could be",
                "possibly",
                "potentially",
                "speculation:",
                "[speculative]",
                "[hypothesis]",
            ]
            for marker in speculation_markers:
                if marker in value_lower:
                    return True
        
        return False
    
    def _requires_grounding(self, key: str) -> bool:
        """Check if artifact key requires grounding check."""
        key_lower = key.lower()
        
        for grounded_key in self.GROUNDING_REQUIRED:
            if grounded_key in key_lower:
                return True
        
        return False
    
    def _is_grounded(
        self,
        key: str,
        value: Any,
        claim_registry: Optional["ClaimRegistry"]
    ) -> bool:
        """Check if artifact content is grounded."""
        # If no registry, cannot verify grounding
        if claim_registry is None:
            return False
        
        # Check grounding ratio
        ratio = claim_registry.get_grounding_ratio()
        
        # If overall grounding is above threshold, consider grounded
        if ratio >= 0.6:
            return True
        
        # Check for source citations in content
        if isinstance(value, str):
            citation_markers = ["[source:", "[cited:", "according to", "based on"]
            for marker in citation_markers:
                if marker in value.lower():
                    return True
        
        return False
    
    def _get_preview(self, value: Any) -> str:
        """Get a preview of artifact value for logging."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value[:200]
        if isinstance(value, (list, dict)):
            return str(value)[:200]
        return str(value)[:200]
    
    def get_promotion_history(self) -> List[PromotionResult]:
        """Get history of promotion results."""
        return self._promotion_history.copy()
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of filtering activity."""
        total_promoted = sum(
            len(r.promoted_artifacts) for r in self._promotion_history
        )
        total_filtered = sum(
            len(r.filtered_artifacts) for r in self._promotion_history
        )
        
        return {
            "total_phases_processed": len(self._promotion_history),
            "total_artifacts_promoted": total_promoted,
            "total_artifacts_filtered": total_filtered,
            "overall_promotion_ratio": (
                total_promoted / (total_promoted + total_filtered)
                if (total_promoted + total_filtered) > 0 else 1.0
            ),
            "phases_with_filtering": sum(
                1 for r in self._promotion_history if r.filtered_artifacts
            ),
        }
    
    def clear_history(self) -> None:
        """Clear promotion history."""
        self._promotion_history.clear()


def create_sanitized_context(
    promoted_result: PromotionResult,
    objective: str,
    phase_name: str
) -> Dict[str, Any]:
    """
    Create a sanitized context for the next phase.
    
    Combines promoted artifacts with objective and grounded claims
    into a clean context dictionary.
    
    Args:
        promoted_result: Result from artifact promotion
        objective: Mission objective
        phase_name: Name of the next phase
        
    Returns:
        Sanitized context dictionary
    """
    context = {
        "objective": objective,
        "from_phase": promoted_result.phase_name,
        "for_phase": phase_name,
        "grounded_claims": [
            c.to_dict() if hasattr(c, "to_dict") else str(c)
            for c in promoted_result.grounded_claims
        ],
        "grounding_summary": {
            "total_grounded_claims": len(promoted_result.grounded_claims),
            "promotion_ratio": promoted_result.promotion_ratio,
            "filtering_applied": len(promoted_result.filtered_artifacts) > 0,
        },
    }
    
    # Add promoted artifacts
    context.update(promoted_result.promoted_artifacts)
    
    return context

