"""
Claim Registry for Epistemic Hardening.

Provides a mission-level claim store for council operations.
Councils must operate on claims, not free text, enabling:
- Claim-level evaluation
- Claim-level disagreement tracking
- Auditable claim promotion/rejection
- Focus area aggregation

Usage:
    registry = ClaimRegistry()
    
    # Register a new claim
    claim_id = registry.register_claim(claim)
    
    # Promote after validation
    registry.promote_claim(claim_id)
    
    # Contest during council debate
    registry.contest_claim(claim_id, "Contradicts source X")
    
    # Get only grounded claims for downstream
    grounded = registry.get_grounded_claims()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging

from .claim_validator import Claim, ClaimType, ClaimStatus, Source

logger = logging.getLogger(__name__)


@dataclass
class ClaimContestRecord:
    """
    Record of a claim being contested.
    
    Attributes:
        claim_id: ID of the contested claim
        reason: Why the claim is contested
        contested_by: Who/what contested (council, model, etc.)
        timestamp: When the contest occurred
        resolution: How it was resolved (promoted, rejected, pending)
    """
    claim_id: str
    reason: str
    contested_by: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolution: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_id": self.claim_id,
            "reason": self.reason,
            "contested_by": self.contested_by,
            "timestamp": self.timestamp.isoformat(),
            "resolution": self.resolution,
        }


class ClaimRegistry:
    """
    Mission-level claim store for council operations.
    
    Provides centralized claim management with:
    - Registration and ID assignment
    - Status transitions (proposed -> grounded/contested/rejected)
    - Focus area grouping
    - Source association
    - Contest tracking for audit
    """
    
    def __init__(self):
        """Initialize the claim registry."""
        self._claims: Dict[str, Claim] = {}
        self._sources: Dict[str, Source] = {}
        self._contest_records: List[ClaimContestRecord] = []
        self._focus_area_claims: Dict[str, Set[str]] = {}  # focus_area -> claim_ids
        
        # Statistics
        self._total_registered: int = 0
        self._total_promoted: int = 0
        self._total_contested: int = 0
        self._total_rejected: int = 0
    
    def register_claim(
        self,
        claim: Claim,
        source: Optional[Source] = None
    ) -> str:
        """
        Register a new claim.
        
        Args:
            claim: The claim to register
            source: Optional source backing the claim
            
        Returns:
            Claim ID for future reference
        """
        claim_id = claim.id
        
        # Store the claim
        self._claims[claim_id] = claim
        self._total_registered += 1
        
        # Register source if provided
        if source:
            self._sources[source.id] = source
            if source.id not in claim.source_ids:
                claim.source_ids.append(source.id)
            if source.url and not claim.source_url:
                claim.source_url = source.url
        
        # Track by focus area
        if claim.focus_area:
            if claim.focus_area not in self._focus_area_claims:
                self._focus_area_claims[claim.focus_area] = set()
            self._focus_area_claims[claim.focus_area].add(claim_id)
        
        logger.debug(
            f"[CLAIM REGISTRY] Registered claim {claim_id}: "
            f"type={claim.claim_type.value}, focus_area={claim.focus_area}"
        )
        
        return claim_id
    
    def register_claims(self, claims: List[Claim]) -> List[str]:
        """
        Register multiple claims.
        
        Args:
            claims: List of claims to register
            
        Returns:
            List of claim IDs
        """
        return [self.register_claim(claim) for claim in claims]
    
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """
        Get a claim by ID.
        
        Args:
            claim_id: The claim ID
            
        Returns:
            Claim if found, None otherwise
        """
        return self._claims.get(claim_id)
    
    def promote_claim(self, claim_id: str) -> bool:
        """
        Mark a claim as grounded.
        
        Args:
            claim_id: The claim ID to promote
            
        Returns:
            True if successful, False if claim not found
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            logger.warning(f"[CLAIM REGISTRY] Cannot promote: claim {claim_id} not found")
            return False
        
        if claim.status == ClaimStatus.REJECTED:
            logger.warning(f"[CLAIM REGISTRY] Cannot promote rejected claim {claim_id}")
            return False
        
        claim.promote_to_grounded()
        self._total_promoted += 1
        
        logger.info(f"[CLAIM REGISTRY] Promoted claim {claim_id} to GROUNDED")
        return True
    
    def contest_claim(
        self,
        claim_id: str,
        reason: str,
        contested_by: str = "unknown"
    ) -> bool:
        """
        Mark a claim as contested.
        
        Args:
            claim_id: The claim ID to contest
            reason: Why the claim is contested
            contested_by: Who/what is contesting
            
        Returns:
            True if successful, False if claim not found
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            logger.warning(f"[CLAIM REGISTRY] Cannot contest: claim {claim_id} not found")
            return False
        
        claim.contest(reason)
        self._total_contested += 1
        
        # Record the contest
        record = ClaimContestRecord(
            claim_id=claim_id,
            reason=reason,
            contested_by=contested_by,
        )
        self._contest_records.append(record)
        
        logger.info(f"[CLAIM REGISTRY] Contested claim {claim_id}: {reason}")
        return True
    
    def reject_claim(
        self,
        claim_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Reject a claim.
        
        Args:
            claim_id: The claim ID to reject
            reason: Optional reason for rejection
            
        Returns:
            True if successful, False if claim not found
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            logger.warning(f"[CLAIM REGISTRY] Cannot reject: claim {claim_id} not found")
            return False
        
        claim.reject(reason)
        self._total_rejected += 1
        
        logger.info(f"[CLAIM REGISTRY] Rejected claim {claim_id}: {reason or 'no reason'}")
        return True
    
    def resolve_contest(
        self,
        claim_id: str,
        resolution: str  # "promoted", "rejected"
    ) -> bool:
        """
        Resolve a contested claim.
        
        Args:
            claim_id: The claim ID
            resolution: How to resolve ("promoted" or "rejected")
            
        Returns:
            True if successful
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return False
        
        if claim.status != ClaimStatus.CONTESTED:
            return False
        
        if resolution == "promoted":
            claim.promote_to_grounded()
            self._total_promoted += 1
        elif resolution == "rejected":
            claim.reject()
            self._total_rejected += 1
        
        # Update contest record
        for record in reversed(self._contest_records):
            if record.claim_id == claim_id and record.resolution == "pending":
                record.resolution = resolution
                break
        
        return True
    
    def get_claims_by_status(self, status: ClaimStatus) -> List[Claim]:
        """
        Get all claims with a specific status.
        
        Args:
            status: The status to filter by
            
        Returns:
            List of claims with that status
        """
        return [c for c in self._claims.values() if c.status == status]
    
    def get_grounded_claims(self) -> List[Claim]:
        """Get only grounded claims for downstream phases."""
        return self.get_claims_by_status(ClaimStatus.GROUNDED)
    
    def get_proposed_claims(self) -> List[Claim]:
        """Get claims that are still proposed (awaiting validation)."""
        return self.get_claims_by_status(ClaimStatus.PROPOSED)
    
    def get_contested_claims(self) -> List[Claim]:
        """Get claims that are contested."""
        return self.get_claims_by_status(ClaimStatus.CONTESTED)
    
    def get_claims_by_focus_area(self, focus_area: str) -> List[Claim]:
        """
        Get all claims for a focus area.
        
        Args:
            focus_area: The focus area name
            
        Returns:
            List of claims in that focus area
        """
        claim_ids = self._focus_area_claims.get(focus_area, set())
        return [self._claims[cid] for cid in claim_ids if cid in self._claims]
    
    def get_grounded_claims_by_focus_area(self, focus_area: str) -> List[Claim]:
        """Get grounded claims for a specific focus area."""
        claims = self.get_claims_by_focus_area(focus_area)
        return [c for c in claims if c.status == ClaimStatus.GROUNDED]
    
    def get_claims_by_type(self, claim_type: ClaimType) -> List[Claim]:
        """
        Get all claims of a specific type.
        
        Args:
            claim_type: The type to filter by
            
        Returns:
            List of claims of that type
        """
        return [c for c in self._claims.values() if c.claim_type == claim_type]
    
    def associate_source(
        self,
        claim_id: str,
        source: Source
    ) -> bool:
        """
        Associate a source with a claim.
        
        Args:
            claim_id: The claim ID
            source: The source to associate
            
        Returns:
            True if successful
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return False
        
        # Register the source
        self._sources[source.id] = source
        
        # Associate with claim
        if source.id not in claim.source_ids:
            claim.source_ids.append(source.id)
        
        if source.url and not claim.source_url:
            claim.source_url = source.url
        
        return True
    
    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return self._sources.get(source_id)
    
    def get_sources_for_claim(self, claim_id: str) -> List[Source]:
        """Get all sources for a claim."""
        claim = self._claims.get(claim_id)
        if claim is None:
            return []
        
        return [
            self._sources[sid]
            for sid in claim.source_ids
            if sid in self._sources
        ]
    
    def get_grounding_ratio(self) -> float:
        """
        Calculate the overall grounding ratio.
        
        Returns:
            Ratio of grounded to total claims (0-1)
        """
        if not self._claims:
            return 0.0
        
        grounded_count = len(self.get_grounded_claims())
        return grounded_count / len(self._claims)
    
    def get_focus_area_grounding_ratios(self) -> Dict[str, float]:
        """
        Get grounding ratios per focus area.
        
        Returns:
            Dictionary mapping focus_area -> grounding_ratio
        """
        ratios = {}
        
        for focus_area in self._focus_area_claims:
            claims = self.get_claims_by_focus_area(focus_area)
            if claims:
                grounded = sum(1 for c in claims if c.status == ClaimStatus.GROUNDED)
                ratios[focus_area] = grounded / len(claims)
            else:
                ratios[focus_area] = 0.0
        
        return ratios
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_registered": self._total_registered,
            "total_claims": len(self._claims),
            "total_sources": len(self._sources),
            "total_promoted": self._total_promoted,
            "total_contested": self._total_contested,
            "total_rejected": self._total_rejected,
            "grounding_ratio": self.get_grounding_ratio(),
            "by_status": {
                "proposed": len(self.get_proposed_claims()),
                "grounded": len(self.get_grounded_claims()),
                "contested": len(self.get_contested_claims()),
                "rejected": len(self.get_claims_by_status(ClaimStatus.REJECTED)),
            },
            "by_type": {
                "fact": len(self.get_claims_by_type(ClaimType.FACT)),
                "inference": len(self.get_claims_by_type(ClaimType.INFERENCE)),
                "speculation": len(self.get_claims_by_type(ClaimType.SPECULATION)),
            },
            "focus_areas": list(self._focus_area_claims.keys()),
            "contest_records": len(self._contest_records),
        }
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get contest records for audit."""
        return [r.to_dict() for r in self._contest_records]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claims": {
                cid: claim.to_dict()
                for cid, claim in self._claims.items()
            },
            "sources": {
                sid: {
                    "id": s.id,
                    "url": s.url,
                    "title": s.title,
                    "quality_score": s.quality_score,
                    "quality_tier": s.quality_tier,
                    "domain": s.domain,
                }
                for sid, s in self._sources.items()
            },
            "focus_area_claims": {
                fa: list(cids)
                for fa, cids in self._focus_area_claims.items()
            },
            "statistics": self.get_statistics(),
            "contest_records": self.get_audit_log(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimRegistry":
        """Create from dictionary."""
        registry = cls()
        
        # Restore sources first
        for sid, sdata in data.get("sources", {}).items():
            source = Source(
                id=sdata.get("id", sid),
                url=sdata.get("url"),
                title=sdata.get("title", ""),
                quality_score=sdata.get("quality_score", 0.5),
                quality_tier=sdata.get("quality_tier", "MEDIUM"),
                domain=sdata.get("domain", ""),
            )
            registry._sources[sid] = source
        
        # Restore claims
        for cid, cdata in data.get("claims", {}).items():
            claim = Claim.from_dict(cdata)
            registry._claims[cid] = claim
        
        # Restore focus area mapping
        for fa, cids in data.get("focus_area_claims", {}).items():
            registry._focus_area_claims[fa] = set(cids)
        
        # Restore statistics
        stats = data.get("statistics", {})
        registry._total_registered = stats.get("total_registered", len(registry._claims))
        registry._total_promoted = stats.get("total_promoted", 0)
        registry._total_contested = stats.get("total_contested", 0)
        registry._total_rejected = stats.get("total_rejected", 0)
        
        return registry
    
    def clear(self) -> None:
        """Clear all claims and sources."""
        self._claims.clear()
        self._sources.clear()
        self._contest_records.clear()
        self._focus_area_claims.clear()
        self._total_registered = 0
        self._total_promoted = 0
        self._total_contested = 0
        self._total_rejected = 0

