"""
Focus Area Manager for Epistemic Hardening.

Prevents conceptual explosion by:
- Limiting active focus areas to 5 (configurable)
- Requiring explicit deprecation to add new areas after lock
- Tracking focus area lifecycle (ACTIVE -> PARKED -> DEPRECATED)
- Logging all focus area decisions for auditability

After the ReconGrounding phase, focus areas are locked. Any new focus
area requires deprecating an existing one with an explicit reason.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FocusAreaStatus(str, Enum):
    """Status of a focus area."""
    ACTIVE = "active"       # Currently being investigated
    PARKED = "parked"       # Temporarily set aside (can be reactivated)
    DEPRECATED = "deprecated"  # Permanently dropped with reason


@dataclass
class FocusArea:
    """
    A single focus area with lifecycle tracking.
    
    Attributes:
        name: Name/description of the focus area
        status: Current lifecycle status
        created_phase: Phase where this focus area was created
        created_at: Timestamp of creation
        deprecation_reason: Reason for deprecation (if deprecated)
        deprecation_phase: Phase where deprecation occurred
        priority: Priority score (0-1, higher = more important)
        claims_count: Number of claims associated with this area
    """
    name: str
    status: FocusAreaStatus = FocusAreaStatus.ACTIVE
    created_phase: str = ""
    created_at: Optional[datetime] = None
    deprecation_reason: Optional[str] = None
    deprecation_phase: Optional[str] = None
    priority: float = 0.5
    claims_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def deprecate(self, reason: str, phase: str) -> None:
        """
        Deprecate this focus area.
        
        Args:
            reason: Why the focus area is being deprecated
            phase: Phase where deprecation occurred
        """
        self.status = FocusAreaStatus.DEPRECATED
        self.deprecation_reason = reason
        self.deprecation_phase = phase
        
        logger.info(
            f"[FOCUS AREA] Deprecated '{self.name}': {reason} (phase: {phase})"
        )
    
    def park(self) -> None:
        """Park this focus area for later consideration."""
        if self.status == FocusAreaStatus.ACTIVE:
            self.status = FocusAreaStatus.PARKED
            logger.debug(f"[FOCUS AREA] Parked '{self.name}'")
    
    def reactivate(self) -> None:
        """Reactivate a parked focus area."""
        if self.status == FocusAreaStatus.PARKED:
            self.status = FocusAreaStatus.ACTIVE
            logger.debug(f"[FOCUS AREA] Reactivated '{self.name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "created_phase": self.created_phase,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deprecation_reason": self.deprecation_reason,
            "deprecation_phase": self.deprecation_phase,
            "priority": self.priority,
            "claims_count": self.claims_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FocusArea":
        """Create from dictionary."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                created_at = datetime.utcnow()
        
        return cls(
            name=data.get("name", ""),
            status=FocusAreaStatus(data.get("status", "active")),
            created_phase=data.get("created_phase", ""),
            created_at=created_at,
            deprecation_reason=data.get("deprecation_reason"),
            deprecation_phase=data.get("deprecation_phase"),
            priority=data.get("priority", 0.5),
            claims_count=data.get("claims_count", 0),
        )


@dataclass
class FocusAreaDecision:
    """
    Record of a focus area management decision.
    
    Used for auditability and understanding why focus areas changed.
    """
    action: str  # add, deprecate, park, reactivate
    focus_area_name: str
    phase: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: Optional[str] = None
    replaced_area: Optional[str] = None  # If adding required deprecating another
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "focus_area_name": self.focus_area_name,
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "replaced_area": self.replaced_area,
        }


class FocusAreaManager:
    """
    Manages focus areas with strict limits to prevent sprawl.
    
    Key constraints:
    - Maximum of MAX_ACTIVE focus areas (default 5)
    - After lock (recon_grounding complete), new areas require deprecation
    - All decisions are logged for auditability
    
    Usage:
        manager = FocusAreaManager()
        
        # During exploration
        manager.add_focus_area("market analysis", "recon_exploration")
        
        # Lock after grounding
        manager.lock("recon_grounding")
        
        # After lock, adding requires deprecation
        success = manager.add_focus_area(
            "competitor analysis", 
            "analysis",
            deprecate_area="market analysis",
            deprecation_reason="Merged into broader competitor view"
        )
    """
    
    MAX_ACTIVE = 5
    
    def __init__(
        self,
        max_active: int = 5,
        auto_park_on_overflow: bool = True
    ):
        """
        Initialize the focus area manager.
        
        Args:
            max_active: Maximum number of active focus areas
            auto_park_on_overflow: Whether to auto-park lowest priority areas on overflow
        """
        self.MAX_ACTIVE = max_active
        self.auto_park_on_overflow = auto_park_on_overflow
        
        self._focus_areas: Dict[str, FocusArea] = {}
        self._locked: bool = False
        self._lock_phase: Optional[str] = None
        self._decision_log: List[FocusAreaDecision] = []
    
    def add_focus_area(
        self,
        name: str,
        phase: str,
        priority: float = 0.5,
        deprecate_area: Optional[str] = None,
        deprecation_reason: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Add a new focus area.
        
        Args:
            name: Name of the focus area
            phase: Phase where the area is being added
            priority: Priority score (0-1)
            deprecate_area: Area to deprecate (required if locked and at max)
            deprecation_reason: Reason for deprecation
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Normalize name
        name = name.strip().lower()
        
        # Check if already exists
        if name in self._focus_areas:
            existing = self._focus_areas[name]
            if existing.status == FocusAreaStatus.DEPRECATED:
                # Cannot reuse deprecated names
                return False, f"Focus area '{name}' was deprecated and cannot be reused"
            elif existing.status == FocusAreaStatus.PARKED:
                # Reactivate parked area
                existing.reactivate()
                self._log_decision("reactivate", name, phase)
                return True, f"Reactivated parked focus area '{name}'"
            else:
                return True, f"Focus area '{name}' already active"
        
        # Count active areas
        active_count = len(self.get_active_areas())
        
        # Check if we're at capacity
        if active_count >= self.MAX_ACTIVE:
            if self._locked:
                # After lock, must deprecate to add
                if deprecate_area is None:
                    return False, (
                        f"Cannot add focus area: at capacity ({self.MAX_ACTIVE}) and locked. "
                        f"Must deprecate an existing area."
                    )
                
                # Deprecate the specified area
                if deprecate_area not in self._focus_areas:
                    return False, f"Cannot deprecate: area '{deprecate_area}' not found"
                
                reason = deprecation_reason or f"Replaced by '{name}'"
                self._focus_areas[deprecate_area].deprecate(reason, phase)
                self._log_decision("deprecate", deprecate_area, phase, reason)
                
            elif self.auto_park_on_overflow:
                # Before lock, auto-park lowest priority
                parked_area = self._park_lowest_priority(phase)
                if parked_area:
                    logger.info(f"[FOCUS AREA] Auto-parked '{parked_area}' to make room for '{name}'")
            else:
                return False, f"Cannot add focus area: at capacity ({self.MAX_ACTIVE})"
        
        # Create new focus area
        focus_area = FocusArea(
            name=name,
            status=FocusAreaStatus.ACTIVE,
            created_phase=phase,
            priority=priority,
        )
        
        self._focus_areas[name] = focus_area
        self._log_decision("add", name, phase, replaced_area=deprecate_area)
        
        logger.info(f"[FOCUS AREA] Added '{name}' (phase: {phase}, priority: {priority})")
        
        return True, f"Added focus area '{name}'"
    
    def deprecate_focus_area(
        self,
        name: str,
        reason: str,
        phase: str
    ) -> Tuple[bool, str]:
        """
        Explicitly deprecate a focus area.
        
        Args:
            name: Name of the area to deprecate
            reason: Reason for deprecation
            phase: Phase where deprecation occurred
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        name = name.strip().lower()
        
        if name not in self._focus_areas:
            return False, f"Focus area '{name}' not found"
        
        area = self._focus_areas[name]
        if area.status == FocusAreaStatus.DEPRECATED:
            return False, f"Focus area '{name}' already deprecated"
        
        area.deprecate(reason, phase)
        self._log_decision("deprecate", name, phase, reason)
        
        return True, f"Deprecated focus area '{name}': {reason}"
    
    def park_focus_area(self, name: str, phase: str) -> Tuple[bool, str]:
        """
        Park a focus area for later consideration.
        
        Args:
            name: Name of the area to park
            phase: Phase where parking occurred
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        name = name.strip().lower()
        
        if name not in self._focus_areas:
            return False, f"Focus area '{name}' not found"
        
        area = self._focus_areas[name]
        if area.status != FocusAreaStatus.ACTIVE:
            return False, f"Cannot park: focus area '{name}' is {area.status.value}"
        
        area.park()
        self._log_decision("park", name, phase)
        
        return True, f"Parked focus area '{name}'"
    
    def get_active_areas(self) -> List[FocusArea]:
        """Get all active focus areas."""
        return [
            area for area in self._focus_areas.values()
            if area.status == FocusAreaStatus.ACTIVE
        ]
    
    def get_active_area_names(self) -> List[str]:
        """Get names of all active focus areas."""
        return [area.name for area in self.get_active_areas()]
    
    def get_parked_areas(self) -> List[FocusArea]:
        """Get all parked focus areas."""
        return [
            area for area in self._focus_areas.values()
            if area.status == FocusAreaStatus.PARKED
        ]
    
    def get_deprecated_areas(self) -> List[FocusArea]:
        """Get all deprecated focus areas."""
        return [
            area for area in self._focus_areas.values()
            if area.status == FocusAreaStatus.DEPRECATED
        ]
    
    def get_all_areas(self) -> List[FocusArea]:
        """Get all focus areas regardless of status."""
        return list(self._focus_areas.values())
    
    def lock(self, phase: str) -> None:
        """
        Lock focus areas after recon grounding.
        
        After locking, new focus areas require explicit deprecation
        of an existing area.
        
        Args:
            phase: Phase where lock occurred
        """
        self._locked = True
        self._lock_phase = phase
        
        # Enforce max limit by parking excess areas
        active_areas = self.get_active_areas()
        if len(active_areas) > self.MAX_ACTIVE:
            # Sort by priority, park lowest
            sorted_areas = sorted(active_areas, key=lambda a: a.priority)
            excess = len(active_areas) - self.MAX_ACTIVE
            
            for area in sorted_areas[:excess]:
                area.park()
                self._log_decision("park", area.name, phase, "Auto-parked at lock time")
            
            logger.warning(
                f"[FOCUS AREA] Parked {excess} areas at lock time to enforce limit"
            )
        
        logger.info(
            f"[FOCUS AREA] Locked focus areas at phase '{phase}' "
            f"with {len(self.get_active_areas())} active areas"
        )
    
    def is_locked(self) -> bool:
        """Check if focus areas are locked."""
        return self._locked
    
    def unlock(self) -> None:
        """
        Unlock focus areas (for recovery/testing).
        
        Should rarely be used in production.
        """
        self._locked = False
        self._lock_phase = None
        logger.warning("[FOCUS AREA] Focus areas unlocked")
    
    def update_claims_count(self, name: str, count: int) -> None:
        """
        Update the claims count for a focus area.
        
        Args:
            name: Name of the focus area
            count: New claims count
        """
        name = name.strip().lower()
        if name in self._focus_areas:
            self._focus_areas[name].claims_count = count
    
    def increment_claims_count(self, name: str, delta: int = 1) -> None:
        """
        Increment the claims count for a focus area.
        
        Args:
            name: Name of the focus area
            delta: Amount to increment by
        """
        name = name.strip().lower()
        if name in self._focus_areas:
            self._focus_areas[name].claims_count += delta
    
    def _park_lowest_priority(self, phase: str) -> Optional[str]:
        """
        Park the lowest priority active focus area.
        
        Args:
            phase: Current phase
            
        Returns:
            Name of parked area, or None if none could be parked
        """
        active_areas = self.get_active_areas()
        if not active_areas:
            return None
        
        # Sort by priority (ascending) and park the first one
        sorted_areas = sorted(active_areas, key=lambda a: a.priority)
        lowest = sorted_areas[0]
        
        lowest.park()
        self._log_decision("park", lowest.name, phase, "Auto-parked due to overflow")
        
        return lowest.name
    
    def _log_decision(
        self,
        action: str,
        name: str,
        phase: str,
        reason: Optional[str] = None,
        replaced_area: Optional[str] = None
    ) -> None:
        """Log a focus area management decision."""
        decision = FocusAreaDecision(
            action=action,
            focus_area_name=name,
            phase=phase,
            reason=reason,
            replaced_area=replaced_area,
        )
        self._decision_log.append(decision)
    
    def get_decision_log(self) -> List[FocusAreaDecision]:
        """Get the full decision log."""
        return self._decision_log.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of focus area state."""
        return {
            "total_areas": len(self._focus_areas),
            "active_count": len(self.get_active_areas()),
            "parked_count": len(self.get_parked_areas()),
            "deprecated_count": len(self.get_deprecated_areas()),
            "is_locked": self._locked,
            "lock_phase": self._lock_phase,
            "max_active": self.MAX_ACTIVE,
            "active_areas": self.get_active_area_names(),
            "decision_count": len(self._decision_log),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "focus_areas": {
                name: area.to_dict()
                for name, area in self._focus_areas.items()
            },
            "locked": self._locked,
            "lock_phase": self._lock_phase,
            "max_active": self.MAX_ACTIVE,
            "decision_log": [d.to_dict() for d in self._decision_log],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FocusAreaManager":
        """Create from dictionary."""
        manager = cls(max_active=data.get("max_active", 5))
        
        for name, area_data in data.get("focus_areas", {}).items():
            manager._focus_areas[name] = FocusArea.from_dict(area_data)
        
        manager._locked = data.get("locked", False)
        manager._lock_phase = data.get("lock_phase")
        
        # Decision log is for audit, not typically restored
        
        return manager

