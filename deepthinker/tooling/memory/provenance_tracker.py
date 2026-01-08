"""
Memory Provenance Tracker - Tracks origin, confidence, and decay of memory entries.
"""

import logging
from typing import Optional, Literal
from datetime import datetime, timedelta

from ...memory.schemas import EvidenceSchema

logger = logging.getLogger(__name__)


class MemoryProvenanceTracker:
    """
    Tracks provenance on all memory writes.
    
    Extends EvidenceSchema with:
    - origin: web, inference, or human
    - decay_rate: confidence decay per day
    - expiry_date: when evidence becomes stale
    """
    
    # Default decay rates by origin
    DEFAULT_DECAY_RATES = {
        "web": 0.01,  # 1% per day (web info can become outdated)
        "inference": 0.005,  # 0.5% per day (inferences may need refresh)
        "human": 0.0,  # No decay (human-provided is stable)
    }
    
    # Default expiry periods (days)
    DEFAULT_EXPIRY_PERIODS = {
        "web": 90,  # Web info expires after 90 days
        "inference": 180,  # Inferences expire after 180 days
        "human": None,  # Human info never expires
    }
    
    def __init__(
        self,
        default_decay_rates: Optional[dict] = None,
        default_expiry_periods: Optional[dict] = None
    ):
        """
        Initialize provenance tracker.
        
        Args:
            default_decay_rates: Optional custom decay rates by origin
            default_expiry_periods: Optional custom expiry periods by origin
        """
        self.decay_rates = default_decay_rates or self.DEFAULT_DECAY_RATES.copy()
        self.expiry_periods = default_expiry_periods or self.DEFAULT_EXPIRY_PERIODS.copy()
    
    def track_evidence(
        self,
        evidence: EvidenceSchema,
        origin: Literal["web", "inference", "human"],
        decay_rate: Optional[float] = None,
        expiry_days: Optional[int] = None
    ) -> EvidenceSchema:
        """
        Track provenance for evidence.
        
        Args:
            evidence: Evidence schema to track
            origin: Origin of the evidence
            decay_rate: Optional custom decay rate (overrides default)
            expiry_days: Optional custom expiry period in days (overrides default)
            
        Returns:
            EvidenceSchema with provenance fields set
        """
        # Set origin
        evidence.origin = origin
        
        # Set decay rate
        if decay_rate is not None:
            evidence.decay_rate = decay_rate
        else:
            evidence.decay_rate = self.decay_rates.get(origin, 0.0)
        
        # Set expiry date
        if expiry_days is not None:
            if expiry_days > 0:
                evidence.expiry_date = datetime.utcnow() + timedelta(days=expiry_days)
        else:
            expiry_period = self.expiry_periods.get(origin)
            if expiry_period is not None:
                evidence.expiry_date = datetime.utcnow() + timedelta(days=expiry_period)
        
        logger.debug(
            f"Tracked provenance for evidence {evidence.id}: "
            f"origin={origin}, decay_rate={evidence.decay_rate}, "
            f"expiry_date={evidence.expiry_date}"
        )
        
        return evidence
    
    def apply_decay(self, evidence: EvidenceSchema) -> EvidenceSchema:
        """
        Apply confidence decay based on age.
        
        Args:
            evidence: Evidence to apply decay to
            
        Returns:
            EvidenceSchema with updated confidence
        """
        if evidence.decay_rate == 0.0 or not evidence.created_at:
            return evidence
        
        # Calculate days since creation
        days_old = (datetime.utcnow() - evidence.created_at).days
        
        if days_old <= 0:
            return evidence
        
        # Apply decay
        decay_amount = evidence.decay_rate * days_old
        new_confidence = max(0.0, evidence.confidence - decay_amount)
        
        if new_confidence != evidence.confidence:
            logger.debug(
                f"Applied decay to evidence {evidence.id}: "
                f"{evidence.confidence:.2f} -> {new_confidence:.2f} "
                f"(decay_rate={evidence.decay_rate}, days_old={days_old})"
            )
            evidence.confidence = new_confidence
        
        return evidence
    
    def is_expired(self, evidence: EvidenceSchema) -> bool:
        """
        Check if evidence has expired.
        
        Args:
            evidence: Evidence to check
            
        Returns:
            True if evidence is expired
        """
        if not evidence.expiry_date:
            return False
        
        return datetime.utcnow() > evidence.expiry_date
    
    def get_effective_confidence(self, evidence: EvidenceSchema) -> float:
        """
        Get effective confidence after decay and expiry checks.
        
        Args:
            evidence: Evidence to check
            
        Returns:
            Effective confidence (0.0 if expired, decayed confidence otherwise)
        """
        if self.is_expired(evidence):
            return 0.0
        
        # Apply decay if needed
        evidence = self.apply_decay(evidence)
        return evidence.confidence
    
    def should_downweight(self, evidence: EvidenceSchema, threshold: float = 0.3) -> bool:
        """
        Check if evidence should be down-weighted due to low confidence or expiry.
        
        Args:
            evidence: Evidence to check
            threshold: Confidence threshold below which to down-weight
            
        Returns:
            True if evidence should be down-weighted
        """
        if self.is_expired(evidence):
            return True
        
        effective_confidence = self.get_effective_confidence(evidence)
        return effective_confidence < threshold

