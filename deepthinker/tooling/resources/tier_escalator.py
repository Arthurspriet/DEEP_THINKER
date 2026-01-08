"""
Execution Tier Escalator - Formalizes capability escalation with justification.
"""

import logging
from typing import Optional, Literal

from ..schemas import EscalationDecision

logger = logging.getLogger(__name__)


class ExecutionTierEscalator:
    """
    Formalizes escalation ladder: SAFE → SEARCH → GPU → BROWSER → TRUSTED
    
    Escalation requires:
    - Justification (claim risk, expected value)
    - Budget check (time remaining)
    """
    
    # Escalation ladder (ordered)
    TIER_LADDER = ["SAFE_ML", "SEARCH", "GPU_ML", "BROWSER", "TRUSTED"]
    
    # Minimum time remaining (minutes) for each tier
    MIN_TIME_REQUIREMENTS = {
        "SAFE_ML": 0.0,
        "SEARCH": 1.0,  # Need at least 1 minute for search
        "GPU_ML": 2.0,  # Need at least 2 minutes for GPU
        "BROWSER": 5.0,  # Need at least 5 minutes for browser
        "TRUSTED": 10.0,  # Need at least 10 minutes for trusted
    }
    
    # Risk thresholds for escalation
    RISK_THRESHOLDS = {
        "SEARCH": 0.4,  # Medium risk or higher
        "GPU_ML": 0.5,  # Medium-high risk
        "BROWSER": 0.6,  # High risk
        "TRUSTED": 0.7,  # Very high risk
    }
    
    def __init__(self):
        """Initialize tier escalator."""
        pass
    
    def can_escalate(
        self,
        current_tier: str,
        target_tier: str,
        claim_risk: Optional[float] = None,
        expected_value: Optional[float] = None,
        time_remaining_minutes: Optional[float] = None
    ) -> EscalationDecision:
        """
        Check if escalation to target tier is allowed.
        
        Args:
            current_tier: Current execution tier
            target_tier: Target execution tier
            claim_risk: Optional risk score of claim (0.0-1.0)
            expected_value: Optional expected value of escalation (0.0-1.0)
            time_remaining_minutes: Optional time remaining in mission
            
        Returns:
            EscalationDecision with allowed flag and justification
        """
        # Normalize tier names
        current_tier = current_tier.upper()
        target_tier = target_tier.upper()
        
        # Same tier - always allowed
        if current_tier == target_tier:
            return EscalationDecision(
                allowed=True,
                target_tier=target_tier,
                justification="Same tier, no escalation needed",
                claim_risk=claim_risk,
                expected_value=expected_value,
                time_budget_remaining=time_remaining_minutes
            )
        
        # Check if target tier is in ladder
        if target_tier not in self.TIER_LADDER:
            return EscalationDecision(
                allowed=False,
                target_tier=target_tier,
                justification=f"Target tier '{target_tier}' not in escalation ladder",
                claim_risk=claim_risk,
                expected_value=expected_value,
                time_budget_remaining=time_remaining_minutes
            )
        
        # Check if escalation is forward (higher capability)
        current_index = self.TIER_LADDER.index(current_tier) if current_tier in self.TIER_LADDER else -1
        target_index = self.TIER_LADDER.index(target_tier)
        
        if target_index <= current_index:
            return EscalationDecision(
                allowed=False,
                target_tier=target_tier,
                justification=f"Cannot escalate backwards or to same tier (current: {current_tier})",
                claim_risk=claim_risk,
                expected_value=expected_value,
                time_budget_remaining=time_remaining_minutes
            )
        
        # Check time requirement
        min_time = self.MIN_TIME_REQUIREMENTS.get(target_tier, 0.0)
        if time_remaining_minutes is not None and time_remaining_minutes < min_time:
            return EscalationDecision(
                allowed=False,
                target_tier=target_tier,
                justification=f"Insufficient time remaining ({time_remaining_minutes:.1f} min < {min_time:.1f} min required)",
                claim_risk=claim_risk,
                expected_value=expected_value,
                time_budget_remaining=time_remaining_minutes
            )
        
        # Check risk threshold
        risk_threshold = self.RISK_THRESHOLDS.get(target_tier, 0.0)
        if claim_risk is not None and claim_risk < risk_threshold:
            return EscalationDecision(
                allowed=False,
                target_tier=target_tier,
                justification=f"Claim risk ({claim_risk:.2f}) below threshold ({risk_threshold:.2f}) for tier {target_tier}",
                claim_risk=claim_risk,
                expected_value=expected_value,
                time_budget_remaining=time_remaining_minutes
            )
        
        # Check expected value (if provided)
        if expected_value is not None and expected_value < 0.3:
            return EscalationDecision(
                allowed=False,
                target_tier=target_tier,
                justification=f"Expected value ({expected_value:.2f}) too low for escalation",
                claim_risk=claim_risk,
                expected_value=expected_value,
                time_budget_remaining=time_remaining_minutes
            )
        
        # Build justification
        justification_parts = [f"Escalation from {current_tier} to {target_tier}"]
        if claim_risk is not None:
            justification_parts.append(f"claim_risk={claim_risk:.2f}")
        if expected_value is not None:
            justification_parts.append(f"expected_value={expected_value:.2f}")
        if time_remaining_minutes is not None:
            justification_parts.append(f"time_remaining={time_remaining_minutes:.1f}min")
        
        justification = " - ".join(justification_parts)
        
        # Escalation allowed
        return EscalationDecision(
            allowed=True,
            target_tier=target_tier,
            justification=justification,
            claim_risk=claim_risk,
            expected_value=expected_value,
            time_budget_remaining=time_remaining_minutes
        )
    
    def get_next_tier(self, current_tier: str) -> Optional[str]:
        """
        Get the next tier in the escalation ladder.
        
        Args:
            current_tier: Current tier name
            
        Returns:
            Next tier name or None if at max tier
        """
        current_tier = current_tier.upper()
        if current_tier not in self.TIER_LADDER:
            return None
        
        current_index = self.TIER_LADDER.index(current_tier)
        if current_index < len(self.TIER_LADDER) - 1:
            return self.TIER_LADDER[current_index + 1]
        
        return None
    
    def get_allowed_tiers(
        self,
        current_tier: str,
        claim_risk: Optional[float] = None,
        time_remaining_minutes: Optional[float] = None
    ) -> List[str]:
        """
        Get list of tiers that can be escalated to.
        
        Args:
            current_tier: Current tier
            claim_risk: Optional claim risk
            time_remaining_minutes: Optional time remaining
            
        Returns:
            List of allowed tier names
        """
        allowed = [current_tier]  # Current tier is always allowed
        
        current_tier = current_tier.upper()
        if current_tier not in self.TIER_LADDER:
            return allowed
        
        current_index = self.TIER_LADDER.index(current_tier)
        
        # Check each higher tier
        for tier in self.TIER_LADDER[current_index + 1:]:
            decision = self.can_escalate(
                current_tier=current_tier,
                target_tier=tier,
                claim_risk=claim_risk,
                time_remaining_minutes=time_remaining_minutes
            )
            if decision.allowed:
                allowed.append(tier)
        
        return allowed

