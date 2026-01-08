"""
Escalation policy engine for controlled capability escalation.

Enforces rules for when code can escalate to higher-capability execution profiles.
"""

from typing import Optional, Literal
from .provenance import CodeProvenance


class EscalationPolicy:
    """
    Policy engine for execution profile escalation.
    
    Enforces strict rules:
    - SAFE_ML → GPU_ML: Auto-allowed if trust_score >= 0.6
    - SAFE_ML → BROWSER: Requires human approval OR trust_score >= 0.9
    - Any → TRUSTED: Requires human approval only
    """
    
    # Escalation rules: (from_profile, to_profile) -> (auto_allowed_trust, requires_human)
    ESCALATION_RULES = {
        ("SAFE_ML", "GPU_ML"): (0.6, False),
        ("SAFE_ML", "BROWSER"): (0.9, True),
        ("GPU_ML", "BROWSER"): (0.9, True),
        ("SAFE_ML", "TRUSTED"): (None, True),  # Always requires human
        ("GPU_ML", "TRUSTED"): (None, True),
        ("BROWSER", "TRUSTED"): (None, True),
    }
    
    def can_escalate(
        self,
        from_profile: str,
        to_profile: str,
        provenance: CodeProvenance,
        human_approved: bool = False
    ) -> tuple[bool, str]:
        """
        Check if escalation is allowed.
        
        Args:
            from_profile: Current execution profile name
            to_profile: Target execution profile name
            provenance: Code provenance information
            human_approved: Whether human has explicitly approved escalation
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Same profile - always allowed
        if from_profile == to_profile:
            return True, "Same profile"
        
        # Check if escalation path exists
        rule_key = (from_profile, to_profile)
        if rule_key not in self.ESCALATION_RULES:
            return False, f"Escalation from {from_profile} to {to_profile} not defined"
        
        auto_trust, requires_human = self.ESCALATION_RULES[rule_key]
        
        # TRUSTED always requires human approval
        if to_profile == "TRUSTED":
            if human_approved:
                return True, "Human-approved escalation to TRUSTED"
            return False, "TRUSTED profile requires explicit human approval"
        
        # Check human approval requirement
        if requires_human and not human_approved:
            # Check if trust score is high enough to bypass
            if auto_trust is not None and provenance.trust_score >= auto_trust:
                return True, f"High trust score ({provenance.trust_score:.2f}) bypasses human approval requirement"
            return False, f"Escalation to {to_profile} requires human approval (or trust_score >= {auto_trust})"
        
        # Check auto-allowed trust threshold
        if auto_trust is not None:
            if provenance.trust_score >= auto_trust:
                return True, f"Trust score ({provenance.trust_score:.2f}) meets threshold ({auto_trust})"
            return False, f"Trust score ({provenance.trust_score:.2f}) below threshold ({auto_trust})"
        
        # If we get here, escalation is allowed
        return True, "Escalation allowed"
    
    def should_escalate_on_failure(
        self,
        profile: str,
        error: str,
        provenance: CodeProvenance
    ) -> Optional[str]:
        """
        Suggest escalation profile based on failure error.
        
        Args:
            profile: Current execution profile
            error: Error message from execution
            provenance: Code provenance
            
        Returns:
            Suggested profile name, or None if no escalation recommended
        """
        error_lower = error.lower()
        
        # GPU-related errors
        if any(keyword in error_lower for keyword in ["cuda", "gpu", "device", "cudnn"]):
            if profile == "SAFE_ML" and provenance.trust_score >= 0.6:
                return "GPU_ML"
        
        # Network-related errors
        if any(keyword in error_lower for keyword in ["network", "connection", "timeout", "dns"]):
            if profile in ["SAFE_ML", "GPU_ML"]:
                # Only suggest if trust score is very high
                if provenance.trust_score >= 0.9 or provenance.is_human_reviewed():
                    return "BROWSER"
        
        # Resource exhaustion
        if any(keyword in error_lower for keyword in ["memory", "oom", "out of memory"]):
            if profile == "SAFE_ML" and provenance.trust_score >= 0.6:
                return "GPU_ML"  # GPU_ML has more RAM
        
        return None
    
    def get_allowed_profiles(
        self,
        current_profile: str,
        provenance: CodeProvenance,
        human_approved: bool = False
    ) -> list[str]:
        """
        Get list of profiles that can be escalated to.
        
        Args:
            current_profile: Current execution profile
            provenance: Code provenance
            human_approved: Whether human has approved escalation
            
        Returns:
            List of allowed profile names
        """
        allowed = [current_profile]  # Current profile is always allowed
        
        for to_profile in ["GPU_ML", "BROWSER", "TRUSTED"]:
            if to_profile == current_profile:
                continue
            
            can_escalate, _ = self.can_escalate(
                current_profile,
                to_profile,
                provenance,
                human_approved
            )
            if can_escalate:
                allowed.append(to_profile)
        
        return allowed


# Global policy instance
_default_policy = EscalationPolicy()


def get_default_policy() -> EscalationPolicy:
    """Get the default escalation policy."""
    return _default_policy

