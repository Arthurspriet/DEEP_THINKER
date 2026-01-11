"""
Enforcement Actions for Constitution.

Defines enforcement flags and helper functions for applying
constitution decisions to the system.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class EnforcementAction(str, Enum):
    """Actions that can be taken on constitution violations."""
    OK = "ok"
    """No violation - proceed normally."""
    
    WARN = "warn"
    """Log warning but don't block."""
    
    BLOCK_LEARNING = "block_learning"
    """Block learning updates (bandit, router, predictor)."""
    
    STOP_DEEPENING = "stop_deepening"
    """Stop additional depth/rounds."""
    
    FORCE_EVIDENCE_MODE = "force_evidence_mode"
    """Force evidence-gathering mode next step."""


@dataclass
class ConstitutionFlags:
    """
    Flags output by constitution engine.
    
    Used by orchestrator to make enforcement decisions.
    
    Attributes:
        ok: No violations detected
        warn: Warnings present but not blocking
        block_learning: Block all learning updates
        stop_deepening: Stop additional depth/rounds
        force_evidence_mode: Force evidence gathering
        violations: List of violation messages
        actions: List of enforcement actions taken
    """
    ok: bool = True
    warn: bool = False
    block_learning: bool = False
    stop_deepening: bool = False
    force_evidence_mode: bool = False
    violations: List[str] = field(default_factory=list)
    actions: List[EnforcementAction] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def all_ok(cls) -> "ConstitutionFlags":
        """Create flags indicating no violations."""
        return cls(ok=True, actions=[EnforcementAction.OK])
    
    @classmethod
    def with_warning(cls, message: str) -> "ConstitutionFlags":
        """Create flags with a warning."""
        return cls(
            ok=False,
            warn=True,
            violations=[message],
            actions=[EnforcementAction.WARN],
        )
    
    @classmethod
    def with_block_learning(cls, message: str) -> "ConstitutionFlags":
        """Create flags that block learning."""
        return cls(
            ok=False,
            block_learning=True,
            violations=[message],
            actions=[EnforcementAction.BLOCK_LEARNING],
        )
    
    @classmethod
    def with_stop_deepening(cls, message: str) -> "ConstitutionFlags":
        """Create flags that stop deepening."""
        return cls(
            ok=False,
            stop_deepening=True,
            violations=[message],
            actions=[EnforcementAction.STOP_DEEPENING],
        )
    
    def add_violation(
        self,
        message: str,
        action: EnforcementAction,
    ) -> None:
        """Add a violation with its enforcement action."""
        self.ok = False
        self.violations.append(message)
        self.actions.append(action)
        
        if action == EnforcementAction.WARN:
            self.warn = True
        elif action == EnforcementAction.BLOCK_LEARNING:
            self.block_learning = True
        elif action == EnforcementAction.STOP_DEEPENING:
            self.stop_deepening = True
        elif action == EnforcementAction.FORCE_EVIDENCE_MODE:
            self.force_evidence_mode = True
    
    def merge(self, other: "ConstitutionFlags") -> "ConstitutionFlags":
        """Merge with another set of flags (OR logic for booleans)."""
        return ConstitutionFlags(
            ok=self.ok and other.ok,
            warn=self.warn or other.warn,
            block_learning=self.block_learning or other.block_learning,
            stop_deepening=self.stop_deepening or other.stop_deepening,
            force_evidence_mode=self.force_evidence_mode or other.force_evidence_mode,
            violations=self.violations + other.violations,
            actions=self.actions + other.actions,
            details={**self.details, **other.details},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ok": self.ok,
            "warn": self.warn,
            "block_learning": self.block_learning,
            "stop_deepening": self.stop_deepening,
            "force_evidence_mode": self.force_evidence_mode,
            "violations": self.violations,
            "actions": [a.value for a in self.actions],
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstitutionFlags":
        """Create from dictionary."""
        return cls(
            ok=data.get("ok", True),
            warn=data.get("warn", False),
            block_learning=data.get("block_learning", False),
            stop_deepening=data.get("stop_deepening", False),
            force_evidence_mode=data.get("force_evidence_mode", False),
            violations=data.get("violations", []),
            actions=[EnforcementAction(a) for a in data.get("actions", [])],
            details=data.get("details", {}),
        )


def should_block_learning(flags: ConstitutionFlags) -> bool:
    """Check if learning should be blocked based on flags."""
    return flags.block_learning


def should_stop_deepening(flags: ConstitutionFlags) -> bool:
    """Check if deepening should be stopped based on flags."""
    return flags.stop_deepening


def get_blocked_reason(flags: ConstitutionFlags) -> Optional[str]:
    """Get the reason for blocking, if any."""
    if not flags.violations:
        return None
    return "; ".join(flags.violations)




