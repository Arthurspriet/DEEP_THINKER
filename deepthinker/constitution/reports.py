"""
Constitution Reports for DeepThinker.

Provides summary reports and analysis of constitution events
for a mission or across missions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConstitutionConfig, get_constitution_config
from .ledger import ConstitutionLedger, get_ledger
from .types import ConstitutionEventType, ConstitutionViolationEvent
from .constitution_spec import InvariantType


@dataclass
class ViolationSummary:
    """Summary of a single violation."""
    invariant: str
    severity: float
    message: str
    phase_id: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant": self.invariant,
            "severity": self.severity,
            "message": self.message,
            "phase_id": self.phase_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ConstitutionReport:
    """
    Summary report of constitution events for a mission.
    
    Attributes:
        mission_id: Mission identifier
        generated_at: When the report was generated
        total_events: Total number of events
        total_violations: Total number of violations
        violations_by_invariant: Count per invariant type
        violations_by_severity: Count per severity level
        phases_evaluated: List of phases evaluated
        learning_blocked_count: Times learning was blocked
        deepening_stopped_count: Times deepening was stopped
        top_violations: Most severe violations
    """
    mission_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    total_events: int = 0
    total_violations: int = 0
    violations_by_invariant: Dict[str, int] = field(default_factory=dict)
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    phases_evaluated: List[str] = field(default_factory=list)
    learning_blocked_count: int = 0
    deepening_stopped_count: int = 0
    top_violations: List[ViolationSummary] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "generated_at": self.generated_at.isoformat(),
            "total_events": self.total_events,
            "total_violations": self.total_violations,
            "violations_by_invariant": self.violations_by_invariant,
            "violations_by_severity": self.violations_by_severity,
            "phases_evaluated": self.phases_evaluated,
            "learning_blocked_count": self.learning_blocked_count,
            "deepening_stopped_count": self.deepening_stopped_count,
            "top_violations": [v.to_dict() for v in self.top_violations],
        }
    
    @classmethod
    def from_ledger(
        cls,
        ledger: ConstitutionLedger,
        top_k: int = 10,
    ) -> "ConstitutionReport":
        """
        Generate a report from a ledger.
        
        Args:
            ledger: ConstitutionLedger to analyze
            top_k: Number of top violations to include
            
        Returns:
            ConstitutionReport instance
        """
        events = ledger.read_all()
        
        report = cls(mission_id=ledger.mission_id)
        report.total_events = len(events)
        
        phases = set()
        violations = []
        
        for event in events:
            event_type = event.get("event_type", "")
            phase_id = event.get("phase_id", "")
            
            if phase_id:
                phases.add(phase_id)
            
            if event_type == ConstitutionEventType.VIOLATION.value:
                report.total_violations += 1
                
                invariant = event.get("invariant", "unknown")
                report.violations_by_invariant[invariant] = (
                    report.violations_by_invariant.get(invariant, 0) + 1
                )
                
                severity = event.get("severity", 0.0)
                severity_bucket = _severity_bucket(severity)
                report.violations_by_severity[severity_bucket] = (
                    report.violations_by_severity.get(severity_bucket, 0) + 1
                )
                
                violations.append(ViolationSummary(
                    invariant=invariant,
                    severity=severity,
                    message=event.get("message", ""),
                    phase_id=phase_id,
                    timestamp=event.get("timestamp", ""),
                ))
            
            elif event_type == ConstitutionEventType.LEARNING_UPDATE.value:
                if not event.get("allowed", True):
                    report.learning_blocked_count += 1
        
        report.phases_evaluated = sorted(phases)
        
        # Sort violations by severity (descending) and take top-K
        violations.sort(key=lambda v: v.severity, reverse=True)
        report.top_violations = violations[:top_k]
        
        return report


def _severity_bucket(severity: float) -> str:
    """Categorize severity into buckets."""
    if severity >= 0.8:
        return "critical"
    elif severity >= 0.5:
        return "high"
    elif severity >= 0.3:
        return "medium"
    else:
        return "low"


def generate_report(
    mission_id: str,
    config: Optional[ConstitutionConfig] = None,
) -> ConstitutionReport:
    """
    Generate a constitution report for a mission.
    
    Args:
        mission_id: Mission identifier
        config: Optional configuration
        
    Returns:
        ConstitutionReport instance
    """
    ledger = get_ledger(mission_id, config)
    return ConstitutionReport.from_ledger(ledger)


def list_missions_with_violations(
    base_dir: Optional[Path] = None,
    config: Optional[ConstitutionConfig] = None,
) -> List[str]:
    """
    List all missions that have constitution violations.
    
    Args:
        base_dir: Optional base directory
        config: Optional configuration
        
    Returns:
        List of mission IDs with violations
    """
    config = config or get_constitution_config()
    base_dir = base_dir or Path(config.ledger_base_dir)
    
    missions_with_violations = []
    
    if not base_dir.exists():
        return missions_with_violations
    
    for ledger_file in base_dir.glob("*.jsonl"):
        mission_id = ledger_file.stem
        ledger = ConstitutionLedger(mission_id, config, base_dir)
        
        # Check if there are any violations
        for _ in ledger.read_events(event_type=ConstitutionEventType.VIOLATION):
            missions_with_violations.append(mission_id)
            break
    
    return missions_with_violations




