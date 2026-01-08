"""
Phase Outcome Logger for DeepThinker.

Structured per-phase outcome vector collected automatically
by MissionOrchestrator to enable learning.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class PhaseOutcome:
    """
    Structured outcome record for a single phase execution.
    
    Captures all information needed to learn which orchestration
    decisions were worth the cost.
    """
    # Identification
    mission_id: str
    phase_name: str
    phase_type: str  # reconnaissance, analysis, deep_analysis, synthesis, etc.
    timestamp_start: datetime
    timestamp_end: datetime
    
    # Councils and Models
    councils_invoked: List[str] = field(default_factory=list)  # ordered
    models_used: List[Tuple[str, str]] = field(default_factory=list)  # (model_name, tier)
    consensus_executed: bool = False
    consensus_skipped_reason: Optional[str] = None
    
    # Resource Consumption
    tokens_consumed: int = 0
    wall_time_seconds: float = 0.0
    gpu_seconds: float = 0.0  # wall_time * utilization
    vram_peak_mb: int = 0
    
    # Memory Operations
    memory_chars_written: int = 0
    memory_stable_chars: int = 0
    memory_ephemeral_chars: int = 0
    
    # Quality Metrics (from Arbiter/Evaluator)
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    arbiter_raw_output: Optional[str] = None
    
    # Outcome Linkage
    mission_outcome_success: Optional[bool] = None  # Linked post-mission
    mission_final_quality: Optional[float] = None
    
    # Context
    time_remaining_at_start: float = 0.0
    effort_level: str = "standard"
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Depth Control (optional)
    depth_achieved: Optional[float] = None
    depth_target: Optional[float] = None
    enrichment_passes: int = 0
    termination_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime to ISO format
        result["timestamp_start"] = self.timestamp_start.isoformat()
        result["timestamp_end"] = self.timestamp_end.isoformat()
        # Convert tuples to lists for JSON
        result["models_used"] = [list(m) for m in self.models_used]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseOutcome":
        """Create from dictionary (e.g., from JSON)."""
        # Convert ISO strings back to datetime
        if isinstance(data.get("timestamp_start"), str):
            data["timestamp_start"] = datetime.fromisoformat(data["timestamp_start"])
        if isinstance(data.get("timestamp_end"), str):
            data["timestamp_end"] = datetime.fromisoformat(data["timestamp_end"])
        # Convert lists back to tuples
        if "models_used" in data and isinstance(data["models_used"], list):
            data["models_used"] = [tuple(m) if isinstance(m, list) else m for m in data["models_used"]]
        return cls(**data)

