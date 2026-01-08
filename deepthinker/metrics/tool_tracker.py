"""
Tool Tracker for DeepThinker Metrics.

Records per-step tool usage with:
- Tool name and params hash
- Latency and success/failure
- Output size metrics
- Heuristic score delta attribution

Provides @tracked_tool decorator for easy instrumentation.
"""

import functools
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from .config import MetricsConfig, get_metrics_config, should_sample

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ToolUsageRecord:
    """
    Record of a single tool invocation.
    
    Attributes:
        tool_name: Name of the tool
        params_hash: Hash of parameters (not raw sensitive data)
        latency_ms: Execution time in milliseconds
        success: Whether the tool call succeeded
        output_size: Size of output in characters/bytes
        error_type: Error type if failed
        timestamp: When the tool was called
        step_id: Associated step identifier
        phase_id: Associated phase identifier
        mission_id: Associated mission identifier
        score_delta_attributed: Attributed score delta (heuristic)
    """
    tool_name: str
    params_hash: str = ""
    latency_ms: float = 0.0
    success: bool = True
    output_size: int = 0
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step_id: str = ""
    phase_id: str = ""
    mission_id: str = ""
    score_delta_attributed: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "params_hash": self.params_hash,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "output_size": self.output_size,
            "error_type": self.error_type,
            "timestamp": self.timestamp.isoformat(),
            "step_id": self.step_id,
            "phase_id": self.phase_id,
            "mission_id": self.mission_id,
            "score_delta_attributed": self.score_delta_attributed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUsageRecord":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            tool_name=data.get("tool_name", ""),
            params_hash=data.get("params_hash", ""),
            latency_ms=data.get("latency_ms", 0.0),
            success=data.get("success", True),
            output_size=data.get("output_size", 0),
            error_type=data.get("error_type"),
            timestamp=timestamp,
            step_id=data.get("step_id", ""),
            phase_id=data.get("phase_id", ""),
            mission_id=data.get("mission_id", ""),
            score_delta_attributed=data.get("score_delta_attributed", 0.0),
        )


def _hash_params(params: Dict[str, Any]) -> str:
    """
    Create a hash of parameters for tracking without exposing sensitive data.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Short hash string
    """
    # Create a stable string representation
    param_str = str(sorted(params.items()))
    return hashlib.sha256(param_str.encode()).hexdigest()[:12]


def tracked_tool(
    tool_name: Optional[str] = None,
    track_output_size: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to track tool invocations.
    
    Usage:
        @tracked_tool("web_search")
        def search(query: str) -> str:
            ...
    
    Or without name (uses function name):
        @tracked_tool()
        def my_tool(params: dict) -> str:
            ...
    
    Args:
        tool_name: Optional tool name. Uses function name if None.
        track_output_size: Whether to track output size
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        name = tool_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = get_metrics_config()
            
            # Check if tracking is enabled and should sample
            if not should_sample(config.tool_track_sample_rate):
                return func(*args, **kwargs)
            
            # Get tracker and record
            tracker = get_tool_tracker()
            
            start_time = time.time()
            success = True
            error_type = None
            output = None
            
            try:
                output = func(*args, **kwargs)
                return output
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                # Compute output size
                output_size = 0
                if track_output_size and output is not None:
                    if isinstance(output, str):
                        output_size = len(output)
                    elif isinstance(output, (list, dict)):
                        output_size = len(str(output))
                    elif hasattr(output, '__len__'):
                        output_size = len(output)
                
                # Hash params (excluding self if method)
                params_to_hash = dict(kwargs)
                for i, arg in enumerate(args):
                    if i == 0 and hasattr(arg, '__class__'):
                        continue  # Skip self
                    params_to_hash[f"arg_{i}"] = str(arg)[:100]  # Truncate
                
                record = ToolUsageRecord(
                    tool_name=name,
                    params_hash=_hash_params(params_to_hash),
                    latency_ms=latency_ms,
                    success=success,
                    output_size=output_size,
                    error_type=error_type,
                )
                
                tracker.add_record(record)
        
        return cast(F, wrapper)
    
    return decorator


class ToolTracker:
    """
    Tracks tool usage across steps and phases.
    
    Records are collected per-step and can be attributed
    score deltas via heuristic (equal split or weighted by latency).
    
    Usage:
        tracker = ToolTracker()
        
        # Record a tool call
        tracker.add_record(ToolUsageRecord(
            tool_name="web_search",
            latency_ms=500,
            success=True,
            output_size=1024,
        ))
        
        # Attribute score delta to tools in step
        tracker.attribute_score_delta(
            step_id="step_1",
            score_delta=0.1,
            attribution_method="latency_weighted",
        )
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize the tool tracker.
        
        Args:
            config: Optional MetricsConfig. Uses global if None.
        """
        self.config = config or get_metrics_config()
        self._records: List[ToolUsageRecord] = []
        self._current_step_id: str = ""
        self._current_phase_id: str = ""
        self._current_mission_id: str = ""
    
    def set_context(
        self,
        mission_id: str = "",
        phase_id: str = "",
        step_id: str = "",
    ) -> None:
        """
        Set context for subsequent tool recordings.
        
        Args:
            mission_id: Current mission ID
            phase_id: Current phase ID
            step_id: Current step ID
        """
        self._current_mission_id = mission_id
        self._current_phase_id = phase_id
        self._current_step_id = step_id
    
    def add_record(self, record: ToolUsageRecord) -> None:
        """
        Add a tool usage record.
        
        Args:
            record: ToolUsageRecord to add
        """
        # Fill in context if not set
        if not record.mission_id:
            record.mission_id = self._current_mission_id
        if not record.phase_id:
            record.phase_id = self._current_phase_id
        if not record.step_id:
            record.step_id = self._current_step_id
        
        self._records.append(record)
        
        logger.debug(
            f"[TOOL_TRACKER] Recorded {record.tool_name}: "
            f"latency={record.latency_ms:.1f}ms, success={record.success}"
        )
    
    def record_step(
        self,
        step_id: str,
        step_result: Any,
        tools_used: Optional[List[str]] = None,
        step_latency_ms: float = 0.0,
    ) -> List[ToolUsageRecord]:
        """
        Record tool usage for a completed step.
        
        This is a convenience method for step-level tracking when
        individual tool calls weren't instrumented.
        
        Args:
            step_id: Step identifier
            step_result: Result from step execution
            tools_used: List of tool names used (from step definition)
            step_latency_ms: Total step latency
            
        Returns:
            List of created ToolUsageRecord objects
        """
        if not tools_used:
            return []
        
        records = []
        
        # Distribute latency equally among tools
        latency_per_tool = step_latency_ms / len(tools_used) if tools_used else 0
        
        for tool_name in tools_used:
            record = ToolUsageRecord(
                tool_name=tool_name,
                latency_ms=latency_per_tool,
                success=True,  # Assume success if step completed
                step_id=step_id,
                phase_id=self._current_phase_id,
                mission_id=self._current_mission_id,
            )
            self._records.append(record)
            records.append(record)
        
        return records
    
    def attribute_score_delta(
        self,
        step_id: str,
        score_delta: float,
        attribution_method: str = "equal",
    ) -> None:
        """
        Attribute a score delta to tools used in a step.
        
        Args:
            step_id: Step to attribute to
            score_delta: Score change to attribute
            attribution_method: "equal" or "latency_weighted"
        """
        step_records = [r for r in self._records if r.step_id == step_id]
        
        if not step_records:
            return
        
        if attribution_method == "equal":
            delta_per_tool = score_delta / len(step_records)
            for record in step_records:
                record.score_delta_attributed = delta_per_tool
        
        elif attribution_method == "latency_weighted":
            total_latency = sum(r.latency_ms for r in step_records)
            if total_latency > 0:
                for record in step_records:
                    weight = record.latency_ms / total_latency
                    record.score_delta_attributed = score_delta * weight
        
        logger.debug(
            f"[TOOL_TRACKER] Attributed delta={score_delta:.3f} to "
            f"{len(step_records)} tools in step {step_id}"
        )
    
    def get_records(
        self,
        mission_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> List[ToolUsageRecord]:
        """
        Get tool usage records with optional filtering.
        
        Args:
            mission_id: Filter by mission
            phase_id: Filter by phase
            step_id: Filter by step
            
        Returns:
            List of matching ToolUsageRecord objects
        """
        records = self._records
        
        if mission_id:
            records = [r for r in records if r.mission_id == mission_id]
        if phase_id:
            records = [r for r in records if r.phase_id == phase_id]
        if step_id:
            records = [r for r in records if r.step_id == step_id]
        
        return records
    
    def get_phase_summary(self, phase_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a phase.
        
        Args:
            phase_id: Phase to summarize
            
        Returns:
            Summary dictionary
        """
        records = self.get_records(phase_id=phase_id)
        
        if not records:
            return {
                "phase_id": phase_id,
                "total_tools": 0,
                "total_latency_ms": 0,
                "success_rate": 0,
            }
        
        by_tool: Dict[str, List[ToolUsageRecord]] = {}
        for r in records:
            if r.tool_name not in by_tool:
                by_tool[r.tool_name] = []
            by_tool[r.tool_name].append(r)
        
        return {
            "phase_id": phase_id,
            "total_tools": len(records),
            "unique_tools": list(by_tool.keys()),
            "total_latency_ms": sum(r.latency_ms for r in records),
            "success_rate": sum(1 for r in records if r.success) / len(records),
            "total_output_size": sum(r.output_size for r in records),
            "total_score_delta": sum(r.score_delta_attributed for r in records),
            "by_tool": {
                name: {
                    "count": len(recs),
                    "total_latency_ms": sum(r.latency_ms for r in recs),
                    "success_rate": sum(1 for r in recs if r.success) / len(recs),
                }
                for name, recs in by_tool.items()
            },
        }
    
    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all records to list of dictionaries."""
        return [r.to_dict() for r in self._records]


# Global tracker instance
_tracker: Optional[ToolTracker] = None


def get_tool_tracker(config: Optional[MetricsConfig] = None) -> ToolTracker:
    """Get global tool tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ToolTracker(config=config)
    return _tracker


def reset_tool_tracker() -> None:
    """Reset global tool tracker (mainly for testing)."""
    global _tracker
    _tracker = None

