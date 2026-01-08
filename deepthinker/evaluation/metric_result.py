"""
Data structures for metric-based evaluation results.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.execution_metrics import ExecutionMetrics


@dataclass
class ExecutionResult:
    """
    Results from executing generated code on a dataset.
    
    Attributes:
        success: Whether execution completed without errors
        predictions: Model predictions (if successful)
        error_type: Type of error if execution failed
        error_message: Detailed error message
        traceback: Full traceback for debugging
        execution_time: Time taken to execute in seconds
        metrics: Execution metrics (resource usage, network, etc.)
    """
    
    success: bool
    predictions: Optional[List] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    execution_time: float = 0.0
    metrics: Optional["ExecutionMetrics"] = None
    
    def error_summary(self) -> str:
        """Generate a concise error summary for feedback."""
        if self.success:
            return "Execution successful"
        
        lines = [f"Execution failed: {self.error_type}"]
        if self.error_message:
            lines.append(f"Message: {self.error_message}")
        
        return "\n".join(lines)


@dataclass
class MetricResult:
    """
    Contains computed metrics and metadata from code execution on dataset.
    
    Attributes:
        task_type: Type of task (classification or regression)
        metrics: Dictionary of metric names to values
        execution_result: Details of code execution
        num_samples: Number of test samples evaluated
    """
    
    task_type: str
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None
    num_samples: int = 0
    
    def normalized_score(self) -> float:
        """
        Normalize metrics to a 0-10 scale for comparison with LLM scores.
        
        For classification: Uses accuracy (0-1) * 10
        For regression: Uses normalized R² score
        
        Returns:
            Score on 0-10 scale
        """
        if not self.metrics:
            return 0.0
        
        if self.task_type == "classification":
            # Use accuracy as primary metric
            accuracy = self.metrics.get("accuracy", 0.0)
            return accuracy * 10
        
        elif self.task_type == "regression":
            # Use R² score, normalized to 0-10
            # R² can be negative, so we clamp to [0, 1]
            r2 = self.metrics.get("r2", 0.0)
            r2_clamped = max(0.0, min(1.0, r2))
            return r2_clamped * 10
        
        return 0.0
    
    def summary(self) -> str:
        """Generate human-readable summary of metrics."""
        lines = [f"Task Type: {self.task_type}"]
        lines.append(f"Samples: {self.num_samples}")
        
        if self.execution_result and not self.execution_result.success:
            lines.append(f"\n{self.execution_result.error_summary()}")
        elif self.metrics:
            lines.append("\nMetrics:")
            for name, value in self.metrics.items():
                lines.append(f"  {name}: {value:.4f}")
            lines.append(f"\nNormalized Score: {self.normalized_score():.1f}/10")
        
        return "\n".join(lines)

