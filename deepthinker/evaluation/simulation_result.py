"""
Data structures for simulation and scenario results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .metric_result import ExecutionResult


@dataclass
class SamplePrediction:
    """
    Single sample prediction with true label.
    
    Attributes:
        index: Sample index in test set
        true_value: Ground truth label/value
        predicted_value: Model prediction
        error: Absolute error (for analysis)
    """
    
    index: int
    true_value: Any
    predicted_value: Any
    error: float
    
    def __str__(self) -> str:
        """String representation."""
        return f"Sample {self.index}: true={self.true_value}, pred={self.predicted_value}, error={self.error:.4f}"


@dataclass
class ScenarioResult:
    """
    Results from executing code on a single scenario.
    
    Attributes:
        scenario_name: Name of the scenario
        scenario_description: Description of the scenario
        metrics: Computed performance metrics
        execution_result: Execution metadata and status
        sample_predictions: Sample predictions for analysis
        sample_errors: Worst prediction errors
        num_samples: Number of test samples
        success: Whether scenario executed successfully
        error_message: Error message if failed
    """
    
    scenario_name: str
    scenario_description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None
    sample_predictions: List[SamplePrediction] = field(default_factory=list)
    sample_errors: List[SamplePrediction] = field(default_factory=list)
    num_samples: int = 0
    success: bool = True
    error_message: str = ""
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Scenario: {self.scenario_name}",
            f"Description: {self.scenario_description}",
            f"Status: {'✅ SUCCESS' if self.success else '❌ FAILED'}",
        ]
        
        if not self.success:
            lines.append(f"Error: {self.error_message}")
            return "\n".join(lines)
        
        lines.append(f"Samples: {self.num_samples}")
        
        if self.metrics:
            lines.append("\nMetrics:")
            for name, value in self.metrics.items():
                lines.append(f"  {name}: {value:.4f}")
        
        if self.sample_predictions:
            lines.append(f"\nSample Predictions ({len(self.sample_predictions)}):")
            for sample in self.sample_predictions[:3]:
                lines.append(f"  {sample}")
        
        if self.sample_errors:
            lines.append(f"\nWorst Errors ({len(self.sample_errors)}):")
            for sample in self.sample_errors[:3]:
                lines.append(f"  {sample}")
        
        return "\n".join(lines)


@dataclass
class SimulationSummary:
    """
    Aggregated results from all simulation scenarios.
    
    Attributes:
        scenario_results: Results for each scenario
        cross_scenario_analysis: Comparative statistics across scenarios
        overall_success: Whether all scenarios succeeded
        total_scenarios: Total number of scenarios run
        successful_scenarios: Number of successful scenarios
    """
    
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    cross_scenario_analysis: Dict[str, Any] = field(default_factory=dict)
    overall_success: bool = True
    total_scenarios: int = 0
    successful_scenarios: int = 0
    
    def __post_init__(self):
        """Compute aggregate statistics."""
        self.total_scenarios = len(self.scenario_results)
        self.successful_scenarios = sum(1 for r in self.scenario_results if r.success)
        self.overall_success = self.successful_scenarios == self.total_scenarios
        
        # Compute cross-scenario analysis
        self._compute_cross_scenario_analysis()
    
    def _compute_cross_scenario_analysis(self):
        """Compute comparative statistics across scenarios."""
        if not self.scenario_results:
            return
        
        # Collect metrics across scenarios
        successful_results = [r for r in self.scenario_results if r.success]
        
        if not successful_results:
            return
        
        # Get all metric names
        all_metric_names = set()
        for result in successful_results:
            all_metric_names.update(result.metrics.keys())
        
        # Compute statistics per metric
        metric_stats = {}
        for metric_name in all_metric_names:
            values = [
                r.metrics[metric_name]
                for r in successful_results
                if metric_name in r.metrics
            ]
            
            if values:
                metric_stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": float(np.max(values) - np.min(values))
                }
        
        self.cross_scenario_analysis = {
            "metric_statistics": metric_stats,
            "scenario_names": [r.scenario_name for r in self.scenario_results],
            "failed_scenarios": [
                r.scenario_name for r in self.scenario_results if not r.success
            ]
        }
    
    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            "=" * 60,
            "SIMULATION SUMMARY",
            "=" * 60,
            f"\nTotal Scenarios: {self.total_scenarios}",
            f"Successful: {self.successful_scenarios}",
            f"Failed: {self.total_scenarios - self.successful_scenarios}",
            f"Overall Status: {'✅ ALL PASSED' if self.overall_success else '⚠️ SOME FAILED'}",
        ]
        
        # Cross-scenario metrics
        if self.cross_scenario_analysis.get("metric_statistics"):
            lines.append("\n" + "-" * 60)
            lines.append("CROSS-SCENARIO METRICS")
            lines.append("-" * 60)
            
            for metric_name, stats in self.cross_scenario_analysis["metric_statistics"].items():
                lines.append(f"\n{metric_name}:")
                lines.append(f"  Mean: {stats['mean']:.4f} (±{stats['std']:.4f})")
                lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                lines.append(f"  Variance: {stats['range']:.4f}")
        
        # Individual scenario summaries
        lines.append("\n" + "-" * 60)
        lines.append("SCENARIO DETAILS")
        lines.append("-" * 60)
        
        for result in self.scenario_results:
            lines.append("\n" + result.summary())
        
        return "\n".join(lines)
    
    def get_scenario_result(self, scenario_name: str) -> Optional[ScenarioResult]:
        """Get result for a specific scenario by name."""
        for result in self.scenario_results:
            if result.scenario_name == scenario_name:
                return result
        return None
    
    def get_metric_comparison(self, metric_name: str) -> Dict[str, float]:
        """
        Get comparison of a specific metric across all scenarios.
        
        Args:
            metric_name: Name of metric to compare
            
        Returns:
            Dictionary mapping scenario names to metric values
        """
        comparison = {}
        for result in self.scenario_results:
            if result.success and metric_name in result.metrics:
                comparison[result.scenario_name] = result.metrics[metric_name]
        return comparison

