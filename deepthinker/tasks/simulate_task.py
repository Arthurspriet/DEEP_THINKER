"""
Simulation task definition.
"""

from crewai import Task, Agent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluation.simulation_result import SimulationSummary


def create_simulate_task(
    agent: Agent,
    code: str,
    simulation_summary: "SimulationSummary"
) -> Task:
    """
    Create a task for scenario-based simulation with rich result data.
    
    Args:
        agent: Simulator agent to execute the task
        code: Code that was simulated
        simulation_summary: Complete simulation results
        
    Returns:
        Configured CrewAI Task
    """
    # Build detailed scenario results section
    scenario_details = []
    
    for result in simulation_summary.scenario_results:
        detail = f"\n### Scenario: {result.scenario_name}"
        detail += f"\nDescription: {result.scenario_description}"
        detail += f"\nStatus: {'✅ SUCCESS' if result.success else '❌ FAILED'}"
        
        if not result.success:
            detail += f"\nError: {result.error_message}"
        else:
            detail += f"\nSamples: {result.num_samples}"
            
            if result.metrics:
                detail += "\nMetrics:"
                for metric_name, value in result.metrics.items():
                    detail += f"\n  - {metric_name}: {value:.4f}"
            
            if result.sample_predictions:
                detail += f"\n\nSample Predictions (first {len(result.sample_predictions)}):"
                for sample in result.sample_predictions[:3]:
                    detail += f"\n  - Sample {sample.index}: true={sample.true_value:.4f}, pred={sample.predicted_value:.4f}, error={sample.error:.4f}"
            
            if result.sample_errors:
                detail += f"\n\nWorst Prediction Errors:"
                for sample in result.sample_errors[:3]:
                    detail += f"\n  - Sample {sample.index}: true={sample.true_value:.4f}, pred={sample.predicted_value:.4f}, error={sample.error:.4f}"
        
        scenario_details.append(detail)
    
    scenarios_text = "\n".join(scenario_details)
    
    # Build cross-scenario analysis section
    cross_analysis = ""
    if simulation_summary.cross_scenario_analysis.get("metric_statistics"):
        cross_analysis = "\n\n### Cross-Scenario Analysis\n"
        
        for metric_name, stats in simulation_summary.cross_scenario_analysis["metric_statistics"].items():
            cross_analysis += f"\n{metric_name}:"
            cross_analysis += f"\n  - Mean: {stats['mean']:.4f} (±{stats['std']:.4f})"
            cross_analysis += f"\n  - Range: [{stats['min']:.4f}, {stats['max']:.4f}]"
            cross_analysis += f"\n  - Variance: {stats['range']:.4f}"
        
        if simulation_summary.cross_scenario_analysis.get("failed_scenarios"):
            cross_analysis += f"\n\nFailed Scenarios: {', '.join(simulation_summary.cross_scenario_analysis['failed_scenarios'])}"
    
    description = f"""
Analyze the simulation results from testing the model across multiple scenarios.

## Code Tested:
```python
{code}
```

## Simulation Results

**Overall Status:** {'✅ ALL SCENARIOS PASSED' if simulation_summary.overall_success else '⚠️ SOME SCENARIOS FAILED'}
**Total Scenarios:** {simulation_summary.total_scenarios}
**Successful:** {simulation_summary.successful_scenarios}
**Failed:** {simulation_summary.total_scenarios - simulation_summary.successful_scenarios}

{scenarios_text}

{cross_analysis}

## Your Task

Provide a comprehensive simulation analysis report that includes:

1. **Performance Summary**
   - Overall model behavior across scenarios
   - Key performance metrics and trends
   - Comparison of performance across different scenarios

2. **Robustness Assessment**
   - How consistent is the model across scenarios?
   - Which scenarios showed best/worst performance?
   - Identify any performance degradation patterns

3. **Edge Cases and Failure Modes**
   - Analyze the worst prediction errors
   - Identify conditions where the model struggles
   - Note any scenarios that failed to execute

4. **Risk Analysis**
   - What are the main risks in deploying this model?
   - Under what conditions might it fail?
   - Are there data patterns that cause problems?

5. **Recommendations**
   - Suggested improvements for robustness
   - Additional scenarios worth testing
   - Data or feature engineering suggestions
   - Training or architecture changes to consider

Be specific and actionable. Focus on insights that help developers understand model behavior and improve reliability.
"""
    
    return Task(
        description=description,
        expected_output="Comprehensive simulation analysis with performance assessment, risk analysis, and actionable recommendations",
        agent=agent
    )

