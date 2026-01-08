"""
Code revision task definition.
"""

from crewai import Task, Agent
from typing import Optional, Any


def create_revise_task(
    agent: Agent,
    objective: str,
    previous_code: str,
    evaluation: Any,
    iteration: int,
    metric_result: Optional[Any] = None,
    previous_metrics: Optional[Any] = None
) -> Task:
    """
    Create a task for code revision based on evaluation feedback.
    
    Args:
        agent: Coder agent to execute the task
        objective: Original objective
        previous_code: Code from previous iteration
        evaluation: Evaluation result with feedback
        iteration: Current iteration number
        metric_result: Current iteration metrics
        previous_metrics: Previous iteration metrics for comparison
        
    Returns:
        Configured CrewAI Task
    """
    description_parts = [
        f"Revise the following code (Iteration {iteration}) to address evaluation feedback.\n",
        f"\nOriginal Objective:\n{objective}\n",
        f"\nPrevious Code:\n```python\n{previous_code}\n```\n",
        f"\nEvaluation Feedback:\n{evaluation.summary()}\n"
    ]
    
    if metric_result and not metric_result.execution_result.success:
        description_parts.append(f"""
⚠️ CRITICAL: Previous code failed to execute!
{metric_result.execution_result.error_summary()}

You MUST fix these execution errors.
""")
    
    if metric_result and previous_metrics:
        description_parts.append(f"""
Performance Comparison:
- Current metrics: {metric_result.metrics}
- Previous metrics: {previous_metrics.metrics if hasattr(previous_metrics, 'metrics') else previous_metrics}

{'📈 Performance improved!' if metric_result.normalized_score() > previous_metrics.normalized_score() else '📉 Performance regressed - investigate why'}
""")
    
    description_parts.append("""
Revision Instructions:
1. Address all CRITICAL issues immediately
2. Fix MAJOR issues that impact functionality or design
3. Improve MINOR issues where practical
4. Preserve strengths and working code
5. Maintain or improve performance metrics
6. Ensure code still meets original objective

Return ONLY the revised Python code, no explanations or markdown formatting.
""")
    
    description = "\n".join(description_parts)
    
    return Task(
        description=description,
        expected_output="Improved Python code that addresses feedback while preserving what works",
        agent=agent
    )

