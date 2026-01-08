"""
Code evaluation task definition.
"""

from crewai import Task, Agent
from typing import Optional, Any


def create_evaluate_task(
    agent: Agent,
    objective: str,
    code: str,
    metric_result: Optional[Any] = None
) -> Task:
    """
    Create a task for code evaluation.
    
    Args:
        agent: Evaluator agent to execute the task
        objective: Original objective for context
        code: Code to evaluate
        metric_result: Optional metric results from execution
        
    Returns:
        Configured CrewAI Task
    """
    description_parts = [
        f"Evaluate the following Python code against this objective:\n{objective}\n",
        f"\nCode to Evaluate:\n```python\n{code}\n```\n"
    ]
    
    if metric_result and metric_result.execution_result:
        if not metric_result.execution_result.success:
            description_parts.append(f"""
⚠️ EXECUTION ERROR:
{metric_result.execution_result.error_summary()}

This is a CRITICAL issue that must be addressed.
""")
        else:
            description_parts.append(f"""
✅ Execution Results:
{metric_result.summary()}

Consider both code quality AND performance metrics in your evaluation.
""")
    
    description_parts.append("""
Provide a comprehensive evaluation in the following format:

Quality Score: [0-10]/10

Issues:
- [CRITICAL] Description of any critical issues (code doesn't work, security problems)
- [MAJOR] Description of major issues (inefficiency, poor design, missing error handling)
- [MINOR] Description of minor issues (style, documentation improvements)

Recommendations:
- Specific suggestion 1
- Specific suggestion 2
- etc.

Strengths:
- What the code does well
- Good practices observed
- etc.

Pass/Fail: [PASS or FAIL]

Scoring Rubric:
- 9-10: Excellent code, production-ready with minor or no issues
- 7-8: Good code, works well with some improvements needed
- 5-6: Acceptable code, significant improvements needed
- 3-4: Poor code, major issues need addressing
- 0-2: Code doesn't work or has critical problems
""")
    
    description = "\n".join(description_parts)
    
    return Task(
        description=description,
        expected_output="Structured evaluation with quality score, categorized issues, recommendations, and pass/fail assessment",
        agent=agent
    )

