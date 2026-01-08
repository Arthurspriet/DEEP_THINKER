"""
Task definition for secure code execution monitoring.
"""

from crewai import Task, Agent
from typing import Optional, Dict, Any


def create_execute_task(
    agent: Agent,
    code: str,
    security_scan_results: Optional[Dict[str, Any]] = None,
    execution_results: Optional[Dict[str, Any]] = None
) -> Task:
    """
    Create a task for secure code execution monitoring and analysis.
    
    Args:
        agent: Executor agent to perform the task
        code: Code that was or will be executed
        security_scan_results: Results from pre-execution security scanning
        execution_results: Results from code execution
        
    Returns:
        Configured CrewAI Task
    """
    
    # Build security context
    security_context = ""
    if security_scan_results:
        total_issues = security_scan_results.get("total_issues", 0)
        max_risk = security_scan_results.get("max_risk_level", "unknown")
        is_safe = security_scan_results.get("is_safe", False)
        
        security_context = f"""
## Security Scan Results
- Total Issues: {total_issues}
- Maximum Risk Level: {max_risk}
- Safety Status: {'SAFE' if is_safe else 'POTENTIALLY UNSAFE'}

### Detected Issues:
"""
        for issue in security_scan_results.get("issues", []):
            security_context += f"- [{issue['risk_level'].upper()}] {issue['description']}"
            if issue.get('line_number'):
                security_context += f" (line {issue['line_number']})"
            security_context += "\n"
    
    # Build execution context
    execution_context = ""
    if execution_results:
        success = execution_results.get("success", False)
        execution_context = f"""
## Execution Results
- Status: {'SUCCESS' if success else 'FAILED'}
- Execution Time: {execution_results.get('execution_time', 'N/A')}s
"""
        if not success:
            execution_context += f"""
- Error Type: {execution_results.get('error_type', 'Unknown')}
- Error Message: {execution_results.get('error_message', 'No details')}
"""
    
    description = f"""
Analyze the security and execution of the following generated code.

## Code to Analyze:
```python
{code}
```

{security_context}

{execution_context}

## Your Task:
1. **Security Assessment**: Review any security issues detected during scanning
   - Categorize risks (false positives vs genuine threats)
   - Assess the overall security posture
   - Identify any patterns that could lead to security vulnerabilities

2. **Execution Analysis**: If execution results are available, analyze:
   - Whether execution completed successfully
   - Any runtime errors or anomalies
   - Resource usage and performance characteristics
   - Potential security implications of runtime behavior

3. **Recommendations**: Provide specific, actionable recommendations:
   - Security improvements for the code
   - Best practices to follow
   - Risk mitigation strategies
   - Alternative approaches if needed

4. **Risk Summary**: Provide a clear summary of:
   - Overall risk level (LOW/MEDIUM/HIGH/CRITICAL)
   - Key concerns that need attention
   - Whether the code is safe to deploy in production

Be thorough but practical. Focus on genuine security concerns while avoiding
unnecessary alarm over benign patterns common in ML code.
"""
    
    expected_output = """
A comprehensive security and execution analysis report containing:
1. Security risk assessment with specific issues categorized
2. Execution analysis with performance and correctness evaluation
3. Actionable recommendations for improvements
4. Overall risk summary with deployment readiness assessment
"""
    
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )

