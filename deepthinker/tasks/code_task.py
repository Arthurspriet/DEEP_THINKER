"""
Code generation task definition.
"""

from crewai import Task, Agent
from typing import Optional, Dict, Any


def create_code_task(
    agent: Agent,
    objective: str,
    context: Optional[Dict[str, Any]] = None,
    data_config: Optional[Any] = None
) -> Task:
    """
    Create a task for code generation.
    
    Args:
        agent: Coder agent to execute the task
        objective: Code generation objective
        context: Additional context and constraints
        data_config: Dataset configuration if applicable
        
    Returns:
        Configured CrewAI Task
    """
    # Build description
    description_parts = [
        f"Generate Python code that accomplishes the following objective:\n{objective}\n"
    ]
    
    if context:
        description_parts.append("\nAdditional Context:")
        for key, value in context.items():
            description_parts.append(f"- {key}: {value}")
    
    if data_config and data_config.is_enabled():
        description_parts.append(f"""
Dataset-Based Requirements:
- The code will be executed on a real dataset: {data_config.data_path}
- Task type: {data_config.task_type}
- Target column: {data_config.target_column or 'last column'}

IMPORTANT: Create a class with the following scikit-learn-like interface:
1. A fit(X, y) method that trains the model
   - X: numpy array of shape (n_samples, n_features)
   - y: numpy array of shape (n_samples,)
2. A predict(X) method that makes predictions
   - X: numpy array of shape (n_samples, n_features)
   - Returns: numpy array of predictions

Your code will be evaluated on actual performance metrics ({data_config.task_type} metrics).
""")
    
    description_parts.append("""
Code Requirements:
- Write clean, readable Python code
- Include docstrings and comments
- Add type hints where appropriate
- Handle edge cases and errors
- Follow PEP 8 style guidelines
- Make it production-ready

Return ONLY the Python code, no explanations or markdown formatting.
""")
    
    description = "\n".join(description_parts)
    
    return Task(
        description=description,
        expected_output="Clean, well-documented Python code that meets all requirements",
        agent=agent
    )

