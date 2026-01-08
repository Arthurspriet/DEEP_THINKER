"""
Planning task definition - Strategic workflow planning and task decomposition.
"""

from crewai import Task, Agent
from typing import Optional, Dict, Any


def create_planning_task(
    agent: Agent,
    objective: str,
    context: Optional[Dict[str, Any]] = None,
    data_config: Optional[Any] = None,
    simulation_config: Optional[Any] = None,
    iteration_config: Optional[Any] = None
) -> Task:
    """
    Create a task for workflow planning and coordination.
    
    Args:
        agent: Planner agent to execute the task
        objective: Primary objective to plan for
        context: Additional context and constraints
        data_config: Dataset configuration if applicable
        simulation_config: Simulation configuration if applicable
        iteration_config: Iteration configuration
        
    Returns:
        Configured CrewAI Task
    """
    # Build description
    description_parts = [
        f"Analyze the following objective and create a comprehensive execution plan:\n{objective}\n"
    ]
    
    if context:
        description_parts.append("\nAdditional Context:")
        for key, value in context.items():
            if key != "research_findings":  # Don't include research in planning
                description_parts.append(f"- {key}: {value}")
    
    # Add configuration details
    config_info = []
    
    if data_config and data_config.is_enabled():
        config_info.append(f"""
Dataset Configuration:
- Dataset: {data_config.data_path}
- Task Type: {data_config.task_type}
- Target Column: {data_config.target_column or 'last column'}
- Execution Backend: {data_config.execution_backend}
- Metric Weight: {data_config.metric_weight}
""")
    
    if simulation_config and simulation_config.is_enabled():
        config_info.append(f"""
Simulation Configuration:
- Mode: {simulation_config.mode}
- Scenarios: {len(simulation_config.scenarios) if simulation_config.scenarios else 0}
""")
    
    if iteration_config:
        config_info.append(f"""
Iteration Configuration:
- Max Iterations: {iteration_config.max_iterations}
- Quality Threshold: {iteration_config.quality_threshold}/10
- Enabled: {iteration_config.enabled}
""")
    
    if config_info:
        description_parts.append("\nSystem Configuration:")
        description_parts.extend(config_info)
    
    description_parts.append("""
Available Agents:
1. **Researcher** - Searches web for documentation, examples, and best practices
2. **Coder** - Generates and revises Python code
3. **Evaluator** - Assesses code quality and provides feedback
4. **Executor** - Runs code on datasets and computes metrics
5. **Simulator** - Tests code against various scenarios

Your Task:
Create a detailed execution plan that specifies:

1. OBJECTIVE ANALYSIS
   - What needs to be accomplished
   - Key challenges and considerations
   - Complexity assessment (simple/moderate/complex)

2. WORKFLOW STRATEGY
   - Research: Yes/No (and why)
   - Code Generation: Yes (always required)
   - Evaluation: Yes (always required)
   - Execution: Yes/No (based on dataset availability)
   - Simulation: Yes/No (based on simulation config)
   
   For each phase, briefly explain why it's needed or can be skipped.

3. AGENT REQUIREMENTS
   
   For Researcher (if needed):
   - Specific search queries or topics to research
   - Focus areas (libraries, APIs, algorithms, patterns)
   - What information would be most valuable
   
   For Coder:
   - What to build (class, function, module)
   - Required interface (method signatures, parameters)
   - Key constraints and requirements
   - Edge cases to handle
   - Performance or style considerations
   
   For Evaluator:
   - Primary evaluation criteria to focus on
   - What aspects are most critical for this task
   - Specific quality metrics to emphasize
   
   For Executor (if dataset provided):
   - Which metrics are most important
   - Any special execution considerations
   - Security or performance concerns
   
   For Simulator (if enabled):
   - What scenarios to test
   - Edge cases to validate
   - Robustness checks to perform

4. SUCCESS CRITERIA
   - List 3-5 specific, measurable criteria for success
   - Include quality thresholds, metric targets, or functional requirements
   - Be realistic based on task complexity

5. ITERATION STRATEGY
   - How many iterations are likely needed
   - What improvements to focus on in each iteration
   - When to stop iterating (quality threshold, diminishing returns)

Format your plan clearly with headers and bullet points. Be specific and actionable.
The plan should be concise but comprehensive - focus on what matters most for this objective.
""")
    
    description = "\n".join(description_parts)
    
    return Task(
        description=description,
        expected_output="""A structured execution plan with:
- Objective analysis with complexity assessment
- Workflow strategy (which agents to use and why)
- Specific requirements for each agent
- Clear success criteria (3-5 measurable goals)
- Iteration strategy with quality thresholds""",
        agent=agent
    )

