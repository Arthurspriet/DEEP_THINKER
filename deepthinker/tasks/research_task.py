"""
Research task definition - Web search for information before code generation.
"""

from crewai import Task, Agent
from typing import Optional, Dict, Any


def create_research_task(
    agent: Agent,
    objective: str,
    context: Optional[Dict[str, Any]] = None
) -> Task:
    """
    Create a task for web research before code generation.
    
    Args:
        agent: WebSearch agent to execute the task
        objective: Code generation objective to research
        context: Additional context for research focus
        
    Returns:
        Configured CrewAI Task
    """
    # Build description
    description_parts = [
        f"Research web resources to help accomplish this coding objective:\n{objective}\n"
    ]
    
    if context:
        description_parts.append("\nAdditional Context:")
        for key, value in context.items():
            description_parts.append(f"- {key}: {value}")
    
    description_parts.append("""
IMPORTANT: You MUST use the "Web Search" tool to search the web. Do NOT rely solely on your training data.
For each research area below, invoke the Web Search tool with a specific query.

Research Focus Areas (use Web Search tool for each):
1. Search for official documentation for relevant libraries and APIs
2. Search for code examples and implementation patterns
3. Search for best practices and recommended approaches
4. Search for common pitfalls and how to avoid them
5. Search for library compatibility and version information

Search Strategy:
- Execute multiple Web Search tool calls with specific queries
- Target official documentation sources
- Focus on practical implementation guidance
- Consider edge cases and error handling

Expected Output Format:
Provide a structured research summary with:

## Key Libraries/APIs Found
- Library name and purpose
- Official documentation links
- Key features relevant to the task

## Implementation Approaches
- Recommended approaches and patterns
- Code examples or references
- Best practices to follow

## Important Considerations
- Common pitfalls to avoid
- Edge cases to handle
- Performance or security considerations

## Helpful Resources
- Documentation URLs
- Tutorial or example links
- Stack Overflow or community resources

Keep your research focused and actionable. Prioritize information that will directly help 
with implementing the objective.
""")
    
    description = "\n".join(description_parts)
    
    return Task(
        description=description,
        expected_output="Structured research summary with libraries, approaches, considerations, and resource links",
        agent=agent
    )

