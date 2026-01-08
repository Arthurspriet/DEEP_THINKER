"""
WebSearch Agent - Researches information, documentation, and best practices.
"""

from crewai import Agent
from typing import Any, List, Optional


def create_websearch_agent(llm: Any, tools: Optional[List] = None) -> Agent:
    """
    Create a WebSearch Agent that researches information from the web.
    
    Args:
        llm: Language model instance (Ollama)
        tools: List of tools to equip the agent with (WebSearchTool)
        
    Returns:
        Configured CrewAI Agent
    """
    return Agent(
        role="Research and Information Specialist",
        goal="Search the web for relevant documentation, code examples, and best practices to help solve coding tasks",
        backstory="""You are a skilled technical researcher with expertise in finding and synthesizing
        information from the web. You specialize in:
        - Finding official documentation and API references
        - Discovering code examples and implementation patterns
        - Identifying best practices and common pitfalls
        - Researching library usage and compatibility
        - Gathering current technical information
        
        Your research approach:
        - Start with official documentation sources
        - Look for established libraries and frameworks
        - Search for code examples and tutorials
        - Identify potential gotchas and edge cases
        - Synthesize findings into actionable insights
        
        You provide concise, relevant summaries that help other agents:
        - Understand how to use specific libraries or APIs
        - Learn about best practices for the task at hand
        - Avoid common mistakes and pitfalls
        - Make informed decisions about implementation approaches
        
        Your research is thorough but focused, prioritizing quality over quantity.""",
        llm=llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False
    )

