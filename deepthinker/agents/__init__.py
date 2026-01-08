"""
Agent definitions for the DeepThinker 2.0 system.

Provides both legacy flat imports and new subdirectory-based organization.
"""

# New subdirectory-based imports
from .planner import create_planner_agent
from .coder import create_coder_agent
from .evaluator import create_evaluator_agent
from .simulator import create_simulator_agent
from .researcher import create_websearch_agent
from .executor import create_executor_agent

__all__ = [
    "create_coder_agent",
    "create_evaluator_agent",
    "create_simulator_agent",
    "create_executor_agent",
    "create_websearch_agent",
    "create_planner_agent",
]
