"""
Task definitions for the DeepThinker system.
"""

from .code_task import create_code_task
from .evaluate_task import create_evaluate_task
from .revise_task import create_revise_task
from .simulate_task import create_simulate_task
from .execute_task import create_execute_task
from .research_task import create_research_task
from .planning_task import create_planning_task

__all__ = [
    "create_code_task",
    "create_evaluate_task",
    "create_revise_task",
    "create_simulate_task",
    "create_execute_task",
    "create_research_task",
    "create_planning_task",
]

