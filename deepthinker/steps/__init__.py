"""
Step Engine Module for DeepThinker 2.0.

The Step Engine provides fine-grained execution of mission phases by breaking
them into discrete steps, each executed by a single specialized model.

Key components:
- StepDefinition: Defines what a step should accomplish
- StepExecutor: Executes steps using single models with retry logic
- StepResult: Captures execution outcomes and artifacts

This layer sits BELOW councils:
- Councils (Planner, Researcher, Evaluator) handle strategy and reflection
- StepExecutor handles the actual execution of individual work units
"""

from .step_types import (
    StepStatus,
    StepType,
    StepExecutionContext,
    StepResult,
    StepDefinition,
    StepEvaluationResult,
)

from .step_executor import StepExecutor

__all__ = [
    "StepStatus",
    "StepType",
    "StepExecutionContext",
    "StepResult",
    "StepDefinition",
    "StepEvaluationResult",
    "StepExecutor",
]

