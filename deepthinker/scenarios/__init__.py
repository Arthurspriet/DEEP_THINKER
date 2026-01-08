"""
Scenarios Module for DeepThinker.

Provides structured scenario modeling for strategic analysis.
Scenarios are first-class objects with explicit drivers, timelines,
winners/losers, and failure modes.

Components:
- Scenario: Structured scenario representation
- ScenarioFactory: Generates and validates scenario sets
- ScenarioParser: Extracts scenarios from LLM outputs
"""

from .scenario_model import (
    Scenario,
    ScenarioDriver,
    ScenarioFactory,
    ScenarioParser,
    ScenarioSet,
    ScenarioTimeline,
    get_scenario_factory,
)

__all__ = [
    "Scenario",
    "ScenarioDriver",
    "ScenarioFactory",
    "ScenarioParser",
    "ScenarioSet",
    "ScenarioTimeline",
    "get_scenario_factory",
]

