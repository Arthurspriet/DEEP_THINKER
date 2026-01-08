"""
Utilities for DeepThinker 2.0.

Common utilities including convergence tracking, time management,
and helper functions.
"""

from .convergence import ConvergenceTracker, ConvergenceStatus, StoppingReason

__all__ = [
    "ConvergenceTracker",
    "ConvergenceStatus",
    "StoppingReason",
]

