"""
Arbiter for DeepThinker 2.0.

The final decision-maker that resolves contradictions between councils
and ensures output consistency and coherence.
"""

from .arbiter import Arbiter, ArbiterDecision, CouncilOutput

__all__ = ["Arbiter", "ArbiterDecision", "CouncilOutput"]

