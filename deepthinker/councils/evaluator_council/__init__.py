"""
Evaluator Council for DeepThinker 2.0.

Evaluation council that scores code quality using multiple LLMs
and weighted blend consensus.
"""

from .evaluator_council import EvaluatorCouncil, EvaluatorContext

__all__ = ["EvaluatorCouncil", "EvaluatorContext"]

