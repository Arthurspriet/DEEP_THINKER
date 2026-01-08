"""
Training module for Web Search Predictor.

Provides training pipeline for learning from historical mission data.
"""

from .train_predictor import train, load_training_data

__all__ = [
    "train",
    "load_training_data",
]

