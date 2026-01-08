"""
Computation of task-specific evaluation metrics.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class MetricComputer:
    """
    Computes task-specific metrics for model evaluation.
    
    Supports classification and regression tasks with appropriate metrics.
    """
    
    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metric names to values
        """
        # Determine if binary or multi-class
        n_classes = len(np.unique(y_true))
        average = "binary" if n_classes == 2 else "macro"
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
        }
        
        # Add precision, recall, F1 with zero_division handling
        try:
            metrics["precision"] = precision_score(
                y_true, y_pred, average=average, zero_division=0
            )
            metrics["recall"] = recall_score(
                y_true, y_pred, average=average, zero_division=0
            )
            metrics["f1"] = f1_score(
                y_true, y_pred, average=average, zero_division=0
            )
        except Exception:
            # Fallback if metrics fail (e.g., single class in predictions)
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names to values
        """
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str
    ) -> Dict[str, float]:
        """
        Compute metrics based on task type.
        
        Args:
            y_true: True values/labels
            y_pred: Predicted values/labels
            task_type: "classification" or "regression"
            
        Returns:
            Dictionary of computed metrics
        """
        if task_type == "classification":
            return MetricComputer.compute_classification_metrics(y_true, y_pred)
        elif task_type == "regression":
            return MetricComputer.compute_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

