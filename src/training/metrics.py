"""
Metrics Module

Provides evaluation metrics for model training and validation.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import numpy as np
from typing import Callable, Dict, Any, List
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score
)


class MetricFactory:
    """
    Factory for creating metric functions.
    """
    
    @staticmethod
    def get_metric(metric_name: str) -> Callable:
        """
        Get a metric function by name.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Metric function
            
        Raises:
            ValueError: If metric not found
        """
        metric_functions = {
            'accuracy': MetricFactory.accuracy,
            'f1': MetricFactory.f1,
            'precision': MetricFactory.precision,
            'recall': MetricFactory.recall,
            'roc_auc': MetricFactory.roc_auc,
            'pr_auc': MetricFactory.pr_auc,
            'mcc': MetricFactory.mcc,
            'balanced_accuracy': MetricFactory.balanced_accuracy,
        }
        
        if metric_name not in metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}. "
                           f"Available: {list(metric_functions.keys())}")
        
        return metric_functions[metric_name]
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Calculate F1 score."""
        return f1_score(y_true, y_pred, average=average)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Calculate precision."""
        return precision_score(y_true, y_pred, average=average)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Calculate recall."""
        return recall_score(y_true, y_pred, average=average)
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate ROC AUC score."""
        return roc_auc_score(y_true, y_pred_proba)
    
    @staticmethod
    def pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Precision-Recall AUC score."""
        return average_precision_score(y_true, y_pred_proba)
    
    @staticmethod
    def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews Correlation Coefficient."""
        return matthews_corrcoef(y_true, y_pred)
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        return balanced_accuracy_score(y_true, y_pred)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'accuracy': MetricFactory.accuracy(y_true, y_pred),
            'f1': MetricFactory.f1(y_true, y_pred),
            'precision': MetricFactory.precision(y_true, y_pred),
            'recall': MetricFactory.recall(y_true, y_pred),
            'mcc': MetricFactory.mcc(y_true, y_pred),
            'balanced_accuracy': MetricFactory.balanced_accuracy(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = MetricFactory.roc_auc(y_true, y_pred_proba)
            metrics['pr_auc'] = MetricFactory.pr_auc(y_true, y_pred_proba)
        
        return metrics
