"""
Loss Functions Module

Provides loss functions for model training.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import numpy as np
from typing import Callable, Dict, Any
from sklearn.metrics import log_loss


class LossFactory:
    """
    Factory for creating loss functions.
    """
    
    @staticmethod
    def get_loss(loss_name: str) -> Callable:
        """
        Get a loss function by name.
        
        Args:
            loss_name: Name of the loss function
            
        Returns:
            Loss function
            
        Raises:
            ValueError: If loss function not found
        """
        loss_functions = {
            'binary_crossentropy': LossFactory.binary_crossentropy,
            'categorical_crossentropy': LossFactory.categorical_crossentropy,
            'mse': LossFactory.mse,
            'mae': LossFactory.mae,
        }
        
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}. "
                           f"Available: {list(loss_functions.keys())}")
        
        return loss_functions[loss_name]
    
    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
            
        Returns:
            Binary cross-entropy loss
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Categorical cross-entropy loss.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Categorical cross-entropy loss
        """
        # Clip predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean squared error loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean absolute error loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Mean absolute error
        """
        return np.mean(np.abs(y_true - y_pred))
