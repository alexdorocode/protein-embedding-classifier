"""
Training Loop Module

Handles the training loop for models in the Protein Embedding Classifier.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles model training with validation and logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
        }
    
    def train(self, model: any, data: any, validation_data: Optional[any] = None) -> any:
        """
        Train a model.
        
        Args:
            model: Model to train
            data: Training data
            validation_data: Optional validation data
            
        Returns:
            Trained model
        """
        logger.info("Starting training...")
        
        # Extract features and labels from data
        if isinstance(data, pd.DataFrame):
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        else:
            X, y = data
        
        # Train the model
        if hasattr(model, 'fit'):
            model.fit(X, y)
        elif hasattr(model, 'train'):
            model.train(X, y)
        else:
            raise ValueError(f"Model {type(model)} does not have fit() or train() method")
        
        # Validate if validation data provided
        if validation_data is not None:
            self._validate(model, validation_data)
        
        logger.info("Training completed successfully")
        return model
    
    def _validate(self, model: any, data: any) -> Dict[str, float]:
        """
        Validate model on validation data.
        
        Args:
            model: Trained model
            data: Validation data
            
        Returns:
            Dictionary of validation metrics
        """
        if isinstance(data, pd.DataFrame):
            X_val = data.iloc[:, :-1].values
            y_val = data.iloc[:, -1].values
        else:
            X_val, y_val = data
        
        # Predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
        elif hasattr(model, 'forward'):
            y_pred = model.forward(X_val)
        else:
            raise ValueError(f"Model {type(model)} does not have predict() or forward() method")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred, average='weighted'),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
        }
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    
    def train_with_split(self, model: any, data: any, 
                        test_size: float = 0.2, random_state: int = 42) -> Tuple[any, Dict[str, float]]:
        """
        Train a model with automatic train/test split.
        
        Args:
            model: Model to train
            data: Input data
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (trained model, validation metrics)
        """
        if isinstance(data, pd.DataFrame):
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        else:
            X, y = data
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train
        self.train(model, (X_train, y_train))
        
        # Validate
        metrics = self._validate(model, (X_val, y_val))
        
        return model, metrics
