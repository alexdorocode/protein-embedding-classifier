"""
Predictor Module

Handles single and batch predictions using trained models.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


class Predictor:
    """
    Handles predictions using a trained model.
    """
    
    def __init__(self, model: any):
        """
        Initialize the predictor.
        
        Args:
            model: Trained model to use for predictions
        """
        self.model = model
    
    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict, list], 
               **kwargs) -> any:
        """
        Make a prediction.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction arguments
            
        Returns:
            Prediction result
        """
        # Convert data to appropriate format
        X = self._prepare_data(data)
        
        # Make prediction based on model type
        if hasattr(self.model, 'predict'):
            return self.model.predict(X, **kwargs)
        elif hasattr(self.model, 'forward'):
            return self.model.forward(X, **kwargs)
        elif hasattr(self.model, '__call__'):
            return self.model(X, **kwargs)
        else:
            raise ValueError(f"Model {type(self.model)} does not have a prediction method")
    
    def predict_proba(self, data: Union[np.ndarray, pd.DataFrame, Dict, list], 
                      **kwargs) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction arguments
            
        Returns:
            Probability predictions
        """
        X = self._prepare_data(data)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X, **kwargs)
        elif hasattr(self.model, 'forward'):
            # Assume forward returns probabilities
            return self.model.forward(X, **kwargs)
        else:
            raise ValueError(f"Model {type(self.model)} does not support probability predictions")
    
    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, Dict, list]) -> np.ndarray:
        """
        Prepare data for prediction.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Data as numpy array
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, dict):
            # Convert dict to array (assuming single sample)
            return np.array([list(data.values())])
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


class BatchPredictor:
    """
    Handles batch predictions.
    """
    
    def __init__(self, model: any, batch_size: int = 32):
        """
        Initialize the batch predictor.
        
        Args:
            model: Trained model
            batch_size: Size of each batch
        """
        self.model = model
        self.batch_size = batch_size
        self.predictor = Predictor(model)
    
    def predict_batch(self, data_list: list, **kwargs) -> list:
        """
        Make predictions on a batch of data.
        
        Args:
            data_list: List of input data
            **kwargs: Additional prediction arguments
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batch_result = self.predictor.predict(batch, **kwargs)
            if isinstance(batch_result, np.ndarray):
                results.extend(batch_result.tolist())
            else:
                results.extend(batch_result)
        
        return results
    
    def predict_batch_proba(self, data_list: list, **kwargs) -> list:
        """
        Make probability predictions on a batch of data.
        
        Args:
            data_list: List of input data
            **kwargs: Additional prediction arguments
            
        Returns:
            List of probability prediction results
        """
        results = []
        
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batch_result = self.predictor.predict_proba(batch, **kwargs)
            if isinstance(batch_result, np.ndarray):
                results.extend(batch_result.tolist())
            else:
                results.extend(batch_result)
        
        return results
