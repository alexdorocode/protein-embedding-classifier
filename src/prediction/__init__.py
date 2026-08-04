"""
Prediction Module

This module handles all model prediction operations for the Protein Embedding Classifier.

Responsibilities:
- Loading trained models
- Making predictions on new data
- Batch prediction support
- Prediction post-processing

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from .predictor import Predictor, BatchPredictor

# Public API
__all__ = [
    'Predictor',
    'BatchPredictor',
    'predict',
    'batch_predict',
]


def predict(model: any, data: any, **kwargs) -> any:
    """
    Make a prediction using a trained model.
    
    Args:
        model: Trained model
        data: Input data for prediction
        **kwargs: Additional prediction arguments
        
    Returns:
        Prediction result
    """
    predictor = Predictor(model)
    return predictor.predict(data, **kwargs)


def batch_predict(model: any, data_list: list, **kwargs) -> list:
    """
    Make batch predictions.
    
    Args:
        model: Trained model
        data_list: List of input data for prediction
        **kwargs: Additional prediction arguments
        
    Returns:
        List of prediction results
    """
    batch_predictor = BatchPredictor(model)
    return batch_predictor.predict_batch(data_list, **kwargs)
