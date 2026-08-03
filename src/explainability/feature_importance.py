"""
Feature Importance Module

Handles feature importance analysis for model explainability.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureImportance:
    """
    Calculates feature importance for trained models.
    """
    
    def __init__(self, model: any):
        """
        Initialize the feature importance calculator.
        
        Args:
            model: Trained model
        """
        self.model = model
    
    def explain(self, data: Union[np.ndarray, pd.DataFrame], 
               method: str = 'auto', **kwargs) -> Dict[str, Any]:
        """
        Calculate feature importance.
        
        Args:
            data: Input data
            method: Method to use for feature importance
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with feature importance results
        """
        if method == 'auto':
            method = self._detect_method()
        
        if method == 'coef':
            return self._coef_importance(data, **kwargs)
        elif method == 'feature_importances_':
            return self._feature_importances_importance(data, **kwargs)
        elif method == 'permutation':
            return self._permutation_importance(data, **kwargs)
        else:
            raise ValueError(f"Unknown feature importance method: {method}")
    
    def _detect_method(self) -> str:
        """Detect the appropriate method for the model."""
        if hasattr(self.model, 'coef_'):
            return 'coef'
        elif hasattr(self.model, 'feature_importances_'):
            return 'feature_importances_'
        else:
            return 'permutation'
    
    def _coef_importance(self, data: Union[np.ndarray, pd.DataFrame], 
                        **kwargs) -> Dict[str, Any]:
        """Calculate importance from model coefficients."""
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            X = data.values
        else:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            X = data
        
        coef = self.model.coef_
        if len(coef.shape) > 1:
            # For multi-class, take mean across classes
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        
        importance = {name: float(val) for name, val in zip(feature_names, coef)}
        
        return {
            'method': 'coef',
            'importance': importance,
            'feature_names': feature_names
        }
    
    def _feature_importances_importance(self, data: Union[np.ndarray, pd.DataFrame], 
                                       **kwargs) -> Dict[str, Any]:
        """Calculate importance from feature_importances_ attribute."""
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        importance = self.model.feature_importances_
        
        return {
            'method': 'feature_importances_',
            'importance': {name: float(val) for name, val in zip(feature_names, importance)},
            'feature_names': feature_names
        }
    
    def _permutation_importance(self, data: Union[np.ndarray, pd.DataFrame], 
                               n_repeats: int = 10, **kwargs) -> Dict[str, Any]:
        """Calculate permutation feature importance."""
        from sklearn.inspection import permutation_importance
        
        if isinstance(data, pd.DataFrame):
            X = data.values
            feature_names = data.columns.tolist()
        else:
            X = data
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        # Need labels for permutation importance
        if 'y' in kwargs:
            y = kwargs['y']
        else:
            raise ValueError("Permutation importance requires labels (y)")
        
        result = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, random_state=42
        )
        
        return {
            'method': 'permutation',
            'importance': {name: float(val) for name, val in zip(feature_names, result.importances_mean)},
            'std': {name: float(val) for name, val in zip(feature_names, result.importances_std)},
            'feature_names': feature_names
        }
    
    def get_top_features(self, importance: Dict[str, Any], n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            importance: Feature importance dictionary
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if 'importance' in importance:
            sorted_features = sorted(
                importance['importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return sorted_features[:n]
        return []
