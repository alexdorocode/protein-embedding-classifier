"""
Explainability Module

This module handles all model explainability operations for the Protein Embedding Classifier.

Responsibilities:
- Feature importance analysis
- Embedding saliency maps
- SHAP value analysis
- Visualization of explanations

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from .feature_importance import FeatureImportance
from .embedding_saliency import EmbeddingSaliency
from .visualization.plotter import Plotter

# Public API
__all__ = [
    'FeatureImportance',
    'EmbeddingSaliency',
    'Plotter',
    'explain',
    'visualize',
]


def explain(model: any, data: any, method: str = 'feature_importance', **kwargs) -> any:
    """
    Generate explanations for model predictions.
    
    Args:
        model: Trained model
        data: Input data
        method: Explainability method to use
        **kwargs: Additional arguments for the method
        
    Returns:
        Explanation results
    """
    explainers = {
        'feature_importance': FeatureImportance,
        'embedding_saliency': EmbeddingSaliency,
    }
    
    if method not in explainers:
        raise ValueError(f"Unknown explainability method: {method}. "
                        f"Available: {list(explainers.keys())}")
    
    explainer_class = explainers[method]
    explainer = explainer_class(model)
    return explainer.explain(data, **kwargs)


def visualize(explanation: any, visualization_type: str = 'bar', **kwargs) -> any:
    """
    Visualize explanation results.
    
    Args:
        explanation: Explanation results to visualize
        visualization_type: Type of visualization
        **kwargs: Additional visualization arguments
        
    Returns:
        Visualization (plot, figure, etc.)
    """
    plotter = Plotter()
    return plotter.plot(explanation, visualization_type, **kwargs)
