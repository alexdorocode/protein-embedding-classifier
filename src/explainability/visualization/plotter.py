"""
Plotter Module

Provides plotting utilities for explainability results.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class Plotter:
    """
    Handles plotting of explainability results.
    """
    
    def __init__(self):
        """Initialize the plotter."""
        plt.style.use('seaborn-v0_8')
    
    def plot(self, data: Dict[str, Any], plot_type: str = 'bar', **kwargs) -> plt.Figure:
        """
        Create a plot from explanation data.
        
        Args:
            data: Explanation data to plot
            plot_type: Type of plot to create
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        plot_methods = {
            'bar': self._plot_bar,
            'heatmap': self._plot_heatmap,
            'line': self._plot_line,
            'scatter': self._plot_scatter,
        }
        
        if plot_type not in plot_methods:
            raise ValueError(f"Unknown plot type: {plot_type}. "
                           f"Available: {list(plot_methods.keys())}")
        
        return plot_methods[plot_type](data, **kwargs)
    
    def _plot_bar(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create a bar plot."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if 'importance' in data:
            # Feature importance plot
            features = list(data['importance'].keys())
            values = list(data['importance'].values())
            
            # Sort by value
            sorted_idx = np.argsort(values)[::-1]
            features = [features[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            ax.bar(range(len(features)), values)
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def _plot_heatmap(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create a heatmap plot."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        if 'saliency_maps' in data:
            # Plot first saliency map as heatmap
            saliency = data['saliency_maps'][0]
            
            # Reshape if needed
            if len(saliency.shape) == 1:
                saliency = saliency.reshape(1, -1)
            
            im = ax.imshow(saliency, aspect='auto', cmap='RdBu')
            plt.colorbar(im, ax=ax)
            ax.set_title('Saliency Map')
        
        plt.tight_layout()
        return fig
    
    def _plot_line(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create a line plot."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if 'importance' in data:
            features = list(data['importance'].keys())
            values = list(data['importance'].values())
            
            ax.plot(features, values, marker='o')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _plot_scatter(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create a scatter plot."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if 'saliency_maps' in data and len(data['saliency_maps']) > 0:
            saliency = data['saliency_maps'][0]
            
            # Plot each dimension
            x = range(len(saliency))
            y = saliency
            
            ax.scatter(x, y, alpha=0.6)
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Saliency')
            ax.set_title('Embedding Saliency')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance: Dict[str, float], 
                               n: int = 10, **kwargs) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance: Dictionary of feature names and importance values
            n: Number of top features to show
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n]
        features = [x[0] for x in sorted_importance]
        values = [x[1] for x in sorted_importance]
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        ax.bar(range(len(features)), values)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title(f'Top {n} Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_saliency_map(self, saliency: np.ndarray, **kwargs) -> plt.Figure:
        """
        Plot a saliency map.
        
        Args:
            saliency: Saliency map array
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        if len(saliency.shape) == 1:
            saliency = saliency.reshape(1, -1)
        
        im = ax.imshow(saliency, aspect='auto', cmap='RdBu')
        plt.colorbar(im, ax=ax)
        ax.set_title('Saliency Map')
        
        plt.tight_layout()
        return fig
