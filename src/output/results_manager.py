"""
Results Manager Module

Manages results saving and organization.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import json
import pickle

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Manages saving and organizing of results.
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize the results manager.
        
        Args:
            output_dir: Base directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: any, name: str, 
                     format: str = 'auto', **kwargs) -> Path:
        """
        Save results.
        
        Args:
            results: Results to save
            name: Name for the results
            format: Format to save in ('auto', 'json', 'pkl', 'csv')
            **kwargs: Additional arguments
            
        Returns:
            Path to saved results
        """
        if format == 'auto':
            format = self._detect_format(results)
        
        output_path = self.output_dir / f"{name}.{format}"
        
        if format == 'json':
            self._save_json(results, output_path)
        elif format == 'pkl':
            self._save_pickle(results, output_path)
        elif format == 'csv':
            self._save_csv(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def _detect_format(self, results: any) -> str:
        """Detect appropriate format for results."""
        import pandas as pd
        
        if isinstance(results, dict):
            return 'json'
        elif isinstance(results, pd.DataFrame):
            return 'csv'
        else:
            return 'pkl'
    
    def _save_json(self, results: any, path: Path) -> None:
        """Save results as JSON."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_pickle(self, results: any, path: Path) -> None:
        """Save results as pickle."""
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    
    def _save_csv(self, results: any, path: Path) -> None:
        """Save results as CSV."""
        import pandas as pd
        
        if isinstance(results, pd.DataFrame):
            results.to_csv(path, index=False)
        else:
            raise ValueError("CSV format requires DataFrame input")
    
    def save_experiment(self, experiment_name: str, results: Dict[str, Any]) -> Path:
        """
        Save a complete experiment with all results.
        
        Args:
            experiment_name: Name for the experiment
            results: Dictionary containing all experiment results
            
        Returns:
            Path to saved experiment directory
        """
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        self.save_results(results, 'results', format='json')
        
        # Save individual components if present
        if 'model' in results:
            self.save_results(results['model'], 'model', format='pkl')
        if 'predictions' in results:
            self.save_results(results['predictions'], 'predictions', format='csv')
        if 'metrics' in results:
            self.save_results(results['metrics'], 'metrics', format='json')
        
        logger.info(f"Experiment saved to: {exp_dir}")
        return exp_dir
