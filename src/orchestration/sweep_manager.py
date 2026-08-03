"""
Sweep Manager Module

Handles hyperparameter sweeps.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from typing import Dict, Any, List
import logging
import itertools

logger = logging.getLogger(__name__)


class SweepManager:
    """
    Manages hyperparameter sweeps.
    """
    
    def __init__(self):
        """Initialize the sweep manager."""
        pass
    
    def run_sweep(self, sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run a hyperparameter sweep.
        
        Args:
            sweep_config: Sweep configuration containing:
                - experiment: Base experiment name
                - parameters: Dictionary of parameter grids
                - runs: Number of runs per combination (optional)
                
        Returns:
            List of sweep results
        """
        from .runner import ExperimentRunner
        
        experiment_name = sweep_config.get('experiment')
        param_grid = sweep_config.get('parameters', {})
        num_runs = sweep_config.get('runs', 1)
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        for i, combo in enumerate(combinations):
            # Create config for this combination
            combo_config = {name: val for name, val in zip(param_names, combo)}
            
            # Add sweep metadata
            combo_config['sweep_id'] = i
            combo_config['total_combinations'] = len(combinations)
            
            for run in range(num_runs):
                # Add run number to config
                run_config = combo_config.copy()
                run_config['run_number'] = run
                
                logger.info(f"Running sweep combination {i+1}/{len(combinations)}, "
                          f"run {run+1}/{num_runs}")
                
                # Run experiment
                runner = ExperimentRunner()
                result = runner.run(experiment_name, run_config)
                results.append(result)
        
        return results
    
    def run_grid_search(self, sweep_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a grid search and return best configuration.
        
        Args:
            sweep_config: Sweep configuration
            
        Returns:
            Dictionary with best configuration and results
        """
        results = self.run_sweep(sweep_config)
        
        # Find best result based on validation metric
        best_result = None
        best_metric = float('-inf')
        metric_name = sweep_config.get('metric', 'accuracy')
        
        for result in results:
            if 'evaluation' in result and metric_name in result['evaluation']:
                metric_value = result['evaluation'][metric_name]
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_result = result
        
        return {
            'best_result': best_result,
            'best_metric': best_metric,
            'metric_name': metric_name,
            'all_results': results
        }
