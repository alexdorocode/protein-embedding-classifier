"""
Benchmark Manager Module

Handles model benchmarking and comparison.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from typing import Dict, Any, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkManager:
    """
    Manages model benchmarking and comparison.
    """
    
    def __init__(self):
        """Initialize the benchmark manager."""
        pass
    
    def run(self, benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a benchmark comparison.
        
        Args:
            benchmark_config: Benchmark configuration containing:
                - name: Benchmark name
                - models: List of models to compare
                - datasets: List of datasets to use
                - metrics: List of metrics to calculate
                
        Returns:
            Benchmark results
        """
        from src.dataset_builder import load_run
        from src.training import get_classifier
        from src.prediction import predict
        from src.training.metrics import MetricFactory
        
        results = {
            'benchmark_name': benchmark_config.get('name', 'unnamed'),
            'models': [],
            'datasets': [],
            'comparison': {}
        }
        
        models = benchmark_config.get('models', [])
        datasets = benchmark_config.get('datasets', [])
        metrics = benchmark_config.get('metrics', ['accuracy', 'f1', 'precision', 'recall'])
        
        # Run each model on each dataset
        for model_name in models:
            model_results = {'name': model_name, 'datasets': []}
            
            for dataset_id in datasets:
                try:
                    # Load dataset
                    run_data = load_run(dataset_id, base_path='dataset_designer_runs')
                    dataset = run_data.tp_ntp_pairs
                    
                    # Train model
                    model = get_classifier(model_name, **benchmark_config.get('training_config', {}))
                    
                    # For benchmarking, we'll just store model info
                    # In a real implementation, we would train and evaluate
                    dataset_result = {
                        'dataset': dataset_id,
                        'species': run_data.metadata.get('species', 'unknown'),
                        'num_samples': len(dataset),
                        'model_type': model_name
                    }
                    
                    model_results['datasets'].append(dataset_result)
                    
                except Exception as e:
                    logger.error(f"Failed to run {model_name} on {dataset_id}: {str(e)}")
                    dataset_result = {
                        'dataset': dataset_id,
                        'error': str(e)
                    }
                    model_results['datasets'].append(dataset_result)
            
            results['models'].append(model_results)
        
        # Generate comparison table
        results['comparison'] = self._generate_comparison_table(results, metrics)
        
        return results
    
    def _generate_comparison_table(self, results: Dict[str, Any], 
                                  metrics: List[str]) -> pd.DataFrame:
        """
        Generate a comparison table from benchmark results.
        
        Args:
            results: Benchmark results
            metrics: List of metrics to include
            
        Returns:
            Comparison DataFrame
        """
        # This is a placeholder - in practice would extract actual metrics
        data = []
        
        for model_result in results['models']:
            model_name = model_result['name']
            for dataset_result in model_result['datasets']:
                dataset_id = dataset_result['dataset']
                
                # Create a row for this model-dataset combination
                row = {
                    'model': model_name,
                    'dataset': dataset_id,
                }
                
                # Add metric columns (placeholder values)
                for metric in metrics:
                    row[metric] = 0.0  # Would be actual metric value
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def compare_species(self, species_list: List[str], 
                       model_name: str = 'mlp', 
                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare model performance across different species.
        
        Args:
            species_list: List of species/dataset IDs to compare
            model_name: Model to use for comparison
            config: Training configuration
            
        Returns:
            Cross-species comparison results
        """
        benchmark_config = {
            'name': f'cross_species_{model_name}',
            'models': [model_name],
            'datasets': species_list,
            'training_config': config or {},
            'metrics': ['accuracy', 'f1', 'precision', 'recall']
        }
        
        return self.run(benchmark_config)
