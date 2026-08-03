"""
Orchestration Module

This module handles experiment orchestration for the Protein Embedding Classifier.

Responsibilities:
- Running experiments with different configurations
- Managing hyperparameter sweeps
- Benchmarking models
- Coordinating complex workflows

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from .experiment_definitions import ExperimentDefinitions
from .runner import ExperimentRunner
from .sweep_manager import SweepManager
from .benchmark import BenchmarkManager

# Public API
__all__ = [
    'ExperimentDefinitions',
    'ExperimentRunner',
    'SweepManager',
    'BenchmarkManager',
    'run_experiment',
    'run_sweep',
    'run_benchmark',
]


def run_experiment(experiment_name: str, config: dict = None) -> dict:
    """
    Run a single experiment.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        
    Returns:
        Experiment results
    """
    runner = ExperimentRunner()
    return runner.run(experiment_name, config)


def run_sweep(sweep_config: dict) -> list:
    """
    Run a hyperparameter sweep.
    
    Args:
        sweep_config: Sweep configuration
        
    Returns:
        List of sweep results
    """
    manager = SweepManager()
    return manager.run_sweep(sweep_config)


def run_benchmark(benchmark_config: dict) -> dict:
    """
    Run a benchmark comparison.
    
    Args:
        benchmark_config: Benchmark configuration
        
    Returns:
        Benchmark results
    """
    manager = BenchmarkManager()
    return manager.run(benchmark_config)
