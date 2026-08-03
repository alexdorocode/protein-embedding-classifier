"""
Experiment Definitions Module

Defines available experiments and their configurations.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    dataset: str
    model: str
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    explainability: Dict[str, Any] = field(default_factory=dict)


class ExperimentDefinitions:
    """
    Defines all available experiments.
    """
    
    # Experiment registry
    experiments: Dict[str, ExperimentConfig] = {}
    
    def __init__(self):
        """Initialize with default experiments."""
        self._register_default_experiments()
    
    def _register_default_experiments(self):
        """Register default experiments."""
        # Humans dataset experiment
        self.experiments['humans_mlp'] = ExperimentConfig(
            name='humans_mlp',
            description='Train MLP on humans dataset',
            dataset='20260803_0258_7672b947',
            model='mlp',
            training={
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
            },
            evaluation={
                'test_size': 0.2,
                'random_state': 42,
            }
        )
        
        # Model organisms dataset experiment
        self.experiments['model_organisms_mlp'] = ExperimentConfig(
            name='model_organisms_mlp',
            description='Train MLP on model organisms dataset',
            dataset='20260803_0304_a68aa0bb',
            model='mlp',
            training={
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
            },
            evaluation={
                'test_size': 0.2,
                'random_state': 42,
            }
        )
        
        # Cross-species experiment
        self.experiments['cross_species'] = ExperimentConfig(
            name='cross_species',
            description='Train on model organisms, test on humans',
            dataset='20260803_0304_a68aa0bb',
            model='mlp',
            training={
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
            },
            evaluation={
                'test_dataset': '20260803_0258_7672b947',
                'random_state': 42,
            }
        )
        
        # Random Forest experiment
        self.experiments['humans_rf'] = ExperimentConfig(
            name='humans_rf',
            description='Train Random Forest on humans dataset',
            dataset='20260803_0258_7672b947',
            model='random_forest',
            training={
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42,
            },
            evaluation={
                'test_size': 0.2,
                'random_state': 42,
            }
        )
    
    def get_experiment(self, experiment_name: str) -> ExperimentConfig:
        """
        Get an experiment configuration by name.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            ExperimentConfig object
            
        Raises:
            ValueError: If experiment not found
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_name}. "
                           f"Available: {list(self.experiments.keys())}")
        return self.experiments[experiment_name]
    
    def list_experiments(self) -> List[str]:
        """
        List all available experiments.
        
        Returns:
            List of experiment names
        """
        return list(self.experiments.keys())
    
    def add_experiment(self, config: ExperimentConfig) -> None:
        """
        Add a new experiment configuration.
        
        Args:
            config: Experiment configuration to add
        """
        self.experiments[config.name] = config
    
    def remove_experiment(self, experiment_name: str) -> None:
        """
        Remove an experiment configuration.
        
        Args:
            experiment_name: Name of experiment to remove
        """
        if experiment_name in self.experiments:
            del self.experiments[experiment_name]
