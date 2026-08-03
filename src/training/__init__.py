"""
Training Module

This module handles all model training operations for the Protein Embedding Classifier.

Responsibilities:
- Defining model architectures
- Training models on datasets
- Validating model performance
- Managing training configurations
- Integrating with Weights & Biases

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

# Import model classes
from .models.base import BaseModel
from .models.mlp_protein_classifier import MLPProteinClassifier
from .models.linear import LinearModel
from .models.random_forest import RandomForestModel
from .models.registry import ModelRegistry

# Import training utilities
from .embedding_handler import EmbeddingHandler
from .train_loop import Trainer
from .wandb_integration import WandbIntegration

# Import metrics and losses
from .losses import LossFactory
from .metrics import MetricFactory

# Import decision and statistics
from .decision.decision_policy import DecisionPolicy
from .statistics.friedman_test import FriedmanTest
from .statistics.nemenyi_test import NemenyiTest
from .statistics.ranking_utils import RankingUtils

# Public API
__all__ = [
    # Model classes
    'BaseModel',
    'MLPProteinClassifier',
    'LinearModel',
    'RandomForestModel',
    'ModelRegistry',
    
    # Training classes
    'Trainer',
    'EmbeddingHandler',
    'WandbIntegration',
    
    # Utilities
    'LossFactory',
    'MetricFactory',
    'DecisionPolicy',
    'FriedmanTest',
    'NemenyiTest',
    'RankingUtils',
    
    # Functions
    'train_model',
    'load_model',
    'register_model',
]


def train_model(model_name: str, data: any, config: dict = None) -> any:
    """
    Train a model.
    
    Args:
        model_name: Name of the model to train
        data: Training data
        config: Training configuration
        
    Returns:
        Trained model
    """
    registry = ModelRegistry()
    model_class = registry.get_model(model_name)
    model = model_class()
    
    trainer = Trainer()
    return trainer.train(model, data, config)


def load_model(model_path: str) -> any:
    """
    Load a trained model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    # Implementation depends on model type
    # This is a placeholder
    import pickle
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def register_model(name: str, model_class: type) -> None:
    """
    Register a model class.
    
    Args:
        name: Name to register the model under
        model_class: Model class to register
    """
    registry = ModelRegistry()
    registry.register_model(name, model_class)
