"""
Dataset Builder Module

This module handles all dataset construction operations for the Protein Embedding Classifier.

Responsibilities:
- Building datasets from raw data and embeddings
- Transforming and filtering data
- Generating TP/NTP pairs
- Managing dataset metadata and lineage

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

# Import from submodules
from .contracts import DatasetContract
from .generators.generator import DatasetGenerator
from .generators.models import GeneratorConfig
from .export.exporter import DatasetExporter
from .export.models import ExportConfig
from .splits.strategies import SplitStrategy
from .splits.models import SplitConfig
from .lineage.builder import LineageBuilder
from .lineage.models import LineageConfig
from .policies.validator import PolicyValidator
from .policies.models import PolicyConfig
from .builders.dataset_builder import DatasetBuilder
from .label_loader import LabelLoader
from .run_loader import RunLoader, RunData
from .transformers import Normalizer, Filter
from .embedding_integration import EmbeddingIntegrator

# Public API
__all__ = [
    # Main classes
    'DatasetBuilder',
    'DatasetGenerator',
    'DatasetExporter',
    'RunLoader',
    'RunData',
    
    # Config classes
    'DatasetContract',
    'GeneratorConfig',
    'ExportConfig',
    'SplitConfig',
    'LineageConfig',
    'PolicyConfig',
    
    # Strategy classes
    'SplitStrategy',
    
    # Utility classes
    'LineageBuilder',
    'PolicyValidator',
    'LabelLoader',
    'Normalizer',
    'Filter',
    'EmbeddingIntegrator',
    
    # Functions
    'build_dataset',
    'export_dataset',
    'load_run',
    'list_runs',
]


def build_dataset(config: dict, data: any = None) -> any:
    """
    Build a dataset from configuration.
    
    Args:
        config: Dataset configuration
        data: Optional input data
        
    Returns:
        Built dataset
    """
    builder = DatasetBuilder(config)
    return builder.build(data)


def export_dataset(dataset: any, config: dict) -> None:
    """
    Export a dataset.
    
    Args:
        dataset: Dataset to export
        config: Export configuration
    """
    exporter = DatasetExporter(config)
    exporter.export(dataset)


def load_run(run_id: str, base_path: str = 'datasets') -> RunData:
    """
    Load a dataset designer run.
    
    Args:
        run_id: Run identifier
        base_path: Base path for runs
        
    Returns:
        RunData object
    """
    loader = RunLoader(base_path)
    return loader.load_run(run_id)


def list_runs(base_path: str = 'datasets') -> list:
    """
    List all available runs.
    
    Args:
        base_path: Base path for runs
        
    Returns:
        List of run IDs
    """
    loader = RunLoader(base_path)
    return loader.list_runs()
