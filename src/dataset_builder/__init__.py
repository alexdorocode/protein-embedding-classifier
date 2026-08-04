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
from .builders.dataset_builder import DatasetBuilder
from .label_loader import LabelLoader
from .run_loader import RunLoader, RunData
from .models import UniverseRecord, PoolConstraints, PoolMetadata, UniverseManifest
# from .contracts import DatasetContract  # Not available in current contracts.py

# Import from other modules
from .policies.validator import PolicyValidator
from .policies.models import DatasetPolicy
from .generator.generator import DatasetVariantGenerator
from .generator.models import DatasetVariant
from .splits.strategies import SplitStrategy, GroupByTargetSplitStrategy
from .lineage.builder import LineageBuilder
from .lineage.models import LineageManifest
from .export.exporter import BundleExporter
from .export.models import DatasetBundle

# Public API
__all__ = [
    # Builders
    'DatasetBuilder',
    'LabelLoader',
    'RunLoader',
    'RunData',
    
    # Models
    'UniverseRecord',
    'PoolConstraints',
    'PoolMetadata',
    'UniverseManifest',
    'DatasetContract',
    
    # Policies
    'DatasetPolicy',
    'PolicyValidator',
    
    # Generator
    'DatasetVariantGenerator',
    'DatasetVariant',
    
    # Splits
    'SplitStrategy',
    'GroupByTargetSplitStrategy',
    
    # Lineage
    'LineageBuilder',
    'LineageManifest',
    
    # Export
    'BundleExporter',
    'DatasetBundle',
    
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
    exporter = BundleExporter(config)
    exporter.export(dataset)


def load_run(run_id: str, base_path: str = 'dataset_designer_runs') -> RunData:
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


def list_runs(base_path: str = 'dataset_designer_runs') -> list:
    """
    List all available runs.
    
    Args:
        base_path: Base path for runs
        
    Returns:
        List of run IDs
    """
    loader = RunLoader(base_path)
    return loader.list_runs()
