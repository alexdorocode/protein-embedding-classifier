"""
Input Module

This module handles all raw data loading operations for the Protein Embedding Classifier.

Responsibilities:
- Loading data from various sources (CSV, databases, APIs)
- Validating input data structure and content
- Providing a unified interface for data access

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from .csv_loader import CSVLoader
from .db_loader import DatabaseLoader
from .api_loader import APILoader
from .protein_loader import ProteinLoader
from .reader import Reader
from .normalizer import Normalizer
from .validators import DataValidator
from .models import InputConfig

# Public API
__all__ = [
    'CSVLoader',
    'DatabaseLoader', 
    'APILoader',
    'ProteinLoader',
    'Reader',
    'Normalizer',
    'DataValidator',
    'InputConfig',
    'load_data',
    'validate_data',
]


def load_data(source: str, source_type: str = 'csv', **kwargs) -> any:
    """
    Load data from a specified source.
    
    Args:
        source: Path or identifier for the data source
        source_type: Type of source ('csv', 'db', 'api', 'protein')
        **kwargs: Additional arguments for the specific loader
        
    Returns:
        Loaded data (DataFrame, dict, etc.)
        
    Raises:
        ValueError: If source_type is not supported
    """
    loaders = {
        'csv': CSVLoader,
        'db': DatabaseLoader,
        'api': APILoader,
        'protein': ProteinLoader,
    }
    
    if source_type not in loaders:
        raise ValueError(f"Unsupported source type: {source_type}. "
                        f"Supported types: {list(loaders.keys())}")
    
    loader_class = loaders[source_type]
    loader = loader_class()
    return loader.load(source, **kwargs)


def validate_data(data: any, schema: dict = None) -> bool:
    """
    Validate loaded data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema (optional)
        
    Returns:
        bool: True if data is valid
    """
    validator = DataValidator()
    
    if schema:
        return validator.validate_dataframe(data, schema)
    
    # Basic validation
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        return DataValidator.check_nulls(data)
    
    return True
