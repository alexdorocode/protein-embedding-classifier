"""
CSV Loader Module

Handles loading of CSV files for the Protein Embedding Classifier.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import pandas as pd
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CSVLoader:
    """
    Loader for CSV files.
    
    Provides methods to load CSV files with various options.
    """
    
    def __init__(self):
        """Initialize the CSV loader."""
        pass
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv()
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file does not exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            # Set default kwargs
            default_kwargs = {
                'sep': ',',
                'header': 0,
                'index_col': None,
                'low_memory': False,
            }
            default_kwargs.update(kwargs)
            
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, **default_kwargs)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
