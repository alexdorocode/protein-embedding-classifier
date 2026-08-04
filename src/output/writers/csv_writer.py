"""
CSV Writer Module

Handles writing data to CSV format.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CSVWriter:
    """
    Writes data to CSV format.
    """
    
    def write(self, data: any, output_path: str, **kwargs) -> None:
        """
        Write data to CSV file.
        
        Args:
            data: Data to write (DataFrame, dict, or list)
            output_path: Path to output CSV file
            **kwargs: Additional arguments for pd.to_csv()
        """
        output_path = Path(output_path)
        
        # Convert to DataFrame if needed
        df = self._to_dataframe(data)
        
        # Set default kwargs
        default_kwargs = {
            'index': False,
        }
        default_kwargs.update(kwargs)
        
        # Write to CSV
        df.to_csv(output_path, **default_kwargs)
        logger.info(f"CSV written to: {output_path}")
    
    def _to_dataframe(self, data: any) -> pd.DataFrame:
        """Convert data to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Cannot convert {type(data)} to DataFrame")
