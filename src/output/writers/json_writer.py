"""
JSON Writer Module

Handles writing data to JSON format.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class JSONWriter:
    """
    Writes data to JSON format.
    """
    
    def write(self, data: any, output_path: str, **kwargs) -> None:
        """
        Write data to JSON file.
        
        Args:
            data: Data to write
            output_path: Path to output JSON file
            **kwargs: Additional arguments for json.dump()
        """
        output_path = Path(output_path)
        
        # Set default kwargs
        default_kwargs = {
            'indent': 2,
            'default': str,
            'ensure_ascii': False,
        }
        default_kwargs.update(kwargs)
        
        # Write to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **default_kwargs)
        
        logger.info(f"JSON written to: {output_path}")
