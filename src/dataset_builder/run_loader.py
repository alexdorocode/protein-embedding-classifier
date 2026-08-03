"""
Run Loader Module

This module provides functions to load and validate dataset designer runs
in the Protein Embedding Classifier project.

IMPORTANT TERMINOLOGY NOTE:
- This project uses TP (Target Protein) and NTP (Non-Target Protein)
- Legacy filenames may contain 'mf_' or 'MF' but refer to TP/NTP concepts
- 'mf_id' in files actually means TP (Target Protein)
- 'candidates' in files actually means NTP (Non-Target Protein)

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RunData:
    """Container for loaded run data."""
    run_id: str
    metadata: Dict[str, Any]
    tp_ntp_pairs: pd.DataFrame
    assignments: Optional[pd.DataFrame] = None
    tp_metrics: Optional[pd.DataFrame] = None
    ntp_metrics: Optional[pd.DataFrame] = None
    
    def __repr__(self) -> str:
        return (f"RunData(run_id='{self.run_id}', "
                f"species='{self.metadata.get('species', 'unknown')}', "
                f"pairs={len(self.tp_ntp_pairs)})")


class RunLoader:
    """
    Loader for dataset designer runs.
    """
    
    def __init__(self, base_path: Union[str, Path] = 'datasets'):
        """
        Initialize the run loader.
        
        Args:
            base_path: Base directory containing run folders
        """
        self.base_path = Path(base_path)
        
    def list_runs(self) -> list:
        """List all available run IDs."""
        runs = []
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir() and (item / 'run_metadata.json').exists():
                    runs.append(item.name)
        return sorted(runs)
    
    def load_run(self, run_id: str) -> RunData:
        """
        Load a complete run dataset.
        
        Args:
            run_id: The run identifier
            
        Returns:
            RunData object with loaded data
        """
        run_path = self.base_path / run_id
        
        # Load metadata
        metadata_path = run_path / 'run_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load main dataset (final_dataset.csv)
        final_dataset_path = run_path / 'final_dataset.csv'
        tp_ntp_pairs = pd.read_csv(final_dataset_path)
        
        # Load assignments
        assignments = None
        for assign_file in ['target_assignments.csv', 'mf_assignments.csv']:
            assign_path = run_path / assign_file
            if assign_path.exists():
                df = pd.read_csv(assign_path)
                if 'mf_id' in df.columns:
                    df = df.rename(columns={'mf_id': 'target_id'})
                assignments = df
                break
        
        return RunData(
            run_id=run_id,
            metadata=metadata,
            tp_ntp_pairs=tp_ntp_pairs,
            assignments=assignments
        )


def load_run(run_id: str, base_path: Union[str, Path] = 'datasets') -> RunData:
    """
    Convenience function to load a run.
    
    Args:
        run_id: The run identifier
        base_path: Base directory for runs
        
    Returns:
        RunData object
    """
    loader = RunLoader(base_path)
    return loader.load_run(run_id)


def list_runs(base_path: Union[str, Path] = 'datasets') -> list:
    """
    Convenience function to list all runs.
    
    Args:
        base_path: Base directory for runs
        
    Returns:
        List of run IDs
    """
    loader = RunLoader(base_path)
    return loader.list_runs()
