#!/usr/bin/env python3
"""
Run Loader Module for Dataset Designer Runs

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
    assignments: pd.DataFrame
    tp_metrics: Optional[pd.DataFrame] = None
    ntp_metrics: Optional[pd.DataFrame] = None
    filter1_output: Optional[pd.DataFrame] = None
    filter2_output: Optional[pd.DataFrame] = None
    filter3_output: Optional[pd.DataFrame] = None
    pipeline_log: Optional[str] = None
    
    def __repr__(self) -> str:
        return (f"RunData(run_id='{self.run_id}', "
                f"species='{self.metadata.get('species', 'unknown')}', "
                f"pairs={len(self.tp_ntp_pairs)})"
                f"assignments={len(self.assignments)})")


@dataclass
class RunValidationResult:
    """Result of run validation."""
    run_id: str
    is_valid: bool
    missing_files: list
    warnings: list
    errors: list
    
    def __repr__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (f"RunValidationResult(run_id='{self.run_id}', "
                f"status='{status}', "
                f"missing_files={self.missing_files}, "
                f"warnings={len(self.warnings)}, "
                f"errors={len(self.errors)})")


# Required files for a valid run
REQUIRED_FILES = [
    'final_dataset.csv',
    'run_metadata.json',
    'pipeline.log',
    'target_assignments.csv',
    'mf_assignments.csv',
    'mf_metricas.csv',
    'candidatas_metricas.csv',
    'filter1_output.csv',
    'filter2_output.csv',
    'filter3_output.csv'
]

# Primary training files (in order of preference)
PRIMARY_TRAINING_FILES = [
    'final_dataset.csv',
    'target_non_target_dataset.csv',
    'target_non_target_pairs.csv'
]

# Assignment files (in order of preference)
ASSIGNMENT_FILES = [
    'target_assignments.csv',  # Correct terminology
    'mf_assignments.csv'       # Legacy name
]


class RunLoader:
    """
    Loader for dataset designer runs.
    
    This class provides methods to load, validate, and access run data
    with proper terminology handling.
    """
    
    def __init__(self, base_path: Union[str, Path] = 'dataset_designer_runs'):
        """
        Initialize the run loader.
        
        Args:
            base_path: Base directory containing run folders
        """
        self.base_path = Path(base_path)
        self._catalog = None
        
    def _load_catalog(self) -> Optional[Dict[str, Any]]:
        """Load the runs catalog if available."""
        catalog_path = self.base_path / 'runs_catalog.json'
        if catalog_path.exists():
            with open(catalog_path, 'r') as f:
                self._catalog = json.load(f)
        return self._catalog
    
    def list_runs(self) -> list:
        """List all available run IDs."""
        runs = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / 'run_metadata.json').exists():
                runs.append(item.name)
        return sorted(runs)
    
    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific run from the catalog."""
        catalog = self._load_catalog()
        if catalog and 'runs' in catalog:
            return catalog['runs'].get(run_id)
        return None
    
    def validate_run(self, run_id: str) -> RunValidationResult:
        """
        Validate that a run has all required files and consistent data.
        
        Args:
            run_id: The run identifier
            
        Returns:
            RunValidationResult with validation details
        """
        run_path = self.base_path / run_id
        missing_files = []
        warnings = []
        errors = []
        
        # Check required files
        for required_file in REQUIRED_FILES:
            if not (run_path / required_file).exists():
                missing_files.append(required_file)
        
        # Check metadata file
        metadata_path = run_path / 'run_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check for required metadata fields
                required_metadata = ['run_id', 'species', 'species_category']
                for field in required_metadata:
                    if field not in metadata:
                        warnings.append(f"Metadata missing field: {field}")
                        
                # Check terminology field
                if 'terminology' not in metadata:
                    warnings.append("Metadata missing terminology field")
                    
            except Exception as e:
                errors.append(f"Failed to load metadata: {str(e)}")
        else:
            errors.append("run_metadata.json not found")
        
        # Check final dataset structure
        final_dataset_path = run_path / 'final_dataset.csv'
        if final_dataset_path.exists():
            try:
                df = pd.read_csv(final_dataset_path)
                required_columns = ['tp_id', 'ntp_id']
                for col in required_columns:
                    if col not in df.columns:
                        errors.append(f"final_dataset.csv missing column: {col}")
            except Exception as e:
                errors.append(f"Failed to read final_dataset.csv: {str(e)}")
        
        # Check assignments structure
        for assign_file in ASSIGNMENT_FILES:
            assign_path = run_path / assign_file
            if assign_path.exists():
                try:
                    df = pd.read_csv(assign_path)
                    # Check for expected columns (accounting for legacy naming)
                    if 'target_id' in df.columns:
                        # Correct terminology
                        pass
                    elif 'mf_id' in df.columns:
                        # Legacy naming
                        warnings.append(f"{assign_file} uses legacy 'mf_id' column (should be 'target_id')")
                    else:
                        errors.append(f"{assign_file} missing expected columns")
                        
                    if 'candidates' not in df.columns:
                        errors.append(f"{assign_file} missing 'candidates' column")
                        
                except Exception as e:
                    errors.append(f"Failed to read {assign_file}: {str(e)}")
                break  # Only check the first found assignment file
        
        is_valid = (len(missing_files) == 0 and 
                   len(errors) == 0)
        
        return RunValidationResult(
            run_id=run_id,
            is_valid=is_valid,
            missing_files=missing_files,
            warnings=warnings,
            errors=errors
        )
    
    def load_run(self, run_id: str, load_all: bool = False) -> RunData:
        """
        Load a complete run dataset.
        
        Args:
            run_id: The run identifier
            load_all: If True, load all available files (slower but complete)
            
        Returns:
            RunData object with loaded data
            
        Raises:
            FileNotFoundError: If run directory or required files not found
            ValueError: If data structure is invalid
        """
        run_path = self.base_path / run_id
        
        # Validate run exists
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        # Load metadata
        metadata_path = run_path / 'run_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load primary training file (final_dataset.csv preferred)
        tp_ntp_pairs = None
        for training_file in PRIMARY_TRAINING_FILES:
            file_path = run_path / training_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Standardize column names to tp_id, ntp_id
                if 'tp_id' in df.columns and 'ntp_id' in df.columns:
                    tp_ntp_pairs = df
                elif 'target_id' in df.columns and 'non_target_id' in df.columns:
                    tp_ntp_pairs = df.rename(columns={
                        'target_id': 'tp_id',
                        'non_target_id': 'ntp_id'
                    })
                else:
                    # Try to handle other column name variations
                    logger.warning(f"Unexpected column names in {training_file}: {list(df.columns)}")
                    tp_ntp_pairs = df
                break
        
        if tp_ntp_pairs is None:
            raise FileNotFoundError(f"No primary training file found in run {run_id}")
        
        # Load assignments (prefer correct terminology file)
        assignments = None
        for assign_file in ASSIGNMENT_FILES:
            file_path = run_path / assign_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Standardize column names
                if 'mf_id' in df.columns:
                    df = df.rename(columns={'mf_id': 'target_id'})
                assignments = df
                break
        
        if assignments is None:
            raise FileNotFoundError(f"No assignment file found in run {run_id}")
        
        # Load other files if requested
        tp_metrics = None
        ntp_metrics = None
        filter1_output = None
        filter2_output = None
        filter3_output = None
        pipeline_log = None
        
        if load_all:
            # Load TP metrics (mf_metricas.csv - legacy name)
            tp_metrics_path = run_path / 'mf_metricas.csv'
            if tp_metrics_path.exists():
                tp_metrics = pd.read_csv(tp_metrics_path)
            
            # Load NTP metrics
            ntp_metrics_path = run_path / 'candidatas_metricas.csv'
            if ntp_metrics_path.exists():
                ntp_metrics = pd.read_csv(ntp_metrics_path)
            
            # Load filter outputs
            for i in range(1, 4):
                filter_path = run_path / f'filter{i}_output.csv'
                if filter_path.exists():
                    if i == 1:
                        filter1_output = pd.read_csv(filter_path)
                    elif i == 2:
                        filter2_output = pd.read_csv(filter_path)
                    elif i == 3:
                        filter3_output = pd.read_csv(filter_path)
            
            # Load pipeline log
            log_path = run_path / 'pipeline.log'
            if log_path.exists():
                with open(log_path, 'r') as f:
                    pipeline_log = f.read()
        
        return RunData(
            run_id=run_id,
            metadata=metadata,
            tp_ntp_pairs=tp_ntp_pairs,
            assignments=assignments,
            tp_metrics=tp_metrics,
            ntp_metrics=ntp_metrics,
            filter1_output=filter1_output,
            filter2_output=filter2_output,
            filter3_output=filter3_output,
            pipeline_log=pipeline_log
        )
    
    def load_tp_ntp_pairs(self, run_id: str) -> pd.DataFrame:
        """
        Load only the TP/NTP pairs from a run (lightweight loading).
        
        Args:
            run_id: The run identifier
            
        Returns:
            DataFrame with tp_id and ntp_id columns
        """
        run_data = self.load_run(run_id, load_all=False)
        return run_data.tp_ntp_pairs
    
    def get_species(self, run_id: str) -> str:
        """Get the species category for a run."""
        run_info = self.get_run_info(run_id)
        if run_info:
            return run_info.get('species_category', 'unknown')
        
        # Fallback to loading metadata
        try:
            run_data = self.load_run(run_id, load_all=False)
            return run_data.metadata.get('species_category', 'unknown')
        except Exception:
            return 'unknown'
    
    def get_humans_run(self) -> Optional[str]:
        """Get the run ID for humans dataset."""
        catalog = self._load_catalog()
        if catalog and 'runs' in catalog:
            for run_id, info in catalog['runs'].items():
                if info.get('species_category') == 'humans':
                    return run_id
        
        # Fallback: check for known humans run
        humans_run = '20260803_0258_7672b947'
        if (self.base_path / humans_run / 'run_metadata.json').exists():
            return humans_run
        return None
    
    def get_model_organisms_run(self) -> Optional[str]:
        """Get the run ID for model organisms dataset."""
        catalog = self._load_catalog()
        if catalog and 'runs' in catalog:
            for run_id, info in catalog['runs'].items():
                if info.get('species_category') == 'model_organisms':
                    return run_id
        
        # Fallback: check for known model organisms run
        model_run = '20260803_0304_a68aa0bb'
        if (self.base_path / model_run / 'run_metadata.json').exists():
            return model_run
        return None


# Convenience functions (module-level)

def load_run(run_id: str, base_path: Union[str, Path] = 'dataset_designer_runs', **kwargs) -> RunData:
    """
    Convenience function to load a run.
    
    Args:
        run_id: The run identifier
        base_path: Base directory for runs
        **kwargs: Additional arguments passed to RunLoader.load_run()
        
    Returns:
        RunData object
    """
    loader = RunLoader(base_path)
    return loader.load_run(run_id, **kwargs)


def validate_run(run_id: str, base_path: Union[str, Path] = 'dataset_designer_runs') -> RunValidationResult:
    """
    Convenience function to validate a run.
    
    Args:
        run_id: The run identifier
        base_path: Base directory for runs
        
    Returns:
        RunValidationResult
    """
    loader = RunLoader(base_path)
    return loader.validate_run(run_id)


def list_runs(base_path: Union[str, Path] = 'dataset_designer_runs') -> list:
    """
    Convenience function to list all runs.
    
    Args:
        base_path: Base directory for runs
        
    Returns:
        List of run IDs
    """
    loader = RunLoader(base_path)
    return loader.list_runs()


def get_species(run_id: str, base_path: Union[str, Path] = 'dataset_designer_runs') -> str:
    """
    Convenience function to get species for a run.
    
    Args:
        run_id: The run identifier
        base_path: Base directory for runs
        
    Returns:
        Species category
    """
    loader = RunLoader(base_path)
    return loader.get_species(run_id)


if __name__ == '__main__':
    # Example usage
    print("Available runs:", list_runs())
    
    # Load humans run
    humans_run = '20260803_0258_7672b947'
    print(f"\nLoading run {humans_run}...")
    run_data = load_run(humans_run)
    print(run_data)
    
    # Validate run
    print(f"\nValidating run {humans_run}...")
    validation = validate_run(humans_run)
    print(validation)
    
    # Get species
    print(f"\nSpecies for {humans_run}: {get_species(humans_run)}")
