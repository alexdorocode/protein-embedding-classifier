"""
Data Validators Module

This module provides data validation utilities for the Input Module.
It ensures that loaded data meets the required standards before processing.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import pandas as pd
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validator for input data.
    
    Provides methods to validate data structure, content, and quality.
    """
    
    @staticmethod
    def validate_schema(data: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that data contains required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if all required columns are present
        """
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True
    
    @staticmethod
    def check_nulls(data: pd.DataFrame, threshold: float = 0.0) -> bool:
        """
        Check for null values in data.
        
        Args:
            data: DataFrame to check
            threshold: Maximum allowed proportion of nulls (0.0 to 1.0)
            
        Returns:
            bool: True if nulls are within threshold
        """
        null_counts = data.isnull().sum()
        total_cells = data.size
        null_proportion = null_counts.sum() / total_cells if total_cells > 0 else 0
        
        if null_proportion > threshold:
            logger.warning(f"Null proportion {null_proportion:.2%} exceeds threshold {threshold:.2%}")
            return False
        return True
    
    @staticmethod
    def validate_types(data: pd.DataFrame, expected_types: Dict[str, type]) -> bool:
        """
        Validate that columns have expected data types.
        
        Args:
            data: DataFrame to validate
            expected_types: Dictionary of {column_name: expected_type}
            
        Returns:
            bool: True if all types match
        """
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = data[col].dtype
                # Handle pandas specific types
                if expected_type == str and actual_type != object:
                    logger.error(f"Column {col} expected str but got {actual_type}")
                    return False
                elif expected_type == int and not pd.api.types.is_integer_dtype(actual_type):
                    logger.error(f"Column {col} expected int but got {actual_type}")
                    return False
                elif expected_type == float and not pd.api.types.is_float_dtype(actual_type):
                    logger.error(f"Column {col} expected float but got {actual_type}")
                    return False
        return True
    
    @staticmethod
    def validate_range(data: pd.DataFrame, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
        """
        Validate that a column's values are within a specified range.
        
        Args:
            data: DataFrame containing the column
            column: Column name to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            
        Returns:
            bool: True if values are within range
        """
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            return False
        
        col_data = data[column]
        
        if min_val is not None:
            below_min = (col_data < min_val).any()
            if below_min:
                logger.error(f"Column {column} has values below minimum {min_val}")
                return False
        
        if max_val is not None:
            above_max = (col_data > max_val).any()
            if above_max:
                logger.error(f"Column {column} has values above maximum {max_val}")
                return False
        
        return True
    
    @staticmethod
    def validate_unique(data: pd.DataFrame, column: str) -> bool:
        """
        Validate that a column contains unique values.
        
        Args:
            data: DataFrame containing the column
            column: Column name to validate
            
        Returns:
            bool: True if all values are unique
        """
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            return False
        
        duplicates = data[column].duplicated().any()
        if duplicates:
            logger.error(f"Column {column} contains duplicate values")
            return False
        return True
    
    @staticmethod
    def validate_dataframe(data: pd.DataFrame, schema: Dict[str, Any]) -> bool:
        """
        Comprehensive validation of a DataFrame against a schema.
        
        Args:
            data: DataFrame to validate
            schema: Dictionary containing validation rules:
                - required_columns: List of required column names
                - types: Dictionary of {column: expected_type}
                - ranges: Dictionary of {column: (min, max)}
                - unique: List of columns that must be unique
                - null_threshold: Maximum allowed null proportion
            
        Returns:
            bool: True if data passes all validations
        """
        valid = True
        
        # Check required columns
        if 'required_columns' in schema:
            valid &= DataValidator.validate_schema(data, schema['required_columns'])
        
        # Check types
        if 'types' in schema:
            valid &= DataValidator.validate_types(data, schema['types'])
        
        # Check ranges
        if 'ranges' in schema:
            for col, (min_val, max_val) in schema['ranges'].items():
                valid &= DataValidator.validate_range(data, col, min_val, max_val)
        
        # Check unique
        if 'unique' in schema:
            for col in schema['unique']:
                valid &= DataValidator.validate_unique(data, col)
        
        # Check nulls
        null_threshold = schema.get('null_threshold', 0.0)
        valid &= DataValidator.check_nulls(data, null_threshold)
        
        return valid
