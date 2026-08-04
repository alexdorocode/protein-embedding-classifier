"""
Input Models

Data models for the Input Module.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class SourceType(Enum):
    """Enumeration of supported data source types."""
    CSV = "csv"
    DATABASE = "db"
    API = "api"
    PROTEIN = "protein"
    JSON = "json"
    EXCEL = "excel"


class DataFormat(Enum):
    """Enumeration of supported data formats."""
    DATAFRAME = "dataframe"
    DICT = "dict"
    LIST = "list"
    JSON = "json"


@dataclass
class InputConfig:
    """
    Configuration for data loading.
    
    Attributes:
        source: Path or identifier for the data source
        source_type: Type of data source
        format: Expected data format
        options: Additional loading options
        validation: Validation schema
    """
    source: str
    source_type: SourceType = SourceType.CSV
    format: DataFormat = DataFormat.DATAFRAME
    options: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[Dict[str, Any]] = None


@dataclass
class CSVConfig(InputConfig):
    """Configuration for CSV loading."""
    sep: str = ","
    header: int = 0
    index_col: Optional[str] = None
    usecols: Optional[List[str]] = None
    dtype: Optional[Dict[str, str]] = None


@dataclass
class DatabaseConfig(InputConfig):
    """Configuration for database loading."""
    db_url: str = ""
    query: str = ""
    params: Optional[Dict[str, Any]] = None


@dataclass
class APIConfig(InputConfig):
    """Configuration for API loading."""
    base_url: str = ""
    endpoint: str = ""
    params: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None
    records_path: Optional[str] = None


@dataclass
class ValidationSchema:
    """
    Schema for data validation.
    
    Attributes:
        required_columns: List of columns that must be present
        types: Dictionary mapping column names to expected types
        ranges: Dictionary mapping column names to (min, max) tuples
        unique: List of columns that must contain unique values
        null_threshold: Maximum allowed proportion of null values
    """
    required_columns: List[str] = field(default_factory=list)
    types: Dict[str, type] = field(default_factory=dict)
    ranges: Dict[str, tuple] = field(default_factory=dict)
    unique: List[str] = field(default_factory=list)
    null_threshold: float = 0.0


@dataclass
class LoadedData:
    """
    Container for loaded data and metadata.
    
    Attributes:
        data: The loaded data
        source: Source of the data
        source_type: Type of source
        config: Configuration used for loading
        metadata: Additional metadata
    """
    data: Any
    source: str
    source_type: SourceType
    config: InputConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
