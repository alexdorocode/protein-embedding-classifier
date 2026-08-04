"""
API Loader Module

Handles loading of data from bioinformatics APIs for the Protein Embedding Classifier.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import requests
import pandas as pd
from typing import Optional, Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)


class APILoader:
    """
    Loader for bioinformatics APIs.
    
    Provides methods to load data from various bioinformatics APIs.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the API loader.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def load(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Any:
        """
        Load data from an API endpoint.
        
        Args:
            endpoint: API endpoint (relative or absolute URL)
            params: Query parameters
            **kwargs: Additional arguments
            
        Returns:
            API response data (dict, list, or DataFrame)
        """
        url = endpoint if endpoint.startswith('http') else f"{self.base_url}/{endpoint}"
        
        try:
            logger.info(f"Loading from API: {url}")
            response = self.session.get(url, params=params, **kwargs)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Loaded data from {url}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            raise
    
    def load_to_dataframe(self, endpoint: str, params: Optional[Dict] = None, 
                         records_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from an API endpoint and convert to DataFrame.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            records_path: Path to records in JSON response (e.g., 'results.data')
            **kwargs: Additional arguments
            
        Returns:
            pd.DataFrame: Data from API
        """
        data = self.load(endpoint, params, **kwargs)
        
        if records_path:
            # Navigate to records
            keys = records_path.split('.')
            for key in keys:
                data = data[key]
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # If single record, wrap in list
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Cannot convert {type(data)} to DataFrame")
    
    def load_with_retry(self, endpoint: str, params: Optional[Dict] = None, 
                       max_retries: int = 3, delay: float = 1.0, **kwargs) -> Any:
        """
        Load data from API with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds
            **kwargs: Additional arguments
            
        Returns:
            API response data
        """
        for attempt in range(max_retries):
            try:
                return self.load(endpoint, params, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
