"""
Database Loader Module

Handles loading of data from databases for the Protein Embedding Classifier.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import pandas as pd
from typing import Optional, Dict, Any
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class DatabaseLoader:
    """
    Loader for database connections.
    
    Provides methods to load data from SQL databases.
    """
    
    def __init__(self):
        """Initialize the database loader."""
        self.engine = None
    
    def connect(self, db_url: str) -> Any:
        """
        Connect to a database.
        
        Args:
            db_url: Database connection URL
            
        Returns:
            SQLAlchemy engine
        """
        try:
            self.engine = create_engine(db_url)
            logger.info(f"Connected to database: {db_url}")
            return self.engine
        except Exception as e:
            logger.error(f"Failed to connect to database {db_url}: {str(e)}")
            raise
    
    def load(self, query: str, db_url: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a database query.
        
        Args:
            query: SQL query to execute
            db_url: Database connection URL (optional if already connected)
            **kwargs: Additional arguments for pd.read_sql()
            
        Returns:
            pd.DataFrame: Query results
        """
        if db_url:
            self.connect(db_url)
        
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")
        
        try:
            logger.info(f"Executing query: {query[:100]}...")
            df = pd.read_sql(query, self.engine, **kwargs)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
