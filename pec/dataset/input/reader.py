"""
Universe Reader

Responsible for reading target-candidate universe input files.
Supports CSV format (matches_primer_filtro.csv-style).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, TextIO
import logging

from pec.dataset.input.models import UniverseRecord, PoolConstraints, PoolMetadata


class UniverseReader:
    """
    Reads and parses target-candidate universe input files.
    
    The reader supports CSV files in matches_primer_filtro.csv-style format.
    Each row should contain at minimum:
    - target_id: The target protein identifier
    - candidate_ids: Semicolon-separated or comma-separated list of candidate IDs
    
    Optional columns:
    - target_label: Label for the target (default: "positive")
    - organism: Organism identifier
    - taxonomy_id: Taxonomy identifier
    - Any other metadata columns (stored in raw_payload)
    
    Attributes:
        logger: Logger instance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def read_csv(
        self,
        file_path: Path,
        candidate_separator: str = ";",
        target_label_default: str = "positive",
        pool_metadata: Optional[PoolMetadata] = None,
    ) -> List[UniverseRecord]:
        """
        Read a CSV file and return a list of UniverseRecord instances.
        
        Args:
            file_path: Path to the CSV file
            candidate_separator: Separator used in candidate_ids column (default: ";")
            target_label_default: Default label for targets (default: "positive")
            pool_metadata: Optional pool metadata to attach to all records
            
        Returns:
            List of UniverseRecord instances
            
        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If file is malformed or required columns are missing
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        records = []
        pool_metadata = pool_metadata or PoolMetadata(
            generation_source="matches_primer_filtro"
        )
        
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            required_cols = {"target_id", "candidate_ids"}
            available_cols = set(reader.fieldnames or [])
            missing = required_cols - available_cols
            
            if missing:
                raise ValueError(
                    f"Missing required columns in {file_path}: {missing}"
                )
            
            for row_num, row in enumerate(reader, start=1):
                try:
                    record = self._parse_row(
                        row=row,
                        row_num=row_num,
                        file_path=str(file_path),
                        candidate_separator=candidate_separator,
                        target_label_default=target_label_default,
                        pool_metadata=pool_metadata,
                    )
                    records.append(record)
                except ValueError as e:
                    self.logger.warning(
                        f"Skipping row {row_num} in {file_path}: {e}"
                    )
                    continue
        
        self.logger.info(
            f"Read {len(records)} universe records from {file_path}"
        )
        return records
    
    def _parse_row(
        self,
        row: Dict[str, str],
        row_num: int,
        file_path: str,
        candidate_separator: str,
        target_label_default: str,
        pool_metadata: PoolMetadata,
    ) -> UniverseRecord:
        """
        Parse a single CSV row into a UniverseRecord.
        
        Args:
            row: Dictionary of column names to values
            row_num: Row number (for source_row_id)
            file_path: Source file path
            candidate_separator: Separator for candidate_ids
            target_label_default: Default target label
            pool_metadata: Pool metadata to attach
            
        Returns:
            UniverseRecord instance
            
        Raises:
            ValueError: If row is malformed
        """
        target_id = row.get("target_id", "").strip()
        if not target_id:
            raise ValueError("target_id is empty or missing")
        
        # Parse candidate IDs
        candidate_ids_str = row.get("candidate_ids", "").strip()
        if not candidate_ids_str:
            raise ValueError(f"candidate_ids is empty for target {target_id}")
        
        candidate_ids = [
            cid.strip() for cid in candidate_ids_str.split(candidate_separator)
            if cid.strip()
        ]
        
        if not candidate_ids:
            raise ValueError(f"No valid candidate IDs for target {target_id}")
        
        # Extract optional fields
        target_label = row.get("target_label", target_label_default).strip()
        organism = row.get("organism")
        if organism:
            organism = organism.strip()
        taxonomy_id = row.get("taxonomy_id")
        if taxonomy_id:
            taxonomy_id = taxonomy_id.strip()
        
        # Store raw payload (all other columns)
        raw_payload = {
            k: v for k, v in row.items()
            if k not in {"target_id", "candidate_ids", "target_label", "organism", "taxonomy_id"}
        }
        if raw_payload:
            raw_payload = {k: v.strip() if isinstance(v, str) else v 
                          for k, v in raw_payload.items()}
        
        return UniverseRecord(
            target_id=target_id,
            target_label=target_label or target_label_default,
            candidate_ids=candidate_ids,
            source_file=file_path,
            source_row_id=row_num,
            organism=organism if organism else None,
            taxonomy_id=taxonomy_id if taxonomy_id else None,
            pool_metadata=pool_metadata,
            raw_payload=raw_payload if raw_payload else None,
        )
    
    def read_jsonl(
        self,
        file_path: Path,
        pool_metadata: Optional[PoolMetadata] = None,
    ) -> List[UniverseRecord]:
        """
        Read a JSONL file and return a list of UniverseRecord instances.
        
        Each line should be a JSON object matching the UniverseRecord schema.
        
        Args:
            file_path: Path to the JSONL file
            pool_metadata: Optional pool metadata to attach
            
        Returns:
            List of UniverseRecord instances
        """
        import json
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        records = []
        pool_metadata = pool_metadata or PoolMetadata(
            generation_source="jsonl"
        )
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Ensure pool_metadata is included
                    if "pool_metadata" not in data:
                        data["pool_metadata"] = pool_metadata.to_dict()
                    record = UniverseRecord.from_dict(data)
                    # Override pool_metadata if provided
                    if pool_metadata:
                        record.pool_metadata = pool_metadata
                    records.append(record)
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Skipping line {line_num} in {file_path}: {e}"
                    )
                    continue
        
        self.logger.info(
            f"Read {len(records)} universe records from {file_path}"
        )
        return records
