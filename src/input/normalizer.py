"""
Universe Normalizer

Responsible for normalizing raw universe records into a consistent internal representation.
Ensures uniqueness of target_id and validates the universe structure.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import json

from pec.dataset.input.models import UniverseRecord, UniverseManifest


class UniverseNormalizer:
    """
    Normalizes a list of UniverseRecord instances into a validated universe.
    
    Normative Requirements (from contract §7.3):
    - MUST normalize every input row into exactly one UniverseRecord
    - MUST guarantee that target_id is unique within a normalized universe
    - MUST parse candidate_ids deterministically
    - MUST NOT sample, rank, or discard candidates at this stage
    - SHOULD preserve organism/taxonomy metadata
    - MAY retain raw source row payloads for debugging
    
    Attributes:
        logger: Logger instance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def normalize(
        self,
        records: List[UniverseRecord],
        universe_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> tuple[List[UniverseRecord], UniverseManifest]:
        """
        Normalize a list of UniverseRecord instances.
        
        Validates uniqueness of target_id and creates a universe manifest.
        
        Args:
            records: List of UniverseRecord instances to normalize
            universe_id: Optional unique identifier for this universe
            source_file: Optional source file path
            
        Returns:
            Tuple of (normalized_records, universe_manifest)
            
        Raises:
            ValueError: If target_id is not unique or other validation fails
        """
        if not records:
            raise ValueError("Cannot normalize empty universe")
        
        # Validate uniqueness of target_id
        target_ids = [r.target_id for r in records]
        seen = set()
        duplicates = []
        for tid in target_ids:
            if tid in seen:
                duplicates.append(tid)
            seen.add(tid)
        
        if duplicates:
            raise ValueError(
                f"Duplicate target_ids found in universe: {duplicates}"
            )
        
        # Calculate statistics
        target_count = len(records)
        total_candidates = sum(r.candidate_count for r in records)
        
        # Generate universe_id if not provided
        if universe_id is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            source_name = Path(source_file).stem if source_file else "universe"
            universe_id = f"{source_name}_{timestamp}"
        
        # Create manifest
        manifest = UniverseManifest(
            universe_id=universe_id,
            source_file=source_file or records[0].source_file,
            record_count=len(records),
            target_count=target_count,
            total_candidates=total_candidates,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
        
        self.logger.info(
            f"Normalized universe {universe_id}: "
            f"{target_count} targets, {total_candidates} total candidates"
        )
        
        return records, manifest
    
    def validate(
        self,
        records: List[UniverseRecord],
    ) -> List[str]:
        """
        Validate a list of UniverseRecord instances without normalizing.
        
        Args:
            records: List of UniverseRecord instances to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not records:
            errors.append("Universe is empty")
            return errors
        
        # Check for duplicate target_ids
        target_ids = [r.target_id for r in records]
        seen = set()
        for tid in target_ids:
            if tid in seen:
                errors.append(f"Duplicate target_id: {tid}")
            seen.add(tid)
        
        # Check for empty candidate lists
        for r in records:
            if not r.candidate_ids:
                errors.append(f"target_id {r.target_id} has no candidates")
        
        # Check for empty target_ids
        for r in records:
            if not r.target_id:
                errors.append("Found record with empty target_id")
        
        return errors
    
    def to_jsonl(
        self,
        records: List[UniverseRecord],
        output_path: Path,
    ) -> None:
        """
        Write normalized universe records to a JSONL file.
        
        Args:
            records: List of UniverseRecord instances
            output_path: Path to output JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_json() + "\n")
        
        self.logger.info(f"Wrote {len(records)} records to {output_path}")
    
    def to_manifest(
        self,
        manifest: UniverseManifest,
        output_path: Path,
    ) -> None:
        """
        Write universe manifest to JSON file.
        
        Args:
            manifest: UniverseManifest instance
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        self.logger.info(f"Wrote universe manifest to {output_path}")
