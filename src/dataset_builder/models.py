"""
Data Models for Universe Input

This module defines the canonical internal entities for the pre-embedding dataset layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json


@dataclass
class PoolConstraints:
    """
    Constraints applied to the candidate pool during generation.
    
    Attributes:
        len_variance: Variance in candidate sequence lengths (float or None)
        max_sequence_identity: Maximum sequence identity threshold (float or None)
        min_candidates: Minimum number of candidates required (int or None)
    """
    len_variance: Optional[float] = None
    max_sequence_identity: Optional[float] = None
    min_candidates: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoolConstraints":
        """Create from dictionary."""
        return cls(
            len_variance=data.get("len_variance"),
            max_sequence_identity=data.get("max_sequence_identity"),
            min_candidates=data.get("min_candidates"),
        )


@dataclass
class PoolMetadata:
    """
    Metadata about the candidate pool generation.
    
    Attributes:
        generation_source: Source of the candidate pool (e.g., "matches_primer_filtro")
        constraints_snapshot: Constraints applied during pool generation
    """
    generation_source: str
    constraints_snapshot: PoolConstraints = field(default_factory=PoolConstraints)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation_source": self.generation_source,
            "constraints_snapshot": self.constraints_snapshot.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoolMetadata":
        """Create from dictionary."""
        return cls(
            generation_source=data.get("generation_source", "unknown"),
            constraints_snapshot=PoolConstraints.from_dict(
                data.get("constraints_snapshot", {})
            ),
        )


@dataclass
class UniverseRecord:
    """
    Canonical internal entity representing one row from a target-candidate universe.
    
    This is the primary data structure for the pre-embedding dataset layer.
    Each row from the input file is normalized into exactly one UniverseRecord.
    
    Attributes:
        target_id: Unique identifier for the target protein
        target_label: Label for the target (typically "positive")
        candidate_ids: List of candidate protein IDs for this target
        candidate_count: Number of candidates (len(candidate_ids))
        source_file: Path to the source file
        source_row_id: Row identifier in the source file
        organism: Organism identifier (optional)
        taxonomy_id: Taxonomy identifier (optional)
        pool_metadata: Metadata about candidate pool generation
        raw_payload: Optional raw data from source for debugging
    
    Normative Requirements (from contract §7.3):
    - target_id MUST be unique within a normalized universe
    - candidate_ids MUST be parsed deterministically
    - NO sampling, ranking, or discarding at this stage
    - Organism/taxonomy metadata SHOULD be preserved
    """
    target_id: str
    target_label: str
    candidate_ids: List[str]
    candidate_count: int = field(init=False)
    source_file: str
    source_row_id: Any  # Can be string or integer
    organism: Optional[str] = None
    taxonomy_id: Optional[str] = None
    pool_metadata: PoolMetadata = field(default_factory=lambda: PoolMetadata(
        generation_source="matches_primer_filtro"
    ))
    raw_payload: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and set derived fields."""
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        if not isinstance(self.candidate_ids, list):
            raise ValueError("candidate_ids must be a list")
        self.candidate_count = len(self.candidate_ids)
        if self.candidate_count == 0:
            raise ValueError(f"target_id {self.target_id} has no candidates")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "target_id": self.target_id,
            "target_label": self.target_label,
            "candidate_ids": self.candidate_ids,
            "candidate_count": self.candidate_count,
            "source_file": self.source_file,
            "source_row_id": str(self.source_row_id),
            "organism": self.organism,
            "taxonomy_id": self.taxonomy_id,
            "pool_metadata": self.pool_metadata.to_dict(),
        }
        if self.raw_payload:
            result["raw_payload"] = self.raw_payload
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniverseRecord":
        """Create UniverseRecord from dictionary."""
        return cls(
            target_id=data["target_id"],
            target_label=data["target_label"],
            candidate_ids=data["candidate_ids"],
            source_file=data["source_file"],
            source_row_id=data["source_row_id"],
            organism=data.get("organism"),
            taxonomy_id=data.get("taxonomy_id"),
            pool_metadata=PoolMetadata.from_dict(data.get("pool_metadata", {})),
            raw_payload=data.get("raw_payload"),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "UniverseRecord":
        """Create UniverseRecord from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class UniverseManifest:
    """
    Manifest for a normalized target-candidate universe.
    
    This manifest accompanies the universe_manifest.json output file.
    
    Attributes:
        universe_id: Unique identifier for this universe
        source_file: Path to the source input file
        record_count: Number of UniverseRecord instances
        target_count: Number of unique targets
        total_candidates: Total number of candidate entries
        generated_at: Timestamp of universe creation
        schema_version: Version of the universe schema
    """
    universe_id: str
    source_file: str
    record_count: int
    target_count: int
    total_candidates: int
    generated_at: str
    schema_version: str = "0.1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniverseManifest":
        """Create from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
