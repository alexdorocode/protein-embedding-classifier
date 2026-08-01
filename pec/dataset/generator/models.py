"""
Generator Models

Defines the canonical entities for dataset variant generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
from datetime import datetime


@dataclass
class AssignmentRecord:
    """
    One row in the canonical assignments table.
    
    Canonical assignments table columns (from contract §9.3):
    - target_id: MF target that anchors the local assignment
    - protein_id: Protein accession of the realized instance
    - role: "positive" or "negative"
    - paired_target_id: Target to which the instance belongs
    - variant_id: Dataset variant identity
    
    Attributes:
        target_id: The target that anchors this assignment
        protein_id: The protein accession (target or candidate)
        role: Role of this instance (positive or negative)
        paired_target_id: The target this instance belongs to
        variant_id: The variant this assignment belongs to
    """
    target_id: str
    protein_id: str
    role: str  # "positive" or "negative"
    paired_target_id: str
    variant_id: str
    
    def __post_init__(self):
        """Validate the assignment record."""
        if self.role not in ("positive", "negative"):
            raise ValueError(f"Invalid role: {self.role}. Must be 'positive' or 'negative'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssignmentRecord":
        """Create from dictionary."""
        return cls(**data)
    
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row."""
        return [
            self.target_id,
            self.protein_id,
            self.role,
            self.paired_target_id,
            self.variant_id,
        ]
    
    @staticmethod
    def csv_headers() -> List[str]:
        """Get CSV headers."""
        return ["target_id", "protein_id", "role", "paired_target_id", "variant_id"]


@dataclass
class ScarcityEvent:
    """
    Records a scarcity event during variant generation.
    
    Attributes:
        target_id: The target that was affected
        reason: Reason for the scarcity event
        details: Additional details about the event
    """
    target_id: str
    reason: str
    details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"target_id": self.target_id, "reason": self.reason}
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class DatasetStatistics:
    """
    Statistics for a dataset variant.
    
    Attributes:
        ratio_realized: The actual ratio achieved
        organism_distribution: Distribution of organisms
        candidate_pool_size_distribution: Distribution of candidate pool sizes
    """
    ratio_realized: str
    organism_distribution: Dict[str, int] = field(default_factory=dict)
    candidate_pool_size_distribution: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class VariantManifest:
    """
    Canonical variant manifest (from contract §9.2).
    
    Attributes:
        variant_id: Unique identifier for this variant
        policy_id: Reference to the policy used
        source_universe_id: Reference to the source universe
        seed_used: Random seed used for this variant
        targets_included: Number of targets included
        targets_dropped: Number of targets dropped due to scarcity
        total_positive_instances: Total positive instances
        total_negative_instances: Total negative instances
        assignment_mode: Assignment mode used
        scarcity_events: List of scarcity events
        dataset_statistics: Statistics for this variant
    """
    variant_id: str
    policy_id: str
    source_universe_id: str
    seed_used: int
    targets_included: int
    targets_dropped: int
    total_positive_instances: int
    total_negative_instances: int
    assignment_mode: str
    scarcity_events: List[Dict[str, Any]] = field(default_factory=list)
    dataset_statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariantManifest":
        """Create from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DatasetVariant:
    """
    Concrete dataset variant generated from universe + policy + seed.
    
    Normative Requirements (from contract §9.3):
    - MUST be reconstructible from source_universe + policy + seed
    - MUST produce both realized instances and a machine-readable variant manifest
    - MUST record dropped targets and scarcity events
    - MUST NOT silently relax ratio rules in v0.1
    - MUST forbid candidate reuse within the same variant
    - SHOULD expose deterministic replay for any variant_id
    
    Attributes:
        variant_id: Unique identifier
        policy_id: Reference to policy
        source_universe_id: Reference to source universe
        seed: Random seed used
        assignments: List of assignment records
        manifest: Variant manifest
    """
    variant_id: str
    policy_id: str
    source_universe_id: str
    seed: int
    assignments: List[AssignmentRecord]
    manifest: VariantManifest
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "policy_id": self.policy_id,
            "source_universe_id": self.source_universe_id,
            "seed": self.seed,
            "assignments": [a.to_dict() for a in self.assignments],
            "manifest": self.manifest.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVariant":
        """Create from dictionary."""
        return cls(
            variant_id=data["variant_id"],
            policy_id=data["policy_id"],
            source_universe_id=data["source_universe_id"],
            seed=data["seed"],
            assignments=[AssignmentRecord.from_dict(a) for a in data["assignments"]],
            manifest=VariantManifest.from_dict(data["manifest"]),
        )
    
    def get_positive_assignments(self) -> List[AssignmentRecord]:
        """Get all positive assignments."""
        return [a for a in self.assignments if a.role == "positive"]
    
    def get_negative_assignments(self) -> List[AssignmentRecord]:
        """Get all negative assignments."""
        return [a for a in self.assignments if a.role == "negative"]
    
    def get_target_ids(self) -> List[str]:
        """Get all unique target IDs in this variant."""
        return list(set(a.target_id for a in self.assignments))
    
    def get_protein_ids(self) -> List[str]:
        """Get all unique protein IDs in this variant."""
        return list(set(a.protein_id for a in self.assignments))
