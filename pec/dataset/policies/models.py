"""
Policy Models

Defines the canonical schema for dataset generation policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal
import json


# Type aliases for policy fields
SelectionMode = Literal["sample_without_replacement", "sample_with_replacement", "use_all"]
CandidateScope = Literal["per_target", "global"]
AssignmentStrategy = Literal["global_unique_candidates", "per_target_unique"]
PositiveUnit = Literal["target"]
NegativeUnit = Literal["candidate_assignment"]
ScarcityMode = Literal["drop_target", "relax_ratio", "use_available"]
SeedScope = Literal["global", "per_target"]
OrganismMode = Literal["preserve_source", "filter_by_organism", "balance_by_organism"]
DuplicateMode = Literal["allow", "forbid"]


@dataclass
class SelectionStrategy:
    """
    Defines how candidates are selected from the pool.
    
    Attributes:
        mode: Sampling mode (sample_without_replacement, sample_with_replacement, use_all)
        candidate_scope: Scope of candidate selection (per_target, global)
        assignment_strategy: How candidates are assigned to targets
    """
    mode: SelectionMode = "sample_without_replacement"
    candidate_scope: CandidateScope = "per_target"
    assignment_strategy: AssignmentStrategy = "global_unique_candidates"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionStrategy":
        return cls(**data)


@dataclass
class RatioPolicy:
    """
    Defines the ratio of positive to negative instances.
    
    Attributes:
        positive_unit: Unit of positive instances (target)
        negative_unit: Unit of negative instances (candidate_assignment)
        target_to_negative_ratio: Ratio string (e.g., "1:1", "1:3", "1:5")
    """
    positive_unit: PositiveUnit = "target"
    negative_unit: NegativeUnit = "candidate_assignment"
    target_to_negative_ratio: str = "1:1"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RatioPolicy":
        return cls(**data)
    
    def get_ratio_tuple(self) -> tuple[int, int]:
        """Parse ratio string into (positive, negative) tuple."""
        parts = self.target_to_negative_ratio.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid ratio format: {self.target_to_negative_ratio}"
            )
        return int(parts[0]), int(parts[1])


@dataclass
class CandidatePoolPolicy:
    """
    Defines constraints on the candidate pool.
    
    Attributes:
        min_pool_size: Minimum number of candidates required per target
        max_pool_size: Maximum number of candidates to use per target (None = no limit)
        scarcity_mode: How to handle targets with insufficient candidates
    """
    min_pool_size: int = 5
    max_pool_size: Optional[int] = None
    scarcity_mode: ScarcityMode = "drop_target"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "min_pool_size": self.min_pool_size,
            "scarcity_mode": self.scarcity_mode,
        }
        if self.max_pool_size is not None:
            result["max_pool_size"] = self.max_pool_size
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidatePoolPolicy":
        return cls(
            min_pool_size=data.get("min_pool_size", 5),
            max_pool_size=data.get("max_pool_size"),
            scarcity_mode=data.get("scarcity_mode", "drop_target"),
        )


@dataclass
class RandomizationConfig:
    """
    Defines randomization behavior for dataset generation.
    
    Attributes:
        enabled: Whether randomization is enabled
        seed_scope: Scope of random seed (global, per_target)
    """
    enabled: bool = True
    seed_scope: SeedScope = "global"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RandomizationConfig":
        return cls(**data)


@dataclass
class OrganismPolicy:
    """
    Defines organism-related policy for dataset generation.
    
    Attributes:
        mode: How to handle organism information
    """
    mode: OrganismMode = "preserve_source"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrganismPolicy":
        return cls(**data)


@dataclass
class DuplicatePolicy:
    """
    Defines policy for handling duplicate candidates.
    
    Attributes:
        allow_same_candidate_across_targets: Whether same candidate can be used for different targets
        allow_same_target_across_variants: Whether same target can appear in different variants
    """
    allow_same_candidate_across_targets: bool = False
    allow_same_target_across_variants: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DuplicatePolicy":
        return cls(**data)


@dataclass
class DatasetPolicy:
    """
    Canonical policy schema for dataset generation.
    
    This is the main configuration object for the pre-embedding dataset layer.
    It captures all decisions that could affect composition, reproducibility,
    or scientific interpretation.
    
    Normative Requirements (from contract §8.3):
    - MUST declare ratio behavior, scarcity handling, randomization behavior, and duplicate behavior
    - MUST NOT rely on hidden defaults that materially affect sampling outcomes
    - SHOULD be validatable against a JSON schema
    - MAY define organism-aware filtering or balancing behavior in future versions
    
    Attributes:
        policy_id: Unique identifier for this policy
        source_universe_id: Reference to the source universe
        selection_strategy: Strategy for selecting candidates
        ratio_policy: Ratio constraints
        candidate_pool_policy: Pool constraints and scarcity handling
        randomization: Randomization configuration
        split_policy_ref: Reference to split policy
        organism_policy: Organism-related policy
        duplicate_policy: Duplicate handling policy
    """
    policy_id: str
    source_universe_id: str
    selection_strategy: SelectionStrategy = field(default_factory=SelectionStrategy)
    ratio_policy: RatioPolicy = field(default_factory=lambda: RatioPolicy(target_to_negative_ratio="1:1"))
    candidate_pool_policy: CandidatePoolPolicy = field(default_factory=CandidatePoolPolicy)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    split_policy_ref: str = "group_by_target_v1"
    organism_policy: OrganismPolicy = field(default_factory=OrganismPolicy)
    duplicate_policy: DuplicatePolicy = field(default_factory=DuplicatePolicy)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "policy_id": self.policy_id,
            "source_universe_id": self.source_universe_id,
            "selection_strategy": self.selection_strategy.to_dict(),
            "ratio_policy": self.ratio_policy.to_dict(),
            "candidate_pool_policy": self.candidate_pool_policy.to_dict(),
            "randomization": self.randomization.to_dict(),
            "split_policy_ref": self.split_policy_ref,
            "organism_policy": self.organism_policy.to_dict(),
            "duplicate_policy": self.duplicate_policy.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetPolicy":
        """Create DatasetPolicy from dictionary."""
        return cls(
            policy_id=data["policy_id"],
            source_universe_id=data["source_universe_id"],
            selection_strategy=SelectionStrategy.from_dict(
                data.get("selection_strategy", {})
            ),
            ratio_policy=RatioPolicy.from_dict(
                data.get("ratio_policy", {})
            ),
            candidate_pool_policy=CandidatePoolPolicy.from_dict(
                data.get("candidate_pool_policy", {})
            ),
            randomization=RandomizationConfig.from_dict(
                data.get("randomization", {})
            ),
            split_policy_ref=data.get("split_policy_ref", "group_by_target_v1"),
            organism_policy=OrganismPolicy.from_dict(
                data.get("organism_policy", {})
            ),
            duplicate_policy=DuplicatePolicy.from_dict(
                data.get("duplicate_policy", {})
            ),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DatasetPolicy":
        """Create DatasetPolicy from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_variant_count(self) -> int:
        """Get the default number of variants for this policy (25 from contract)."""
        return 25
    
    def get_ratio_families(self) -> List[str]:
        """Get the initial ratio families from contract §15."""
        return ["1:1", "1:3", "1:5"]
