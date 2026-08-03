"""
Split Models

Defines the canonical entities for dataset splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json


@dataclass
class LeakageGuards:
    """
    Leakage prevention guards for splits.
    
    Attributes:
        keep_same_target_in_one_split: Whether to keep all instances of same target in one split
        keep_linked_instances_together: Whether to keep linked instances together
    """
    keep_same_target_in_one_split: bool = True
    keep_linked_instances_together: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitStrategyConfig:
    """
    Configuration for a split strategy.
    
    Attributes:
        type: Type of split strategy
        group_key: Key to group by
        stratify_by: Field to stratify by
        train_ratio: Ratio for training
        val_ratio: Ratio for validation
        test_ratio: Ratio for testing
    """
    type: str
    group_key: str = "target_id"
    stratify_by: Optional[str] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "group_key": self.group_key,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }
        if self.stratify_by:
            result["stratify_by"] = self.stratify_by
        return result


@dataclass
class SplitManifest:
    """
    Canonical split manifest (from contract §10.2).
    
    Attributes:
        split_id: Unique identifier for this split
        variant_id: Reference to the variant being split
        split_strategy: Configuration of the split strategy
        random_seed: Random seed used
        leakage_guards: Leakage prevention guards
    """
    split_id: str
    variant_id: str
    split_strategy: SplitStrategyConfig
    random_seed: int
    leakage_guards: LeakageGuards = field(default_factory=LeakageGuards)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitManifest":
        return cls(
            split_id=data["split_id"],
            variant_id=data["variant_id"],
            split_strategy=SplitStrategyConfig(**data["split_strategy"]),
            random_seed=data["random_seed"],
            leakage_guards=LeakageGuards(**data.get("leakage_guards", {})),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SplitArtifact:
    """
    Represents a split partition (train, val, or test).
    
    Attributes:
        name: Name of the partition (train, val, test)
        assignment_records: List of assignment records in this partition
    """
    name: str
    assignment_records: List[Any]  # AssignmentRecord instances
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": len(self.assignment_records),
        }
