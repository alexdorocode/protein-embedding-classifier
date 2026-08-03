"""
Split Strategies

Implements split strategies with leakage guards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
import logging

from pec.dataset.generator.models import AssignmentRecord
from pec.dataset.splits.models import SplitManifest, SplitStrategyConfig, LeakageGuards, SplitArtifact


class SplitStrategy(ABC):
    """
    Abstract base class for split strategies.
    
    Normative Requirements (from contract §10.3):
    - A split MUST be generated from an explicit split strategy artifact or policy reference
    - Positive and negative instances tied to the same target_id MUST remain within the same split in v0.1
    - The split stage MUST emit train, validation, and test artifacts plus a split manifest
    - The split stage SHOULD support future organism-aware strategies
    - The split stage MUST NOT introduce target leakage across partitions
    """
    
    @abstractmethod
    def split(
        self,
        assignments: List[AssignmentRecord],
        seed: int,
        variant_id: str,
    ) -> tuple[SplitArtifact, SplitArtifact, SplitArtifact, SplitManifest]:
        """
        Split assignments into train, val, test partitions.
        
        Args:
            assignments: List of AssignmentRecord instances
            seed: Random seed for reproducibility
            variant_id: Variant ID for the manifest
            
        Returns:
            Tuple of (train_artifact, val_artifact, test_artifact, split_manifest)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> SplitStrategyConfig:
        """Get the configuration for this strategy."""
        pass


class GroupByTargetSplitStrategy(SplitStrategy):
    """
    Split strategy that groups by target_id to prevent leakage.
    
    This is the primary split strategy for v0.1 as specified in contract §15.
    All instances (positive and negative) tied to the same target_id remain
    in the same partition.
    
    Attributes:
        config: Split strategy configuration
        logger: Logger instance
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: Optional[str] = None,
    ):
        """
        Initialize the split strategy.
        
        Args:
            train_ratio: Ratio for training (default: 0.7)
            val_ratio: Ratio for validation (default: 0.15)
            test_ratio: Ratio for testing (default: 0.15)
            stratify_by: Field to stratify by (optional)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = SplitStrategyConfig(
            type="group_by_target",
            group_key="target_id",
            stratify_by=stratify_by,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total}: "
                f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )
    
    def get_config(self) -> SplitStrategyConfig:
        """Get the configuration."""
        return self.config
    
    def split(
        self,
        assignments: List[AssignmentRecord],
        seed: int,
        variant_id: str,
    ) -> tuple[SplitArtifact, SplitArtifact, SplitArtifact, SplitManifest]:
        """
        Split assignments by grouping targets together.
        
        All instances (positive and negative) for the same target_id
        are assigned to the same partition.
        
        Args:
            assignments: List of AssignmentRecord instances
            seed: Random seed for reproducibility
            variant_id: Variant ID for the manifest
            
        Returns:
            Tuple of (train_artifact, val_artifact, test_artifact, split_manifest)
        """
        if not assignments:
            raise ValueError("Cannot split empty assignments")
        
        # Set random seed
        rng = random.Random(seed)
        
        # Group assignments by target_id
        target_to_assignments: Dict[str, List[AssignmentRecord]] = {}
        for assignment in assignments:
            target_id = assignment.target_id
            if target_id not in target_to_assignments:
                target_to_assignments[target_id] = []
            target_to_assignments[target_id].append(assignment)
        
        # Get unique targets
        targets = list(target_to_assignments.keys())
        
        # Shuffle targets for random assignment
        rng.shuffle(targets)
        
        # Calculate split sizes
        n_targets = len(targets)
        n_train = int(n_targets * self.config.train_ratio)
        n_val = int(n_targets * self.config.val_ratio)
        # Test gets the remainder
        n_test = n_targets - n_train - n_val
        
        # Assign targets to partitions
        train_targets = set(targets[:n_train])
        val_targets = set(targets[n_train:n_train + n_val])
        test_targets = set(targets[n_train + n_val:])
        
        # Create artifacts
        train_assignments = []
        val_assignments = []
        test_assignments = []
        
        for target_id, target_assignments in target_to_assignments.items():
            if target_id in train_targets:
                train_assignments.extend(target_assignments)
            elif target_id in val_targets:
                val_assignments.extend(target_assignments)
            else:
                test_assignments.extend(target_assignments)
        
        # Generate split ID
        split_id = f"split_{variant_id}_seed{seed}"
        
        # Create manifest
        manifest = SplitManifest(
            split_id=split_id,
            variant_id=variant_id,
            split_strategy=self.config,
            random_seed=seed,
            leakage_guards=LeakageGuards(
                keep_same_target_in_one_split=True,
                keep_linked_instances_together=True,
            ),
        )
        
        self.logger.info(
            f"Split variant {variant_id}: "
            f"{len(train_assignments)} train, {len(val_assignments)} val, {len(test_assignments)} test"
        )
        
        return (
            SplitArtifact(name="train", assignment_records=train_assignments),
            SplitArtifact(name="val", assignment_records=val_assignments),
            SplitArtifact(name="test", assignment_records=test_assignments),
            manifest,
        )
    
    def split_to_dicts(
        self,
        assignments: List[AssignmentRecord],
        seed: int,
        variant_id: str,
    ) -> Dict[str, Any]:
        """
        Split and return as dictionary with partition lists.
        
        Args:
            assignments: List of AssignmentRecord instances
            seed: Random seed
            variant_id: Variant ID
            
        Returns:
            Dictionary with train, val, test lists and manifest
        """
        train, val, test, manifest = self.split(assignments, seed, variant_id)
        
        return {
            "train": [a.to_dict() for a in train.assignment_records],
            "val": [a.to_dict() for a in val.assignment_records],
            "test": [a.to_dict() for a in test.assignment_records],
            "manifest": manifest.to_dict(),
        }
