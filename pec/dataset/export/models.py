"""
Export Models

Defines the canonical entities for dataset bundle export.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


@dataclass
class DatasetBundle:
    """
    Represents a complete dataset bundle for export.
    
    Canonical bundle layout (from contract §12.2):
    dataset_bundle/
    ├── dataset_instances.csv
    ├── assignments.csv
    ├── split/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── split_manifest.json
    ├── manifests/
    │   ├── universe_manifest.json
    │   ├── dataset_policy.json
    │   ├── variant_manifest.json
    │   └── lineage.json
    └── reports/
        └── dataset_summary.json
    
    Attributes:
        bundle_path: Path to the bundle directory
        variant: The dataset variant being exported
        split_artifacts: The split artifacts (train, val, test)
        universe_manifest: The universe manifest
        policy: The dataset policy
        variant_manifest: The variant manifest
        lineage_manifest: The lineage manifest
    """
    bundle_path: Path
    variant: Any  # DatasetVariant
    split_artifacts: Dict[str, Any]  # train, val, test SplitArtifact
    universe_manifest: Any  # UniverseManifest
    policy: Any  # DatasetPolicy
    variant_manifest: Any  # VariantManifest
    split_manifest: Any  # SplitManifest
    lineage_manifest: Any  # LineageManifest
    
    def get_structure(self) -> Dict[str, List[str]]:
        """Get the expected file structure of the bundle."""
        return {
            "": ["dataset_instances.csv", "assignments.csv"],
            "split": ["train.csv", "val.csv", "test.csv", "split_manifest.json"],
            "manifests": [
                "universe_manifest.json",
                "dataset_policy.json",
                "variant_manifest.json",
                "lineage.json",
            ],
            "reports": ["dataset_summary.json"],
        }
    
    def validate(self) -> List[str]:
        """
        Validate that the bundle contains all required files.
        
        Returns:
            List of missing files (empty if valid)
        """
        missing = []
        structure = self.get_structure()
        
        for dir_path, files in structure.items():
            full_dir = self.bundle_path / dir_path if dir_path else self.bundle_path
            for file_name in files:
                file_path = full_dir / file_name
                if not file_path.exists():
                    missing.append(str(file_path.relative_to(self.bundle_path)))
        
        return missing


@dataclass
class DatasetSummary:
    """
    Summary report for a dataset bundle.
    
    Attributes:
        variant_id: Variant identifier
        policy_id: Policy identifier
        total_instances: Total number of instances
        positive_count: Number of positive instances
        negative_count: Number of negative instances
        train_count: Number of training instances
        val_count: Number of validation instances
        test_count: Number of test instances
        targets_included: Number of targets included
        targets_dropped: Number of targets dropped
        ratio_realized: The realized ratio
    """
    variant_id: str
    policy_id: str
    total_instances: int
    positive_count: int
    negative_count: int
    train_count: int
    val_count: int
    test_count: int
    targets_included: int
    targets_dropped: int
    ratio_realized: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
