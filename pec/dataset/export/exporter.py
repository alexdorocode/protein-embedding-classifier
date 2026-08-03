"""
Bundle Exporter

Exports self-contained dataset bundles for downstream PEC use.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import csv
import json
import logging

from pec.dataset.generator.models import DatasetVariant, AssignmentRecord
from pec.dataset.splits.models import SplitArtifact, SplitManifest
from pec.dataset.input.models import UniverseManifest
from pec.dataset.policies.models import DatasetPolicy
from pec.dataset.generator.models import VariantManifest
from pec.dataset.lineage.models import LineageManifest
from pec.dataset.export.models import DatasetBundle, DatasetSummary


class BundleExporter:
    """
    Exports dataset bundles to the filesystem.
    
    Normative Requirements (from contract §12.3):
    - Every exported dataset MUST be packaged as a self-contained bundle
    - The bundle MUST include the realized instances, split artifacts, and all relevant manifests
    - The bundle SHOULD include a machine-readable summary report
    - The bundle MUST be consumable without requiring notebook state or ad hoc path assumptions
    
    Attributes:
        logger: Logger instance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def export(
        self,
        variant: DatasetVariant,
        split_artifacts: Dict[str, SplitArtifact],
        split_manifest: SplitManifest,
        universe_manifest: UniverseManifest,
        policy: DatasetPolicy,
        lineage_manifest: LineageManifest,
        bundle_path: Path,
    ) -> DatasetBundle:
        """
        Export a complete dataset bundle.
        
        Args:
            variant: The dataset variant
            split_artifacts: Dictionary of split artifacts (train, val, test)
            split_manifest: The split manifest
            universe_manifest: The universe manifest
            policy: The dataset policy
            lineage_manifest: The lineage manifest
            bundle_path: Path to export the bundle to
            
        Returns:
            DatasetBundle instance
        """
        # Create bundle directory
        bundle_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        split_dir = bundle_path / "split"
        manifests_dir = bundle_path / "manifests"
        reports_dir = bundle_path / "reports"
        
        split_dir.mkdir(exist_ok=True)
        manifests_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)
        
        # Export dataset_instances.csv
        self._export_instances_csv(variant, bundle_path / "dataset_instances.csv")
        
        # Export assignments.csv
        self._export_assignments_csv(variant, bundle_path / "assignments.csv")
        
        # Export split artifacts
        self._export_split_artifacts(split_artifacts, split_dir)
        
        # Export split manifest
        self._export_json(split_manifest.to_dict(), split_dir / "split_manifest.json")
        
        # Export manifests
        self._export_json(universe_manifest.to_dict(), manifests_dir / "universe_manifest.json")
        self._export_json(policy.to_dict(), manifests_dir / "dataset_policy.json")
        self._export_json(variant.manifest.to_dict(), manifests_dir / "variant_manifest.json")
        self._export_json(lineage_manifest.to_dict(), manifests_dir / "lineage.json")
        
        # Export summary report
        summary = self._build_summary(variant, split_artifacts)
        self._export_json(summary.to_dict(), reports_dir / "dataset_summary.json")
        
        self.logger.info(f"Exported dataset bundle to {bundle_path}")
        
        return DatasetBundle(
            bundle_path=bundle_path,
            variant=variant,
            split_artifacts=split_artifacts,
            universe_manifest=universe_manifest,
            policy=policy,
            variant_manifest=variant.manifest,
            split_manifest=split_manifest,
            lineage_manifest=lineage_manifest,
        )
    
    def _export_instances_csv(self, variant: DatasetVariant, path: Path) -> None:
        """Export dataset instances to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["protein_id", "role", "target_id"])
            for assignment in variant.assignments:
                writer.writerow([
                    assignment.protein_id,
                    assignment.role,
                    assignment.target_id,
                ])
    
    def _export_assignments_csv(self, variant: DatasetVariant, path: Path) -> None:
        """Export assignments table to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(AssignmentRecord.csv_headers())
            for assignment in variant.assignments:
                writer.writerow(assignment.to_csv_row())
    
    def _export_split_artifacts(
        self,
        split_artifacts: Dict[str, SplitArtifact],
        split_dir: Path,
    ) -> None:
        """Export split artifacts to CSV files."""
        for name, artifact in split_artifacts.items():
            path = split_dir / f"{name}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(AssignmentRecord.csv_headers())
                for assignment in artifact.assignment_records:
                    writer.writerow(assignment.to_csv_row())
    
    def _export_json(self, data: Dict[str, Any], path: Path) -> None:
        """Export data to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def _build_summary(
        self,
        variant: DatasetVariant,
        split_artifacts: Dict[str, SplitArtifact],
    ) -> DatasetSummary:
        """Build a dataset summary."""
        train_count = len(split_artifacts.get("train", SplitArtifact(name="train", assignment_records=[])).assignment_records)
        val_count = len(split_artifacts.get("val", SplitArtifact(name="val", assignment_records=[])).assignment_records)
        test_count = len(split_artifacts.get("test", SplitArtifact(name="test", assignment_records=[])).assignment_records)
        
        positive_count = len(variant.get_positive_assignments())
        negative_count = len(variant.get_negative_assignments())
        
        return DatasetSummary(
            variant_id=variant.variant_id,
            policy_id=variant.policy_id,
            total_instances=len(variant.assignments),
            positive_count=positive_count,
            negative_count=negative_count,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            targets_included=variant.manifest.targets_included,
            targets_dropped=variant.manifest.targets_dropped,
            ratio_realized=variant.manifest.dataset_statistics.get("ratio_realized", "unknown"),
        )
    
    def validate_bundle(self, bundle_path: Path) -> List[str]:
        """
        Validate that a bundle contains all required files.
        
        Args:
            bundle_path: Path to the bundle
            
        Returns:
            List of missing files (empty if valid)
        """
        bundle = DatasetBundle(
            bundle_path=bundle_path,
            variant=None,
            split_artifacts={},
            universe_manifest=None,
            policy=None,
            variant_manifest=None,
            split_manifest=None,
            lineage_manifest=None,
        )
        return bundle.validate()
