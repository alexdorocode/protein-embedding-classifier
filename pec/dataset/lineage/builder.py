"""
Lineage Builder

Builds provenance manifests for dataset artifacts.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path

from pec.dataset.lineage.models import (
    LineageManifest,
    SourceArtifact,
    TransformStep,
    RuntimeInfo,
)


class LineageBuilder:
    """
    Builds lineage manifests for dataset artifacts.
    
    Normative Requirements (from contract §11.3):
    - Every dataset variant MUST have one lineage manifest
    - A lineage manifest MUST reference the exact source artifact identity
    - A lineage manifest MUST record policy identity, split identity, code version, and seed
    - A dataset artifact MUST NOT be considered valid for downstream PEC consumption if lineage is missing
    - The lineage system SHOULD support hash-based source and output identification
    
    Attributes:
        logger: Logger instance
        generated_by: Identifier for the generator
    """
    
    def __init__(self, generated_by: str = "pec.dataset_generator"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generated_by = generated_by
    
    def build(
        self,
        lineage_id: Optional[str] = None,
        source_artifacts: Optional[List[SourceArtifact]] = None,
        transforms: Optional[List[TransformStep]] = None,
        runtime: Optional[RuntimeInfo] = None,
        random_seed: Optional[int] = None,
    ) -> LineageManifest:
        """
        Build a lineage manifest.
        
        Args:
            lineage_id: Optional lineage ID (generated if None)
            source_artifacts: List of source artifacts
            transforms: List of transformation steps
            runtime: Runtime information
            random_seed: Random seed used
            
        Returns:
            LineageManifest instance
        """
        if source_artifacts is None:
            source_artifacts = []
        if transforms is None:
            transforms = []
        
        # Generate lineage_id if not provided
        if lineage_id is None:
            lineage_id = self._generate_lineage_id(source_artifacts, transforms, random_seed)
        
        # Build runtime if not provided
        if runtime is None:
            runtime = RuntimeInfo(
                generated_at=datetime.utcnow().isoformat() + "Z",
                generated_by=self.generated_by,
                random_seed=random_seed,
            )
        
        manifest = LineageManifest(
            lineage_id=lineage_id,
            source_artifacts=source_artifacts,
            transforms=transforms,
            runtime=runtime,
        )
        
        self.logger.info(f"Built lineage manifest: {lineage_id}")
        return manifest
    
    def _generate_lineage_id(
        self,
        source_artifacts: List[SourceArtifact],
        transforms: List[TransformStep],
        random_seed: Optional[int],
    ) -> str:
        """Generate a deterministic lineage ID."""
        parts = []
        
        # Include source artifact IDs
        for sa in source_artifacts:
            parts.append(sa.artifact_id)
        
        # Include transform steps
        for t in transforms:
            parts.append(t.step)
        
        # Include seed if provided
        if random_seed is not None:
            parts.append(str(random_seed))
        
        # Include timestamp
        parts.append(datetime.utcnow().isoformat())
        
        # Hash the parts
        hash_input = "|".join(parts)
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        return f"lineage_{hash_value}"
    
    def build_from_file(
        self,
        source_path: str,
        artifact_type: str = "target_candidate_universe",
        artifact_id: Optional[str] = None,
    ) -> SourceArtifact:
        """
        Create a source artifact from a file path.
        
        Args:
            source_path: Path to the source file
            artifact_type: Type of artifact
            artifact_id: Optional artifact ID (derived from path if None)
            
        Returns:
            SourceArtifact instance
        """
        path = Path(source_path)
        if artifact_id is None:
            artifact_id = path.stem
        
        # Calculate hash
        source_version = self._calculate_file_hash(path)
        
        return SourceArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            source_path=str(path),
            source_version=source_version,
        )
    
    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        if not path.exists():
            return "unknown"
        
        hash_obj = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def build_transform_step(
        self,
        step: str,
        code_version: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> TransformStep:
        """
        Create a transform step.
        
        Args:
            step: Name of the transformation step
            code_version: Code version (default: current git commit)
            details: Optional additional details
            
        Returns:
            TransformStep instance
        """
        if code_version is None:
            code_version = self._get_git_commit()
        
        return TransformStep(
            step=step,
            code_version=code_version,
            details=details,
        )
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def build_complete_lineage(
        self,
        source_file: str,
        universe_id: str,
        policy_id: str,
        variant_id: str,
        split_id: str,
        random_seed: int,
    ) -> LineageManifest:
        """
        Build a complete lineage manifest for a dataset variant.
        
        Args:
            source_file: Path to source file
            universe_id: Universe identifier
            policy_id: Policy identifier
            variant_id: Variant identifier
            split_id: Split identifier
            random_seed: Random seed used
            
        Returns:
            Complete LineageManifest
        """
        source_artifact = self.build_from_file(
            source_path=source_file,
            artifact_type="target_candidate_universe",
            artifact_id=universe_id,
        )
        
        transforms = [
            self.build_transform_step("normalize_input"),
            self.build_transform_step("apply_dataset_policy", details={"policy_id": policy_id}),
            self.build_transform_step("generate_variant", details={"variant_id": variant_id}),
            self.build_transform_step("generate_split", details={"split_id": split_id}),
        ]
        
        return self.build(
            source_artifacts=[source_artifact],
            transforms=transforms,
            random_seed=random_seed,
        )
