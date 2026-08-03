"""
Lineage Models

Defines the canonical entities for provenance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json
from datetime import datetime


@dataclass
class SourceArtifact:
    """
    Reference to a source artifact in the lineage chain.
    
    Attributes:
        artifact_id: Unique identifier for the artifact
        artifact_type: Type of artifact
        source_path: Path to the source file
        source_version: Version identifier (e.g., SHA256 hash)
    """
    artifact_id: str
    artifact_type: str
    source_path: str
    source_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get("source_version") is None:
            del result["source_version"]
        return result


@dataclass
class TransformStep:
    """
    Record of a transformation step in the lineage chain.
    
    Attributes:
        step: Name of the transformation step
        code_version: Version of the code that performed the step
        details: Optional additional details
    """
    step: str
    code_version: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"step": self.step, "code_version": self.code_version}
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class RuntimeInfo:
    """
    Runtime information for lineage tracking.
    
    Attributes:
        generated_at: Timestamp of generation
        generated_by: Identifier of the generator
        random_seed: Random seed used
    """
    generated_at: str
    generated_by: str
    random_seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get("random_seed") is None:
            del result["random_seed"]
        return result


@dataclass
class LineageManifest:
    """
    Canonical lineage manifest (from contract §11.2).
    
    Normative Requirements (from contract §11.3):
    - Every dataset variant MUST have one lineage manifest
    - A lineage manifest MUST reference the exact source artifact identity
    - A lineage manifest MUST record policy identity, split identity, code version, and seed
    - A dataset artifact MUST NOT be considered valid for downstream PEC consumption if lineage is missing
    - The lineage system SHOULD support hash-based source and output identification
    
    Attributes:
        lineage_id: Unique identifier for this lineage
        source_artifacts: List of source artifact references
        transforms: List of transformation steps
        runtime: Runtime information
    """
    lineage_id: str
    source_artifacts: List[SourceArtifact]
    transforms: List[TransformStep]
    runtime: RuntimeInfo
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageManifest":
        return cls(
            lineage_id=data["lineage_id"],
            source_artifacts=[
                SourceArtifact(**sa) for sa in data["source_artifacts"]
            ],
            transforms=[
                TransformStep(**t) for t in data["transforms"]
            ],
            runtime=RuntimeInfo(**data["runtime"]),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "LineageManifest":
        return cls.from_dict(json.loads(json_str))
