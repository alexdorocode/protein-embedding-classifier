"""
Lineage Module

Responsible for building provenance manifests and artifact identities.

Key Classes:
- LineageBuilder: Builds lineage manifests
- LineageManifest: Canonical lineage manifest
- SourceArtifact: Reference to a source artifact
- TransformStep: Record of a transformation step
"""

from pec.dataset.lineage.models import LineageManifest, SourceArtifact, TransformStep, RuntimeInfo
from pec.dataset.lineage.builder import LineageBuilder

__all__ = ["LineageBuilder", "LineageManifest", "SourceArtifact", "TransformStep", "RuntimeInfo"]
