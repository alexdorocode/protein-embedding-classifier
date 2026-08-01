"""
Generator Module

Responsible for building concrete dataset variants from universes and policies.

Key Classes:
- DatasetVariant: Represents a concrete dataset variant
- DatasetVariantGenerator: Generates variants from universe + policy + seed
- AssignmentTable: Canonical assignments table for a variant
"""

from pec.dataset.generator.models import DatasetVariant, AssignmentRecord, VariantManifest
from pec.dataset.generator.generator import DatasetVariantGenerator

__all__ = ["DatasetVariant", "AssignmentRecord", "VariantManifest", "DatasetVariantGenerator"]
