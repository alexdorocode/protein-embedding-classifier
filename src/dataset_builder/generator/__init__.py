"""
Generator Module

Responsible for building concrete dataset variants from universes and policies.

Key Classes:
- DatasetVariant: Represents a concrete dataset variant
- DatasetVariantGenerator: Generates variants from universe + policy + seed
- AssignmentTable: Canonical assignments table for a variant
"""

from src.dataset_builder.generator.models import DatasetVariant, AssignmentRecord, VariantManifest
from src.dataset_builder.generator.generator import DatasetVariantGenerator

__all__ = ["DatasetVariant", "AssignmentRecord", "VariantManifest", "DatasetVariantGenerator"]
