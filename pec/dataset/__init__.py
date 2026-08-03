"""
PEC Dataset Layer

This module implements the pre-embedding dataset layer contract.

Core Concepts (from contract):
- Universe: Normalized target-candidate input representation
- Policy: Explicit rules for dataset generation (ratio, scarcity, randomization)
- Variant: Concrete dataset instance from universe + policy + seed
- Split: Partitioning of variant into train/val/test with leakage guards
- Lineage: Complete provenance chain from source to export
- Bundle: Self-contained export package for downstream PEC stages
"""

from src.dataset_builder.input import UniverseReader, UniverseNormalizer, UniverseRecord
from src.dataset_builder.policies import DatasetPolicy, PolicyValidator
from src.dataset_builder.generator import DatasetVariantGenerator, DatasetVariant
from src.dataset_builder.splits import SplitStrategy, GroupByTargetSplitStrategy
from src.dataset_builder.lineage import LineageBuilder, LineageManifest
from src.dataset_builder.export import BundleExporter, DatasetBundle

__all__ = [
    # Input
    "UniverseReader",
    "UniverseNormalizer",
    "UniverseRecord",
    # Policies
    "DatasetPolicy",
    "PolicyValidator",
    # Generator
    "DatasetVariantGenerator",
    "DatasetVariant",
    # Splits
    "SplitStrategy",
    "GroupByTargetSplitStrategy",
    # Lineage
    "LineageBuilder",
    "LineageManifest",
    # Export
    "BundleExporter",
    "DatasetBundle",
]
