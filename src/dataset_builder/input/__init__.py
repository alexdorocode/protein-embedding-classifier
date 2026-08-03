"""
Input Module

Responsible for reading, validating, and normalizing target-candidate universe inputs
from matches_primer_filtro.csv-style files.

Key Classes:
- UniverseRecord: Canonical internal representation of a target-candidate universe row
- UniverseReader: Reads and parses input files
- UniverseNormalizer: Normalizes raw input into UniverseRecord instances
"""

from src.dataset_builder.input.models import UniverseRecord, UniverseManifest
from src.dataset_builder.input.reader import UniverseReader
from src.dataset_builder.input.normalizer import UniverseNormalizer

__all__ = ["UniverseRecord", "UniverseManifest", "UniverseReader", "UniverseNormalizer"]
