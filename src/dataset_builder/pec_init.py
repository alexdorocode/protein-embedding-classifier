"""
PEC (Protein Embedding Classifier) - Pre-Embedding Dataset Layer

This package implements the pre-embedding dataset layer as defined in
PEC Pre-Embedding Dataset Contract v0.1.

The layer operates BEFORE embedding loading and is responsible for:
- Consuming target-candidate universe inputs (matches_primer_filtro.csv-style)
- Building normalized target-candidate universes
- Defining explicit dataset policies
- Generating reproducible dataset variants
- Creating leakage-safe splits
- Recording complete lineage/provenance
- Exporting self-contained dataset bundles

Module Structure:
- pec.dataset.input: Input reading, validation, and normalization
- pec.dataset.policies: Dataset generation policy definitions and validation
- pec.dataset.generator: Dataset variant generation from universes and policies
- pec.dataset.splits: Split strategy implementations with leakage guards
- pec.dataset.lineage: Provenance manifest building
- pec.dataset.export: Dataset bundle export

All modules are designed to be:
- Configuration-first
- Reproducible (deterministic with fixed seeds)
- Traceable (complete lineage records)
- Extensible (abstract base classes, clean interfaces)
"""

__version__ = "0.1.0"
