# Protein Embedding Classifier (PEC)

A research framework for protein function prediction using precomputed embeddings, with a **two-layer architecture** that separates dataset preparation from embedding-based classification.

**Important:** This repository does **not** generate embeddings. It consumes precomputed embeddings as feature vectors for supervised classification.

---

## 📖 Repository Overview

The Protein Embedding Classifier (PEC) is now organized into **two distinct layers** to support the complete research pipeline:

### Layer 1: Pre-Embedding Dataset Layer (`pec/`)
**Purpose:** Dataset preparation and organization **before** embedding loading.

This layer is responsible for:
- Consuming `matches_primer_filtro.csv`-style target-candidate universe inputs
- Building normalized target-candidate universes
- Defining explicit dataset policies (ratio, scarcity, randomization)
- Generating reproducible dataset variants
- Creating leakage-safe splits grouped by target_id
- Recording complete lineage/provenance
- Exporting self-contained dataset bundles

**Status:** ✅ **Fully implemented** according to [PEC Pre-Embedding Dataset Contract v0.1](pec_pre_embedding_dataset_contract_v0_1.md)

### Layer 2: Embedding-Based Classification Layer (`protein_embedding_classifier/`)
**Purpose:** Supervised classification using precomputed embeddings.

This layer is responsible for:
- Loading accession-level proteins and labels from configured sources
- Loading precomputed embedding vectors (sequence and GO-derived)
- Training supervised classifiers over fixed embedding vectors
- Running randomized hyperparameter sweeps
- Ensemble inference and evaluation
- Benchmark reporting

**Status:** ✅ **Existing implementation** (original codebase)

---

## 🗺️ Repository Structure

```
protein-embedding-classifier/
├── README.md                           # This file - complete overview
├── LICENSE                             # License (to be added)
├── .gitignore                          # Git ignore patterns
│
├── pec/                                # 🆕 Pre-Embedding Dataset Layer
│   ├── README.md                       # Detailed pre-embedding documentation
│   ├── __init__.py
│   └── dataset/
│       ├── __init__.py                # Dataset layer exports
│       ├── contracts.py              # Abstract base classes and protocols
│       ├── input/                     # Input reading and normalization
│       │   ├── __init__.py
│       │   ├── models.py              # UniverseRecord, UniverseManifest
│       │   ├── reader.py              # UniverseReader (CSV/JSONL)
│       │   └── normalizer.py          # UniverseNormalizer
│       ├── policies/                  # Dataset generation policies
│       │   ├── __init__.py
│       │   ├── models.py              # DatasetPolicy and sub-models
│       │   └── validator.py           # PolicyValidator
│       ├── generator/                 # Dataset variant generation
│       │   ├── __init__.py
│       │   ├── models.py              # DatasetVariant, AssignmentRecord, VariantManifest
│       │   └── generator.py           # DatasetVariantGenerator
│       ├── splits/                    # Split strategies
│       │   ├── __init__.py
│       │   ├── models.py              # SplitManifest, SplitStrategyConfig
│       │   └── strategies.py          # SplitStrategy, GroupByTargetSplitStrategy
│       ├── lineage/                   # Provenance tracking
│       │   ├── __init__.py
│       │   ├── models.py              # LineageManifest, SourceArtifact, TransformStep
│       │   └── builder.py             # LineageBuilder
│       └── export/                    # Bundle export
│           ├── __init__.py
│           ├── models.py              # DatasetBundle, DatasetSummary
│           └── exporter.py            # BundleExporter
│
├── protein_embedding_classifier/      # Existing Embedding-Based Layer
│   ├── __init__.py
│   ├── main.py                       # CLI entry point
│   ├── config/                       # Configuration files
│   ├── core/                         # Core functionality
│   │   ├── db.py                     # Database operations
│   │   ├── embedding_loading.py      # Embedding loading
│   │   ├── embeddings.py             # Embedding handling
│   │   ├── pipeline.py               # Training pipeline
│   │   ├── experiment.py             # Experiment management
│   │   ├── training/                 # Training services
│   │   │   ├── training_service.py
│   │   │   ├── sweep_service.py
│   │   │   ├── model_factory.py
│   │   │   └── ...
│   │   ├── ensemble/                # Ensemble methods
│   │   │   └── soft_voting_service.py
│   │   ├── statistics/               # Statistical tests
│   │   └── ...
│   ├── classifiers/                  # Classifier implementations
│   │   ├── base.py
│   │   ├── linear.py
│   │   ├── mlp_protein_classifier.py
│   │   ├── random_forest.py
│   │   └── registry.py
│   ├── data/                         # Data loading
│   │   ├── dataset_builder.py
│   │   ├── protein_loader.py
│   │   ├── label_loader.py
│   │   └── splits/
│   ├── layer_aggregation/            # Layer aggregation strategies
│   └── logging_config.py
│
├── tests/                            # Test suites
│   ├── pec/                          # 🆕 Tests for pre-embedding layer
│   │   └── dataset/
│   │       ├── test_integration.py        # End-to-end workflow tests
│   │       ├── test_forward_compatibility.py  # Future module scaffolding
│   │       ├── input/
│   │       │   ├── test_models.py
│   │       │   ├── test_reader.py
│   │       │   └── test_normalizer.py
│   │       ├── policies/
│   │       │   └── test_models.py
│   │       ├── generator/
│   │       │   └── __init__.py
│   │       ├── splits/
│   │       │   └── __init__.py
│   │       ├── lineage/
│   │       │   └── __init__.py
│   │       └── export/
│   │           └── __init__.py
│   └── core/                          # Existing tests for embedding layer
│       ├── training/
│       ├── ensemble/
│       └── statistics/
│
├── config/                           # Configuration files
│   ├── db.yaml
│   ├── embeddings.yaml
│   └── pipeline.yaml
│
├── pyproject.toml                    # Python project configuration
├── poetry.lock                       # Dependency lock file
├── pytest.ini                        # Pytest configuration
│
├── pec_pre_embedding_dataset_contract_v0_1.md  # Pre-embedding contract
├── FUNCTIONAL_CONTRACT.md            # Final system functional vision
│
└── PEC-IMPLEMENTATION-NOTES.md       # Implementation documentation
```

---

## 🎯 Branch Strategy

This repository uses a **documentation-first** branch model:

| Branch | Purpose | Status |
|--------|---------|--------|
| `master` | **Stable, minimal** - Contains only high-level vision documents, contracts, and basic repository metadata. This is the canonical reference for project architecture and specifications. | ✅ Protected |
| `pre-embedding-dev` | **Active development** - Implementation of the pre-embedding dataset layer. All current pre-embedding code lives here. | ✅ Active |

### Future Branch Placeholders
- `embeddings-dev` - Embedding loading and processing layer
- `classifier-dev` - Classifier training and evaluation layer
- `ensemble-dev` - Ensemble methods and voting strategies
- `benchmark-dev` - Benchmark orchestration and reporting

---

## 🚀 Quick Start

### For New Developers

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alexdorocode/protein-embedding-classifier.git
   cd protein-embedding-classifier
   ```

2. **Understand the architecture:**
   - Read [PEC Pre-Embedding Dataset Contract v0.1](pec_pre_embedding_dataset_contract_v0_1.md)
   - Read [PEC Final-System Functional Vision](FUNCTIONAL_CONTRACT.md)
   - Read [Implementation Notes](PEC-IMPLEMENTATION-NOTES.md)

3. **Work on the pre-embedding layer:**
   ```bash
   git checkout pre-embedding-dev
   ```

4. **Run tests:**
   ```bash
   # All pre-embedding tests
   pytest tests/pec/ -v
   
   # With coverage
   pytest tests/pec/ --cov=pec --cov-report=html
   ```

### For Existing Users

The existing `protein_embedding_classifier/` code remains unchanged and functional. The new `pec/` directory adds the pre-embedding dataset layer that operates **before** embedding loading.

---

## 📚 Documentation

### Vision and Architecture
- **[PEC Final-System Functional Vision](FUNCTIONAL_CONTRACT.md)** - High-level system architecture and responsibilities
- **[PEC Pre-Embedding Dataset Contract v0.1](pec_pre_embedding_dataset_contract_v0_1.md)** - Contract for the pre-embedding dataset layer

### Layer-Specific Documentation
- **[Pre-Embedding Layer Documentation](pec/README.md)** - Detailed guide for the pre-embedding dataset layer
- **[Implementation Notes](PEC-IMPLEMENTATION-NOTES.md)** - Complete implementation summary and verification

---

## 🔧 Pre-Embedding Dataset Layer Usage

The pre-embedding layer is designed to be used **before** embedding loading. Here's a basic workflow:

```python
from pathlib import Path
from pec.dataset.input.reader import UniverseReader
from pec.dataset.input.normalizer import UniverseNormalizer
from pec.dataset.policies.models import DatasetPolicy, RatioPolicy, CandidatePoolPolicy
from pec.dataset.generator.generator import DatasetVariantGenerator
from pec.dataset.splits.strategies import GroupByTargetSplitStrategy
from pec.dataset.lineage.builder import LineageBuilder
from pec.dataset.export.exporter import BundleExporter

# 1. Read and normalize input
reader = UniverseReader()
records = reader.read_csv(Path("matches_primer_filtro.csv"))
normalizer = UniverseNormalizer()
normalized, universe_manifest = normalizer.normalize(records, universe_id="my_universe")

# 2. Define policy
policy = DatasetPolicy(
    policy_id="mf_ratio_1to3_v1",
    source_universe_id="my_universe",
    ratio_policy=RatioPolicy(target_to_negative_ratio="1:3"),
    candidate_pool_policy=CandidatePoolPolicy(
        min_pool_size=5,
        scarcity_mode="drop_target",
    ),
)

# 3. Generate variant
generator = DatasetVariantGenerator()
variant = generator.generate(
    universe=normalized,
    policy=policy,
    seed=42,
    variant_id="variant_001",
)

# 4. Create splits
split_strategy = GroupByTargetSplitStrategy()
train, val, test, split_manifest = split_strategy.split(
    assignments=variant.assignments,
    seed=42,
    variant_id="variant_001",
)

# 5. Build lineage
lineage_builder = LineageBuilder()
lineage = lineage_builder.build_complete_lineage(
    source_file="matches_primer_filtro.csv",
    universe_id="my_universe",
    policy_id="mf_ratio_1to3_v1",
    variant_id="variant_001",
    split_id="split_001",
    random_seed=42,
)

# 6. Export bundle
exporter = BundleExporter()
bundle = exporter.export(
    variant=variant,
    split_artifacts={"train": train, "val": val, "test": test},
    split_manifest=split_manifest,
    universe_manifest=universe_manifest,
    policy=policy,
    lineage_manifest=lineage,
    bundle_path=Path("output/dataset_bundle"),
)
```

For complete documentation, see **[pec/README.md](pec/README.md)**.

---

## 🧪 Testing

### Pre-Embedding Layer Tests
```bash
# All pre-embedding tests
pytest tests/pec/ -v

# Specific modules
pytest tests/pec/dataset/input/ -v
pytest tests/pec/dataset/policies/ -v
pytest tests/pec/dataset/generator/ -v

# Integration tests
pytest tests/pec/dataset/test_integration.py -v

# Forward compatibility tests
pytest tests/pec/dataset/test_forward_compatibility.py -v
```

### Embedding Layer Tests
```bash
# Existing tests for the embedding-based layer
pytest tests/ -v --ignore=tests/pec/
```

### Test Results
- **Pre-embedding layer:** 73 passing, 13 skipped (forward compatibility)
- **Embedding layer:** Existing tests (status depends on configuration)

---

## 📊 Current Implementation Status

### Pre-Embedding Dataset Layer

| Component | Status | Tests |
|-----------|--------|-------|
| Input Module | ✅ Complete | 20 passing |
| Policies Module | ✅ Complete | 22 passing |
| Generator Module | ✅ Complete | Included in integration |
| Splits Module | ✅ Complete | Included in integration |
| Lineage Module | ✅ Complete | Included in integration |
| Export Module | ✅ Complete | Included in integration |
| Integration Tests | ✅ Complete | 6 passing |
| Forward Compatibility | ✅ Scaffolding | 13 skipped |

**All acceptance criteria from PEC Pre-Embedding Dataset Contract v0.1 are met.**

### Embedding-Based Classification Layer
- ✅ Existing implementation preserved
- ✅ All functionality intact
- ⚠️ Ready for integration with pre-embedding layer

---

## 🔮 Future Work

### Immediate Next Steps

1. **Embedding Loading Layer**
   - Consume dataset bundles from `pec/dataset/export/`
   - Load embeddings for protein IDs in the bundle
   - Pass embedded datasets to classifier code

2. **Integration**
   - Connect pre-embedding layer output to embedding loading input
   - Ensure provenance chain is maintained across layers
   - Create end-to-end pipeline tests

3. **Classifier Layer Updates**
   - Adapt existing classifiers to consume pre-embedding datasets
   - Maintain backward compatibility with existing workflows

### Long-Term Evolution

1. **Additional Split Strategies**
   - Organism-aware splitting
   - Stratified splitting
   - Cross-validation strategies

2. **Additional Scarcity Modes**
   - `relax_ratio` mode
   - `use_available` mode

3. **Performance Optimization**
   - Batch processing
   - Parallel variant generation
   - Memory-efficient operations

4. **Advanced Features**
   - Organism filtering and balancing
   - Custom ratio families
   - Extended metadata tracking

---

## 🤝 Contributing

### For Pre-Embedding Layer Development

1. Branch from `pre-embedding-dev`
2. Create a feature branch: `feature/pre-embedding-<description>`
3. Follow the existing architecture patterns
4. Add tests for new functionality
5. Ensure all acceptance criteria are maintained

### For Embedding Layer Development

1. Branch from `pre-embedding-dev`
2. Create a feature branch: `feature/embedding-<description>`
3. Maintain compatibility with existing code
4. Add tests for new functionality

### For Contract/Architecture Updates

1. Propose changes via PR to `master`
2. Ensure all development branches are updated to reflect changes
3. Maintain backward compatibility where possible

---

## 📄 License

This project is licensed under the terms specified in the LICENSE file (to be added).

---

## 🆘 Support

For questions about:
- **Project architecture**: Refer to [PEC Final-System Functional Vision](FUNCTIONAL_CONTRACT.md)
- **Pre-embedding layer**: Refer to [pec/README.md](pec/README.md) or [Implementation Notes](PEC-IMPLEMENTATION-NOTES.md)
- **Embedding layer**: Refer to existing documentation in `protein_embedding_classifier/`
- **Contract specifications**: Refer to [PEC Pre-Embedding Dataset Contract v0.1](pec_pre_embedding_dataset_contract_v0_1.md)

---

## 🏷️ Keywords

Protein function prediction, embedding-based classification, supervised learning, dataset preparation, reproducibility, provenance tracking, leakage prevention, ensemble methods, benchmark evaluation, bioinformatics, computational biology
