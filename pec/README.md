# PEC Pre-Embedding Dataset Layer

This directory contains the **pre-embedding dataset layer** implementation for the Protein Embedding Classifier (PEC) project.

## Overview

The pre-embedding dataset layer operates **before** embedding loading and is responsible for:

1. **Input Processing**: Reading and normalizing `matches_primer_filtro.csv`-style target-candidate universe files
2. **Policy Definition**: Defining explicit dataset generation policies (ratio, scarcity, randomization)
3. **Variant Generation**: Creating reproducible dataset variants from universes and policies
4. **Splitting**: Generating leakage-safe train/val/test partitions
5. **Lineage Tracking**: Recording complete provenance chains for all artifacts
6. **Bundle Export**: Packaging datasets as self-contained bundles for downstream PEC stages

## Contract Compliance

This implementation follows the **PEC Pre-Embedding Dataset Contract v0.1** specification. All normative requirements from the contract are implemented:

- ✅ Input contract (§7): UniverseRecord normalization, deterministic parsing
- ✅ Dataset policy contract (§8): Explicit policy schema, validation
- ✅ Dataset generation contract (§9): Variant generation, scarcity handling, ratio enforcement
- ✅ Split contract (§10): Group-by-target splitting, leakage guards
- ✅ Lineage contract (§11): Complete provenance manifests
- ✅ Export bundle contract (§12): Self-contained bundle structure

## Module Structure

```
pec/
├── __init__.py                    # Package initialization
├── README.md                      # This file
└── dataset/
    ├── __init__.py                # Dataset layer exports
    ├── contracts.py              # Abstract base classes and protocols
    ├── input/
    │   ├── __init__.py           # Input module exports
    │   ├── models.py             # UniverseRecord, UniverseManifest
    │   ├── reader.py             # UniverseReader (CSV/JSONL)
    │   └── normalizer.py         # UniverseNormalizer
    ├── policies/
    │   ├── __init__.py           # Policies module exports
    │   ├── models.py             # DatasetPolicy and sub-models
    │   └── validator.py           # PolicyValidator
    ├── generator/
    │   ├── __init__.py           # Generator module exports
    │   ├── models.py             # DatasetVariant, AssignmentRecord, VariantManifest
    │   └── generator.py          # DatasetVariantGenerator
    ├── splits/
    │   ├── __init__.py           # Splits module exports
    │   ├── models.py             # SplitManifest, SplitStrategyConfig, etc.
    │   └── strategies.py         # SplitStrategy, GroupByTargetSplitStrategy
    ├── lineage/
    │   ├── __init__.py           # Lineage module exports
    │   ├── models.py             # LineageManifest, SourceArtifact, etc.
    │   └── builder.py            # LineageBuilder
    └── export/
        ├── __init__.py           # Export module exports
        ├── models.py             # DatasetBundle, DatasetSummary
        └── exporter.py           # BundleExporter
```

## Quick Start

### Basic Usage

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

### Using Multiple Ratio Families

```python
from pec.dataset.policies.models import DatasetPolicy

# Generate variants for all standard ratio families
ratio_families = ["1:1", "1:3", "1:5"]
all_variants = []

for ratio in ratio_families:
    policy = DatasetPolicy(
        policy_id=f"ratio_{ratio.replace(':', 'to')}",
        source_universe_id="my_universe",
        ratio_policy=RatioPolicy(target_to_negative_ratio=ratio),
        candidate_pool_policy=CandidatePoolPolicy(
            min_pool_size=5,
            scarcity_mode="drop_target",
        ),
    )
    
    # Generate 25 variants for this ratio (as per contract §15)
    variants = generator.generate_multiple(
        universe=normalized,
        policy=policy,
        num_variants=25,
    )
    all_variants.extend(variants)
```

## Key Features

### 1. Input Processing

- **CSV Support**: Reads `matches_primer_filtro.csv`-style files
- **JSONL Support**: Reads and writes JSONL format
- **Normalization**: Ensures unique target_ids, validates structure
- **Metadata Preservation**: Preserves organism, taxonomy, and custom metadata

### 2. Policy System

- **Explicit Configuration**: All dataset generation rules are explicit
- **Validation**: Policies are validated against schema
- **Ratio Families**: Supports `1:1`, `1:3`, `1:5` (contract §15)
- **Scarcity Handling**: `drop_target` mode (contract §15)
- **Randomization**: Global seed per variant (contract §15)

### 3. Variant Generation

- **Deterministic**: Same seed + universe + policy = same variant
- **No Candidate Reuse**: Candidates not reused within same variant
- **Scarcity Events**: Records targets dropped due to insufficient candidates
- **Statistics**: Tracks ratio realization, organism distribution, pool sizes

### 4. Splitting

- **Group by Target**: All instances for same target in same partition
- **Leakage Guards**: Prevents target leakage across partitions
- **Configurable Ratios**: Train/val/test ratios are configurable
- **Reproducible**: Same seed produces same split

### 5. Lineage Tracking

- **Complete Provenance**: Tracks from source file to final bundle
- **Hash-Based**: Supports hash-based artifact identification
- **Transform Steps**: Records each transformation step
- **Runtime Info**: Captures generation timestamp, code version, seed

### 6. Bundle Export

- **Self-Contained**: All artifacts included in bundle
- **Standard Structure**: Follows contract §12.2 layout
- **Manifests**: Includes all manifests (universe, policy, variant, split, lineage)
- **Reports**: Includes dataset summary report

## Configuration Options

### DatasetPolicy

```python
policy = DatasetPolicy(
    policy_id="my_policy",
    source_universe_id="my_universe",
    selection_strategy=SelectionStrategy(
        mode="sample_without_replacement",  # or "sample_with_replacement", "use_all"
        candidate_scope="per_target",        # or "global"
        assignment_strategy="global_unique_candidates",
    ),
    ratio_policy=RatioPolicy(
        positive_unit="target",
        negative_unit="candidate_assignment",
        target_to_negative_ratio="1:3",
    ),
    candidate_pool_policy=CandidatePoolPolicy(
        min_pool_size=5,
        max_pool_size=None,  # or specific number
        scarcity_mode="drop_target",  # or "relax_ratio", "use_available"
    ),
    randomization=RandomizationConfig(
        enabled=True,
        seed_scope="global",  # or "per_target"
    ),
    split_policy_ref="group_by_target_v1",
    organism_policy=OrganismPolicy(
        mode="preserve_source",  # or "filter_by_organism", "balance_by_organism"
    ),
    duplicate_policy=DuplicatePolicy(
        allow_same_candidate_across_targets=False,
        allow_same_target_across_variants=True,
    ),
)
```

## Testing

The pre-embedding dataset layer includes comprehensive tests:

- **Unit Tests**: 66 passing tests for all modules
- **Integration Tests**: Complete workflow tests
- **Forward Compatibility Tests**: Scaffolding for future modules

Run tests with:

```bash
# All tests
pytest tests/pec/

# Specific module
pytest tests/pec/dataset/input/

# With coverage
pytest tests/pec/ --cov=pec --cov-report=html
```

## Future Extensions

The architecture is designed for future extension:

### 1. Embedding Loading
- Implement `EmbeddingLoaderProtocol` from `pec.dataset.contracts`
- Consume dataset bundles from this layer
- Maintain provenance chain

### 2. Classifier Training
- Implement `ClassifierProtocol` from `pec.dataset.contracts`
- Use dataset bundles as input
- Extend lineage with training information

### 3. Aggregation
- Implement `AggregatorProtocol` from `pec.dataset.contracts`
- Combine predictions from multiple classifiers
- Maintain decomposability and traceability

### 4. Pipeline Orchestration
- Implement `PECStep` abstract base class
- Support isolated and chained execution
- Maintain artifact passing between steps

## Contract References

- **PEC Pre-Embedding Dataset Contract v0.1**: `../pec_pre_embedding_dataset_contract_v0_1.md`
- **PEC Final-System Functional Vision**: `../FUNCTIONAL_CONTRACT.md`

## License

This code is part of the Protein Embedding Classifier project and is licensed under the same terms as the main project.
