# PEC Pre-Embedding Dataset Contract v0.1

## 1. Purpose

This document defines the proposed v0.1 contract for the **pre-embedding dataset layer** of the Protein Embedding Classifier (PEC). Its purpose is to formalize the part of the system that operates **before** embedding loading, turning a target–candidate input source into reproducible, versioned, and traceable dataset variants that can later be consumed by the rest of PEC.[cite:1]

This contract is aligned with the final functional vision of PEC, which requires the system to become a configurable, problem-agnostic, and scientifically rigorous experimentation engine, with explicit configuration, dataset plurality, artifact traceability, and manifest-based orchestration.[cite:1]

## 2. Scope

This contract covers only the dataset preparation layer that sits between the raw `matches_primer_filtro.csv`-style input and the future embedding consumption layer.[cite:1] It is intentionally limited to the stages that organize targets, candidate pools, dataset variants, sampling policies, proportion studies, splits, lineage, and export bundles.[cite:1]

This contract does **not** define embedding loading, pooling, classifier training, ensemble logic, or inference behavior. Those layers belong to later PEC contracts, although this document is designed so that they can consume the resulting artifacts cleanly and without ambiguity.[cite:1]

## 3. Design principles

The contract is governed by the same principles described in the PEC functional vision: configuration-first execution, dataset plurality, artifact traceability, and support for both isolated step execution and future uninterrupted end-to-end pipelines.[cite:1]

The input must be interpreted as a reusable constrained candidate universe rather than as a fixed benchmark file. This is essential because PEC is expected to generate many alternative dataset organizations under controlled rules rather than rely on a single canonical composition.[cite:1]

## 4. Core concepts

The pre-embedding dataset layer is built around six core concepts: **Universe**, **Policy**, **Variant**, **Split**, **Lineage**, and **Bundle**. These concepts separate raw candidate availability from experimental choices, realized dataset instances, evaluation partitions, provenance, and the final export interface consumed by downstream PEC stages.[cite:1]

Each concept is a first-class artifact. No downstream stage should infer missing rules from filenames, scripts, or notebook state, because the PEC vision requires every major artifact to be self-descriptive and traceable through explicit manifests.[cite:1]

## 5. Normative decisions

The following decisions define the proposed baseline behavior for v0.1 of the contract.[cite:1]

| Decision area | Proposed v0.1 rule | Rationale |
|---|---|---|
| Input interpretation | Each row defines one `target_id` and its candidate universe | PEC must treat target–candidate input as a reusable search universe, not as a fixed dataset.[cite:1] |
| Positive unit | One positive instance per `target_id` | Keeps the initial dataset layer simple and stable before feature loading. |
| Negative unit | Each selected candidate becomes one negative instance linked to a target | Makes ratio policies explicit and target-scoped. |
| Candidate reuse within same variant | Not allowed | Reduces ambiguity and prevents artificial inflation of negatives. |
| Scarcity handling | `drop_target` in v0.1 | Gives deterministic behavior and avoids implicit ratio relaxation. |
| Initial ratio families | `1:1`, `1:3`, `1:5` | Matches the need to make class-proportion experimentation a first-class axis.[cite:1] |
| Variant multiplicity | 25 variants per ratio policy by default | Enough to test randomized organization behavior without exploding complexity. |
| Seed policy | One global seed per variant | Simpler provenance and easier replay. |
| Split strategy | Group by `target_id` | Protects against leakage across linked positive/negative instances. |
| Manifest format | JSON | Machine-friendly, explicit, and easy to validate. |
| Lineage requirement | Mandatory | PEC must consume datasets through self-descriptive contracts.[cite:1] |

## 6. Contract overview

The full contract is divided into six subcontracts that form a strict chain:[cite:1]

1. **Input contract**
2. **Dataset policy contract**
3. **Dataset generation contract**
4. **Split contract**
5. **Lineage contract**
6. **Export bundle contract**

Each subcontract must consume explicit inputs and produce explicit outputs. This follows the PEC requirement that steps remain executable both independently and as part of an eventual continuous end-to-end pipeline.[cite:1]

## 7. Input contract

### 7.1 Purpose

The input contract defines how a `matches_primer_filtro.csv`-style file is parsed and normalized into an internal representation of target–candidate universes.[cite:1] The purpose of this stage is not to produce a trainable dataset, but to formalize the admissible search space from which later dataset variants will be generated.[cite:1]

### 7.2 Canonical internal entity

The canonical internal entity is `UniverseRecord`.

```json
{
  "target_id": "string",
  "target_label": "positive",
  "candidate_ids": ["string"],
  "candidate_count": 0,
  "source_file": "string",
  "source_row_id": "string_or_integer",
  "organism": "string_or_null",
  "taxonomy_id": "string_or_null",
  "pool_metadata": {
    "generation_source": "matches_primer_filtro",
    "constraints_snapshot": {
      "len_variance": "float_or_null",
      "max_sequence_identity": "float_or_null",
      "min_candidates": "integer_or_null"
    }
  }
}
```

### 7.3 Normative requirements

- The system **must** normalize every input row into exactly one `UniverseRecord`.
- The system **must** guarantee that `target_id` is unique within a normalized universe.
- The system **must** parse `candidate_ids` deterministically.
- The system **must not** sample, rank, or discard candidates at this stage.
- The system **should** preserve any organism or taxonomy metadata available in the source file.
- The system **may** retain raw source row payloads for debugging and audit.

### 7.4 Outputs

This stage outputs:
- `target_candidate_universe.jsonl`
- `universe_manifest.json`

## 8. Dataset policy contract

### 8.1 Purpose

The dataset policy contract defines the explicit rules that transform a target–candidate universe into a family of dataset variants.[cite:1] This is the main configuration object of the pre-embedding layer and must capture all decisions that could affect composition, reproducibility, or scientific interpretation.[cite:1]

### 8.2 Canonical policy schema

```json
{
  "policy_id": "mf_ratio_1to3_seedset_v1",
  "source_universe_id": "matches_primer_filtro_2026_07",
  "selection_strategy": {
    "mode": "sample_without_replacement",
    "candidate_scope": "per_target",
    "assignment_strategy": "global_unique_candidates"
  },
  "ratio_policy": {
    "positive_unit": "target",
    "negative_unit": "candidate_assignment",
    "target_to_negative_ratio": "1:3"
  },
  "candidate_pool_policy": {
    "min_pool_size": 5,
    "max_pool_size": null,
    "scarcity_mode": "drop_target"
  },
  "randomization": {
    "enabled": true,
    "seed_scope": "global"
  },
  "split_policy_ref": "group_by_target_v1",
  "organism_policy": {
    "mode": "preserve_source"
  },
  "duplicate_policy": {
    "allow_same_candidate_across_targets": false,
    "allow_same_target_across_variants": true
  }
}
```

### 8.3 Normative requirements

- Every dataset family **must** be defined through a policy artifact.
- A policy **must** declare ratio behavior, scarcity handling, randomization behavior, and duplicate behavior.
- A policy **must not** rely on hidden defaults that materially affect sampling outcomes.
- A policy **should** be validatable against a JSON schema.
- A policy **may** define organism-aware filtering or balancing behavior in future versions.

## 9. Dataset generation contract

### 9.1 Purpose

The dataset generation contract defines how one concrete dataset variant is created from one normalized universe, one dataset policy, and one explicit seed.[cite:1] This stage operationalizes the PEC requirement for randomized dataset organizations under controlled and reproducible constraints.[cite:1]

### 9.2 Canonical variant manifest

```json
{
  "variant_id": "variant_000173",
  "policy_id": "mf_ratio_1to3_seedset_v1",
  "source_universe_id": "matches_primer_filtro_2026_07",
  "seed_used": 42,
  "targets_included": 0,
  "targets_dropped": 0,
  "total_positive_instances": 0,
  "total_negative_instances": 0,
  "assignment_mode": "global_unique_candidates",
  "scarcity_events": [],
  "dataset_statistics": {
    "ratio_realized": "1:3",
    "organism_distribution": {},
    "candidate_pool_size_distribution": {}
  }
}
```

### 9.3 Canonical assignments table

| Column | Meaning |
|---|---|
| `target_id` | MF target that anchors the local assignment |
| `protein_id` | Protein accession of the realized instance |
| `role` | `positive` or `negative` |
| `paired_target_id` | Target to which the instance belongs |
| `variant_id` | Dataset variant identity |

### 9.4 Normative requirements

- One variant **must** be reconstructible from `source_universe + policy + seed`.
- Variant generation **must** produce both realized instances and a machine-readable variant manifest.
- The generator **must** record dropped targets and scarcity events.
- The generator **must not** silently relax ratio rules in v0.1.
- The generator **must** forbid candidate reuse within the same variant.
- The generator **should** expose deterministic replay for any `variant_id`.

## 10. Split contract

### 10.1 Purpose

The split contract defines how a realized dataset variant is partitioned into train, validation, and test artifacts.[cite:1] The split policy is separate from dataset generation because PEC requires explicit split strategies in experiment manifests and traceable continuity between dataset construction and downstream experimentation.[cite:1]

### 10.2 Canonical split manifest

```json
{
  "split_id": "split_group_seed42_foldsetA",
  "variant_id": "variant_000173",
  "split_strategy": {
    "type": "group_by_target",
    "group_key": "target_id",
    "stratify_by": "role",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15
  },
  "random_seed": 42,
  "leakage_guards": {
    "keep_same_target_in_one_split": true,
    "keep_linked_instances_together": true
  }
}
```

### 10.3 Normative requirements

- A split **must** be generated from an explicit split strategy artifact or policy reference.
- Positive and negative instances tied to the same `target_id` **must** remain within the same split in v0.1.
- The split stage **must** emit train, validation, and test artifacts plus a split manifest.
- The split stage **should** support future organism-aware strategies.
- The split stage **must not** introduce target leakage across partitions.

## 11. Lineage contract

### 11.1 Purpose

The lineage contract captures the complete provenance chain of the dataset artifact, from source universe to policy application and split generation.[cite:1] This is mandatory because the PEC vision requires every dataset variant to carry explicit metadata on source target list, candidate pool definition, selection rules, random seeds, split policy, organism distribution, ratio policy, and upstream version identifiers.[cite:1]

### 11.2 Canonical lineage manifest

```json
{
  "lineage_id": "lineage_variant_000173",
  "source_artifacts": [
    {
      "artifact_id": "matches_primer_filtro_2026_07",
      "artifact_type": "target_candidate_universe",
      "source_path": "data/raw/matches_primer_filtro.csv",
      "source_version": "sha256:..."
    }
  ],
  "transforms": [
    {
      "step": "normalize_input",
      "code_version": "git:abc123"
    },
    {
      "step": "apply_dataset_policy",
      "policy_id": "mf_ratio_1to3_seedset_v1"
    },
    {
      "step": "generate_split",
      "split_id": "split_group_seed42_foldsetA"
    }
  ],
  "runtime": {
    "generated_at": "2026-07-31T00:00:00Z",
    "generated_by": "pec.dataset_generator",
    "random_seed": 42
  }
}
```

### 11.3 Normative requirements

- Every dataset variant **must** have one lineage manifest.
- A lineage manifest **must** reference the exact source artifact identity.
- A lineage manifest **must** record policy identity, split identity, code version, and seed.
- A dataset artifact **must not** be considered valid for downstream PEC consumption if lineage is missing.
- The lineage system **should** support hash-based source and output identification.

## 12. Export bundle contract

### 12.1 Purpose

The export bundle contract defines the exact interface that the next PEC layer will consume.[cite:1] The bundle must be self-descriptive so that embedding-loading and later experimentation layers do not need to reconstruct hidden assumptions about dataset construction.[cite:1]

### 12.2 Canonical bundle layout

```text
dataset_bundle/
├── dataset_instances.csv
├── assignments.csv
├── split/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── split_manifest.json
├── manifests/
│   ├── universe_manifest.json
│   ├── dataset_policy.json
│   ├── variant_manifest.json
│   └── lineage.json
└── reports/
    └── dataset_summary.json
```

### 12.3 Normative requirements

- Every exported dataset **must** be packaged as a self-contained bundle.
- The bundle **must** include the realized instances, split artifacts, and all relevant manifests.
- The bundle **should** include a machine-readable summary report.
- The bundle **must** be consumable without requiring notebook state or ad hoc path assumptions.

## 13. Module design proposal

To keep the repository aligned with step-based execution and future orchestration, the implementation should be organized into the following internal modules:[cite:1]

| Module | Responsibility |
|---|---|
| `pec/dataset/input/` | Read, validate, and normalize `matches_primer_filtro.csv`-style inputs |
| `pec/dataset/policies/` | Define and validate dataset generation policies |
| `pec/dataset/generator/` | Build concrete dataset variants from universes and policies |
| `pec/dataset/splits/` | Create split artifacts with leakage guards |
| `pec/dataset/lineage/` | Build provenance manifests and artifact identities |
| `pec/dataset/export/` | Emit self-contained dataset bundles for downstream PEC use |

This separation preserves the PEC requirement that steps be executable independently while still supporting future continuous end-to-end orchestration around experiment manifests.[cite:1]

## 14. Mandatory open questions before implementation

The following design questions should be explicitly confirmed before the first formal prompt is written for implementation:

1. Whether one positive instance per `target_id` is sufficient for v0.1.
2. Whether negative instances must always remain target-linked in downstream tables.
3. Whether candidate reuse across targets should remain globally forbidden in all policies.
4. Whether `drop_target` is the definitive v0.1 scarcity behavior.
5. Whether ratio families `1:1`, `1:3`, and `1:5` are the official initial benchmark set.
6. Whether 25 variants per ratio policy is the correct initial breadth.
7. Whether the seed model should remain one global seed per variant.
8. Whether `group_by_target` is the right first split policy.
9. Whether JSON alone is sufficient for manifests.
10. Whether artifact identity should be hash-based, timestamp-based, or hybrid.

## 15. Initial v0.1 recommendation

The recommended initial configuration for implementation is the following:[cite:1]

- Canonical input abstraction: `target_id + candidate_ids`.
- Positive unit: one instance per target.
- Negative unit: one selected candidate per negative instance.
- Candidate reuse inside one variant: forbidden.
- Scarcity mode: `drop_target`.
- Initial ratio families: `1:1`, `1:3`, `1:5`.
- Default number of variants per ratio: `25`.
- Seed behavior: one global seed per variant.
- Initial split strategy: `group_by_target`.
- Manifest format: JSON.
- Lineage: mandatory.
- Export bundle: mandatory.

## 16. Acceptance criteria for v0.1

The pre-embedding dataset layer should be considered ready for the next PEC stage only if all of the following are true:

- A raw target–candidate file can be normalized into a stable universe artifact.
- A dataset policy can be declared explicitly and validated.
- Multiple variants can be generated from the same universe under different seeds and ratio policies.[cite:1]
- Each variant has explicit manifests, deterministic replay, and a complete lineage record.[cite:1]
- Leakage-safe splits can be generated independently from variant construction.[cite:1]
- The final output is a self-contained dataset bundle consumable by later PEC stages without hidden assumptions.[cite:1]

