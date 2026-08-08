# Functional Contract Document
## protein-embedding-classifier

## Introduction
This document specifies the implemented functional contract of the repository `protein-embedding-classifier` as an engineering system specification for ML engineers, bioinformatics researchers, and scientific reviewers.

**Mandatory global statement**:

**This system does NOT generate embeddings. It strictly consumes precomputed embeddings as input features. It assumes embeddings are externally generated.**

The contract below is implementation-grounded (code and tests), not aspirational.

---

# 1. System Scope and Responsibilities

## 1.1 In-Scope Responsibilities
The system is responsible for:
- Loading accession-level proteins and labels from configured sources (DB/CSV).
- Building train/validation/test and optional zero-shot partitions.
- Loading precomputed embedding vectors (sequence and GO-derived) and aligning them to split accessions.
- Training supervised classifiers over fixed embedding vectors.
- Running randomized hyperparameter sweeps per classifier and embedding view.
- Persisting model artifacts and run metadata.
- Running ensemble inference from persisted base-model artifacts.
- Running benchmark comparisons (best single vs ensemble variants), including multi-seed aggregation.

## 1.2 Explicitly External / Out-of-Scope
The system does **not** implement:
- Embedding generation pipelines (sequence model inference, GO embedding training, structure embedding generation).
- Biological annotation curation or ontology construction.
- Dataset authoring policy (except split application and alignment checks).
- Continuous/online model updates.
- Active learning loops.

## 1.3 Operational Boundaries
- Inputs are accession-indexed and must be joinable across label and embedding sources.
- Embeddings are treated as immutable feature vectors at training/inference time.
- Output artifacts are filesystem-based (`pec_data` layout for sweep/benchmark/reporting).
- Orchestration is step-based CLI (`dataset`, `embeddings`, `train`, `sweep`, `ensemble`, `benchmark`, `evaluate`).

---

# 2. End-to-End Pipeline Specification

The implemented logical pipeline is below.

| Stage | Inputs | Outputs | Determinism / Randomness | Failure Modes |
|---|---|---|---|---|
| 1) Dataset loading | `dataset` config, DB engine, protein query, label source/query/file | `DatasetBundle` (`train_ids`, `val_ids`, `test_ids`, optional `zero_shot_ids`, labels arrays) | Deterministic if source data and split seeds fixed | Missing DB config/query, invalid label source, malformed CSV/SQL, empty/invalid split assignment, split coverage mismatch |
| 2) Embedding loading and validation | `embeddings.yaml`, ordered accessions from dataset bundle | `EmbeddingBundle` with per-view `X_train/X_val/X_test` (+ optional zero-shot matrices) | Deterministic for fixed input files/DB and aggregation mode | Missing embedding records (unless explicitly tolerated), malformed vectors, unknown aggregation mode, missing GO columns |
| 3) Label parsing (single-label / multilabel) | Raw label values from `LabelLoader`; training labels in `ProblemSpecification` | Canonical task type (`binary`/`multiclass`/`multilabel`), class set, encoded labels for training | Deterministic | Empty labels for task inference; incompatible shapes in multilabel transformations |
| 4) Split orchestration | Split config (`validation`, `train_test`, `zero_shot` strategies), accession metadata (organism) | Disjoint train/validation/test and optional zero-shot partitions | Randomized for random/cross-validation strategies using configured seeds; deterministic for CSV-defined splits | Duplicate accession split assignment in CSV, unknown strategy, leakage overlap, partition coverage mismatch |
| 5) Training stage | `EmbeddingBundle`, `train` config, model params | Fitted models per `(classifier, embedding)`, validation probabilities, metrics | Deterministic for deterministic backends + fixed seeds; stochastic for some models/hardware (not globally forced) | Unsupported classifier, missing optional deps (`xgboost`, `torch`), invalid normalization mode, model without `predict_proba` |
| 6) Validation scoring | Validation labels and canonical probabilities | Validation metrics used for model ranking/selection | Deterministic for fixed outputs | Probability shape/range contract violations, class mismatch, metric computation errors |
| 7) Ensemble decision logic | Persisted model artifacts + metadata, validation/test matrices | Ensemble probabilities, labels, ensemble metadata, optional ensemble artifact | Deterministic for uniform/validation-score strategies; trainable weights add seeded random search | Missing run/artifacts/metadata, inconsistent class order/problem type across models, `<2` models, unsupported weighting mode |
| 8) Test evaluation | Optional in training/final training, or evaluate-last-sweep confusion matrix step | Test metrics and optional confusion matrix CSVs | Deterministic given fixed model and test set | Missing model artifacts, unsupported serializer, confusion-matrix skipped for multilabel |
| 9) Zero-shot evaluation | Zero-shot IDs and matrices from split stage, benchmark runner | Zero-shot metrics per variant (if non-empty split) | Deterministic for fixed model/split | Zero-shot split absent/too small warnings, leakage checks fail if overlap detected |
| 10) Benchmark aggregation | Per-seed benchmark rows, variant set, ablation selections | Aggregated mean/std summaries, ranking, deltas vs best single | Seed-dependent; aggregation via `nanmean/nanstd` | All seeds/ablations fail; invalid model selection causing insufficient models |
| 11) Artifact persistence | Run context, configs, fitted models, metrics | CSV/JSON/YAML/PKL/PT artifacts (reports, configs, predictions, model files) | Deterministic serialization for fixed payload | Filesystem errors, serializer incompatibility, unpicklable models |

## Stage Ordering Notes
- In independent split mode, order is: **validation selection → train/test split on remaining IDs → zero-shot selection and removal from train/val/test**.
- Ensemble weight fitting uses **validation only**.
- Benchmark compares all enabled variants to best single and computes deltas on test/zero-shot.

---

# 3. Embedding Modalities (Feature Types)

## Contract-Wide Rule
- Embeddings are consumed as fixed vectors.
- No gradient updates are applied to embedding generators.
- No embedding fine-tuning stage exists.

## 3.1 ESM3c
- **Config key**: `sequencePE.models.ESM3c`.
- **Source**: DB table join over `sequence_embeddings` + accession mapping.
- **Expected format**: vector payload parseable by `coerce_embedding_vector` (list/tuple, bytes buffer, or stringified array).
- **Dimensionality**: not hard-coded; inferred from loaded vectors.
- **Layer handling**:
  - `aggregation.mode = none` with multiple layers creates views like `ESM3c__layer_<idx>`.
  - `mean`, `max`, `mean_max`, `concat` aggregate layer vectors per accession.
- **Accession alignment**: matrices are built in split accession order from `DatasetBundle`.
- **Normalization/scaling**: not in loader; applied later in training (`none`/`l2`/`standard`).
- **Memory constraints**: full split matrices are materialized in memory for each view.

## 3.2 ProtT5
- **Naming note (implemented)**:
  - Sequence model key present in config: `Prot-T5`.
  - Separate key also present: `Prost-T5` (enabled in current config sample).
  - Training embedding group token `ProtT5` is a selection alias token, not a loader key.
- **Source/format/alignment/normalization**: same sequence-embedding contract as ESM3c.
- **Dimensionality**: not fixed in code; inferred at runtime from vectors.
- **Memory constraints**: same as other sequence views.

## 3.3 Ankh3
- **Implemented key**: `Ankh3-Large` under `sequencePE.models`.
- **Source**: DB sequence embedding loading by model name and layer index.
- **Expected format**: DB embedding field parseable to 1D float vector.
- **Dimensionality**: runtime-inferred, not schema-fixed.
- **Alignment/normalization/scaling**: same contract as ESM3c.
- **Memory constraints**: per-split dense matrices held in memory.

## 3.4 GeOKG
- **Config key**: `GOPE.models.GeOKG`.
- **Source**: CSV files (`BP`, `MF`, `CC`) under configured folder.
- **Expected format**:
  - Must contain configured accession column and embedding column.
  - Embedding values parseable via `coerce_embedding_vector`.
- **Loading behavior**:
  - Reads configured ontology files.
  - Filters by accessions in current dataset.
  - Builds per-ontology vectors; missing ontology vector for an accession is zero-padded at that ontology dimension.
  - Concatenates ontology vectors into one `GeOKG` vector per accession.
- **Dimensionality**: sum of loaded ontology vector lengths; not globally fixed in code.
- **Alignment**: same split-order matrix construction.
- **Normalization/scaling**: handled only in training stage.
- **Missing embeddings handling**: can be explicitly tolerated (`missing_embeddings.allow_models.GeOKG`) and reported to artifact file.

## 3.5 Loader-Level Invariants
- By default, any missing embedding in a split raises `ValueError`.
- If a model is listed in `missing_embeddings.allow_models`, missing entries are zero-filled and missing accession list is written.

---

# 4. Classifier Capabilities

## 4.1 Supported Classifiers in Active Pipeline (`ModelFactory`)
- `LR` → `sklearn.linear_model.LogisticRegression`
- `SVM` → `sklearn.svm.SVC` (`probability=True` forced)
- `RF` → `sklearn.ensemble.RandomForestClassifier`
- `KNN-2` → `sklearn.neighbors.KNeighborsClassifier`
- `XGB` → `xgboost.XGBClassifier` (optional dependency)
- `MLP` → internal `TorchTrainingWrapper` + `MLPProteinClassifier`

## 4.2 Multilabel Handling Strategy
- Non-MLP models: wrapped with `OneVsRestClassifier` when `problem_type=multilabel`.
- MLP: trained directly on multilabel binary matrices with `BCEWithLogitsLoss`.

## 4.3 Mathematical / Objective Contract by Classifier

### Logistic Regression (`LR`)
- **Logic**: linear logits with logistic link.
- **Objective**: scikit-learn logistic loss (binary or multinomial depending class count/solver).
- **Regularization**: L2 by default (unless overridden in params).
- **Hyperparameters**: from YAML trial params or `training_config.model_params.LR`.
- **Strengths**: fast, interpretable linear baseline.
- **Limitations**: linear decision boundary; sensitive to feature scaling and separability.

### SVM (`SVM`)
- **Logic**: maximum-margin classifier with configured kernel.
- **Objective**: hinge-style margin optimization (probabilities produced via `SVC(probability=True)`).
- **Regularization**: margin parameter `C`.
- **Hyperparameters**: kernel family/options, `C`, `tol`, class weight, etc.
- **Strengths**: effective on high-dimensional features.
- **Limitations**: probability calibration depends on SVC internals; can be slow on large datasets.

### Random Forest (`RF`)
- **Logic**: bagged decision trees with feature subsampling.
- **Objective**: impurity reduction (`gini`, `entropy`, `log_loss`).
- **Regularization controls**: depth/split/leaf constraints, bootstrap sampling.
- **Strengths**: non-linear decision surfaces, robust baseline.
- **Limitations**: large memory footprint, less calibrated probabilities.

### XGBoost (`XGB`)
- **Logic**: gradient-boosted trees.
- **Objective**: XGBoost default objective unless overridden.
- **Regularization controls**: depth, child weight, subsampling, learning rate, etc.
- **Strengths**: strong tabular performance.
- **Limitations**: optional dependency; label encoding path required for non-multilabel string labels.

### KNN (`KNN-2`)
- **Logic**: distance-based neighborhood voting.
- **Objective**: non-parametric instance-based prediction.
- **Regularization controls**: neighborhood size and distance metric.
- **Strengths**: simple local baseline.
- **Limitations**: expensive at inference for large datasets; sensitive to scaling/metric choice.

### MLP (`MLP`)
- **Logic**: feed-forward neural network (`MLPProteinClassifier`) with configurable hidden topology.
- **Objective**:
  - `CrossEntropyLoss` for multiclass.
  - `BCEWithLogitsLoss` for binary/multilabel paths.
- **Regularization controls**: dropout, optional batch norm, early stopping patience.
- **Optimization**: Adam/RMSprop/SGD/Adagrad.
- **Strengths**: non-linear function approximation.
- **Limitations**: dependency on torch availability and stochastic training behavior.

## 4.4 Hyperparameter Passing from YAML
- Sweep path:
  1. Classifier-specific sweep YAML loaded by `run_sweep_step`.
  2. `SweepService` samples one trial config per trial.
  3. `TrainingService` receives sampled config via `wandb_config` when `sweep_mode=True`.
  4. `ModelFactory` flattens nested helper groups (`kernel_config`, `bootstrap_config`, `p_metric`, `custom_layer_config`).
- Non-sweep training path:
  - Uses `training_config.model_params[MODEL_TYPE]` directly.

## 4.5 Default Parameters
- For `LR`, `SVM`, `RF`, `KNN-2`, `XGB`: library defaults apply unless overridden.
- `SVM`: `probability=True` is always forced.
- `MLP` defaults are defined in `TorchTrainingWrapper` constructor (e.g., `num_hidden_layers=2`, `dropout_rate=0.1`, `learning_rate=1e-3`, `num_epochs=100`, `batch_size=64`).

---

# 5. Sweep System Specification

## 5.1 What a Sweep Is in This Repository
A sweep is a repeated trial process per classifier where each trial samples one hyperparameter configuration from YAML-defined distributions and trains/evaluates that configuration across selected embedding views.

## 5.2 Iteration Structure
Effective execution structure is:
- Selected classifiers × selected embeddings × sampled trial configurations.

Important implementation detail:
- Hyperparameters are **sampled randomly** (uniform/int/log-uniform or categorical choices).
- This is **not** an exhaustive grid and not a Cartesian enumeration of full hyperparameter space.

## 5.3 Storage of Results
Per sweep run:
- `reports/best_config_by_classifier.yaml`
- `reports/sweep_results_full.csv`
- `reports/best_per_classifier.csv`
- `reports/best_classifier_per_embedding.csv`
- optional `reports/final_test_results.csv`
- `configs/resolved_pipeline.yaml`, `configs/resolved_training.yaml`, `configs/run_metadata.json`
- `predictions/predictions_test.csv`
- model artifacts under `models/` when final training saves models.

## 5.4 Seed Handling
- `SweepService` sampling RNG defaults to `42` and is not configured from pipeline YAML in current wiring.
- Trainable ensemble weighting uses configurable/random-seeded trainer params (`random_seed`, `n_trials`).
- Benchmark uses `benchmark.seeds` (default `[42]`) for repeated evaluations.

## 5.5 Artifact Reuse
- Ensemble and benchmark steps read persisted model artifacts + metadata from latest sweep run (`best_classifier_per_embedding.csv` + model files).
- They do not retrain base classifiers.

## 5.6 Failure Handling
- Missing optional dependencies per classifier sweep (`ImportError`, e.g., missing `torch`/`xgboost`) are logged and that classifier is skipped.
- If all selected classifiers are skipped, sweep exits with warning and no benchmarkable results.
- Per-trial internal failures are not isolated inside `SweepService`; unhandled exceptions can abort the classifier sweep.

## 5.7 Tunable Hyperparameters by Classifier (Configured Search Spaces)

| Classifier | Search-Space Source | Tunable Parameters |
|---|---|---|
| `LR` | `sweep_config_lr.yaml` | `solver`, `C`, `tol`, `fit_intercept`, `class_weight`, `max_iter` |
| `SVM` | `sweep_config_svm.yaml` | `kernel_config` (linear/rbf/poly/sigmoid variants), `C`, `shrinking`, `tol`, `cache_size`, `class_weight`, `max_iter` |
| `RF` | `sweep_config_rf.yaml` | `n_estimators`, `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`, `bootstrap_config` |
| `KNN-2` | `sweep_config_knn.yaml` | `n_neighbors`, `weights`, `p_metric` (metric/p combinations) |
| `XGB` | `sweep_config_xgb.yaml` | `learning_rate`, `n_estimators`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree` |
| `MLP` | `sweep_config_mlp.yaml` | architecture params (`hidden_layers_mode`, `num_hidden_layers`, `custom_layer_config`, `dropout_rate`, activations, batch norm, initialization) |

**Additional MLP training sweep file exists** (`sweep_config_mlp_training.yaml`) with optimizer/loss/training params, but default pipeline mapping for `MLP` points to `sweep_config_mlp.yaml` unless overridden in `sweep.config_paths`.

---

# 6. Split Strategy System

## 6.1 Implemented Strategy Families
### `validation.strategy`
- `random`
- `organism`
- `csv`

### `train_test.strategy`
- `random`
- `cross_validation`

### `zero_shot.strategy`
- `random`
- `organism`
- `csv`

## 6.2 Order of Application (Independent Split Manager)
1. Select validation IDs from full accession set.
2. Split remaining IDs into train/test.
3. Select zero-shot IDs from full accession set.
4. Remove zero-shot IDs from train/validation/test.
5. Enforce overlap and coverage invariants.

## 6.3 Leakage Safeguards
Enforced checks include:
- No intersection between zero-shot and train/validation/test.
- CSV duplicate accession assignment across split labels raises error.
- Coverage check ensures every aligned accession belongs to exactly one of train/val/test/zero-shot.

## 6.4 Invariants
- `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅` after split logic.
- `zero_shot ∩ train = ∅`, `zero_shot ∩ val = ∅`, `zero_shot ∩ test = ∅`.
- Zero-shot labels are never used for model fitting or ensemble weighting.

## 6.5 Legacy Split Path
If `split` config does not contain both `validation` and `train_test`, pipeline falls back to legacy single-strategy constructors (`cross_validation`, `zero_shot_csv`, `zero_shot_organism`, `zero_shot_random`).

---

# 7. Ensemble System Specification

## 7.1 Implemented Ensemble Variants
1. **Best single** (benchmark baseline, not a weight-learning ensemble).
2. **Uniform soft voting**.
3. **Validation-weighted soft voting**.
4. **Trainable weight soft voting**.
5. **Majority voting variants**: `majority_global`, `majority_by_embedding`, `majority_by_classifier`.

## 7.2 Input Contract
- Requires persisted base model artifacts with metadata containing:
  - `problem_type`, `classes`, `num_classes`, `normalization`, `threshold_policy`, classifier and embedding names.
- At least 2 models required after filtering/loading.
- All selected models must share same `problem_type`, class ordering, and class count.

## 7.3 Probability Combination Logic
For soft voting:
\[
\hat{P}(y\mid x) = \sum_{i=1}^{M} w_i P_i(y\mid x), \quad w_i \ge 0, \quad \sum_i w_i = 1
\]

### Weight computation
- **Uniform**: \(w_i = 1/M\).
- **Validation-score-based**: per-model macro-F1 on validation converted to non-negative normalized weights.
- **Trainable**: validation-time random Dirichlet search maximizing benchmark macro-F1 (optional L2 penalty term).

## 7.4 Where Weights Are Learned
- Learned exclusively from validation probabilities/labels via `fit_with_validation`.
- Base models are never retrained inside ensemble service.

## 7.5 Majority Voting Logic
- Base model probabilities are converted to labels with decision policy.
- Majority service resolves class by vote count and deterministic tie-break (`argmax` over bincount).
- Grouped majority modes aggregate voters by embedding or by classifier before final majority.

## 7.6 Threshold and Label Decision Logic
- Decision function resolves threshold in this precedence:
  1. `thresholds.classifier_embedding["Classifier::Embedding"]`
  2. `thresholds.classifier[Classifier]`
  3. `thresholds.default` (fallback 0.5)
- In global soft ensemble prediction, service currently uses fixed threshold config `{default: 0.5}`.

## 7.7 Regularization Options
- Trainable weights support optional `l2_regularization` in trainer params.

## 7.8 Multilabel Behavior
- Soft-voting probability contract supports multilabel matrices.
- Majority voting implementation expects 1D label vectors; multilabel majority may fail contract checks and be skipped in benchmark if exceptions occur.

## 7.9 Standalone Ensemble Step vs Benchmark Variant Runner
- `run_ensemble_step` creates `SoftVotingService` **without** injected majority service; majority modes in that step are not executable.
- Benchmark variant runner injects `SimpleMajorityVotingService`, enabling majority variants there.

---

# 8. Benchmark System

## 8.1 What `--step benchmark` Performs
`run_benchmark_step`:
1. Loads latest sweep run artifacts.
2. Selects model subset (default or ablations).
3. Rebuilds dataset + embeddings per seed.
4. Evaluates each base model (validation/test/zero-shot).
5. Builds best-single baseline.
6. Evaluates configured ensemble variants.
7. Computes deltas and generalization gaps.
8. Aggregates across seeds and exports artifacts.

## 8.2 Metrics Computed
For non-multilabel problems:
- Accuracy, Precision (macro), Recall (macro), F1 (macro).

For multilabel problems:
- Accuracy = `NaN` (explicitly set), Precision (macro), Recall (macro), F1 (macro).

## 8.3 F1 Macro/Micro Handling
- Benchmark ranking/comparison uses macro F1 (`f1`).
- Micro F1 is computed only in best-single diagnostics (`_evaluate_model_artifact_scores`) and not used as primary benchmark ranking metric.

## 8.4 Multilabel Normalization Logic
- True/pred labels are converted to binary matrices with `MultiLabelBinarizer` using explicit class ordering when available.
- Shape mismatch between true/pred matrices raises error.

## 8.5 Delta vs Best Single
For each variant row:
- `delta_vs_best_single_test = variant_test_f1 - best_single_test_f1`
- `delta_vs_best_single_zero_shot = variant_zero_f1 - best_single_zero_f1`

## 8.6 Multi-Seed Aggregation Formula
Aggregated fields are computed with:
- mean = `np.nanmean(values)`
- std = `np.nanstd(values)`

for validation/test/zero-shot F1 and delta metrics.

## 8.7 Exported Benchmark Artifacts
Under latest run `results/`:
- `benchmark_summary.csv`
- `benchmark_summary.json`
- `benchmark_multiseed_summary.csv`
- `benchmark_multiseed_summary.json`
- `benchmark_ablation_summary.csv`
- `benchmark_weights_analysis.json`

### CSV/JSON Content Contract
- Summary CSV includes category/variant, validation/test/zero-shot metrics, deltas, model count, weighting strategy.
- Multiseed CSV includes mean/std F1 and mean/std deltas.
- JSON payload includes per-seed rows, aggregated metrics, failed runs, reproducibility metadata, and config snapshot.

---

# 9. Zero-Shot Evaluation Contract

## 9.1 Purpose
Zero-shot evaluation measures model behavior on an isolated holdout partition meant to represent out-of-distribution deployment conditions.

## 9.2 Isolation Guarantee
- Independent split manager removes zero-shot IDs from train/validation/test.
- Benchmark re-validates isolation (`train/val/test` intersections with zero-shot must be empty), otherwise raises leakage error.

## 9.3 Prohibited Operations on Zero-Shot Split
Zero-shot samples are not used for:
- Base model fitting.
- Hyperparameter selection.
- Threshold fitting.
- Ensemble weight learning.

## 9.4 Expected Usage
- Configure `zero_shot.strategy` in dataset split config.
- Run sweep/final training as usual.
- Run benchmark to obtain zero-shot metrics.

## 9.5 Statistical Caveats
- Empty zero-shot split is allowed but scoring is skipped.
- Very small zero-shot sample count (`n < 10`) triggers explicit instability warning.

---

# 10. Configuration Contract (YAML Schema)

## 10.1 Configuration Files
- `config/pipeline.yaml` (step orchestration + dataset/split + optional per-step blocks).
- `config/training/training_config.yaml` (wandb/reporting/final training/selection groups).
- `config/embeddings.yaml` (embedding source and model toggles).
- `config/model_sweep/*.yaml` (classifier search spaces).
- `config/db.yaml` (DB connection).

## 10.2 Schema Summary

### `pipeline.yaml` (top-level)
| Key | Required | Default | Notes |
|---|---:|---|---|
| `dataset` | Yes for dataset-dependent steps | — | Contains DB, loader, and split configs |
| `training_config_path` | No | `config/training/training_config.yaml` | Loaded if file exists |
| `embeddings.config_path` | No | `config/embeddings.yaml` | Used by embeddings/train/sweep/ensemble/benchmark/evaluate |
| `train` | No | `{}` | Training-stage overrides |
| `sweep` | No | `{}` | Sweep behavior and config paths |
| `ensemble` | No | `{}` | Ensemble behavior |
| `benchmark` | No | `{}` | Benchmark behavior |

### `dataset.label_loader`
| Key | Required | Notes |
|---|---:|---|
| `source` | Yes | `file` or `db` |
| `file_path` | Conditional | Required when `source=file` |
| `db_query` or `db_query_file` | Conditional | Required when `source=db` |
| `accession_column`, `label_column` | No | Column names for CSV/SQL row access |

### `dataset.split` (independent mode)
- Independent mode is activated when both `validation` and `train_test` sections exist.
- `validation.strategy`: `random` \| `organism` \| `csv`
- `train_test.strategy`: `random` \| `cross_validation`
- `zero_shot.strategy`: optional; `random` \| `organism` \| `csv`

### `training_config.yaml`
| Path | Required | Default / Behavior |
|---|---:|---|
| `wandb.enabled` | No | `true` in sample config |
| `embedding_groups` | No | Used by `--embedding_group` runtime filter |
| `sweep.enabled_classifiers` | No | If non-empty, filters sweep classifiers |
| `final_training.enabled` | No | Controls retrain-on-train+val stage |
| `reporting.output_root` | No | `../../pec_data` |
| `reporting.thresholds` | No | Default threshold + optional per-classifier / per-(classifier,embedding) overrides |

### `embeddings.yaml`
| Path | Purpose |
|---|---|
| `aggregation.mode` | Layer aggregation mode (`none`, `mean`, `max`, `mean_max`, `concat`) |
| `sequencePE.models.*.enabled` | Enable/disable sequence embedding model views |
| `sequencePE.models.*.layer_index` | Layers to load |
| `GOPE.file_info.*` | GO CSV source definitions |
| `GOPE.models.GeOKG.enabled` | Enable GeOKG view |
| `missing_embeddings.allow_models` | Models allowed to have missing vectors (zero-filled) |

## 10.3 Default Behavior and Validation Logic
- No centralized JSON-schema validator exists.
- Validation is runtime/branch-local and fail-fast (e.g., missing columns, unknown strategies, missing artifacts).

## 10.4 Structured Example Snippet
```yaml
dataset:
  db_config_path: config/db.yaml
  label_loader:
    source: file
    file_path: /path/to/labels.csv
    accession_column: uniprot_id
    label_column: data_class
  split:
    validation:
      strategy: csv
      csv:
        csv_path: /path/to/splits.csv
        accession_column: uniprot_id
        split_column: data_split
        validation_values: [val, validation]
    train_test:
      strategy: cross_validation
      cross_validation:
        n_splits: 5
        fold_index: 0
        random_state: 42
    zero_shot:
      strategy: csv
      csv:
        csv_path: /path/to/splits.csv
        accession_column: uniprot_id
        split_column: data_split
        validation_values: [zs, zero_shot]

training_config_path: config/training/training_config.yaml
```

---

# 11. Determinism and Reproducibility

## 11.1 Seed Handling
- Random split strategies consume configured `random_state` values.
- KFold-based splitting uses configured `random_state` and `fold_index`.
- Sweep hyperparameter sampling uses `np.random.default_rng(42)` in `SweepService` unless constructor is overridden.
- Trainable ensemble weights use seeded random generator (`random_seed`, default 42).

## 11.2 Cross-Validation Fold Selection
- Fold selected as `fold_index % n_folds`; deterministic for fixed accession ordering and seed.

## 11.3 Artifact Hashing
- Benchmark computes SHA-256 hashes for loaded model artifact files and stores them under reproducibility payload in benchmark JSON.

## 11.4 Config Snapshot Persistence
Sweep run stores:
- `configs/resolved_pipeline.yaml`
- `configs/resolved_training.yaml`
- `configs/run_metadata.json` (includes runtime filters/context, CLI argv, git commit, package versions, timing).

## 11.5 Determinism Limitations
- No global deterministic seed orchestration for all backends (e.g., torch/cuda deterministic flags).
- Hardware/backend nondeterminism may affect MLP and some external model implementations.

---

# 12. Performance and Scalability Characteristics

## 12.1 Memory Usage Drivers
- Full split matrices are materialized per embedding view (`train`, `val`, `test`, optional `zero_shot`).
- Sequence loader accumulates accession-layer embeddings before matrix materialization.
- Multi-view sweeps train one model per view per trial; cumulative memory pressure scales with view dimension and batch model object size.

## 12.2 Parallelism
- Pipeline orchestration itself is sequential.
- Classifier internals may use their own parallelism (e.g., RF `n_jobs`, backend-specific behavior).
- No built-in distributed sweep coordinator exists.

## 12.3 Sweep Explosion Risk
Runtime complexity scales approximately with:
- number of selected classifiers × number of selected embeddings × `num_trials`.

Benchmark complexity further multiplies by:
- number of seeds × number of ablations × number of enabled ensemble variants.

## 12.4 Disk Usage Drivers
- Persisted base model artifacts (`.pkl` or `.pt` + metadata JSON).
- Per-run report CSV/JSON snapshots.
- Benchmark multi-seed payloads can grow due to per-seed per-variant records.

---

# 13. Known Limitations

- No embedding generation is implemented.
- No embedding fine-tuning is implemented.
- No dedicated probability calibration stage (e.g., Platt/Isotonic) is implemented.
- No online/incremental learning loop is implemented.
- Model quality is dependent on external embedding quality and upstream data quality.
- Class imbalance handling is model/config dependent; there is no automatic global rebalancing policy.
- Sweep labeling `method: bayes` in YAML is not interpreted as Bayesian optimization; sampling is randomized.
- Standalone `ensemble` step does not inject majority-voting service; majority modes are effectively benchmark-only in current orchestration.
- `structurePE` appears in embedding config but is not wired into active `EmbeddingService` loading path.
- Several legacy modules exist (`core/embeddings.py`, `classifiers/registry.py`) outside the active pipeline path and are not the authoritative execution route.

---

## Contract Summary Statement
`protein-embedding-classifier` is a supervised classifier orchestration system over externally precomputed protein embeddings, with controlled split logic, per-view model training/sweeps, artifact-based ensemble/benchmark evaluation, and reproducibility-oriented reporting. It is not an embedding generator.
