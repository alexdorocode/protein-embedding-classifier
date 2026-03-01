from __future__ import annotations

import csv
import json
import logging
import pickle
import hashlib
import copy
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from datetime import datetime
import numpy as np
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata

import yaml
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from protein_embedding_classifier.core.decision.decision_policy import decide
from protein_embedding_classifier.core.db import create_engine_from_config, load_db_config
from protein_embedding_classifier.core.ensemble.soft_voting_service import (
    EnsembleConfig,
    EnsembleMode,
    EnsembleSelectionConfig,
    ModelArtifact,
    SimpleMajorityVotingService,
    SoftVotingContractError,
    SoftVotingService,
    WeightingConfig,
    WeightingStrategyType,
)
from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle, EmbeddingService
from protein_embedding_classifier.core.probability.probability_adapter import ProbabilityAdapter
from protein_embedding_classifier.core.training.problem_specification import ProblemSpecification
from protein_embedding_classifier.core.training.sweep_service import SweepService
from protein_embedding_classifier.core.training.training_service import TrainingService
from protein_embedding_classifier.data.dataset_builder import DatasetBuilder
from protein_embedding_classifier.data.label_loader import LabelLoader
from protein_embedding_classifier.data.protein_loader import ProteinLoader
from protein_embedding_classifier.data.splits.cross_validation import CrossValidationSplit
from protein_embedding_classifier.data.splits.independent import IndependentValidationTrainTestSplit
from protein_embedding_classifier.data.splits.zero_shot_csv import ZeroShotCSVSplit
from protein_embedding_classifier.data.splits.zero_shot_organism import ZeroShotOrganismSplit
from protein_embedding_classifier.data.splits.zero_shot_random import ZeroShotRandomSplit


PIPELINE_STEPS = ["dataset", "embeddings", "train", "sweep", "ensemble", "benchmark", "evaluate"]


class _EmbeddingViewPredictor:
    def __init__(self, model: Any, embedding_name: str):
        self.model = model
        self.embedding_name = embedding_name

    def predict_proba(self, X: Any):
        if isinstance(X, Mapping):
            if self.embedding_name not in X:
                raise KeyError(
                    f"Missing embedding view '{self.embedding_name}' in ensemble inference payload"
                )
            return self.model.predict_proba(np.asarray(X[self.embedding_name]))
        return self.model.predict_proba(X)


class _ValidationWeightTrainer:
    def __init__(self, random_seed: int = 42, n_trials: int = 256):
        self.random_seed = int(random_seed)
        self.n_trials = int(max(32, n_trials))

    def fit(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        model_identifiers: list[str],
        problem_type: str,
        classes: list[Any],
        metric: str | None,
        params: Mapping[str, Any],
    ) -> np.ndarray:
        del model_identifiers
        del metric

        probs = np.asarray(validation_probabilities)
        y_true = np.asarray(validation_labels)
        n_models = probs.shape[0]
        if n_models < 2:
            raise ValueError("Weight trainer requires at least two models")

        trials = int(params.get("n_trials", self.n_trials)) if isinstance(params, Mapping) else self.n_trials
        trials = int(max(32, trials))
        l2_regularization = (
            float(params.get("l2_regularization", 0.0)) if isinstance(params, Mapping) else 0.0
        )
        l2_regularization = max(0.0, l2_regularization)
        rng = np.random.default_rng(self.random_seed)

        best_weights = np.ones(n_models, dtype=np.float64) / float(n_models)
        best_score = float("-inf")

        for _ in range(trials):
            candidate = rng.dirichlet(np.ones(n_models, dtype=np.float64))
            blended = np.tensordot(candidate, probs, axes=(0, 0))
            preds = decide(
                probs=blended,
                problem_type=problem_type,
                threshold_config={"default": 0.5},
            )
            score = float(
                _benchmark_f1_score(
                    problem_type=problem_type,
                    y_true=y_true,
                    y_pred=np.asarray(preds),
                    classes=classes,
                )
            )
            if l2_regularization > 0.0:
                score -= float(l2_regularization * np.sum(np.square(candidate)))
            if score > best_score:
                best_score = score
                best_weights = candidate

        return best_weights


def _to_multilabel_matrix(values: Any, classes: Sequence[Any]) -> np.ndarray:
    values_array = np.asarray(values, dtype=object)
    if values_array.ndim == 2 and np.issubdtype(values_array.dtype, np.number):
        return values_array.astype(int)

    converted: list[list[Any]] = []
    for item in values_array:
        if isinstance(item, np.ndarray):
            converted.append(item.tolist())
        elif isinstance(item, (list, tuple, set)):
            converted.append(list(item))
        elif item is None:
            converted.append([])
        else:
            converted.append([item])

    class_list = list(classes)
    mlb = MultiLabelBinarizer(classes=class_list if class_list else None)
    if class_list:
        mlb.fit([class_list])
    else:
        mlb.fit(converted)
    return mlb.transform(converted)


def _benchmark_f1_score(
    problem_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[Any],
) -> float:
    if problem_type == "multilabel":
        y_true_bin = _to_multilabel_matrix(y_true, classes)
        pred_array = np.asarray(y_pred)
        if pred_array.ndim == 2:
            y_pred_bin = pred_array.astype(int)
        else:
            y_pred_bin = _to_multilabel_matrix(pred_array, classes)

        if y_true_bin.shape != y_pred_bin.shape:
            raise ValueError(
                "Multilabel benchmark metric shape mismatch: "
                f"y_true={y_true_bin.shape} y_pred={y_pred_bin.shape}"
            )
        return float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))

    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro", zero_division=0))


class Pipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger("Pipeline")
        self.runtime_filters: dict[str, Any] = {}
        self.runtime_context: dict[str, Any] = {}
        self._active_log_handler = None

    def run(
        self,
        step: str | None = None,
        run_all: bool = False,
        filters: dict[str, Any] | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> None:
        self.logger.info("Step 1: Load pipeline config")
        conf = self._load_yaml(self.config_path)
        self.runtime_filters = {key: value for key, value in (filters or {}).items() if value is not None}
        self.runtime_context = dict(runtime_context or {})

        self.logger.info("Step 2: Resolve execution plan")
        if run_all:
            execution_plan = PIPELINE_STEPS
        else:
            execution_plan = [step or "dataset"]

        self.logger.info("Step 3: Run pipeline steps")
        for step_name in execution_plan:
            self.run_step(step_name, conf)

    def run_step(self, step: str, conf: dict[str, Any]) -> None:
        step_runners = self._step_runner_map()
        if step not in step_runners:
            raise ValueError(f"Unknown step: {step}")

        self.logger.info("Starting step: %s", step)
        step_conf = conf.get(step, conf.get("dataset", conf) if step == "dataset" else conf)
        step_runners[step](step_conf)
        self.logger.info("Finished step: %s", step)

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        requested_path = Path(path)
        project_root = Path(__file__).resolve().parents[2]

        candidates: list[Path] = []
        if requested_path.is_absolute():
            candidates.append(requested_path)
        else:
            candidates.append(requested_path)
            candidates.append(project_root / requested_path)
            if len(requested_path.parts) == 1:
                candidates.append(Path("config") / requested_path.name)
                candidates.append(project_root / "config" / requested_path.name)

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve() if candidate.exists() else candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists() and candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle) or {}

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"YAML config not found: {path}. Searched: {searched}")

    @staticmethod
    def _build_split_strategy(conf: dict[str, Any]):
        if "validation" in conf and "train_test" in conf:
            return IndependentValidationTrainTestSplit(conf)

        strategy_name = conf.get("strategy", "zero_shot_random")

        if strategy_name == "cross_validation":
            return CrossValidationSplit(
                n_splits=conf.get("n_splits", 5),
                fold_index=conf.get("fold_index", 0),
                random_state=conf.get("random_state", 42),
            )

        if strategy_name == "zero_shot_csv":
            return ZeroShotCSVSplit(
                csv_path=conf["csv_path"],
                accession_column=conf.get("accession_column", "accession"),
                split_column=conf.get("split_column", "split"),
            )

        if strategy_name == "zero_shot_organism":
            return ZeroShotOrganismSplit(
                test_organisms=conf.get("test_organisms", []),
                train_test_organisms=conf.get("train_test_organisms", []),
                val_organisms=conf.get("val_organisms", []),
                test_size=conf.get("test_size", 0.2),
                random_state=conf.get("random_state", 42),
            )

        if strategy_name == "zero_shot_random":
            return ZeroShotRandomSplit(
                test_size=conf.get("test_size", 0.2),
                val_size=conf.get("val_size", 0.1),
                random_state=conf.get("random_state", 42),
            )

        raise ValueError(f"Unknown split strategy: {strategy_name}")

    @classmethod
    def _build_dataset_bundle(cls, conf: dict[str, Any]):
        db_conf_path = conf.get("db_config_path", "config/db.yaml")
        db_conf = load_db_config(db_conf_path)
        engine = create_engine_from_config(db_conf)

        protein_conf = conf.get("protein_loader", {})
        label_conf = conf.get("label_loader", {})
        split_conf = conf.get("split", {})

        protein_loader = ProteinLoader(engine=engine, query=protein_conf.get("query"))
        label_loader = LabelLoader(
            source=label_conf.get("source", "db"),
            engine=engine,
            file_path=label_conf.get("file_path"),
            db_query=label_conf.get("db_query"),
            db_query_file=label_conf.get("db_query_file"),
            accession_column=label_conf.get("accession_column", "accession"),
            label_column=label_conf.get("label_column", "label"),
            artifacts_dir=label_conf.get("artifacts_dir", "artifacts"),
        )
        split_strategy = cls._build_split_strategy(split_conf)

        builder = DatasetBuilder(
            protein_loader=protein_loader,
            label_loader=label_loader,
            split_strategy=split_strategy,
        )

        return builder.build()

    def run_dataset_step(self, conf: dict[str, Any]) -> None:
        bundle = self._build_dataset_bundle(conf)
        self.logger.info(
            "Dataset bundle ready: train=%d val=%d test=%d",
            len(bundle.train_ids),
            len(bundle.val_ids),
            len(bundle.test_ids),
        )

    def run_embeddings_step(self, conf: dict[str, Any]) -> None:
        pipeline_conf = conf if "dataset" in conf else self._load_yaml(self.config_path)
        dataset_conf = pipeline_conf.get("dataset", pipeline_conf)
        embeddings_step_conf = pipeline_conf.get("embeddings", {})

        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        self.logger.info("Loading embeddings config: %s", embeddings_config_path)
        embeddings_conf = self._load_yaml(embeddings_config_path)

        embedding_bundle = self._build_embedding_bundle(dataset_conf=dataset_conf, embeddings_conf=embeddings_conf)

        for model_name in sorted(embedding_bundle.X_train.keys()):
            self.logger.info(
                "Embedding view ready model=%s train=%s val=%s test=%s",
                model_name,
                tuple(embedding_bundle.X_train[model_name].shape),
                tuple(embedding_bundle.X_val[model_name].shape),
                tuple(embedding_bundle.X_test[model_name].shape),
            )

    def run_train_step(self, conf: dict[str, Any]) -> None:
        self.logger.info("Running mode=train")
        pipeline_conf = conf if "dataset" in conf else self._load_yaml(self.config_path)
        dataset_conf = pipeline_conf.get("dataset", pipeline_conf)
        embeddings_step_conf = pipeline_conf.get("embeddings", {})
        train_conf = pipeline_conf.get("train", {})

        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        embeddings_conf = self._load_yaml(embeddings_config_path)
        embedding_bundle = self._build_embedding_bundle(dataset_conf=dataset_conf, embeddings_conf=embeddings_conf)
        training_global_conf = self._load_training_config(pipeline_conf)
        threshold_policy = training_global_conf.get("reporting", {}).get("thresholds", {})
        selected_embeddings = self._resolve_selected_embeddings(
            available=list(embedding_bundle.X_train.keys()),
            filters=self.runtime_filters,
            training_global_conf=training_global_conf,
        )
        embedding_bundle = self._filter_embedding_bundle(embedding_bundle, selected_embeddings)

        problem_spec = ProblemSpecification.from_labels(embedding_bundle.y_train)
        self.logger.info(
            "Problem specification type=%s output_size=%d loss=%s",
            problem_spec.problem_type,
            problem_spec.output_size,
            problem_spec.loss_name,
        )

        service = TrainingService()
        results = service.train(
            embedding_bundle=embedding_bundle,
            training_config={
                **dict(train_conf),
                "threshold_policy": threshold_policy,
            },
        )

        artifacts_dir = Path(str(train_conf.get("artifacts_dir", "artifacts")))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / "training_results.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(results, handle)

        self.logger.info("Saved training results artifact: %s", output_path)
        for (model_type, embedding_name), payload in sorted(results.items()):
            f1_value = payload.get("metrics", {}).get("f1_score")
            self.logger.info(
                "Training summary model_type=%s embedding_name=%s validation_f1=%s",
                model_type,
                embedding_name,
                f1_value,
            )

    def run_sweep_step(self, conf: dict[str, Any]) -> None:
        self.logger.info("Running mode=sweep")
        pipeline_conf = conf if "dataset" in conf else self._load_yaml(self.config_path)
        dataset_conf = pipeline_conf.get("dataset", pipeline_conf)
        embeddings_step_conf = pipeline_conf.get("embeddings", {})
        train_conf = pipeline_conf.get("train", {})
        sweep_conf = pipeline_conf.get("sweep", {})
        training_global_conf = self._load_training_config(pipeline_conf)
        reporting_conf = self._build_reporting_config(training_global_conf)

        path_layout = self._ensure_pec_data_layout(reporting_conf)
        run_prefix = str(self.runtime_context.get("run_prefix") or reporting_conf.get("run_prefix", "sweep"))
        run_dir = self._create_timestamped_run_dir(path_layout["sweep"], run_prefix)
        self._attach_file_logger(path_layout["logs"], run_dir.name)

        run_started_at = datetime.utcnow().isoformat() + "Z"
        run_started_ts = time.time()

        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        embeddings_conf = self._load_yaml(embeddings_config_path)

        dataset_bundle = self._build_dataset_bundle(dataset_conf)
        self._persist_dataset_snapshot_if_missing(
            dataset_bundle=dataset_bundle,
            dataset_name=str(reporting_conf.get("dataset_name", "default_dataset")),
            dataset_dir=path_layout["dataset"],
        )

        embedding_bundle = self._build_embedding_bundle_from_dataset(
            dataset_bundle=dataset_bundle,
            dataset_conf=dataset_conf,
            embeddings_conf=embeddings_conf,
        )

        selected_embeddings = self._resolve_selected_embeddings(
            available=list(embedding_bundle.X_train.keys()),
            filters=self.runtime_filters,
            training_global_conf=training_global_conf,
        )
        embedding_bundle = self._filter_embedding_bundle(embedding_bundle, selected_embeddings)
        problem_spec = ProblemSpecification.from_labels(embedding_bundle.y_train)

        explicit_model_type = sweep_conf.get("model_type")
        model_type = str(explicit_model_type) if explicit_model_type is not None else None
        num_trials = int(sweep_conf.get("num_trials", 1))
        wandb_project = str(sweep_conf.get("wandb_project", "protein-embedding-classifier"))
        classifier_sweeps = self._resolve_classifier_sweeps(sweep_conf)
        artifacts_dir = Path(str(sweep_conf.get("artifacts_dir", "artifacts")))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        selected_classifiers = self._resolve_selected_classifiers(
            available=list(classifier_sweeps.keys()),
            filters=self.runtime_filters,
            training_global_conf=training_global_conf,
        )
        classifier_sweeps = {name: classifier_sweeps[name] for name in selected_classifiers if name in classifier_sweeps}

        if model_type is None or model_type.lower() == "all":
            selected_classifier_sweeps = classifier_sweeps
            self.logger.info(
                "Sweep mode configured for all classifiers: %s",
                sorted(selected_classifier_sweeps.keys()),
            )
        else:
            if model_type not in classifier_sweeps:
                raise ValueError(
                    f"No sweep config path configured for model_type={model_type}. "
                    f"Configured model types: {sorted(classifier_sweeps.keys())}"
                )
            selected_classifier_sweeps = {model_type: classifier_sweeps[model_type]}
            self.logger.info("Sweep mode configured for single classifier model_type=%s", model_type)

        best_results: dict[str, dict[str, Any]] = {}
        all_trial_results: list[dict[str, Any]] = []
        final_test_rows: list[dict[str, Any]] = []
        skipped_classifiers: list[tuple[str, str]] = []

        wandb_conf = training_global_conf.get("wandb", {}) if isinstance(training_global_conf.get("wandb", {}), Mapping) else {}
        wandb_enabled = bool(wandb_conf.get("enabled", True))
        wandb_mode = str(wandb_conf.get("mode", "online"))
        wandb_entity = wandb_conf.get("entity")
        wandb_project = str(wandb_conf.get("project", wandb_project))

        for current_model_type, sweep_config_path in selected_classifier_sweeps.items():
            self.logger.info(
                "Starting classifier sweep model_type=%s config_path=%s",
                current_model_type,
                sweep_config_path,
            )
            sweep_yaml = self._load_yaml(sweep_config_path)

            sweep_service = SweepService(model_type=current_model_type)
            try:
                result = sweep_service.run(
                    embedding_bundle=embedding_bundle,
                    sweep_config=sweep_yaml,
                    num_trials=num_trials,
                    training_config={
                        **dict(train_conf),
                        "threshold_policy": reporting_conf.get("thresholds", {}),
                    },
                    wandb_project=wandb_project,
                    artifacts_dir=str(artifacts_dir),
                    wandb_enabled=wandb_enabled,
                    wandb_mode=wandb_mode,
                    wandb_entity=wandb_entity,
                )
            except ImportError as exc:
                self.logger.warning(
                    "Skipping classifier sweep model_type=%s due to missing optional dependency: %s",
                    current_model_type,
                    exc,
                )
                skipped_classifiers.append((current_model_type, str(exc)))
                continue
            except ValueError as exc:
                raise

            best_results[current_model_type] = {
                "best_config": result.best_config,
                "best_metric": result.best_metric,
                "best_key": result.best_key,
                "sweep_config_path": sweep_config_path,
            }
            all_trial_results.extend(result.trial_results)
            self.logger.info(
                "Finished classifier sweep model_type=%s best_embedding=%s validation_f1=%.6f",
                current_model_type,
                result.best_key[1],
                result.best_metric,
            )

            if bool(training_global_conf.get("final_training", {}).get("enabled", False)):
                final_training_conf = dict(training_global_conf.get("final_training", {}))
                final_training_conf["output_dir"] = str(run_dir / "models")
                final_training_conf["threshold_policy"] = reporting_conf.get("thresholds", {})
                final_rows = self._run_final_training_for_classifier(
                    classifier=current_model_type,
                    trial_results=result.trial_results,
                    embedding_bundle=embedding_bundle,
                    train_conf=train_conf,
                    final_training_conf=final_training_conf,
                    wandb_enabled=wandb_enabled,
                    wandb_mode=wandb_mode,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                )
                final_test_rows.extend(final_rows)

        if not best_results:
            if skipped_classifiers:
                skipped_summary = "; ".join(f"{name}: {reason}" for name, reason in skipped_classifiers)
                self.logger.warning(
                    "No successful sweep results were produced; all selected classifiers were skipped. %s",
                    skipped_summary,
                )
            else:
                self.logger.warning("No successful sweep results were produced")
            return

        configs_dir = run_dir / "configs"
        reports_dir = run_dir / "reports"
        predictions_dir = run_dir / "predictions"
        models_dir = run_dir / "models"
        for folder in (configs_dir, reports_dir, predictions_dir, models_dir):
            folder.mkdir(parents=True, exist_ok=True)

        per_classifier_path = reports_dir / "best_config_by_classifier.yaml"
        with per_classifier_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(best_results, handle, sort_keys=False)
        self.logger.info("Saved per-classifier sweep results artifact: %s", per_classifier_path)

        full_results_csv = reports_dir / "sweep_results_full.csv"
        self._write_full_sweep_results_csv(full_results_csv, all_trial_results)
        self.logger.info("Saved full sweep results CSV artifact: %s", full_results_csv)

        best_per_classifier_csv = reports_dir / "best_per_classifier.csv"
        self._write_best_per_classifier_csv(best_per_classifier_csv, best_results)
        self.logger.info("Saved per-classifier sweep CSV artifact: %s", best_per_classifier_csv)

        best_classifier_per_embedding = self._select_best_classifier_per_embedding(all_trial_results)
        best_classifier_per_embedding_csv = reports_dir / "best_classifier_per_embedding.csv"
        self._write_best_classifier_per_embedding_csv(best_classifier_per_embedding_csv, best_classifier_per_embedding)
        self.logger.info("Saved best classifier per embedding CSV artifact: %s", best_classifier_per_embedding_csv)

        summary_table = SweepService.build_summary_table(
            trial_results=all_trial_results,
            model_order=sorted(selected_classifier_sweeps.keys()),
        )
        self.logger.info("Validation summary table (F1):\n%s", summary_table)

        if final_test_rows:
            final_test_path = reports_dir / "final_test_results.csv"
            self._write_final_test_results_csv(final_test_path, final_test_rows)
            self.logger.info("Saved final test results CSV artifact: %s", final_test_path)

        self._save_run_methodology_snapshot(
            configs_dir=configs_dir,
            pipeline_conf=pipeline_conf,
            training_conf=training_global_conf,
            sweep_conf=sweep_conf,
            selected_embeddings=selected_embeddings,
            selected_classifiers=sorted(selected_classifier_sweeps.keys()),
            run_started_at=run_started_at,
            run_duration_seconds=float(time.time() - run_started_ts),
        )

        predictions_csv = predictions_dir / "predictions_test.csv"
        self._write_test_predictions_csv(
            path=predictions_csv,
            dataset_bundle=dataset_bundle,
            embedding_bundle=embedding_bundle,
            final_test_rows=final_test_rows,
            best_classifier_per_embedding=best_classifier_per_embedding,
            threshold_conf=reporting_conf.get("thresholds", {}),
        )
        self.logger.info("Saved test predictions CSV artifact: %s", predictions_csv)

    def run_ensemble_step(self, conf: dict[str, Any]) -> None:
        pipeline_conf = conf if "dataset" in conf else self._load_yaml(self.config_path)
        dataset_conf = pipeline_conf.get("dataset", pipeline_conf)
        embeddings_step_conf = pipeline_conf.get("embeddings", {})
        ensemble_conf = pipeline_conf.get("ensemble", conf if isinstance(conf, Mapping) else {})
        if not isinstance(ensemble_conf, Mapping):
            ensemble_conf = {}

        if not bool(ensemble_conf.get("enabled", True)):
            self.logger.info("Ensemble step skipped because ensemble.enabled=false")
            return

        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        embeddings_conf = self._load_yaml(embeddings_config_path)
        training_global_conf = self._load_training_config(pipeline_conf)
        reporting_conf = self._build_reporting_config(training_global_conf)
        path_layout = self._ensure_pec_data_layout(reporting_conf)

        source_conf = ensemble_conf.get("source", {}) if isinstance(ensemble_conf.get("source", {}), Mapping) else {}
        configured_run_dir = source_conf.get("run_dir", ensemble_conf.get("run_dir"))
        if configured_run_dir:
            latest_run_dir = Path(str(configured_run_dir)).expanduser().resolve()
        else:
            latest_run_dir = self._get_latest_sweep_run(path_layout["sweep"])

        if latest_run_dir is None or not latest_run_dir.exists():
            raise FileNotFoundError("No sweep run directory available for ensemble step")

        best_csv = latest_run_dir / "reports" / "best_classifier_per_embedding.csv"
        if not best_csv.exists():
            raise FileNotFoundError(
                f"best_classifier_per_embedding.csv not found in {latest_run_dir}. "
                "Ensemble step requires persisted model artifacts from previous sweep/final training."
            )

        rows = self._read_best_classifier_per_embedding_csv(best_csv)
        if not rows:
            raise FileNotFoundError(f"No model rows found in {best_csv}")

        selected_rows, requested_count = self._select_ensemble_rows(
            rows=rows,
            ensemble_conf=dict(ensemble_conf),
            training_global_conf=training_global_conf,
        )

        if len(selected_rows) < requested_count:
            self.logger.warning(
                "Ensemble selection requested=%d available=%d after artifact/config filtering; continuing",
                requested_count,
                len(selected_rows),
            )

        model_artifacts, selected_payloads = self._load_ensemble_model_artifacts(
            run_dir=latest_run_dir,
            selected_rows=selected_rows,
        )
        if len(model_artifacts) < 2:
            raise SoftVotingContractError(
                f"Ensemble step requires at least 2 models after loading artifacts, got {len(model_artifacts)}"
            )

        dataset_bundle = self._build_dataset_bundle(dataset_conf)
        embedding_bundle = self._build_embedding_bundle_from_dataset(
            dataset_bundle=dataset_bundle,
            dataset_conf=dataset_conf,
            embeddings_conf=embeddings_conf,
        )

        service_config, strategy_type, weighting_params = self._build_soft_voting_service_config(
            ensemble_conf=dict(ensemble_conf),
            model_artifacts=model_artifacts,
        )
        weight_trainer = None
        if strategy_type == WeightingStrategyType.TRAINABLE_WEIGHTS:
            weight_trainer = _ValidationWeightTrainer(
                random_seed=int(weighting_params.get("random_seed", 42)),
                n_trials=int(weighting_params.get("n_trials", 256)),
            )

        soft_voting_service = SoftVotingService(
            model_artifacts=model_artifacts,
            config=service_config,
            weight_trainer=weight_trainer,
        )

        x_val_map = {name: np.asarray(values) for name, values in embedding_bundle.X_val.items()}
        x_test_map = {name: np.asarray(values) for name, values in embedding_bundle.X_test.items()}
        y_val = np.asarray(dataset_bundle.y_val)

        val_probabilities = soft_voting_service.collect_validation_probabilities(x_val_map)
        soft_voting_service.fit_with_validation(x_val_map, y_val)
        ensemble_output = soft_voting_service.predict(x_test_map)

        weights = np.asarray(ensemble_output["metadata"].get("ensemble", {}).get("weights", []), dtype=np.float64)
        summary_rows = self._build_ensemble_summary_rows(
            model_artifacts=model_artifacts,
            selected_payloads=selected_payloads,
            validation_probabilities=val_probabilities,
            y_val=y_val,
            weights=weights,
        )
        self.logger.info("Ensemble summary table (model -> validation_f1 -> weight):\n%s", self._format_ensemble_summary(summary_rows))

        output_dir = latest_run_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_ensemble_artifact(
            output_dir=output_dir,
            ensemble_conf=dict(ensemble_conf),
            ensemble_output=ensemble_output,
            summary_rows=summary_rows,
        )
        self.logger.info(
            "Ensemble step completed mode=%s models=%d test_samples=%d",
            ensemble_output["metadata"].get("ensemble", {}).get("mode"),
            len(model_artifacts),
            int(np.asarray(ensemble_output["probabilities"]).shape[0]),
        )

    def run_benchmark_step(self, conf: dict[str, Any]) -> None:
        pipeline_conf = conf if "dataset" in conf else self._load_yaml(self.config_path)
        dataset_conf = pipeline_conf.get("dataset", pipeline_conf)
        embeddings_step_conf = pipeline_conf.get("embeddings", {})
        benchmark_conf = pipeline_conf.get("benchmark", conf if isinstance(conf, Mapping) else {})
        if not isinstance(benchmark_conf, Mapping):
            benchmark_conf = {}

        metric_name = str(benchmark_conf.get("metric", "f1_macro"))
        include_uniform = bool(benchmark_conf.get("include_uniform", True))
        include_validation_weighted = bool(benchmark_conf.get("include_validation_weighted", True))
        include_trainable = bool(benchmark_conf.get("include_trainable", True))
        include_majority = bool(benchmark_conf.get("include_majority", True))
        seeds_raw = benchmark_conf.get("seeds", [42])
        if isinstance(seeds_raw, list) and seeds_raw:
            seeds = [int(seed) for seed in seeds_raw]
        else:
            seeds = [42]
        aggregate_mode = str(benchmark_conf.get("aggregate", "mean_std"))
        ablations_conf = benchmark_conf.get("ablations", [])
        ablation_specs = self._build_benchmark_ablation_specs(ablations_conf)

        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        embeddings_conf = self._load_yaml(embeddings_config_path)
        training_global_conf = self._load_training_config(pipeline_conf)
        reporting_conf = self._build_reporting_config(training_global_conf)
        path_layout = self._ensure_pec_data_layout(reporting_conf)

        latest_run_dir = self._get_latest_sweep_run(path_layout["sweep"])
        if latest_run_dir is None:
            raise FileNotFoundError("No previous sweep run found for benchmark step")

        best_csv = latest_run_dir / "reports" / "best_classifier_per_embedding.csv"
        if not best_csv.exists():
            raise FileNotFoundError(f"best_classifier_per_embedding.csv not found in {latest_run_dir}")

        rows = self._read_best_classifier_per_embedding_csv(best_csv)
        results_dir = latest_run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        benchmark_models_dir = latest_run_dir / "models"
        benchmark_models_dir.mkdir(parents=True, exist_ok=True)

        variant_specs = self._build_benchmark_variant_specs(
            metric_name=metric_name,
            include_uniform=include_uniform,
            include_validation_weighted=include_validation_weighted,
            include_trainable=include_trainable,
            include_majority=include_majority,
            trainable_params=benchmark_conf.get("trainable_params", {}),
        )

        seed_level_rows: list[dict[str, Any]] = []
        seed_failures: list[dict[str, Any]] = []
        weight_records: list[dict[str, Any]] = []
        artifact_hashes: dict[str, str] = {}
        default_payload: dict[str, Any] | None = None

        for ablation in ablation_specs:
            ablation_name = str(ablation.get("name", "default"))
            selection_conf = {"selection": dict(ablation.get("selection", {}))}
            selected_rows, requested_count = self._select_ensemble_rows(
                rows=rows,
                ensemble_conf=selection_conf,
                training_global_conf=training_global_conf,
            )
            if len(selected_rows) < requested_count:
                self.logger.warning(
                    "Benchmark ablation=%s requested=%d available=%d; continuing",
                    ablation_name,
                    requested_count,
                    len(selected_rows),
                )

            model_artifacts, selected_payloads = self._load_ensemble_model_artifacts(
                run_dir=latest_run_dir,
                selected_rows=selected_rows,
            )
            if len(model_artifacts) < 2:
                self.logger.warning(
                    "Skipping ablation=%s because it has <2 models after filtering (got=%d)",
                    ablation_name,
                    len(model_artifacts),
                )
                seed_failures.extend(
                    [
                        {
                            "seed": seed,
                            "ablation": ablation_name,
                            "error": f"insufficient_models:{len(model_artifacts)}",
                        }
                        for seed in seeds
                    ]
                )
                continue

            for payload in selected_payloads:
                artifact_rel = str(payload.get("artifact_path", ""))
                model_path = latest_run_dir / artifact_rel
                if model_path.exists() and artifact_rel not in artifact_hashes:
                    artifact_hashes[artifact_rel] = self._sha256_file(model_path)

            for seed in seeds:
                try:
                    dataset_conf_for_seed = self._with_seeded_dataset_config(dataset_conf, seed)
                    dataset_bundle = self._build_dataset_bundle(dataset_conf_for_seed)
                    embedding_bundle = self._build_embedding_bundle_from_dataset(
                        dataset_bundle=dataset_bundle,
                        dataset_conf=dataset_conf_for_seed,
                        embeddings_conf=embeddings_conf,
                    )

                    x_val_map = {name: np.asarray(values) for name, values in embedding_bundle.X_val.items()}
                    x_test_map = {name: np.asarray(values) for name, values in embedding_bundle.X_test.items()}
                    y_val = np.asarray(dataset_bundle.y_val)
                    y_test = np.asarray(dataset_bundle.y_test)

                    per_model_scores: list[dict[str, Any]] = []
                    for artifact in model_artifacts:
                        val_score, test_score, diagnostics = self._evaluate_model_artifact_scores(
                            artifact=artifact,
                            x_val_map=x_val_map,
                            y_val=y_val,
                            x_test_map=x_test_map,
                            y_test=y_test,
                            include_diagnostics=True,
                        )
                        per_model_scores.append(
                            {
                                "model": f"{artifact.classifier_name}::{artifact.embedding_name}",
                                "classifier_name": artifact.classifier_name,
                                "embedding_name": artifact.embedding_name,
                                "validation_f1": val_score,
                                "test_f1": test_score,
                                "diagnostics": diagnostics,
                            }
                        )

                    best_single = max(per_model_scores, key=lambda item: float(item["validation_f1"]))
                    self._log_best_single_diagnostics(best_single, seed=seed, ablation_name=ablation_name)

                    benchmark_results: list[dict[str, Any]] = [
                        {
                            "category": "Single",
                            "variant": "best_single",
                            "validation_f1": float(best_single["validation_f1"]),
                            "test_f1": float(best_single["test_f1"]),
                            "num_models": 1,
                            "weighting_strategy": "n/a",
                            "mode": "single",
                            "models_used": [best_single["model"]],
                            "weights": None,
                        }
                    ]

                    for spec in variant_specs:
                        try:
                            spec_for_seed = dict(spec)
                            spec_for_seed["params"] = {
                                **dict(spec.get("params", {})),
                                "random_seed": seed,
                            }
                            result_row = self._run_benchmark_ensemble_variant(
                                spec=spec_for_seed,
                                model_artifacts=model_artifacts,
                                x_val_map=x_val_map,
                                y_val=y_val,
                                x_test_map=x_test_map,
                                y_test=y_test,
                                selected_payloads=selected_payloads,
                                benchmark_models_dir=benchmark_models_dir,
                            )
                            benchmark_results.append(result_row)
                        except Exception as exc:
                            self.logger.warning(
                                "Skipping benchmark variant=%s seed=%d ablation=%s due to error: %s",
                                spec.get("variant"),
                                seed,
                                ablation_name,
                                exc,
                            )

                    if len(benchmark_results) < 2:
                        raise RuntimeError(
                            f"Benchmark produced no valid ensemble variants for seed={seed} ablation={ablation_name}"
                        )

                    best_single_test = float(best_single["test_f1"])
                    for row in benchmark_results:
                        delta = float(row.get("test_f1", float("nan"))) - best_single_test
                        gap = float(row.get("validation_f1", float("nan"))) - float(row.get("test_f1", float("nan")))
                        row.update(
                            {
                                "delta_vs_best_single_test": delta,
                                "generalization_gap": gap,
                                "seed": seed,
                                "ablation": ablation_name,
                            }
                        )
                        seed_level_rows.append(dict(row))
                        self.logger.info(
                            "Benchmark seed=%d ablation=%s variant=%s val_f1=%.4f test_f1=%.4f gap=%.4f",
                            seed,
                            ablation_name,
                            row.get("variant"),
                            float(row.get("validation_f1", float("nan"))),
                            float(row.get("test_f1", float("nan"))),
                            gap,
                        )

                    for row in benchmark_results:
                        if str(row.get("variant")) == "trainable_weights_soft_voting" and row.get("weights"):
                            weight_records.append(
                                {
                                    "seed": seed,
                                    "ablation": ablation_name,
                                    "weights": list(row.get("weights") or []),
                                    "models_used": list(row.get("models_used") or []),
                                    "validation_f1": float(row.get("validation_f1", float("nan"))),
                                    "test_f1": float(row.get("test_f1", float("nan"))),
                                    "model_validation_scores": dict(row.get("model_validation_scores", {})),
                                }
                            )

                    if default_payload is None and ablation_name == "default":
                        comparison_rows = self._comparison_rows_from_seed_results(benchmark_results)
                        ranking = self._ranking_from_results(benchmark_results)
                        default_payload = {
                            "config": {
                                "metric": metric_name,
                                "include_majority": include_majority,
                                "include_trainable": include_trainable,
                                "include_validation_weighted": include_validation_weighted,
                                "include_uniform": include_uniform,
                                "seed": seed,
                                "ablation": ablation_name,
                            },
                            "best_single": benchmark_results[0],
                            "ensembles": [row for row in benchmark_results if row.get("category") == "Ensemble"],
                            "ranking_by_test_metric": ranking,
                            "comparison_table": comparison_rows,
                            "models_evaluated": per_model_scores,
                        }
                except Exception as exc:
                    self.logger.warning(
                        "Skipping benchmark seed=%d ablation=%s due to error: %s",
                        seed,
                        ablation_name,
                        exc,
                    )
                    seed_failures.append({"seed": seed, "ablation": ablation_name, "error": str(exc)})

        if not seed_level_rows:
            raise RuntimeError("Benchmark step failed: all seeds/ablations failed")

        aggregated_rows = self._aggregate_benchmark_seed_rows(seed_level_rows)
        default_aggregated = [row for row in aggregated_rows if row.get("ablation") == "default"]
        if not default_aggregated:
            default_aggregated = aggregated_rows

        multiseed_csv = results_dir / "benchmark_multiseed_summary.csv"
        self._write_benchmark_multiseed_summary_csv(multiseed_csv, default_aggregated)

        multiseed_json = results_dir / "benchmark_multiseed_summary.json"
        weights_analysis = self._build_benchmark_weights_analysis(weight_records)
        overfitting_report = self._build_overfitting_report(seed_level_rows)
        multiseed_payload = {
            "aggregate_mode": aggregate_mode,
            "config": {
                "metric": metric_name,
                "seeds": seeds,
                "include_majority": include_majority,
                "include_trainable": include_trainable,
                "include_validation_weighted": include_validation_weighted,
                "include_uniform": include_uniform,
                "ablations": [dict(item) for item in ablation_specs],
            },
            "per_seed_results": seed_level_rows,
            "aggregated_metrics": aggregated_rows,
            "failed_runs": seed_failures,
            "weights_analysis": weights_analysis,
            "overfitting_report": overfitting_report,
            "reproducibility": {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "git_commit": self._git_commit(),
                "seeds": seeds,
                "artifact_hashes": artifact_hashes,
                "config_snapshot": {
                    "pipeline": pipeline_conf,
                    "benchmark": dict(benchmark_conf),
                },
            },
        }
        with multiseed_json.open("w", encoding="utf-8") as handle:
            json.dump(multiseed_payload, handle, indent=2)

        weights_json = results_dir / "benchmark_weights_analysis.json"
        with weights_json.open("w", encoding="utf-8") as handle:
            json.dump(weights_analysis, handle, indent=2)

        ablation_csv = results_dir / "benchmark_ablation_summary.csv"
        self._write_benchmark_ablation_summary_csv(ablation_csv, aggregated_rows)

        benchmark_csv = results_dir / "benchmark_summary.csv"
        benchmark_json = results_dir / "benchmark_summary.json"
        summary_rows = self._benchmark_summary_rows_from_aggregated(default_aggregated)
        self._write_benchmark_summary_csv(benchmark_csv, summary_rows)
        with benchmark_json.open("w", encoding="utf-8") as handle:
            json.dump(default_payload or multiseed_payload, handle, indent=2)

        self.logger.info("Benchmark summary table:\n%s", self._format_benchmark_table(summary_rows))
        self.logger.info("Saved benchmark CSV artifact: %s", benchmark_csv)
        self.logger.info("Saved benchmark JSON artifact: %s", benchmark_json)
        self.logger.info("Saved benchmark multiseed CSV artifact: %s", multiseed_csv)
        self.logger.info("Saved benchmark multiseed JSON artifact: %s", multiseed_json)
        self.logger.info("Saved benchmark weight analysis artifact: %s", weights_json)
        self.logger.info("Saved benchmark ablation summary artifact: %s", ablation_csv)

    def _build_soft_voting_service_config(
        self,
        ensemble_conf: dict[str, Any],
        model_artifacts: list[ModelArtifact],
    ) -> tuple[EnsembleConfig, WeightingStrategyType, dict[str, Any]]:
        mode_raw = str(ensemble_conf.get("mode", "global_soft"))
        mode = EnsembleMode(mode_raw)

        weighting_conf = ensemble_conf.get("weighting", {}) if isinstance(ensemble_conf.get("weighting", {}), Mapping) else {}
        strategy_raw = str(weighting_conf.get("strategy", "uniform")).strip().lower()
        strategy_aliases = {
            "uniform": WeightingStrategyType.UNIFORM,
            "validation_score_based": WeightingStrategyType.VALIDATION_SCORE_BASED,
            "validation": WeightingStrategyType.VALIDATION_SCORE_BASED,
            "trainable": WeightingStrategyType.TRAINABLE_WEIGHTS,
            "trainable_weights": WeightingStrategyType.TRAINABLE_WEIGHTS,
        }
        if strategy_raw not in strategy_aliases:
            raise ValueError(f"Unsupported ensemble weighting strategy: {strategy_raw}")
        strategy = strategy_aliases[strategy_raw]

        prechecks_conf = ensemble_conf.get("prechecks", {}) if isinstance(ensemble_conf.get("prechecks", {}), Mapping) else {}
        selected_embeddings = sorted({artifact.embedding_name for artifact in model_artifacts})
        selected_classifiers = sorted({artifact.classifier_name for artifact in model_artifacts})
        weighting_params = dict(weighting_conf.get("params", {})) if isinstance(weighting_conf.get("params", {}), Mapping) else {}

        config = EnsembleConfig(
            enabled=bool(ensemble_conf.get("enabled", True)),
            mode=mode,
            selection=EnsembleSelectionConfig(
                embeddings=selected_embeddings,
                classifiers=selected_classifiers,
            ),
            weighting=WeightingConfig(
                strategy=strategy,
                metric=weighting_conf.get("metric"),
                params=weighting_params,
            ),
            enforce_same_normalization=bool(prechecks_conf.get("enforce_same_normalization", False)),
            min_models=int(prechecks_conf.get("min_models", 2)),
        )
        return config, strategy, weighting_params

    @staticmethod
    def _build_benchmark_ablation_specs(ablations_conf: Any) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = [{"name": "default", "selection": {}}]
        if not isinstance(ablations_conf, list):
            return specs
        for index, item in enumerate(ablations_conf, start=1):
            if not isinstance(item, Mapping):
                continue
            selection = {}
            if isinstance(item.get("embeddings"), list):
                selection["embeddings"] = [str(v) for v in item.get("embeddings", [])]
            if isinstance(item.get("classifiers"), list):
                selection["classifiers"] = [str(v) for v in item.get("classifiers", [])]
            if not selection:
                continue
            specs.append({"name": f"ablation_{index}", "selection": selection})
        return specs

    @staticmethod
    def _build_benchmark_variant_specs(
        metric_name: str,
        include_uniform: bool,
        include_validation_weighted: bool,
        include_trainable: bool,
        include_majority: bool,
        trainable_params: Any,
    ) -> list[dict[str, Any]]:
        variant_specs: list[dict[str, Any]] = []
        if include_uniform:
            variant_specs.append(
                {
                    "variant": "uniform_soft_voting",
                    "mode": EnsembleMode.GLOBAL_SOFT,
                    "strategy": WeightingStrategyType.UNIFORM,
                    "metric": metric_name,
                    "params": {},
                }
            )
        if include_validation_weighted:
            variant_specs.append(
                {
                    "variant": "validation_weighted_soft_voting",
                    "mode": EnsembleMode.GLOBAL_SOFT,
                    "strategy": WeightingStrategyType.VALIDATION_SCORE_BASED,
                    "metric": metric_name,
                    "params": {},
                }
            )
        if include_trainable:
            variant_specs.append(
                {
                    "variant": "trainable_weights_soft_voting",
                    "mode": EnsembleMode.GLOBAL_SOFT,
                    "strategy": WeightingStrategyType.TRAINABLE_WEIGHTS,
                    "metric": metric_name,
                    "params": dict(trainable_params) if isinstance(trainable_params, Mapping) else {},
                }
            )
        if include_majority:
            variant_specs.extend(
                [
                    {
                        "variant": "majority_global",
                        "mode": EnsembleMode.MAJORITY_GLOBAL,
                        "strategy": WeightingStrategyType.UNIFORM,
                        "metric": metric_name,
                        "params": {},
                    },
                    {
                        "variant": "majority_by_embedding",
                        "mode": EnsembleMode.MAJORITY_BY_EMBEDDING,
                        "strategy": WeightingStrategyType.UNIFORM,
                        "metric": metric_name,
                        "params": {},
                    },
                    {
                        "variant": "majority_by_classifier",
                        "mode": EnsembleMode.MAJORITY_BY_CLASSIFIER,
                        "strategy": WeightingStrategyType.UNIFORM,
                        "metric": metric_name,
                        "params": {},
                    },
                ]
            )
        return variant_specs

    @staticmethod
    def _with_seeded_dataset_config(dataset_conf: dict[str, Any], seed: int) -> dict[str, Any]:
        cloned = copy.deepcopy(dataset_conf)
        if isinstance(cloned.get("split"), Mapping):
            cloned["split"]["random_state"] = int(seed)
        return cloned

    @staticmethod
    def _comparison_rows_from_seed_results(benchmark_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "Category": row.get("category"),
                "Variant": row.get("variant"),
                "Validation F1": float(row.get("validation_f1", float("nan"))),
                "Test F1": float(row.get("test_f1", float("nan"))),
                "Delta vs Best Single (Test)": float(row.get("delta_vs_best_single_test", float("nan"))),
                "Num Models": int(row.get("num_models", 0)),
                "Weighting Strategy": row.get("weighting_strategy", ""),
            }
            for row in benchmark_results
        ]

    @staticmethod
    def _ranking_from_results(benchmark_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [
                {
                    "variant": row.get("variant"),
                    "category": row.get("category"),
                    "test_f1": float(row.get("test_f1", float("nan"))),
                }
                for row in benchmark_results
            ],
            key=lambda item: float(item.get("test_f1", float("nan"))),
            reverse=True,
        )

    @staticmethod
    def _aggregate_benchmark_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in seed_rows:
            key = (str(row.get("ablation", "default")), str(row.get("variant", "")))
            grouped.setdefault(key, []).append(row)

        aggregated: list[dict[str, Any]] = []
        for (ablation, variant), rows in grouped.items():
            val_values = np.asarray([float(row.get("validation_f1", float("nan"))) for row in rows], dtype=float)
            test_values = np.asarray([float(row.get("test_f1", float("nan"))) for row in rows], dtype=float)
            delta_values = np.asarray([float(row.get("delta_vs_best_single_test", float("nan"))) for row in rows], dtype=float)
            gap_values = np.asarray([float(row.get("generalization_gap", float("nan"))) for row in rows], dtype=float)
            base = rows[0]
            aggregated.append(
                {
                    "ablation": ablation,
                    "category": base.get("category"),
                    "variant": variant,
                    "mean_validation_f1": float(np.nanmean(val_values)),
                    "std_validation_f1": float(np.nanstd(val_values)),
                    "mean_test_f1": float(np.nanmean(test_values)),
                    "std_test_f1": float(np.nanstd(test_values)),
                    "mean_delta_vs_best": float(np.nanmean(delta_values)),
                    "std_delta_vs_best": float(np.nanstd(delta_values)),
                    "mean_generalization_gap": float(np.nanmean(gap_values)),
                    "num_models": int(base.get("num_models", 0)),
                    "weighting_strategy": base.get("weighting_strategy", ""),
                    "successful_runs": len(rows),
                }
            )
        return sorted(aggregated, key=lambda row: (str(row.get("ablation")), str(row.get("variant"))))

    @staticmethod
    def _benchmark_summary_rows_from_aggregated(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "Category": row.get("category"),
                "Variant": row.get("variant"),
                "Validation F1": row.get("mean_validation_f1"),
                "Test F1": row.get("mean_test_f1"),
                "Delta vs Best Single (Test)": row.get("mean_delta_vs_best"),
                "Num Models": row.get("num_models"),
                "Weighting Strategy": row.get("weighting_strategy"),
            }
            for row in aggregated_rows
        ]

    @staticmethod
    def _write_benchmark_multiseed_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "Category",
            "Variant",
            "Mean Val F1",
            "Std Val F1",
            "Mean Test F1",
            "Std Test F1",
            "Mean Delta vs Best",
            "Std Delta vs Best",
            "Num Models",
            "Weighting Strategy",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Category": row.get("category"),
                        "Variant": row.get("variant"),
                        "Mean Val F1": row.get("mean_validation_f1"),
                        "Std Val F1": row.get("std_validation_f1"),
                        "Mean Test F1": row.get("mean_test_f1"),
                        "Std Test F1": row.get("std_test_f1"),
                        "Mean Delta vs Best": row.get("mean_delta_vs_best"),
                        "Std Delta vs Best": row.get("std_delta_vs_best"),
                        "Num Models": row.get("num_models"),
                        "Weighting Strategy": row.get("weighting_strategy"),
                    }
                )

    @staticmethod
    def _write_benchmark_ablation_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "Ablation",
            "Category",
            "Variant",
            "Mean Val F1",
            "Std Val F1",
            "Mean Test F1",
            "Std Test F1",
            "Mean Delta vs Best",
            "Std Delta vs Best",
            "Num Models",
            "Weighting Strategy",
            "Successful Runs",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Ablation": row.get("ablation"),
                        "Category": row.get("category"),
                        "Variant": row.get("variant"),
                        "Mean Val F1": row.get("mean_validation_f1"),
                        "Std Val F1": row.get("std_validation_f1"),
                        "Mean Test F1": row.get("mean_test_f1"),
                        "Std Test F1": row.get("std_test_f1"),
                        "Mean Delta vs Best": row.get("mean_delta_vs_best"),
                        "Std Delta vs Best": row.get("std_delta_vs_best"),
                        "Num Models": row.get("num_models"),
                        "Weighting Strategy": row.get("weighting_strategy"),
                        "Successful Runs": row.get("successful_runs"),
                    }
                )

    def _build_benchmark_weights_analysis(self, weight_records: list[dict[str, Any]]) -> dict[str, Any]:
        if not weight_records:
            return {"trainable_weights": [], "aggregated": []}

        aggregated: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in weight_records:
            grouped.setdefault(str(record.get("ablation", "default")), []).append(record)

        for ablation, records in grouped.items():
            model_union: list[str] = []
            for record in records:
                for model in record.get("models_used", []):
                    if model not in model_union:
                        model_union.append(model)

            matrix: list[list[float]] = []
            val_scores: list[list[float]] = []
            for record in records:
                model_to_weight = {
                    model: float(weight)
                    for model, weight in zip(record.get("models_used", []), record.get("weights", []))
                }
                model_to_score = {
                    model: float(score)
                    for model, score in (record.get("model_validation_scores", {}) or {}).items()
                }
                matrix.append([model_to_weight.get(model, 0.0) for model in model_union])
                val_scores.append([model_to_score.get(model, float("nan")) for model in model_union])

            weight_matrix = np.asarray(matrix, dtype=float)
            score_matrix = np.asarray(val_scores, dtype=float)
            mean_weight = np.nanmean(weight_matrix, axis=0)
            std_weight = np.nanstd(weight_matrix, axis=0)
            entropy_values = []
            correlations = []
            for idx in range(weight_matrix.shape[0]):
                w = np.asarray(weight_matrix[idx], dtype=float)
                eps = 1e-12
                entropy = float(-np.sum(w * np.log(w + eps)))
                entropy_values.append(entropy)

                s = np.asarray(score_matrix[idx], dtype=float)
                finite = np.isfinite(s)
                if np.count_nonzero(finite) > 1 and np.std(w[finite]) > 0 and np.std(s[finite]) > 0:
                    correlations.append(float(np.corrcoef(s[finite], w[finite])[0, 1]))

            aggregated.append(
                {
                    "ablation": ablation,
                    "models": model_union,
                    "per_seed_weights": [
                        {
                            "seed": record.get("seed"),
                            "weights": record.get("weights"),
                            "models": record.get("models_used"),
                            "validation_f1": record.get("validation_f1"),
                            "test_f1": record.get("test_f1"),
                        }
                        for record in records
                    ],
                    "mean_weight_per_model": {
                        model: float(mean_weight[index]) for index, model in enumerate(model_union)
                    },
                    "std_weight_per_model": {
                        model: float(std_weight[index]) for index, model in enumerate(model_union)
                    },
                    "mean_entropy": float(np.nanmean(np.asarray(entropy_values, dtype=float))),
                    "std_entropy": float(np.nanstd(np.asarray(entropy_values, dtype=float))),
                    "mean_val_weight_correlation": float(np.nanmean(np.asarray(correlations, dtype=float)))
                    if correlations
                    else float("nan"),
                }
            )

        return {
            "trainable_weights": weight_records,
            "aggregated": aggregated,
        }

    def _build_overfitting_report(self, seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in seed_rows:
            key = (str(row.get("ablation", "default")), str(row.get("variant", "")))
            grouped.setdefault(key, []).append(row)

        summary: list[dict[str, Any]] = []
        for (ablation, variant), rows in grouped.items():
            gaps = np.asarray([float(row.get("generalization_gap", float("nan"))) for row in rows], dtype=float)
            test_values = np.asarray([float(row.get("test_f1", float("nan"))) for row in rows], dtype=float)
            summary.append(
                {
                    "ablation": ablation,
                    "variant": variant,
                    "mean_generalization_gap": float(np.nanmean(gaps)),
                    "std_generalization_gap": float(np.nanstd(gaps)),
                    "test_f1_variance": float(np.nanvar(test_values)),
                }
            )

        for ablation in sorted({row.get("ablation") for row in summary}):
            uniform_row = next((row for row in summary if row.get("ablation") == ablation and row.get("variant") == "uniform_soft_voting"), None)
            trainable_row = next((row for row in summary if row.get("ablation") == ablation and row.get("variant") == "trainable_weights_soft_voting"), None)
            if uniform_row and trainable_row and float(trainable_row.get("test_f1_variance", 0.0)) > float(uniform_row.get("test_f1_variance", 0.0)):
                self.logger.warning(
                    "Potential overfitting ablation=%s: trainable ensemble variance %.6f > uniform variance %.6f",
                    ablation,
                    float(trainable_row.get("test_f1_variance", 0.0)),
                    float(uniform_row.get("test_f1_variance", 0.0)),
                )

        return {"variants": summary}

    @staticmethod
    def _sha256_file(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _log_best_single_diagnostics(self, best_single: dict[str, Any], seed: int, ablation_name: str) -> None:
        diagnostics = best_single.get("diagnostics", {}) if isinstance(best_single.get("diagnostics", {}), Mapping) else {}
        class_distribution = diagnostics.get("class_distribution", {})
        macro_f1 = diagnostics.get("macro_f1", float("nan"))
        micro_f1 = diagnostics.get("micro_f1", float("nan"))
        n_val = diagnostics.get("n_val", 0)
        per_class_f1 = diagnostics.get("per_class_f1", {})

        self.logger.info(
            "Best single diagnostics seed=%d ablation=%s model=%s n_val=%s class_distribution=%s macro_f1=%.4f micro_f1=%.4f",
            seed,
            ablation_name,
            best_single.get("model"),
            n_val,
            class_distribution,
            float(macro_f1),
            float(micro_f1),
        )
        self.logger.info(
            "Best single per-class F1 seed=%d ablation=%s: %s",
            seed,
            ablation_name,
            per_class_f1,
        )

        if float(best_single.get("validation_f1", float("nan"))) >= 0.999999:
            self.logger.warning(
                "Validation F1 == 1.0 for seed=%d ablation=%s model=%s; possible overfitting",
                seed,
                ablation_name,
                best_single.get("model"),
            )
            confusion = diagnostics.get("confusion_matrix")
            if confusion is not None:
                self.logger.warning(
                    "Best single confusion matrix seed=%d ablation=%s model=%s:\n%s",
                    seed,
                    ablation_name,
                    best_single.get("model"),
                    confusion,
                )

    def _evaluate_model_artifact_scores(
        self,
        artifact: ModelArtifact,
        x_val_map: Mapping[str, np.ndarray],
        y_val: np.ndarray,
        x_test_map: Mapping[str, np.ndarray],
        y_test: np.ndarray,
        include_diagnostics: bool = False,
    ) -> tuple[float, float, dict[str, Any] | None]:
        probs_val = ProbabilityAdapter.to_canonical(
            raw_output=np.asarray(artifact.model.predict_proba(x_val_map)),
            problem_type=artifact.problem_type,
            classes=artifact.classes,
            context=f"benchmark/{artifact.classifier_name}/{artifact.embedding_name}/validation",
        )
        preds_val = decide(
            probs=probs_val,
            problem_type=artifact.problem_type,
            threshold_config=artifact.threshold_policy,
        )
        val_labels = self._coerce_predictions_to_label_space(np.asarray(preds_val), classes=artifact.classes)
        val_f1 = _benchmark_f1_score(
            problem_type=artifact.problem_type,
            y_true=np.asarray(y_val),
            y_pred=np.asarray(val_labels),
            classes=artifact.classes,
        )

        probs_test = ProbabilityAdapter.to_canonical(
            raw_output=np.asarray(artifact.model.predict_proba(x_test_map)),
            problem_type=artifact.problem_type,
            classes=artifact.classes,
            context=f"benchmark/{artifact.classifier_name}/{artifact.embedding_name}/test",
        )
        preds_test = decide(
            probs=probs_test,
            problem_type=artifact.problem_type,
            threshold_config=artifact.threshold_policy,
        )
        test_labels = self._coerce_predictions_to_label_space(np.asarray(preds_test), classes=artifact.classes)
        test_f1 = _benchmark_f1_score(
            problem_type=artifact.problem_type,
            y_true=np.asarray(y_test),
            y_pred=np.asarray(test_labels),
            classes=artifact.classes,
        )

        if not include_diagnostics:
            return val_f1, test_f1, None

        diagnostics: dict[str, Any] = {
            "n_val": int(np.asarray(y_val).shape[0]),
            "macro_f1": val_f1,
            "micro_f1": float("nan"),
            "per_class_f1": {},
            "class_distribution": {},
            "confusion_matrix": None,
        }

        if artifact.problem_type == "multilabel":
            y_val_bin = _to_multilabel_matrix(np.asarray(y_val), artifact.classes)
            y_pred_bin = np.asarray(val_labels)
            if y_pred_bin.ndim != 2:
                y_pred_bin = _to_multilabel_matrix(y_pred_bin, artifact.classes)

            diagnostics["micro_f1"] = float(f1_score(y_val_bin, y_pred_bin, average="micro", zero_division=0))
            per_class = f1_score(y_val_bin, y_pred_bin, average=None, zero_division=0)
            diagnostics["per_class_f1"] = {
                str(label): float(per_class[idx])
                for idx, label in enumerate(artifact.classes)
                if idx < len(per_class)
            }
            diagnostics["class_distribution"] = {
                str(label): int(y_val_bin[:, idx].sum())
                for idx, label in enumerate(artifact.classes)
                if idx < y_val_bin.shape[1]
            }
        else:
            y_val_arr = np.asarray(y_val)
            pred_arr = np.asarray(val_labels)
            diagnostics["micro_f1"] = float(f1_score(y_val_arr, pred_arr, average="micro", zero_division=0))
            labels = np.unique(y_val_arr)
            per_class = f1_score(y_val_arr, pred_arr, labels=labels, average=None, zero_division=0)
            diagnostics["per_class_f1"] = {
                str(label): float(per_class[idx]) for idx, label in enumerate(labels) if idx < len(per_class)
            }
            diagnostics["class_distribution"] = {
                str(label): int(np.sum(y_val_arr == label)) for label in labels
            }
            try:
                diagnostics["confusion_matrix"] = confusion_matrix(y_val_arr, pred_arr, labels=labels).tolist()
            except Exception:
                diagnostics["confusion_matrix"] = None

        return val_f1, test_f1, diagnostics

    def _run_benchmark_ensemble_variant(
        self,
        spec: dict[str, Any],
        model_artifacts: list[ModelArtifact],
        x_val_map: Mapping[str, np.ndarray],
        y_val: np.ndarray,
        x_test_map: Mapping[str, np.ndarray],
        y_test: np.ndarray,
        selected_payloads: list[dict[str, Any]],
        benchmark_models_dir: Path,
    ) -> dict[str, Any]:
        variant_name = str(spec.get("variant", "ensemble_variant"))
        mode = EnsembleMode(spec.get("mode", EnsembleMode.GLOBAL_SOFT))
        strategy = WeightingStrategyType(spec.get("strategy", WeightingStrategyType.UNIFORM))

        ensemble_conf = {
            "enabled": True,
            "mode": mode.value,
            "weighting": {
                "strategy": strategy.value,
                "metric": spec.get("metric", "f1_macro"),
                "params": dict(spec.get("params", {})) if isinstance(spec.get("params", {}), Mapping) else {},
            },
        }
        service_config, strategy_type, weighting_params = self._build_soft_voting_service_config(
            ensemble_conf=ensemble_conf,
            model_artifacts=model_artifacts,
        )

        weight_trainer = None
        if strategy_type == WeightingStrategyType.TRAINABLE_WEIGHTS:
            weight_trainer = _ValidationWeightTrainer(
                random_seed=int(weighting_params.get("random_seed", 42)),
                n_trials=int(weighting_params.get("n_trials", 256)),
            )

        majority_service = SimpleMajorityVotingService() if mode != EnsembleMode.GLOBAL_SOFT else None
        service = SoftVotingService(
            model_artifacts=model_artifacts,
            config=service_config,
            weight_trainer=weight_trainer,
            majority_voting_service=majority_service,
        )

        val_probabilities = service.collect_validation_probabilities(x_val_map)
        service.fit_with_validation(x_val_map, np.asarray(y_val))
        val_output = service.predict(x_val_map)
        test_output = service.predict(x_test_map)

        val_labels = self._coerce_predictions_to_label_space(
            np.asarray(val_output.get("labels")),
            classes=np.asarray(test_output.get("metadata", {}).get("classes", []), dtype=object).tolist(),
        )
        test_labels = self._coerce_predictions_to_label_space(
            np.asarray(test_output.get("labels")),
            classes=np.asarray(test_output.get("metadata", {}).get("classes", []), dtype=object).tolist(),
        )
        ensemble_problem_type = str(test_output.get("metadata", {}).get("problem_type", "binary"))
        ensemble_classes = np.asarray(test_output.get("metadata", {}).get("classes", []), dtype=object).tolist()

        val_f1 = _benchmark_f1_score(
            problem_type=ensemble_problem_type,
            y_true=np.asarray(y_val),
            y_pred=np.asarray(val_labels),
            classes=ensemble_classes,
        )
        test_f1 = _benchmark_f1_score(
            problem_type=ensemble_problem_type,
            y_true=np.asarray(y_test),
            y_pred=np.asarray(test_labels),
            classes=ensemble_classes,
        )
        weights = np.asarray(test_output.get("metadata", {}).get("ensemble", {}).get("weights", []), dtype=np.float64)

        summary_rows = self._build_ensemble_summary_rows(
            model_artifacts=model_artifacts,
            selected_payloads=selected_payloads,
            validation_probabilities=val_probabilities,
            y_val=np.asarray(y_val),
            weights=weights,
        )

        if strategy_type == WeightingStrategyType.TRAINABLE_WEIGHTS:
            self._save_ensemble_artifact(
                output_dir=benchmark_models_dir,
                ensemble_conf=ensemble_conf,
                ensemble_output=test_output,
                summary_rows=summary_rows,
                artifact_name=f"benchmark_{variant_name}",
            )

        return {
            "category": "Ensemble",
            "variant": variant_name,
            "validation_f1": val_f1,
            "test_f1": test_f1,
            "num_models": len(model_artifacts),
            "weighting_strategy": strategy.value,
            "mode": mode.value,
            "models_used": [f"{artifact.classifier_name}::{artifact.embedding_name}" for artifact in model_artifacts],
            "weights": weights.tolist() if weights.size else [],
            "model_validation_scores": {
                str(row.get("model")): float(row.get("validation_score", float("nan"))) for row in summary_rows
            },
        }

    @staticmethod
    def _coerce_predictions_to_label_space(predictions: np.ndarray, classes: list[Any]) -> np.ndarray:
        preds = np.asarray(predictions)
        class_array = np.asarray(classes)
        if preds.ndim == 1 and class_array.size > 0 and np.issubdtype(preds.dtype, np.integer):
            if preds.size == 0:
                return preds
            min_index = int(np.min(preds))
            max_index = int(np.max(preds))
            if min_index >= 0 and max_index < class_array.size:
                return class_array[preds]
        return preds

    @staticmethod
    def _write_benchmark_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = [
            "Category",
            "Variant",
            "Validation F1",
            "Test F1",
            "Delta vs Best Single (Test)",
            "Num Models",
            "Weighting Strategy",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _format_benchmark_table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "(empty)"

        category_w = max(len("Category"), *(len(str(row.get("Category", ""))) for row in rows))
        variant_w = max(len("Variant"), *(len(str(row.get("Variant", ""))) for row in rows))
        val_w = len("Validation F1")
        test_w = len("Test F1")
        delta_w = len("Delta")
        num_w = len("Num")
        weight_w = max(len("Weighting Strategy"), *(len(str(row.get("Weighting Strategy", ""))) for row in rows))

        lines = [
            f"{'Category':<{category_w}}  {'Variant':<{variant_w}}  {'Validation F1':>{val_w}}  {'Test F1':>{test_w}}  {'Delta':>{delta_w}}  {'Num':>{num_w}}  {'Weighting Strategy':<{weight_w}}",
            f"{'-' * category_w}  {'-' * variant_w}  {'-' * val_w}  {'-' * test_w}  {'-' * delta_w}  {'-' * num_w}  {'-' * weight_w}",
        ]
        for row in rows:
            lines.append(
                f"{str(row.get('Category', '')):<{category_w}}  "
                f"{str(row.get('Variant', '')):<{variant_w}}  "
                f"{float(row.get('Validation F1', float('nan'))):>{val_w}.4f}  "
                f"{float(row.get('Test F1', float('nan'))):>{test_w}.4f}  "
                f"{float(row.get('Delta vs Best Single (Test)', float('nan'))):>{delta_w}.4f}  "
                f"{int(row.get('Num Models', 0)):>{num_w}d}  "
                f"{str(row.get('Weighting Strategy', '')):<{weight_w}}"
            )
        return "\n".join(lines)

    def _select_ensemble_rows(
        self,
        rows: list[dict[str, Any]],
        ensemble_conf: dict[str, Any],
        training_global_conf: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        selection = ensemble_conf.get("selection", {}) if isinstance(ensemble_conf.get("selection", {}), Mapping) else {}

        available_embeddings = sorted({str(row.get("embedding_name", "")) for row in rows if row.get("embedding_name")})
        available_classifiers = sorted({str(row.get("model_type", "")) for row in rows if row.get("model_type")})

        selected_embeddings = set(
            self._resolve_selected_embeddings(
                available=available_embeddings,
                filters=self.runtime_filters,
                training_global_conf=training_global_conf,
            )
        )
        selected_classifiers = set(
            self._resolve_selected_classifiers(
                available=available_classifiers,
                filters=self.runtime_filters,
                training_global_conf=training_global_conf,
            )
        )

        conf_embeddings_raw = selection.get("embeddings")
        if isinstance(conf_embeddings_raw, list) and conf_embeddings_raw:
            conf_embeddings = {str(item) for item in conf_embeddings_raw}
            selected_embeddings &= conf_embeddings

        conf_classifiers_raw = selection.get("classifiers")
        if isinstance(conf_classifiers_raw, list) and conf_classifiers_raw:
            conf_classifiers = {str(item) for item in conf_classifiers_raw}
            selected_classifiers &= conf_classifiers

        filtered = [
            dict(row)
            for row in rows
            if str(row.get("embedding_name")) in selected_embeddings
            and str(row.get("model_type")) in selected_classifiers
        ]

        if not filtered:
            raise ValueError("No ensemble candidate models after applying config/runtime selection filters")

        requested = len(filtered)
        if isinstance(conf_embeddings_raw, list) and conf_embeddings_raw:
            requested = len(conf_embeddings_raw)
        if isinstance(conf_classifiers_raw, list) and conf_classifiers_raw:
            requested = max(requested, len(conf_classifiers_raw))

        top_k_per_embedding = selection.get("top_k_per_embedding")
        if isinstance(top_k_per_embedding, int) and top_k_per_embedding > 0:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in filtered:
                grouped.setdefault(str(row.get("embedding_name")), []).append(row)
            filtered = []
            for group_rows in grouped.values():
                ranked = sorted(group_rows, key=lambda row: self._safe_float(row.get("validation_f1")), reverse=True)
                filtered.extend(ranked[:top_k_per_embedding])

        top_k_per_classifier = selection.get("top_k_per_classifier")
        if isinstance(top_k_per_classifier, int) and top_k_per_classifier > 0:
            grouped = {}
            for row in filtered:
                grouped.setdefault(str(row.get("model_type")), []).append(row)
            reduced: list[dict[str, Any]] = []
            for group_rows in grouped.values():
                ranked = sorted(group_rows, key=lambda row: self._safe_float(row.get("validation_f1")), reverse=True)
                reduced.extend(ranked[:top_k_per_classifier])
            filtered = reduced

        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for row in filtered:
            key = (str(row.get("model_type")), str(row.get("embedding_name")), str(row.get("artifact_path", "")))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)

        return deduped, max(1, requested)

    def _load_ensemble_model_artifacts(
        self,
        run_dir: Path,
        selected_rows: list[dict[str, Any]],
    ) -> tuple[list[ModelArtifact], list[dict[str, Any]]]:
        model_artifacts: list[ModelArtifact] = []
        payloads: list[dict[str, Any]] = []

        for row in selected_rows:
            classifier = str(row.get("model_type", ""))
            embedding_name = str(row.get("embedding_name", ""))
            artifact_rel = str(row.get("artifact_path", ""))
            serializer = str(row.get("serializer", "pickle"))
            if not artifact_rel:
                self.logger.warning(
                    "Skipping ensemble candidate model_type=%s embedding=%s due to missing artifact_path",
                    classifier,
                    embedding_name,
                )
                continue

            model_path = run_dir / artifact_rel
            if not model_path.exists() and artifact_rel.startswith(f"{run_dir.name}/"):
                model_path = run_dir / artifact_rel.split("/", 1)[1]
            if not model_path.exists():
                self.logger.warning(
                    "Skipping ensemble candidate model_type=%s embedding=%s missing artifact=%s",
                    classifier,
                    embedding_name,
                    model_path,
                )
                continue

            metadata_path = str(row.get("metadata_path", ""))
            resolved_metadata_path = run_dir / metadata_path if metadata_path else model_path.with_suffix(".metadata.json")
            if not resolved_metadata_path.exists() and metadata_path.startswith(f"{run_dir.name}/"):
                resolved_metadata_path = run_dir / metadata_path.split("/", 1)[1]
            if not resolved_metadata_path.exists():
                self.logger.warning(
                    "Skipping ensemble candidate model_type=%s embedding=%s missing metadata=%s",
                    classifier,
                    embedding_name,
                    resolved_metadata_path,
                )
                continue

            with resolved_metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle) or {}
            model = self._load_model_artifact(model_path=model_path, serializer=serializer)
            wrapped_model = _EmbeddingViewPredictor(model=model, embedding_name=embedding_name)

            artifact = ModelArtifact.from_metadata(model=wrapped_model, metadata=metadata)
            if not artifact.classifier_name:
                artifact = ModelArtifact(
                    model=artifact.model,
                    problem_type=artifact.problem_type,
                    classes=artifact.classes,
                    num_classes=artifact.num_classes,
                    normalization=artifact.normalization,
                    threshold_policy=artifact.threshold_policy,
                    classifier_name=classifier,
                    embedding_name=embedding_name,
                    metadata=artifact.metadata,
                )

            model_artifacts.append(artifact)
            payloads.append(
                {
                    "classifier_name": classifier,
                    "embedding_name": embedding_name,
                    "artifact_path": str(artifact_rel),
                    "serializer": serializer,
                    "metadata_path": str(metadata_path or resolved_metadata_path.relative_to(run_dir)),
                    "validation_f1": self._safe_float(row.get("validation_f1")),
                }
            )

        return model_artifacts, payloads

    def _build_ensemble_summary_rows(
        self,
        model_artifacts: list[ModelArtifact],
        selected_payloads: list[dict[str, Any]],
        validation_probabilities: np.ndarray,
        y_val: np.ndarray,
        weights: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, artifact in enumerate(model_artifacts):
            probs = np.asarray(validation_probabilities[index])
            preds = decide(
                probs=probs,
                problem_type=artifact.problem_type,
                threshold_config=artifact.threshold_policy,
            )
            if artifact.problem_type != "multilabel":
                preds_array = np.asarray(preds)
                class_array = np.asarray(artifact.classes)
                if preds_array.ndim == 1 and np.issubdtype(preds_array.dtype, np.integer):
                    if preds_array.size == 0 or (
                        int(np.min(preds_array)) >= 0 and int(np.max(preds_array)) < len(class_array)
                    ):
                        preds = class_array[preds_array]

            val_score = _benchmark_f1_score(
                problem_type=artifact.problem_type,
                y_true=np.asarray(y_val),
                y_pred=np.asarray(preds),
                classes=artifact.classes,
            )
            payload = selected_payloads[index]
            rows.append(
                {
                    "model": f"{artifact.classifier_name}::{artifact.embedding_name}",
                    "classifier_name": artifact.classifier_name,
                    "embedding_name": artifact.embedding_name,
                    "validation_score": val_score,
                    "weight": float(weights[index]) if index < len(weights) else float("nan"),
                    "artifact_path": payload.get("artifact_path"),
                    "serializer": payload.get("serializer"),
                    "metadata_path": payload.get("metadata_path"),
                }
            )
        return rows

    @staticmethod
    def _format_ensemble_summary(rows: list[dict[str, Any]]) -> str:
        headers = ("model", "validation_f1", "weight")
        model_width = max(len(headers[0]), *(len(str(row.get("model", ""))) for row in rows))
        score_width = len(headers[1])
        weight_width = len(headers[2])

        output = [
            f"{headers[0]:<{model_width}}  {headers[1]:>{score_width}}  {headers[2]:>{weight_width}}",
            f"{'-' * model_width}  {'-' * score_width}  {'-' * weight_width}",
        ]
        for row in rows:
            output.append(
                f"{str(row.get('model', '')):<{model_width}}  "
                f"{float(row.get('validation_score', float('nan'))):>{score_width}.4f}  "
                f"{float(row.get('weight', float('nan'))):>{weight_width}.4f}"
            )
        return "\n".join(output)

    def _save_ensemble_artifact(
        self,
        output_dir: Path,
        ensemble_conf: dict[str, Any],
        ensemble_output: dict[str, Any],
        summary_rows: list[dict[str, Any]],
        artifact_name: str = "ensemble_model",
    ) -> None:
        ensemble_pkl = output_dir / f"{artifact_name}.pkl"
        metadata_path = output_dir / f"{artifact_name}.metadata.json"

        metadata = dict(ensemble_output.get("metadata", {}))
        ensemble_meta = metadata.get("ensemble", {}) if isinstance(metadata.get("ensemble", {}), Mapping) else {}

        persistence_payload = {
            "ensemble_config": dict(ensemble_conf),
            "weights": list(ensemble_meta.get("weights", [])),
            "models_used": summary_rows,
            "problem_type": metadata.get("problem_type"),
            "classes": metadata.get("classes"),
            "num_classes": metadata.get("num_classes"),
            "threshold_policy": metadata.get("threshold_policy", {}),
            "weighting_strategy": ensemble_meta.get("weighting", {}),
            "mode": ensemble_meta.get("mode"),
        }

        with ensemble_pkl.open("wb") as handle:
            pickle.dump(persistence_payload, handle)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(persistence_payload, handle, indent=2)

        self.logger.info("Saved ensemble artifact: %s", ensemble_pkl)
        self.logger.info("Saved ensemble metadata: %s", metadata_path)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def run_evaluate_step(self, conf: dict[str, Any]) -> None:
        _ = conf
        if not bool(self.runtime_context.get("evaluate_last_sweep", False)):
            self.logger.info("Evaluate step placeholder (use --evaluate-last-sweep for fast evaluation mode).")
            return

        pipeline_conf = self._load_yaml(self.config_path)
        dataset_conf = pipeline_conf.get("dataset", pipeline_conf)
        embeddings_step_conf = pipeline_conf.get("embeddings", {})
        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        embeddings_conf = self._load_yaml(embeddings_config_path)
        training_global_conf = self._load_training_config(pipeline_conf)
        reporting_conf = self._build_reporting_config(training_global_conf)
        path_layout = self._ensure_pec_data_layout(reporting_conf)

        latest_run_dir = self._get_latest_sweep_run(path_layout["sweep"])
        if latest_run_dir is None:
            raise FileNotFoundError("No previous sweep run found in pec_data/sweep")

        best_csv = latest_run_dir / "reports" / "best_classifier_per_embedding.csv"
        if not best_csv.exists():
            raise FileNotFoundError(f"best_classifier_per_embedding.csv not found in {latest_run_dir}")

        best_rows = self._read_best_classifier_per_embedding_csv(best_csv)

        dataset_bundle = self._build_dataset_bundle(dataset_conf)
        embedding_bundle = self._build_embedding_bundle_from_dataset(
            dataset_bundle=dataset_bundle,
            dataset_conf=dataset_conf,
            embeddings_conf=embeddings_conf,
        )

        reports_dir = latest_run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        for row in best_rows:
            classifier = str(row["model_type"])
            embedding_name = str(row["embedding_name"])
            artifact_rel = row.get("artifact_path")
            serializer = row.get("serializer", "pickle")
            if not artifact_rel:
                self.logger.warning(
                    "Skipping evaluate-last-sweep model_type=%s embedding=%s due to missing artifact path",
                    classifier,
                    embedding_name,
                )
                continue

            model_path = latest_run_dir / str(artifact_rel)
            if not model_path.exists() and str(artifact_rel).startswith(f"{latest_run_dir.name}/"):
                model_path = latest_run_dir / str(artifact_rel).split("/", 1)[1]
            if not model_path.exists():
                self.logger.warning(
                    "Skipping evaluate-last-sweep model_type=%s embedding=%s missing artifact=%s",
                    classifier,
                    embedding_name,
                    model_path,
                )
                continue

            model = self._load_model_artifact(model_path=model_path, serializer=serializer)
            y_train_all = np.asarray(dataset_bundle.y_train, dtype=object)
            problem_spec = ProblemSpecification.from_labels(y_train_all)
            problem_type = problem_spec.problem_type
            if problem_type == "multilabel":
                self.logger.info(
                    "Skipping confusion matrix for model_type=%s embedding=%s because problem_type=multilabel",
                    classifier,
                    embedding_name,
                )
                continue

            x_val = embedding_bundle.X_val[embedding_name]
            y_val = np.asarray(dataset_bundle.y_val, dtype=object)
            probs_val = ProbabilityAdapter.to_canonical(
                raw_output=np.asarray(model.predict_proba(x_val)),
                problem_type=problem_type,
                classes=problem_spec.classes,
                context=f"evaluate/{classifier}/{embedding_name}/validation",
            )
            preds_val = decide(
                probs=probs_val,
                problem_type=problem_type,
                threshold_config={
                    **dict(reporting_conf.get("thresholds", {})),
                    "classifier_name": classifier,
                    "embedding_name": embedding_name,
                },
            )
            if problem_type != "multilabel" and len(problem_spec.classes) > 0:
                preds_val = np.asarray(problem_spec.classes)[preds_val]
            matrix_val = confusion_matrix(y_val, preds_val)
            self.logger.info(
                "Validation confusion matrix model_type=%s embedding=%s:\n%s",
                classifier,
                embedding_name,
                matrix_val,
            )
            self._write_confusion_matrix_csv(
                reports_dir / f"confusion_validation__{classifier}__{embedding_name}.csv",
                matrix_val,
            )

            x_full = np.vstack([
                embedding_bundle.X_train[embedding_name],
                embedding_bundle.X_val[embedding_name],
                embedding_bundle.X_test[embedding_name],
            ])
            y_full = np.concatenate([
                np.asarray(dataset_bundle.y_train, dtype=object),
                np.asarray(dataset_bundle.y_val, dtype=object),
                np.asarray(dataset_bundle.y_test, dtype=object),
            ])
            probs_full = ProbabilityAdapter.to_canonical(
                raw_output=np.asarray(model.predict_proba(x_full)),
                problem_type=problem_type,
                classes=problem_spec.classes,
                context=f"evaluate/{classifier}/{embedding_name}/full",
            )
            preds_full = decide(
                probs=probs_full,
                problem_type=problem_type,
                threshold_config={
                    **dict(reporting_conf.get("thresholds", {})),
                    "classifier_name": classifier,
                    "embedding_name": embedding_name,
                },
            )
            if problem_type != "multilabel" and len(problem_spec.classes) > 0:
                preds_full = np.asarray(problem_spec.classes)[preds_full]
            matrix_full = confusion_matrix(y_full, preds_full)
            self.logger.info(
                "Full confusion matrix model_type=%s embedding=%s:\n%s",
                classifier,
                embedding_name,
                matrix_full,
            )
            self._write_confusion_matrix_csv(
                reports_dir / f"confusion_full__{classifier}__{embedding_name}.csv",
                matrix_full,
            )

    def _step_runner_map(self) -> dict[str, Callable[[dict[str, Any]], None]]:
        return {
            "dataset": self.run_dataset_step,
            "embeddings": self.run_embeddings_step,
            "train": self.run_train_step,
            "sweep": self.run_sweep_step,
            "ensemble": self.run_ensemble_step,
            "benchmark": self.run_benchmark_step,
            "evaluate": self.run_evaluate_step,
        }

    @staticmethod
    def _resolve_allowed_missing_model_artifacts(embeddings_conf: dict[str, Any]) -> dict[str, str]:
        missing_conf = embeddings_conf.get("missing_embeddings", {})
        allow_models_conf = missing_conf.get("allow_models", {})
        allowed_missing_model_artifacts: dict[str, str] = {}
        if isinstance(allow_models_conf, dict):
            allowed_missing_model_artifacts.update(
                {str(model_name): str(path) for model_name, path in allow_models_conf.items()}
            )

        geokg_enabled = embeddings_conf.get("GOPE", {}).get("models", {}).get("GeOKG", {}).get("enabled", False)
        if geokg_enabled and "GeOKG" not in allowed_missing_model_artifacts:
            allowed_missing_model_artifacts["GeOKG"] = "artifacts/missing_gope.txt"
        return allowed_missing_model_artifacts

    def _build_embedding_bundle(self, dataset_conf: dict[str, Any], embeddings_conf: dict[str, Any]) -> EmbeddingBundle:
        dataset_bundle = self._build_dataset_bundle(dataset_conf)
        return self._build_embedding_bundle_from_dataset(dataset_bundle, dataset_conf, embeddings_conf)

    def _build_embedding_bundle_from_dataset(
        self,
        dataset_bundle,
        dataset_conf: dict[str, Any],
        embeddings_conf: dict[str, Any],
    ) -> EmbeddingBundle:
        db_conf_path = dataset_conf.get("db_config_path", "config/db.yaml")
        db_conf = load_db_config(db_conf_path)
        engine = create_engine_from_config(db_conf)

        ordered_accessions = dataset_bundle.train_ids + dataset_bundle.val_ids + dataset_bundle.test_ids
        service = EmbeddingService(
            embeddings_config=embeddings_conf,
            engine=engine,
            accessions=ordered_accessions,
        )
        raw_embeddings = service.load_embeddings()
        allowed_missing_model_artifacts = self._resolve_allowed_missing_model_artifacts(embeddings_conf)
        return EmbeddingBundle.from_dataset(
            dataset_bundle,
            raw_embeddings,
            allowed_missing_model_artifacts=allowed_missing_model_artifacts,
        )

    @staticmethod
    def _resolve_classifier_sweeps(sweep_conf: dict[str, Any]) -> dict[str, str]:
        defaults = {
            "LR": "config/model_sweep/sweep_config_lr.yaml",
            "SVM": "config/model_sweep/sweep_config_svm.yaml",
            "RF": "config/model_sweep/sweep_config_rf.yaml",
            "KNN-2": "config/model_sweep/sweep_config_knn.yaml",
            "XGB": "config/model_sweep/sweep_config_xgb.yaml",
            "MLP": "config/model_sweep/sweep_config_mlp.yaml",
        }

        configured = sweep_conf.get("config_paths")
        if not isinstance(configured, dict):
            configured = {}

        resolved = dict(defaults)
        for model_name, path in configured.items():
            resolved[str(model_name)] = str(path)

        legacy_model_type = sweep_conf.get("model_type")
        legacy_config_path = sweep_conf.get("config_path")
        if isinstance(legacy_model_type, str) and isinstance(legacy_config_path, str):
            resolved[legacy_model_type] = legacy_config_path

        return resolved

    @staticmethod
    def _write_best_per_classifier_csv(path: Path, best_results: dict[str, dict[str, Any]]) -> None:
        rows: list[dict[str, Any]] = []
        for model_type, payload in sorted(best_results.items()):
            best_key = payload.get("best_key", (None, None))
            best_embedding = best_key[1] if isinstance(best_key, tuple) and len(best_key) > 1 else None
            rows.append(
                {
                    "model_type": model_type,
                    "best_embedding": best_embedding,
                    "best_metric": payload.get("best_metric"),
                    "sweep_config_path": payload.get("sweep_config_path"),
                    "best_config": yaml.safe_dump(payload.get("best_config", {}), sort_keys=True).strip(),
                }
            )

        fieldnames = ["model_type", "best_embedding", "best_metric", "sweep_config_path", "best_config"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_global_best_csv(path: Path, global_best: dict[str, Any]) -> None:
        row = {
            "model_type": global_best.get("model_type"),
            "embedding_name": global_best.get("embedding_name"),
            "best_metric": global_best.get("best_metric"),
            "sweep_config_path": global_best.get("sweep_config_path"),
            "best_config": yaml.safe_dump(global_best.get("best_config", {}), sort_keys=True).strip(),
        }

        fieldnames = ["model_type", "embedding_name", "best_metric", "sweep_config_path", "best_config"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _write_full_sweep_results_csv(path: Path, trial_results: list[dict[str, Any]]) -> None:
        fieldnames = [
            "model_type",
            "embedding_name",
            "trial_index",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_roc_auc",
            "val_pr_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
        ]

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for result in trial_results:
                validation_metrics = result.get("validation_metrics", {})
                test_metrics = result.get("test_metrics")

                if not isinstance(validation_metrics, Mapping):
                    validation_metrics = {}
                if not isinstance(test_metrics, Mapping):
                    test_metrics = {}

                writer.writerow(
                    {
                        "model_type": result.get("model_type", ""),
                        "embedding_name": result.get("embedding_name", ""),
                        "trial_index": result.get("trial_index", ""),
                        "val_accuracy": validation_metrics.get("accuracy", ""),
                        "val_precision": validation_metrics.get("precision", ""),
                        "val_recall": validation_metrics.get("recall", ""),
                        "val_f1": validation_metrics.get("f1", validation_metrics.get("macro_f1", "")),
                        "val_roc_auc": validation_metrics.get("roc_auc", ""),
                        "val_pr_auc": validation_metrics.get("pr_auc", ""),
                        "test_accuracy": test_metrics.get("accuracy", ""),
                        "test_precision": test_metrics.get("precision", ""),
                        "test_recall": test_metrics.get("recall", ""),
                        "test_f1": test_metrics.get("f1", test_metrics.get("macro_f1", "")),
                    }
                )

    @staticmethod
    def _load_training_config(pipeline_conf: dict[str, Any]) -> dict[str, Any]:
        training_config_path = pipeline_conf.get("training_config_path", "config/training/training_config.yaml")
        requested_path = Path(str(training_config_path))
        if not requested_path.exists():
            return {}
        with requested_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _resolve_selected_embeddings(
        available: list[str],
        filters: dict[str, Any],
        training_global_conf: dict[str, Any],
    ) -> list[str]:
        def _normalize_token(value: str) -> str:
            return "".join(char.lower() for char in str(value) if char.isalnum())

        available_lookup = {_normalize_token(name): name for name in available}

        def _resolve_embedding_name(name: str) -> str | None:
            direct = str(name)
            if direct in available:
                return direct
            return available_lookup.get(_normalize_token(direct))

        selected = set(available)

        group_name = filters.get("embedding_group")
        if group_name is not None:
            groups = training_global_conf.get("embedding_groups", {})
            if not isinstance(groups, Mapping) or group_name not in groups:
                raise ValueError(f"Unknown embedding_group: {group_name}")
            group_embeddings: set[str] = set()
            for item in groups[group_name]:
                resolved = _resolve_embedding_name(str(item))
                if resolved is not None:
                    group_embeddings.add(resolved)
            selected &= group_embeddings

        embedding_name = filters.get("embedding_name")
        if embedding_name is not None:
            resolved_embedding_name = _resolve_embedding_name(str(embedding_name))
            if resolved_embedding_name is None:
                raise ValueError(
                    "Unknown embedding_name: "
                    f"{embedding_name}. Available embeddings: {sorted(available)}"
                )
            selected &= {resolved_embedding_name}

        if not selected:
            raise ValueError(
                "No embeddings selected after applying filters. "
                f"Available embeddings: {sorted(available)}"
            )

        return [name for name in available if name in selected]

    @staticmethod
    def _resolve_selected_classifiers(
        available: list[str],
        filters: dict[str, Any],
        training_global_conf: dict[str, Any],
    ) -> list[str]:
        def _normalize_token(value: str) -> str:
            return "".join(char.lower() for char in str(value) if char.isalnum())

        classifier_aliases = {
            "rt": "RF",
        }
        available_lookup = {_normalize_token(name): name for name in available}

        def _resolve_classifier_name(name: str) -> str | None:
            direct = str(name)
            if direct in available:
                return direct

            alias_target = classifier_aliases.get(_normalize_token(direct))
            if alias_target is not None and alias_target in available:
                return alias_target

            return available_lookup.get(_normalize_token(direct))

        selected = set(available)

        enabled = training_global_conf.get("sweep", {}).get("enabled_classifiers")
        if isinstance(enabled, list) and enabled:
            resolved_enabled: set[str] = set()
            for item in enabled:
                resolved = _resolve_classifier_name(str(item))
                if resolved is not None:
                    resolved_enabled.add(resolved)
            selected &= resolved_enabled

        classifier_name = filters.get("classifier")
        if classifier_name is not None:
            resolved_classifier_name = _resolve_classifier_name(str(classifier_name))
            if resolved_classifier_name is None:
                raise ValueError(
                    "Unknown classifier: "
                    f"{classifier_name}. Available classifiers: {sorted(available)}"
                )
            selected &= {resolved_classifier_name}

        if not selected:
            raise ValueError(
                "No classifiers selected after applying filters. "
                f"Available classifiers: {sorted(available)}"
            )

        return [name for name in available if name in selected]

    @staticmethod
    def _filter_embedding_bundle(bundle: EmbeddingBundle, selected_embeddings: list[str]) -> EmbeddingBundle:
        return EmbeddingBundle(
            X_train={name: bundle.X_train[name] for name in selected_embeddings},
            X_val={name: bundle.X_val[name] for name in selected_embeddings},
            X_test={name: bundle.X_test[name] for name in selected_embeddings},
            y_train=bundle.y_train,
            y_val=bundle.y_val,
            y_test=bundle.y_test,
        )

    @staticmethod
    def _best_configs_by_embedding(trial_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for row in trial_results:
            embedding = str(row.get("embedding_name"))
            metrics = row.get("validation_metrics", {}) if isinstance(row.get("validation_metrics", {}), Mapping) else {}
            score = metrics.get("f1", metrics.get("macro_f1", float("nan")))
            if not isinstance(score, (int, float)) or not np.isfinite(float(score)):
                continue
            if embedding not in best or float(score) > float(best[embedding]["score"]):
                best[embedding] = {
                    "config": dict(row.get("config", {})),
                    "score": float(score),
                }
        return best

    def _run_final_training_for_classifier(
        self,
        classifier: str,
        trial_results: list[dict[str, Any]],
        embedding_bundle: EmbeddingBundle,
        train_conf: dict[str, Any],
        final_training_conf: dict[str, Any],
        wandb_enabled: bool,
        wandb_mode: str,
        wandb_project: str,
        wandb_entity: str | None,
    ) -> list[dict[str, Any]]:
        if not bool(final_training_conf.get("retrain_on_train_val", True)):
            return []

        evaluate_test = bool(final_training_conf.get("evaluate_test", True))
        save_model = bool(final_training_conf.get("save_model", True))
        output_dir = Path(str(final_training_conf.get("output_dir", "artifacts/final_models")))
        output_dir.mkdir(parents=True, exist_ok=True)

        best_by_embedding = self._best_configs_by_embedding(trial_results)
        final_rows: list[dict[str, Any]] = []

        for embedding_name, payload in best_by_embedding.items():
            best_config = dict(payload["config"])

            x_train = np.vstack([embedding_bundle.X_train[embedding_name], embedding_bundle.X_val[embedding_name]])
            y_train = np.concatenate([np.asarray(embedding_bundle.y_train, dtype=object), np.asarray(embedding_bundle.y_val, dtype=object)])
            final_problem_spec = ProblemSpecification.from_labels(y_train)

            retrain_bundle = EmbeddingBundle(
                X_train={embedding_name: x_train},
                X_val={embedding_name: embedding_bundle.X_val[embedding_name]},
                X_test={embedding_name: embedding_bundle.X_test[embedding_name]},
                y_train=y_train,
                y_val=embedding_bundle.y_val,
                y_test=embedding_bundle.y_test,
            )

            retrain_conf = dict(train_conf)
            retrain_conf["model_types"] = [classifier]
            retrain_conf["evaluate_test"] = evaluate_test
            retrain_conf["threshold_policy"] = final_training_conf.get("threshold_policy", {})
            retrain_conf.setdefault("model_params", {})
            retrain_conf["model_params"] = dict(retrain_conf["model_params"])

            normalize_override = best_config.get("normalize")
            if normalize_override is not None:
                retrain_conf["feature_processing"] = {
                    **dict(retrain_conf.get("feature_processing", {})),
                    "normalize": normalize_override,
                }

            model_specific_params = {k: v for k, v in best_config.items() if k != "normalize"}
            retrain_conf["model_params"][classifier] = model_specific_params

            training_service = TrainingService()
            retrain_result = training_service.train(
                embedding_bundle=retrain_bundle,
                training_config=retrain_conf,
            )
            result_payload = retrain_result[(classifier, embedding_name)]
            metrics_payload = result_payload.get("metrics", {}) if isinstance(result_payload.get("metrics", {}), Mapping) else {}
            validation_metrics = metrics_payload.get("validation", {}) if isinstance(metrics_payload.get("validation", {}), Mapping) else {}
            test_metrics = metrics_payload.get("test") if isinstance(metrics_payload.get("test"), Mapping) else None

            final_rows.append(
                {
                    "model_type": classifier,
                    "embedding_name": embedding_name,
                    "validation_metrics": dict(validation_metrics),
                    "test_metrics": dict(test_metrics) if test_metrics is not None else None,
                    "model": result_payload.get("model"),
                    "classes": getattr(result_payload.get("model"), "classes_", None),
                    "problem_type": final_problem_spec.problem_type,
                }
            )

            if save_model:
                try:
                    model_info = self._save_model_artifact(
                        model=result_payload.get("model"),
                        classifier=classifier,
                        embedding_name=embedding_name,
                        output_dir=output_dir,
                        problem_type=final_problem_spec.problem_type,
                        classes=final_problem_spec.classes,
                        normalization_mode=str(retrain_conf.get("feature_processing", {}).get("normalize", "none")),
                        threshold_policy=retrain_conf.get("threshold_policy", {}),
                    )
                    final_rows[-1]["model_artifact"] = model_info
                except Exception as exc:
                    self.logger.warning(
                        "Skipping model artifact save model_type=%s embedding=%s path=%s reason=%s",
                        classifier,
                        embedding_name,
                        output_dir,
                        exc,
                    )

            if wandb_enabled:
                run_name = f"final_{classifier}_{embedding_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                run = self._wandb_init_run(
                    enabled=wandb_enabled,
                    mode=wandb_mode,
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config={
                        "stage": "final_training",
                        "classifier": classifier,
                        "embedding_name": embedding_name,
                    },
                )
                try:
                    payload_log = {f"test_{k}": v for k, v in (test_metrics or {}).items()}
                    if payload_log:
                        self._wandb_log(payload_log)
                        if run is not None and hasattr(run, "summary"):
                            for key, value in payload_log.items():
                                run.summary[key] = value
                finally:
                    self._wandb_finish_run(run)

        return final_rows

    @staticmethod
    def _write_final_test_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "model_type",
            "embedding_name",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                test_metrics = row.get("test_metrics") if isinstance(row.get("test_metrics"), Mapping) else {}
                writer.writerow(
                    {
                        "model_type": row.get("model_type", ""),
                        "embedding_name": row.get("embedding_name", ""),
                        "test_accuracy": test_metrics.get("accuracy", ""),
                        "test_precision": test_metrics.get("precision", ""),
                        "test_recall": test_metrics.get("recall", ""),
                        "test_f1": test_metrics.get("f1", test_metrics.get("macro_f1", "")),
                        "test_roc_auc": test_metrics.get("roc_auc", ""),
                        "test_pr_auc": test_metrics.get("pr_auc", ""),
                    }
                )

    @staticmethod
    def _build_reporting_config(training_global_conf: dict[str, Any]) -> dict[str, Any]:
        default = {
            "output_root": "../../pec_data",
            "run_prefix": "sweep",
            "dataset_name": "default_dataset",
            "prediction_split": "test",
            "thresholds": {
                "default": 0.5,
                "classifier": {},
                "classifier_embedding": {},
            },
        }
        configured = training_global_conf.get("reporting", {})
        if not isinstance(configured, Mapping):
            return default

        merged = dict(default)
        merged.update({k: v for k, v in configured.items() if k != "thresholds"})
        thresholds = dict(default["thresholds"])
        conf_thresholds = configured.get("thresholds", {})
        if isinstance(conf_thresholds, Mapping):
            thresholds.update(conf_thresholds)
        merged["thresholds"] = thresholds
        return merged

    @staticmethod
    def _ensure_pec_data_layout(reporting_conf: dict[str, Any]) -> dict[str, Path]:
        repo_root = Path(__file__).resolve().parents[2]
        output_root = (repo_root / str(reporting_conf.get("output_root", "../../pec_data"))).resolve()
        layout = {
            "root": output_root,
            "dataset": output_root / "dataset",
            "sweep": output_root / "sweep",
            "results": output_root / "results",
            "logs": output_root / "logs",
        }
        for path in layout.values():
            path.mkdir(parents=True, exist_ok=True)
        return layout

    @staticmethod
    def _create_timestamped_run_dir(base_dir: Path, prefix: str) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        candidate = base_dir / f"{prefix}_{timestamp}"
        suffix = 1
        while candidate.exists():
            candidate = base_dir / f"{prefix}_{timestamp}_{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _attach_file_logger(self, logs_dir: Path, run_name: str) -> None:
        root_logger = logging.getLogger()
        if self._active_log_handler is not None:
            root_logger.removeHandler(self._active_log_handler)
            self._active_log_handler.close()
            self._active_log_handler = None

        file_path = logs_dir / f"{run_name}.log"
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        self._active_log_handler = handler

    def _persist_dataset_snapshot_if_missing(self, dataset_bundle, dataset_name: str, dataset_dir: Path) -> None:
        output_path = dataset_dir / f"{dataset_name}.csv"
        if output_path.exists():
            self.logger.info("Dataset snapshot already exists: %s", output_path)
            return

        rows: list[dict[str, Any]] = []
        for accession, label in zip(dataset_bundle.train_ids, dataset_bundle.y_train):
            rows.append({"accession": accession, "label": self._serialize_label(label), "dataset_split": "train"})
        for accession, label in zip(dataset_bundle.val_ids, dataset_bundle.y_val):
            rows.append({"accession": accession, "label": self._serialize_label(label), "dataset_split": "val"})
        for accession, label in zip(dataset_bundle.test_ids, dataset_bundle.y_test):
            rows.append({"accession": accession, "label": self._serialize_label(label), "dataset_split": "test"})

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["accession", "label", "dataset_split"])
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info("Saved dataset snapshot: %s", output_path)

    @staticmethod
    def _serialize_label(value: Any) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)
        return str(value)

    @staticmethod
    def _select_best_classifier_per_embedding(trial_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for row in trial_results:
            embedding = str(row.get("embedding_name"))
            model_type = str(row.get("model_type"))
            metrics = row.get("validation_metrics", {}) if isinstance(row.get("validation_metrics", {}), Mapping) else {}
            score = metrics.get("f1", float("nan"))
            if not isinstance(score, (int, float)) or not np.isfinite(float(score)):
                continue
            current = best.get(embedding)
            if current is None or float(score) > float(current["validation_f1"]):
                best[embedding] = {
                    "embedding_name": embedding,
                    "model_type": model_type,
                    "validation_f1": float(score),
                    "config": dict(row.get("config", {})),
                }
        return [best[key] for key in sorted(best.keys())]

    @staticmethod
    def _write_best_classifier_per_embedding_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["embedding_name", "model_type", "validation_f1", "config", "artifact_path", "serializer"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "embedding_name": row.get("embedding_name", ""),
                        "model_type": row.get("model_type", ""),
                        "validation_f1": row.get("validation_f1", ""),
                        "config": yaml.safe_dump(row.get("config", {}), sort_keys=True).strip(),
                        "artifact_path": row.get("artifact_path", ""),
                        "serializer": row.get("serializer", ""),
                    }
                )

    def _save_run_methodology_snapshot(
        self,
        configs_dir: Path,
        pipeline_conf: dict[str, Any],
        training_conf: dict[str, Any],
        sweep_conf: dict[str, Any],
        selected_embeddings: list[str],
        selected_classifiers: list[str],
        run_started_at: str,
        run_duration_seconds: float,
    ) -> None:
        metadata = {
            "run_started_at": run_started_at,
            "run_duration_seconds": run_duration_seconds,
            "selected_embeddings": selected_embeddings,
            "selected_classifiers": selected_classifiers,
            "runtime_filters": self.runtime_filters,
            "runtime_context": self.runtime_context,
            "python_version": sys.version,
            "git_commit": self._git_commit(),
            "package_versions": self._package_versions(["numpy", "scikit-learn", "xgboost", "torch", "wandb", "pandas", "pyyaml"]),
            "argv": self.runtime_context.get("argv", []),
            "sweep": sweep_conf,
        }

        with (configs_dir / "resolved_pipeline.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(pipeline_conf, handle, sort_keys=False)
        with (configs_dir / "resolved_training.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(training_conf, handle, sort_keys=False)
        with (configs_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    @staticmethod
    def _package_versions(packages: list[str]) -> dict[str, str | None]:
        versions: dict[str, str | None] = {}
        for package in packages:
            try:
                versions[package] = importlib_metadata.version(package)
            except Exception:
                versions[package] = None
        return versions

    def _write_test_predictions_csv(
        self,
        path: Path,
        dataset_bundle,
        embedding_bundle: EmbeddingBundle,
        final_test_rows: list[dict[str, Any]],
        best_classifier_per_embedding: list[dict[str, Any]],
        threshold_conf: dict[str, Any],
    ) -> None:
        model_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in final_test_rows:
            model_lookup[(str(row.get("model_type")), str(row.get("embedding_name")))] = row

        prediction_payloads: dict[tuple[str, str], np.ndarray] = {}
        for choice in best_classifier_per_embedding:
            classifier = str(choice.get("model_type"))
            embedding_name = str(choice.get("embedding_name"))
            key = (classifier, embedding_name)
            payload = model_lookup.get(key)
            if payload is None:
                continue
            model = payload.get("model")
            if model is None:
                continue
            probs = ProbabilityAdapter.to_canonical(
                raw_output=np.asarray(model.predict_proba(embedding_bundle.X_test[embedding_name])),
                problem_type=str(payload.get("problem_type", "binary")),
                classes=payload.get("classes"),
                context=f"predictions/{classifier}/{embedding_name}/test",
            )
            preds = decide(
                probs=probs,
                problem_type=str(payload.get("problem_type", "binary")),
                threshold_config={
                    **dict(threshold_conf if isinstance(threshold_conf, Mapping) else {}),
                    "classifier_name": classifier,
                    "embedding_name": embedding_name,
                },
            )
            if str(payload.get("problem_type", "binary")) != "multilabel" and payload.get("classes") is not None:
                class_array = np.asarray(payload.get("classes"))
                if class_array.size > 0:
                    preds = class_array[np.asarray(preds, dtype=int)]
            prediction_payloads[key] = preds

            artifact = payload.get("model_artifact")
            if isinstance(artifact, Mapping):
                choice["artifact_path"] = artifact.get("path", "")
                choice["serializer"] = artifact.get("serializer", "")

        self._write_best_classifier_per_embedding_csv(path.parent.parent / "reports" / "best_classifier_per_embedding.csv", best_classifier_per_embedding)

        rows: list[dict[str, Any]] = []
        for index, accession in enumerate(dataset_bundle.test_ids):
            row = {
                "accession": accession,
                "dataset_split": "test",
                "true_label": self._serialize_label(dataset_bundle.y_test[index]),
            }

            for choice in best_classifier_per_embedding:
                classifier = str(choice.get("model_type"))
                embedding_name = str(choice.get("embedding_name"))
                col_key = f"{classifier}__{embedding_name}"
                preds = prediction_payloads.get((classifier, embedding_name))
                if preds is None:
                    row[f"pred_{col_key}"] = ""
                    row[f"missing_reason_{col_key}"] = "model_not_available"
                    continue
                pred_value = preds[index]
                row[f"pred_{col_key}"] = self._serialize_label(pred_value)
                row[f"missing_reason_{col_key}"] = ""

            rows.append(row)

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _save_model_artifact(
        model: Any,
        classifier: str,
        embedding_name: str,
        output_dir: Path,
        problem_type: str,
        classes: tuple[Any, ...] | list[Any] | np.ndarray,
        normalization_mode: str,
        threshold_policy: dict[str, Any] | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{classifier}_{embedding_name}"
        classes_list = np.asarray(classes, dtype=object).tolist()
        metadata_path = output_dir / f"{base_name}.metadata.json"

        metadata: dict[str, Any] = {
            "problem_type": problem_type,
            "classes": classes_list,
            "num_classes": len(classes_list),
            "normalization": normalization_mode,
            "threshold_policy": dict(threshold_policy) if isinstance(threshold_policy, Mapping) else {},
            "classifier": classifier,
            "embedding_name": embedding_name,
        }

        if classifier.upper() == "MLP":
            torch = model._torch if hasattr(model, "_torch") else None
            if torch is None or model.model is None:
                raise RuntimeError("MLP model is not initialized for state_dict export")

            model_path = output_dir / f"{base_name}.pt"
            torch.save(model.model.state_dict(), model_path)

            metadata.update(
                {
                    "serializer": "mlp_state_dict",
                    "mlp": {
                        "input_size": model.input_size,
                        "output_size": model.output_size,
                        "hidden_layers_mode": model.hidden_layers_mode,
                        "num_hidden_layers": model.num_hidden_layers,
                        "custom_hidden_layers": model.custom_hidden_layers,
                        "dropout_rate": model.dropout_rate,
                        "activation_function": model.activation_function,
                        "use_batch_norm": model.use_batch_norm,
                        "output_activation": model.output_activation,
                        "initialization": model.initialization,
                        "optimizer_name": model.optimizer_name,
                        "learning_rate": model.learning_rate,
                        "num_epochs": model.num_epochs,
                        "early_stopping_patience": model.early_stopping_patience,
                        "criterion_name": model.criterion_name,
                        "batch_size": model.batch_size,
                        "optimizer_group": model.optimizer_group,
                    },
                }
            )
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            return {
                "serializer": "mlp_state_dict",
                "path": str(model_path.relative_to(output_dir.parent)),
                "metadata_path": str(metadata_path.relative_to(output_dir.parent)),
            }

        model_path = output_dir / f"{base_name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        metadata.update({"serializer": "pickle"})
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return {
            "serializer": "pickle",
            "path": str(model_path.relative_to(output_dir.parent)),
            "metadata_path": str(metadata_path.relative_to(output_dir.parent)),
        }

    @staticmethod
    def _load_model_artifact(model_path: Path, serializer: str):
        if serializer == "pickle":
            with model_path.open("rb") as handle:
                return pickle.load(handle)

        if serializer == "mlp_state_dict":
            from protein_embedding_classifier.classifiers.mlp_protein_classifier import MLPProteinClassifier
            import torch

            metadata_path = model_path.with_suffix(".metadata.json")
            if not metadata_path.exists():
                metadata_path = model_path.with_suffix(".meta.json")
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            mlp_meta = metadata.get("mlp", metadata)

            net = MLPProteinClassifier(
                input_size=mlp_meta["input_size"],
                output_size=mlp_meta["output_size"],
                num_hidden_layers=mlp_meta["num_hidden_layers"],
                dropout_rate=mlp_meta["dropout_rate"],
                hidden_layers_mode=mlp_meta["hidden_layers_mode"],
                custom_hidden_layers=mlp_meta["custom_hidden_layers"],
                activation_function=mlp_meta["activation_function"],
                use_batch_norm=mlp_meta["use_batch_norm"],
                output_activation=mlp_meta["output_activation"],
                initialization=mlp_meta["initialization"],
            )
            state = torch.load(model_path, map_location="cpu")
            net.load_state_dict(state)
            net.eval()

            class _MLPInferenceWrapper:
                def __init__(self, network, classes):
                    self.network = network
                    self.classes_ = np.asarray(classes) if classes is not None else None

                def predict_proba(self, X):
                    import torch

                    with torch.no_grad():
                        x_tensor = torch.tensor(X, dtype=torch.float32)
                        logits = self.network(x_tensor)
                        if logits.ndim == 1:
                            logits = logits.unsqueeze(1)
                        criterion_name = str(mlp_meta.get("criterion_name", "BCEWithLogitsLoss"))
                        if logits.shape[1] == 1:
                            positive_prob = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                            probs = np.hstack([1.0 - positive_prob, positive_prob])
                            return probs
                        if criterion_name == "BCEWithLogitsLoss":
                            probs = torch.sigmoid(logits).cpu().numpy()
                            return probs
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        return probs

            return _MLPInferenceWrapper(net, metadata.get("classes"))

        raise ValueError(f"Unsupported model serializer: {serializer}")

    @staticmethod
    def _get_latest_sweep_run(sweep_root: Path) -> Path | None:
        candidates = [path for path in sweep_root.iterdir() if path.is_dir()]
        if not candidates:
            return None
        return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]

    @staticmethod
    def _read_best_classifier_per_embedding_csv(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    @staticmethod
    def _write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in np.asarray(matrix):
                writer.writerow(list(row))

    @staticmethod
    def _wandb_init_run(
        enabled: bool,
        mode: str,
        project: str,
        entity: str | None,
        name: str,
        config: dict[str, Any],
    ):
        if not enabled:
            return None
        try:
            import wandb
        except ImportError:
            return None

        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": name,
            "config": config,
            "mode": mode,
        }
        if entity:
            init_kwargs["entity"] = entity
        return wandb.init(**init_kwargs)

    @staticmethod
    def _wandb_log(payload: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return
        if payload:
            wandb.log(payload)

    @staticmethod
    def _wandb_finish_run(run) -> None:
        if run is None:
            return
        run.finish()