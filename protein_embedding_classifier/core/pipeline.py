from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable
from datetime import datetime
import numpy as np

import yaml

from protein_embedding_classifier.core.db import create_engine_from_config, load_db_config
from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle, EmbeddingService
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


PIPELINE_STEPS = ["dataset", "embeddings", "train", "sweep", "ensemble", "evaluate"]


class Pipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger("Pipeline")
        self.runtime_filters: dict[str, Any] = {}

    def run(self, step: str | None = None, run_all: bool = False, filters: dict[str, Any] | None = None) -> None:
        self.logger.info("Step 1: Load pipeline config")
        conf = self._load_yaml(self.config_path)
        self.runtime_filters = {key: value for key, value in (filters or {}).items() if value is not None}

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
            training_config=dict(train_conf),
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

        embeddings_config_path = embeddings_step_conf.get("config_path", "config/embeddings.yaml")
        embeddings_conf = self._load_yaml(embeddings_config_path)
        embedding_bundle = self._build_embedding_bundle(dataset_conf=dataset_conf, embeddings_conf=embeddings_conf)
        training_global_conf = self._load_training_config(pipeline_conf)

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
        global_best: dict[str, Any] | None = None
        all_trial_results: list[dict[str, Any]] = []
        final_test_rows: list[dict[str, Any]] = []
        skipped_classifiers: list[tuple[str, str]] = []

        wandb_conf = training_global_conf.get("wandb", {}) if isinstance(training_global_conf.get("wandb", {}), Mapping) else {}
        wandb_enabled = bool(wandb_conf.get("enabled", True))
        wandb_mode = str(wandb_conf.get("mode", "online"))
        wandb_entity = wandb_conf.get("entity")
        wandb_project = str(wandb_conf.get("project", wandb_project))

        for current_model_type, sweep_config_path in selected_classifier_sweeps.items():
            if current_model_type.upper() == "MLP" and problem_spec.problem_type == "multilabel":
                reason = (
                    "MLP training is not supported for multilabel targets in this pipeline "
                    "(MLP is supported for binary and multiclass tasks)."
                )
                self.logger.warning(
                    "Skipping classifier sweep model_type=%s due to unsupported model/task combination: %s",
                    current_model_type,
                    reason,
                )
                skipped_classifiers.append((current_model_type, reason))
                continue

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
                    training_config=dict(train_conf),
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
                if "MLP training is not supported for multilabel targets" in str(exc):
                    self.logger.warning(
                        "Skipping classifier sweep model_type=%s due to unsupported model/task combination: %s",
                        current_model_type,
                        exc,
                    )
                    skipped_classifiers.append((current_model_type, str(exc)))
                    continue
                raise

            best_results[current_model_type] = {
                "best_config": result.best_config,
                "best_metric": result.best_metric,
                "best_key": result.best_key,
                "sweep_config_path": sweep_config_path,
            }
            if global_best is None or result.best_metric > float(global_best["best_metric"]):
                global_best = {
                    "model_type": current_model_type,
                    "embedding_name": result.best_key[1],
                    "best_metric": result.best_metric,
                    "best_config": result.best_config,
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
                final_rows = self._run_final_training_for_classifier(
                    classifier=current_model_type,
                    trial_results=result.trial_results,
                    embedding_bundle=embedding_bundle,
                    train_conf=train_conf,
                    final_training_conf=dict(training_global_conf.get("final_training", {})),
                    wandb_enabled=wandb_enabled,
                    wandb_mode=wandb_mode,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                )
                final_test_rows.extend(final_rows)

        if not best_results:
            if skipped_classifiers:
                skipped_summary = "; ".join(f"{name}: {reason}" for name, reason in skipped_classifiers)
                only_unsupported_mlp = all(
                    "MLP training is not supported for multilabel targets" in reason
                    for _, reason in skipped_classifiers
                )
                if only_unsupported_mlp:
                    self.logger.info(
                        "Sweep completed with no runs because selected classifiers are unsupported for this task. %s",
                        skipped_summary,
                    )
                else:
                    self.logger.warning(
                        "No successful sweep results were produced; all selected classifiers were skipped. %s",
                        skipped_summary,
                    )
            else:
                self.logger.warning("No successful sweep results were produced")
            return

        per_classifier_path = artifacts_dir / "best_config_by_classifier.yaml"
        with per_classifier_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(best_results, handle, sort_keys=False)
        self.logger.info("Saved per-classifier sweep results artifact: %s", per_classifier_path)

        full_results_csv = artifacts_dir / "sweep_results_full.csv"
        self._write_full_sweep_results_csv(full_results_csv, all_trial_results)
        self.logger.info("Saved full sweep results CSV artifact: %s", full_results_csv)

        best_per_classifier_csv = artifacts_dir / "best_per_classifier.csv"
        self._write_best_per_classifier_csv(best_per_classifier_csv, best_results)
        self.logger.info("Saved per-classifier sweep CSV artifact: %s", best_per_classifier_csv)

        best_config_path = artifacts_dir / "best_config.yaml"
        with best_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump((global_best or {}).get("best_config", {}), handle, sort_keys=False)

        self.logger.info("Saved sweep best config artifact: %s", best_config_path)
        if global_best is not None:
            global_best_csv = artifacts_dir / "global_best.csv"
            self._write_global_best_csv(global_best_csv, global_best)
            self.logger.info("Saved global best CSV artifact: %s", global_best_csv)

            self.logger.info(
                "Sweep global best result model_type=%s embedding_name=%s validation_f1=%.6f",
                global_best["model_type"],
                global_best["embedding_name"],
                float(global_best["best_metric"]),
            )

        summary_table = SweepService.build_summary_table(
            trial_results=all_trial_results,
            model_order=sorted(selected_classifier_sweeps.keys()),
        )
        self.logger.info("Validation summary table (F1):\n%s", summary_table)

        if final_test_rows:
            final_test_path = artifacts_dir / "final_test_results.csv"
            self._write_final_test_results_csv(final_test_path, final_test_rows)
            self.logger.info("Saved final test results CSV artifact: %s", final_test_path)

    def run_ensemble_step(self, conf: dict[str, Any]) -> None:
        _ = conf
        self.logger.info("Ensemble step placeholder (not implemented in this refactor).")

    def run_evaluate_step(self, conf: dict[str, Any]) -> None:
        _ = conf
        self.logger.info("Evaluate step placeholder (not implemented in this refactor).")

    def _step_runner_map(self) -> dict[str, Callable[[dict[str, Any]], None]]:
        return {
            "dataset": self.run_dataset_step,
            "embeddings": self.run_embeddings_step,
            "train": self.run_train_step,
            "sweep": self.run_sweep_step,
            "ensemble": self.run_ensemble_step,
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
                }
            )

            if save_model:
                model_path = output_dir / f"{classifier}_{embedding_name}.pkl"
                with model_path.open("wb") as handle:
                    pickle.dump(result_payload.get("model"), handle)

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