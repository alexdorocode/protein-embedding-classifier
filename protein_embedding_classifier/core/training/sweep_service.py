from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.model_factory import ModelFactory
from protein_embedding_classifier.core.training.problem_specification import ProblemSpecification
from protein_embedding_classifier.core.training.training_service import TrainingService


@dataclass(frozen=True)
class SweepResult:
    best_config: dict[str, Any]
    best_metric: float
    best_key: tuple[str, str]
    trial_results: list[dict[str, Any]]


class SweepService:
    def __init__(
        self,
        model_type: str,
        model_factory: ModelFactory | None = None,
        rng_seed: int = 42,
    ):
        self.model_type = model_type
        self.model_factory = model_factory or ModelFactory()
        self.rng_seed = int(rng_seed)
        self.rng = np.random.default_rng(rng_seed)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        embedding_bundle: EmbeddingBundle,
        sweep_config: Mapping[str, Any],
        num_trials: int = 10,
        training_config: dict[str, Any] | None = None,
        wandb_project: str = "protein-embedding-classifier",
        artifacts_dir: str = "artifacts",
        wandb_enabled: bool = True,
        wandb_mode: str = "online",
        wandb_entity: str | None = None,
    ) -> SweepResult:
        trial_configs = self._build_trial_configs(sweep_config, num_trials)
        if not trial_configs:
            raise ValueError("Sweep config produced no trials")

        metric_name = str(sweep_config.get("metric", {}).get("name", "f1_score"))
        metric_goal = str(sweep_config.get("metric", {}).get("goal", "maximize")).lower()
        maximize_metric = metric_goal != "minimize"
        selection_key = self._metric_name_to_validation_key(metric_name)

        best_metric = float("-inf") if maximize_metric else float("inf")
        best_config: dict[str, Any] = {}
        best_key: tuple[str, str] | None = None
        all_trial_results: list[dict[str, Any]] = []

        run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        embedding_names = sorted(embedding_bundle.X_train.keys())

        for trial_index, params in enumerate(trial_configs, start=1):
            for embedding_name in embedding_names:
                run_name = self._build_run_name(
                    model_type=self.model_type,
                    embedding_name=embedding_name,
                    trial_index=trial_index,
                    timestamp=run_timestamp,
                )

                run = self._wandb_init(
                    wandb_project=wandb_project,
                    config=dict(params),
                    run_name=run_name,
                    enabled=wandb_enabled,
                    mode=wandb_mode,
                    entity=wandb_entity,
                )

                try:
                    single_bundle = self._single_embedding_bundle(embedding_bundle, embedding_name)
                    problem_spec = ProblemSpecification.from_labels(single_bundle.y_train)
                    normalization_mode = self._resolve_normalization_mode(
                        training_config=dict(training_config or {}),
                        trial_params=params,
                    )

                    tracking_config = self._build_tracking_config(
                        trial_params=params,
                        model_type=self.model_type,
                        embedding_name=embedding_name,
                        normalization_mode=normalization_mode,
                        problem_type=problem_spec.problem_type,
                        random_seed=self.rng_seed,
                    )
                    self._wandb_config_update(tracking_config, enabled=wandb_enabled)

                    effective_training_config = dict(training_config or {})
                    effective_training_config["model_types"] = [self.model_type]

                    service = TrainingService(
                        model_factory=self.model_factory,
                        sweep_mode=True,
                        wandb_config=params,
                    )
                    results = service.train(
                        embedding_bundle=single_bundle,
                        training_config=effective_training_config,
                    )

                    result_key, payload = next(iter(results.items()))
                    metrics_payload = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), Mapping) else {}
                    validation_metrics = dict(metrics_payload.get("validation", {})) if isinstance(metrics_payload.get("validation", {}), Mapping) else {}
                    test_metrics_raw = metrics_payload.get("test")
                    test_metrics = dict(test_metrics_raw) if isinstance(test_metrics_raw, Mapping) else None

                    clean_validation = self._clean_metrics(validation_metrics)
                    clean_test = self._clean_metrics(test_metrics) if test_metrics is not None else None
                    if clean_test == {}:
                        clean_test = None

                    val_log_payload = {f"val_{key}": value for key, value in clean_validation.items()}
                    log_payload = {
                        **val_log_payload,
                        "trial_index": trial_index,
                        "model_type": result_key[0],
                        "embedding_name": result_key[1],
                    }
                    if clean_test is not None:
                        log_payload.update({f"test_{key}": value for key, value in clean_test.items()})

                    self._wandb_log(log_payload, enabled=wandb_enabled)
                    self._wandb_update_summary(run, clean_validation, clean_test)

                    metric_value = float(clean_validation.get(selection_key, float("nan")))

                    all_trial_results.append(
                        {
                            "model_type": result_key[0],
                            "embedding_name": result_key[1],
                            "trial_index": trial_index,
                            "config": dict(params),
                            "validation_metrics": dict(clean_validation),
                            "test_metrics": dict(clean_test) if clean_test is not None else None,
                        }
                    )

                    if np.isfinite(metric_value) and self._is_better(metric_value, best_metric, maximize_metric):
                        best_metric = metric_value
                        best_config = dict(params)
                        best_key = result_key
                finally:
                    self._wandb_finish(run)

        if best_key is None:
            raise RuntimeError("Sweep finished but no valid metric was produced")

        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        self._export_trial_results_csv(artifacts_path / "sweep_results_full.csv", all_trial_results)

        self.logger.info(
            "Sweep completed model_type=%s best_embedding=%s %s=%.6f",
            best_key[0],
            best_key[1],
            metric_name,
            best_metric,
        )

        return SweepResult(
            best_config=best_config,
            best_metric=best_metric,
            best_key=best_key,
            trial_results=all_trial_results,
        )

    def _build_trial_configs(self, sweep_config: Mapping[str, Any], num_trials: int) -> list[dict[str, Any]]:
        parameters = sweep_config.get("parameters", {})
        if not isinstance(parameters, Mapping):
            return []
        return [self._sample_trial(parameters) for _ in range(int(max(1, num_trials)))]

    def _sample_trial(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        sampled: dict[str, Any] = {}
        for key, spec in parameters.items():
            if not isinstance(spec, Mapping):
                continue

            if "values" in spec:
                values = list(spec["values"])
                if not values:
                    continue
                sampled[key] = values[int(self.rng.integers(0, len(values)))]
                continue

            if "min" in spec and "max" in spec:
                min_value = spec["min"]
                max_value = spec["max"]
                distribution = str(spec.get("distribution", "uniform"))

                if distribution == "int_uniform":
                    sampled[key] = int(self.rng.integers(int(min_value), int(max_value) + 1))
                elif distribution in {"log_uniform", "log_uniform_values"}:
                    low = np.log(float(min_value))
                    high = np.log(float(max_value))
                    sampled[key] = float(np.exp(self.rng.uniform(low, high)))
                else:
                    sampled[key] = float(self.rng.uniform(float(min_value), float(max_value)))

        return sampled

    @staticmethod
    def _single_embedding_bundle(bundle: EmbeddingBundle, embedding_name: str) -> EmbeddingBundle:
        return EmbeddingBundle(
            X_train={embedding_name: bundle.X_train[embedding_name]},
            X_val={embedding_name: bundle.X_val[embedding_name]},
            X_test={embedding_name: bundle.X_test[embedding_name]},
            y_train=bundle.y_train,
            y_val=bundle.y_val,
            y_test=bundle.y_test,
        )

    @staticmethod
    def _build_run_name(model_type: str, embedding_name: str, trial_index: int, timestamp: str) -> str:
        return f"{model_type}_{embedding_name}_trial{trial_index}_{timestamp}"

    @staticmethod
    def _flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        for key, value in data.items():
            normalized_key = f"{prefix}{key}"
            if isinstance(value, Mapping):
                flattened.update(SweepService._flatten_dict(value, prefix=f"{normalized_key}."))
            else:
                flattened[normalized_key] = value
        return flattened

    def _build_tracking_config(
        self,
        trial_params: Mapping[str, Any],
        model_type: str,
        embedding_name: str,
        normalization_mode: str,
        problem_type: str,
        random_seed: int,
    ) -> dict[str, Any]:
        flattened_trial = self._flatten_dict(trial_params)
        return {
            **{f"trial.{key}": value for key, value in flattened_trial.items()},
            "model_type": model_type,
            "embedding_name": embedding_name,
            "normalization_mode": normalization_mode,
            "problem_type": problem_type,
            "random_seed": random_seed,
        }

    @staticmethod
    def _resolve_normalization_mode(training_config: dict[str, Any], trial_params: Mapping[str, Any]) -> str:
        feature_processing = training_config.get("feature_processing", {})
        mode = "none"
        if isinstance(feature_processing, Mapping):
            mode = str(feature_processing.get("normalize", "none"))

        if "normalize" in trial_params:
            mode = str(trial_params["normalize"])
        nested = trial_params.get("feature_processing")
        if isinstance(nested, Mapping) and "normalize" in nested:
            mode = str(nested["normalize"])

        normalized = mode.lower()
        return normalized if normalized in {"none", "l2", "standard"} else "none"

    @staticmethod
    def _metric_name_to_validation_key(metric_name: str) -> str:
        normalized = metric_name.lower()
        if normalized.startswith("val_"):
            normalized = normalized[4:]

        aliases = {
            "f1_score": "f1",
            "f1": "f1",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
            "pr_auc": "pr_auc",
            "micro_f1": "micro_f1",
            "macro_f1": "macro_f1",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _is_better(candidate: float, current_best: float, maximize: bool) -> bool:
        return candidate > current_best if maximize else candidate < current_best

    @staticmethod
    def _clean_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
        if metrics is None:
            return {}

        cleaned: dict[str, Any] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, dict)):
                cleaned[key] = value
                continue
            if isinstance(value, (int, float, np.floating)):
                if np.isfinite(float(value)):
                    cleaned[key] = float(value)
                continue
        return cleaned

    def _export_trial_results_csv(self, output_path: Path, trial_results: list[dict[str, Any]]) -> None:
        columns = [
            "model_type",
            "embedding_name",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
            "TP",
            "TN",
            "FP",
            "FN",
            "seed_used",
        ]

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()

            for result in trial_results:
                validation = result.get("validation_metrics", {}) if isinstance(result.get("validation_metrics", {}), Mapping) else {}
                test = result.get("test_metrics") if isinstance(result.get("test_metrics"), Mapping) else {}
                confusion = self._resolve_binary_confusion_counts(
                    test if isinstance(test, Mapping) and test else validation
                )

                row = {
                    "model_type": result.get("model_type", ""),
                    "embedding_name": result.get("embedding_name", ""),
                    "validation_accuracy": validation.get("accuracy", ""),
                    "validation_precision": validation.get("precision", ""),
                    "validation_recall": validation.get("recall", ""),
                    "validation_f1": validation.get("f1", validation.get("macro_f1", "")),
                    "test_accuracy": test.get("accuracy", ""),
                    "test_precision": test.get("precision", ""),
                    "test_recall": test.get("recall", ""),
                    "test_f1": test.get("f1", test.get("macro_f1", "")),
                    "test_roc_auc": test.get("roc_auc", ""),
                    "test_pr_auc": test.get("pr_auc", ""),
                    "TP": confusion["TP"],
                    "TN": confusion["TN"],
                    "FP": confusion["FP"],
                    "FN": confusion["FN"],
                    "seed_used": int(self.rng_seed),
                }
                writer.writerow(row)

    @staticmethod
    def _resolve_binary_confusion_counts(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(metrics, Mapping):
            return {"TP": "", "TN": "", "FP": "", "FN": ""}

        if all(key in metrics for key in ("tp", "tn", "fp", "fn")):
            return {
                "TP": metrics.get("tp", ""),
                "TN": metrics.get("tn", ""),
                "FP": metrics.get("fp", ""),
                "FN": metrics.get("fn", ""),
            }

        matrix = metrics.get("confusion_matrix")
        if isinstance(matrix, list) and len(matrix) == 2:
            first_row = matrix[0] if isinstance(matrix[0], list) else None
            second_row = matrix[1] if isinstance(matrix[1], list) else None
            if first_row is not None and second_row is not None and len(first_row) == 2 and len(second_row) == 2:
                return {
                    "TP": second_row[1],
                    "TN": first_row[0],
                    "FP": first_row[1],
                    "FN": second_row[0],
                }

        return {"TP": "", "TN": "", "FP": "", "FN": ""}

    @staticmethod
    def build_summary_table(trial_results: list[dict[str, Any]], model_order: list[str]) -> str:
        header = ["Embedding", *model_order]
        if not trial_results:
            return " | ".join(header)

        table: dict[str, dict[str, float]] = {}
        for row in trial_results:
            embedding_name = str(row.get("embedding_name"))
            model_type = str(row.get("model_type"))
            validation = row.get("validation_metrics", {}) if isinstance(row.get("validation_metrics", {}), Mapping) else {}
            value = validation.get("f1", validation.get("macro_f1"))
            if value is None or not isinstance(value, (int, float, np.floating)) or not np.isfinite(float(value)):
                continue

            table.setdefault(embedding_name, {})
            current = table[embedding_name].get(model_type)
            if current is None or float(value) > current:
                table[embedding_name][model_type] = float(value)

        lines = [" | ".join(header), "-" * (len(" | ".join(header)) + 8)]
        for embedding_name in sorted(table.keys()):
            values = [embedding_name]
            for model_type in model_order:
                score = table[embedding_name].get(model_type)
                values.append("-" if score is None else f"{score:.4f}")
            lines.append(" | ".join(values))

        return "\n".join(lines)

    @staticmethod
    def _wandb_init(
        wandb_project: str,
        config: dict[str, Any],
        run_name: str,
        enabled: bool,
        mode: str,
        entity: str | None,
    ):
        if not enabled:
            return None
        try:
            import wandb
        except ImportError:
            return None
        init_kwargs: dict[str, Any] = {
            "project": wandb_project,
            "config": config,
            "name": run_name,
            "mode": mode,
        }
        if entity:
            init_kwargs["entity"] = entity
        return wandb.init(**init_kwargs)

    @staticmethod
    def _wandb_log(payload: dict[str, Any], enabled: bool) -> None:
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            return
        wandb.log(payload)

    @staticmethod
    def _wandb_config_update(payload: dict[str, Any], enabled: bool) -> None:
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            return
        config_obj = getattr(wandb, "config", None)
        if config_obj is None or not hasattr(config_obj, "update"):
            return
        config_obj.update(payload, allow_val_change=True)

    @staticmethod
    def _wandb_update_summary(run, validation_metrics: dict[str, Any], test_metrics: dict[str, Any] | None) -> None:
        if run is None:
            return

        summary = getattr(run, "summary", None)
        if summary is None:
            return

        if "f1" in validation_metrics:
            summary["val_f1"] = float(validation_metrics["f1"])
        elif "macro_f1" in validation_metrics:
            summary["val_f1"] = float(validation_metrics["macro_f1"])

        for key, value in validation_metrics.items():
            summary[f"val_{key}"] = value

        if test_metrics is not None:
            for key, value in test_metrics.items():
                summary[f"test_{key}"] = value

    @staticmethod
    def _wandb_finish(run) -> None:
        if run is None:
            return
        run.finish()
