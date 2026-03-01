from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

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
    ) -> SweepResult:
        trial_configs = self._build_trial_configs(sweep_config, num_trials)
        if not trial_configs:
            raise ValueError("Sweep config produced no trials")

        metric_name = str(sweep_config.get("metric", {}).get("name", "f1_score"))
        metric_goal = str(sweep_config.get("metric", {}).get("goal", "maximize")).lower()
        maximize_metric = metric_goal != "minimize"

        best_metric = float("-inf") if maximize_metric else float("inf")
        best_config: dict[str, Any] = {}
        best_key: tuple[str, str] | None = None
        all_trial_results: list[dict[str, Any]] = []

        run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        embedding_names = sorted(embedding_bundle.X_train.keys())

        for trial_index, params in enumerate(trial_configs, start=1):
            for embedding_name in embedding_names:
                single_embedding_bundle = self._single_embedding_bundle(embedding_bundle, embedding_name)
                run_name = self._build_run_name(
                    model_type=self.model_type,
                    embedding_name=embedding_name,
                    trial_index=trial_index,
                    timestamp=run_timestamp,
                )
                run = self._wandb_init(
                    wandb_project=wandb_project,
                    config=params,
                    trial_index=trial_index,
                    run_name=run_name,
                )
                try:
                    service = TrainingService(
                        model_factory=self.model_factory,
                        sweep_mode=True,
                        wandb_config=params,
                    )

                    effective_training_config = dict(training_config or {})
                    effective_training_config["model_types"] = [self.model_type]

                    problem_spec = ProblemSpecification.from_labels(single_embedding_bundle.y_train)
                    normalization_mode = self._resolve_normalization_mode(
                        training_config=effective_training_config,
                        trial_params=params,
                    )
                    full_config_payload = self._build_tracking_config(
                        trial_params=params,
                        model_type=self.model_type,
                        embedding_name=embedding_name,
                        normalization_mode=normalization_mode,
                        problem_type=problem_spec.problem_type,
                        random_seed=self.rng_seed,
                    )
                    self._wandb_config_update(full_config_payload)

                    results = service.train(
                        embedding_bundle=single_embedding_bundle,
                        training_config=effective_training_config,
                    )

                    result_key, payload = next(iter(results.items()))
                    val_probs = np.asarray(payload.get("val_probs"))
                    val_metrics = self._compute_metrics(
                        y_true=np.asarray(single_embedding_bundle.y_val),
                        val_probs=val_probs,
                    )
                    test_metrics = self._compute_test_metrics(
                        model=payload.get("model"),
                        x_train=np.asarray(single_embedding_bundle.X_train[embedding_name]),
                        x_test=np.asarray(single_embedding_bundle.X_test[embedding_name]),
                        y_test=np.asarray(single_embedding_bundle.y_test),
                        normalization_mode=normalization_mode,
                    )

                    metric_value = float(
                        payload.get("metrics", {}).get(metric_name, val_metrics.get(metric_name, float("nan")) )
                    )

                    log_payload = {
                        **val_metrics,
                        **{f"test_{name}": value for name, value in test_metrics.items()},
                        "trial_index": trial_index,
                        "model_type": result_key[0],
                        "embedding_name": result_key[1],
                    }
                    self._wandb_log(log_payload)

                    all_trial_results.append(
                        {
                            "model_type": result_key[0],
                            "embedding_name": result_key[1],
                            "trial_index": trial_index,
                            "config": dict(params),
                            "validation_metrics": dict(val_metrics),
                            "test_metrics": dict(test_metrics),
                            "selection_metric_name": metric_name,
                            "selection_metric_value": metric_value,
                        }
                    )

                    if np.isfinite(metric_value) and self._is_better(
                        candidate=metric_value,
                        current_best=best_metric,
                        maximize=maximize_metric,
                    ):
                        best_metric = metric_value
                        best_config = dict(params)
                        best_key = result_key
                finally:
                    self._wandb_finish(run)

        if best_key is None:
            raise RuntimeError("Sweep finished but no valid metric was produced")

        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        full_results_csv_path = artifacts_path / "sweep_results_full.csv"
        self._export_trial_results_csv(full_results_csv_path, all_trial_results)
        self.logger.info("Sweep full results CSV saved: %s", full_results_csv_path)

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
                idx = int(self.rng.integers(0, len(values)))
                sampled[key] = values[idx]
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
    def _wandb_init(wandb_project: str, config: dict[str, Any], trial_index: int):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.init(project=wandb_project, config=config, name=f"sweep-trial-{trial_index}")

    @staticmethod
    def _wandb_log(payload: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return
        wandb.log(payload)

    @staticmethod
    def _wandb_finish(run) -> None:
        if run is None:
            return
        run.finish()

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
    def _wandb_config_update(payload: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return
        config_obj = getattr(wandb, "config", None)
        if config_obj is None or not hasattr(config_obj, "update"):
            return
        config_obj.update(payload, allow_val_change=True)

    @staticmethod
    def _is_better(candidate: float, current_best: float, maximize: bool) -> bool:
        if maximize:
            return candidate > current_best
        return candidate < current_best

    @staticmethod
    def _resolve_normalization_mode(training_config: dict[str, Any], trial_params: Mapping[str, Any]) -> str:
        feature_processing = training_config.get("feature_processing", {})
        normalize_mode = "none"
        if isinstance(feature_processing, Mapping):
            normalize_mode = str(feature_processing.get("normalize", "none"))

        if "normalize" in trial_params:
            normalize_mode = str(trial_params["normalize"])
        trial_feature_processing = trial_params.get("feature_processing")
        if isinstance(trial_feature_processing, Mapping) and "normalize" in trial_feature_processing:
            normalize_mode = str(trial_feature_processing["normalize"])

        normalized = normalize_mode.lower()
        if normalized not in {"none", "l2", "standard"}:
            return "none"
        return normalized

    @staticmethod
    def _preprocess_for_inference(x_train: np.ndarray, x_eval: np.ndarray, normalization_mode: str) -> np.ndarray:
        if normalization_mode == "none":
            return np.asarray(x_eval)
        if normalization_mode == "l2":
            return normalize(np.asarray(x_eval), norm="l2", axis=1)

        scaler = StandardScaler()
        scaler.fit(np.asarray(x_train))
        return scaler.transform(np.asarray(x_eval))

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, val_probs: np.ndarray) -> dict[str, float]:
        y_true_array = np.asarray(y_true)
        probs = np.asarray(val_probs)

        metrics: dict[str, float] = {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1_score": float("nan"),
        }

        if probs.ndim != 2 or y_true_array.ndim != 1:
            return metrics

        if SweepService._is_legacy_multilabel_target(y_true_array):
            y_true_bin = SweepService._binarize_legacy_multilabel(y_true_array)
            y_pred_bin = (probs >= 0.5).astype(int)
            if y_true_bin.shape != y_pred_bin.shape:
                return metrics
            metrics.update(
                {
                    "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
                    "precision": float(precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
                    "recall": float(recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
                    "f1_score": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
                }
            )
            return metrics

        if probs.shape[1] == 2:
            y_pred = np.argmax(probs, axis=1)
            y_score = probs[:, 1]
            metrics.update(
                {
                    "accuracy": float(accuracy_score(y_true_array, y_pred)),
                    "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_true_array, y_pred, zero_division=0)),
                }
            )
            unique_classes = np.unique(y_true_array)
            if unique_classes.size == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score))
                metrics["pr_auc"] = float(average_precision_score(y_true_array, y_score))
            return metrics

        y_pred = np.argmax(probs, axis=1)
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_true_array, y_pred)),
                "precision": float(precision_score(y_true_array, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true_array, y_pred, average="macro", zero_division=0)),
                "f1_score": float(f1_score(y_true_array, y_pred, average="macro", zero_division=0)),
            }
        )
        return metrics

    @staticmethod
    def _is_legacy_multilabel_target(y_true_array: np.ndarray) -> bool:
        if y_true_array.dtype != object or y_true_array.size == 0:
            return False
        first_value = y_true_array[0]
        return isinstance(first_value, (list, tuple, set, np.ndarray))

    @staticmethod
    def _binarize_legacy_multilabel(y_true_array: np.ndarray) -> np.ndarray:
        def _to_list(item: Any) -> list[Any]:
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, (list, tuple, set)):
                return list(item)
            return [item]

        values = [_to_list(item) for item in y_true_array]
        binarizer = MultiLabelBinarizer()
        return binarizer.fit_transform(values)

    def _compute_test_metrics(
        self,
        model,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        normalization_mode: str,
    ) -> dict[str, float]:
        if model is None or not hasattr(model, "predict_proba"):
            return {}
        if x_test.size == 0:
            return {}

        x_test_processed = self._preprocess_for_inference(x_train=x_train, x_eval=x_test, normalization_mode=normalization_mode)
        test_probs = np.asarray(model.predict_proba(x_test_processed))
        return self._compute_metrics(np.asarray(y_test), test_probs)

    def _export_trial_results_csv(self, output_path: Path, trial_results: list[dict[str, Any]]) -> None:
        rows: list[dict[str, Any]] = []
        for result in trial_results:
            config_values = self._flatten_dict(result.get("config", {}))
            val_metrics = result.get("validation_metrics", {})
            test_metrics = result.get("test_metrics", {})
            row = {
                "model_type": result.get("model_type"),
                "embedding_name": result.get("embedding_name"),
                "trial_index": result.get("trial_index"),
                "selection_metric_name": result.get("selection_metric_name"),
                "selection_metric_value": result.get("selection_metric_value"),
                **{f"config.{key}": value for key, value in config_values.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            rows.append(row)

        if not rows:
            return

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def build_summary_table(trial_results: list[dict[str, Any]], model_order: list[str]) -> str:
        if not trial_results:
            header = ["Embedding", *model_order]
            return " | ".join(header)

        by_embedding_model: dict[str, dict[str, float]] = {}
        for row in trial_results:
            embedding = str(row.get("embedding_name"))
            model_type = str(row.get("model_type"))
            val_metrics = row.get("validation_metrics", {})
            value = val_metrics.get("f1_score") if isinstance(val_metrics, Mapping) else None
            if value is None or not np.isfinite(value):
                continue
            by_embedding_model.setdefault(embedding, {})
            current = by_embedding_model[embedding].get(model_type)
            if current is None or value > current:
                by_embedding_model[embedding][model_type] = float(value)

        header = ["Embedding", *model_order]
        lines = [" | ".join(header), "-" * (len(" | ".join(header)) + 8)]
        for embedding in sorted(by_embedding_model.keys()):
            row_values = [embedding]
            for model in model_order:
                score = by_embedding_model[embedding].get(model)
                row_values.append("-" if score is None else f"{score:.4f}")
            lines.append(" | ".join(row_values))
        return "\n".join(lines)

    @staticmethod
    def _wandb_init(wandb_project: str, config: dict[str, Any], trial_index: int, run_name: str):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.init(project=wandb_project, config=config, name=run_name)
