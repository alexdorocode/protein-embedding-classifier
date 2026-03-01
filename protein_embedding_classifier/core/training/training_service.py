from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.model_factory import ModelFactory
from protein_embedding_classifier.core.training.problem_specification import ProblemSpecification


class TrainingService:
    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        sweep_mode: bool = False,
        wandb_config: Mapping[str, Any] | None = None,
    ):
        self.model_factory = model_factory or ModelFactory()
        self.sweep_mode = sweep_mode
        self.wandb_config = wandb_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(
        self,
        embedding_bundle: EmbeddingBundle,
        training_config: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        config = training_config or {}
        model_types = config.get("model_types", ["LR"])
        model_params = config.get("model_params", {})
        metrics_average = str(config.get("metrics_average", "macro"))
        evaluate_test = bool(config.get("evaluate_test", False))
        normalization_mode = self._resolve_normalization_mode(config)

        self.logger.info("Normalization mode: %s", normalization_mode)

        results: dict[tuple[str, str], dict[str, Any]] = {}

        for embedding_name in embedding_bundle.X_train:
            x_train = embedding_bundle.X_train[embedding_name]
            x_val = embedding_bundle.X_val[embedding_name]
            x_test = embedding_bundle.X_test[embedding_name]

            x_train_processed, x_val_processed, x_test_processed = self._apply_normalization(
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                mode=normalization_mode,
            )

            y_train = embedding_bundle.y_train
            y_val = embedding_bundle.y_val
            problem_spec = ProblemSpecification.from_labels(y_train)

            y_train_processed, y_val_processed, multilabel_binarizer = self._prepare_labels(
                problem_spec=problem_spec,
                y_train=y_train,
                y_val=y_val,
            )

            for model_type in model_types:
                params = self._resolve_params(model_type=model_type, model_params=model_params)
                if model_type.upper() == "MLP" and "criterion_name" not in params:
                    params["criterion_name"] = problem_spec.loss_name

                self.logger.info(
                    "Training model_type=%s embedding=%s train_shape=%s val_shape=%s",
                    model_type,
                    embedding_name,
                    tuple(x_train_processed.shape),
                    tuple(x_val_processed.shape),
                )

                model = self.model_factory.create(
                    model_type=model_type,
                    params=params,
                    input_size=int(x_train_processed.shape[1]),
                    output_size=int(problem_spec.output_size),
                )

                if problem_spec.problem_type == "multilabel" and model_type.upper() != "MLP":
                    model = OneVsRestClassifier(model)

                if hasattr(model, "fit") and hasattr(model, "predict_proba") and model_type.upper() != "MLP":
                    model.fit(x_train_processed, y_train_processed)
                    val_probs = model.predict_proba(x_val_processed)
                else:
                    if problem_spec.problem_type == "multilabel":
                        raise ValueError(
                            "MLP training is not supported for multilabel targets in this pipeline "
                            "(MLP is supported for binary and multiclass tasks)."
                        )
                    model.fit(x_train_processed, y_train, x_val_processed, y_val)
                    val_probs = model.predict_proba(x_val_processed)

                validation_metrics = self._compute_metrics(
                    y_true=y_val_processed if problem_spec.problem_type == "multilabel" else y_val,
                    probs=val_probs,
                    problem_spec=problem_spec,
                    metrics_average=metrics_average,
                    class_labels=getattr(model, "classes_", None),
                )

                test_metrics: dict[str, Any] | None = None
                if evaluate_test and hasattr(model, "predict_proba"):
                    test_probs = model.predict_proba(x_test_processed)
                    y_test_true = (
                        self._transform_multilabel_with_binarizer(embedding_bundle.y_test, multilabel_binarizer)
                        if problem_spec.problem_type == "multilabel"
                        else embedding_bundle.y_test
                    )
                    test_metrics = self._compute_metrics(
                        y_true=y_test_true,
                        probs=test_probs,
                        problem_spec=problem_spec,
                        metrics_average=metrics_average,
                        class_labels=getattr(model, "classes_", None),
                    )

                val_f1 = float(
                    validation_metrics.get(
                        "f1",
                        validation_metrics.get("macro_f1", float("nan")),
                    )
                )

                self.logger.info(
                    "Trained model_type=%s embedding=%s f1_score=%.6f",
                    model_type,
                    embedding_name,
                    val_f1,
                )

                results[(model_type, embedding_name)] = {
                    "model": model,
                    "val_probs": val_probs,
                    "metrics": {
                        "validation": validation_metrics,
                        "test": test_metrics,
                    },
                }

        return results

    def _resolve_params(self, model_type: str, model_params: dict[str, Any]) -> dict[str, Any]:
        if self.sweep_mode and self.wandb_config is not None:
            if isinstance(self.wandb_config, Mapping):
                return dict(self.wandb_config)
            return dict(self.wandb_config.items())

        return dict(model_params.get(model_type, model_params.get(model_type.upper(), {})))

    def _resolve_normalization_mode(self, config: dict[str, Any]) -> str:
        feature_processing = config.get("feature_processing", {})
        normalize_mode = "none"
        if isinstance(feature_processing, Mapping):
            normalize_mode = str(feature_processing.get("normalize", "none"))

        if self.sweep_mode and self.wandb_config is not None:
            if isinstance(self.wandb_config, Mapping):
                if "normalize" in self.wandb_config:
                    normalize_mode = str(self.wandb_config["normalize"])
                wandb_feature_processing = self.wandb_config.get("feature_processing")
                if isinstance(wandb_feature_processing, Mapping) and "normalize" in wandb_feature_processing:
                    normalize_mode = str(wandb_feature_processing["normalize"])

        normalized = normalize_mode.lower()
        valid_modes = {"none", "l2", "standard"}
        if normalized not in valid_modes:
            raise ValueError(
                f"Unsupported feature_processing.normalize: {normalize_mode}. "
                f"Expected one of {sorted(valid_modes)}"
            )
        return normalized

    def _apply_normalization(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mode == "none":
            return np.asarray(x_train), np.asarray(x_val), np.asarray(x_test)

        if mode == "l2":
            self.logger.info("Applied L2 normalization (row-wise)")
            return (
                normalize(np.asarray(x_train), norm="l2", axis=1),
                normalize(np.asarray(x_val), norm="l2", axis=1),
                normalize(np.asarray(x_test), norm="l2", axis=1),
            )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(np.asarray(x_train))
        x_val_scaled = scaler.transform(np.asarray(x_val))
        x_test_scaled = scaler.transform(np.asarray(x_test))
        self.logger.info("Applied StandardScaler (fit on train only)")
        return x_train_scaled, x_val_scaled, x_test_scaled

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        problem_spec: ProblemSpecification,
        metrics_average: str,
        class_labels: np.ndarray | None,
    ) -> dict[str, Any]:
        y_true_array = np.asarray(y_true)
        probs_array = np.asarray(probs)

        metrics: dict[str, Any] = {}

        if problem_spec.problem_type == "multilabel":
            y_pred = (probs_array >= 0.5).astype(int)
            metrics["micro_f1"] = float(f1_score(y_true_array, y_pred, average="micro", zero_division=0))
            metrics["macro_f1"] = float(f1_score(y_true_array, y_pred, average="macro", zero_division=0))
            metrics["f1"] = float(metrics["macro_f1"])
            return metrics

        y_pred = self._probs_to_predictions(
            model=None,
            val_probs=probs_array,
            problem_spec=problem_spec,
            class_labels=class_labels,
        )

        metrics["accuracy"] = float(accuracy_score(y_true_array, y_pred))
        if problem_spec.problem_type == "binary":
            metrics["precision"] = float(precision_score(y_true_array, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true_array, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_true_array, y_pred, zero_division=0))

            if probs_array.ndim == 2 and probs_array.shape[1] == 2 and np.unique(y_true_array).size == 2:
                y_score = probs_array[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score))
                metrics["pr_auc"] = float(average_precision_score(y_true_array, y_score))

            labels = np.unique(y_true_array)
            matrix = confusion_matrix(y_true_array, y_pred, labels=labels)
            metrics["confusion_matrix"] = matrix.tolist()
            return metrics

        metrics["precision"] = float(precision_score(y_true_array, y_pred, average=metrics_average, zero_division=0))
        metrics["recall"] = float(recall_score(y_true_array, y_pred, average=metrics_average, zero_division=0))
        metrics["f1"] = float(f1_score(y_true_array, y_pred, average=metrics_average, zero_division=0))
        return metrics

    @staticmethod
    def _probs_to_predictions(
        model,
        val_probs: np.ndarray,
        problem_spec: ProblemSpecification,
        class_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        if problem_spec.problem_type == "multilabel":
            return (np.asarray(val_probs) >= 0.5).astype(int)

        if class_labels is None and model is not None:
            class_labels = getattr(model, "classes_", None)
        pred_index = np.argmax(val_probs, axis=1)

        if class_labels is not None:
            class_array = np.asarray(class_labels)
            return class_array[pred_index]

        return pred_index

    @staticmethod
    def _prepare_labels(
        problem_spec: ProblemSpecification,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, MultiLabelBinarizer | None]:
        if problem_spec.problem_type != "multilabel":
            return np.asarray(y_train), np.asarray(y_val), None

        def _to_list_of_lists(values: np.ndarray) -> list[list[Any]]:
            converted: list[list[Any]] = []
            for item in np.asarray(values, dtype=object):
                if isinstance(item, np.ndarray):
                    converted.append(item.tolist())
                elif isinstance(item, (list, tuple, set)):
                    converted.append(list(item))
                else:
                    converted.append([item])
            return converted

        y_train_list = _to_list_of_lists(y_train)
        y_val_list = _to_list_of_lists(y_val)

        binarizer = MultiLabelBinarizer()
        y_train_bin = binarizer.fit_transform(y_train_list)
        y_val_bin = binarizer.transform(y_val_list)
        return y_train_bin, y_val_bin, binarizer

    @staticmethod
    def _transform_multilabel_with_binarizer(
        values: np.ndarray,
        binarizer: MultiLabelBinarizer | None,
    ) -> np.ndarray:
        if binarizer is None:
            raise RuntimeError("Expected MultiLabelBinarizer for multilabel transformation")

        converted: list[list[Any]] = []
        for item in np.asarray(values, dtype=object):
            if isinstance(item, np.ndarray):
                converted.append(item.tolist())
            elif isinstance(item, (list, tuple, set)):
                converted.append(list(item))
            else:
                converted.append([item])
        return binarizer.transform(converted)
