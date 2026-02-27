from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

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
        metrics_average = config.get("metrics_average", "macro")

        results: dict[tuple[str, str], dict[str, Any]] = {}

        for embedding_name in embedding_bundle.X_train:
            x_train = embedding_bundle.X_train[embedding_name]
            x_val = embedding_bundle.X_val[embedding_name]
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
                    tuple(x_train.shape),
                    tuple(x_val.shape),
                )

                model = self.model_factory.create(
                    model_type=model_type,
                    params=params,
                    input_size=int(x_train.shape[1]),
                    output_size=int(problem_spec.output_size),
                )

                if problem_spec.problem_type == "multilabel" and model_type.upper() != "MLP":
                    model = OneVsRestClassifier(model)

                if hasattr(model, "fit") and hasattr(model, "predict_proba") and model_type.upper() != "MLP":
                    model.fit(x_train, y_train_processed)
                    val_probs = model.predict_proba(x_val)
                else:
                    if problem_spec.problem_type == "multilabel":
                        raise ValueError("MLP training is not supported for multilabel targets in this pipeline")
                    model.fit(x_train, y_train, x_val, y_val)
                    val_probs = model.predict_proba(x_val)

                y_pred = self._probs_to_predictions(
                    model=model,
                    val_probs=val_probs,
                    problem_spec=problem_spec,
                )
                if problem_spec.problem_type == "multilabel":
                    if multilabel_binarizer is None:
                        raise RuntimeError("Expected MultiLabelBinarizer for multilabel problem")
                    val_f1 = f1_score(
                        y_val_processed,
                        y_pred,
                        average=metrics_average,
                        zero_division=0,
                    )
                else:
                    val_f1 = f1_score(y_val, y_pred, average=metrics_average)
                metrics = {"f1_score": float(val_f1)}

                self.logger.info(
                    "Trained model_type=%s embedding=%s f1_score=%.6f",
                    model_type,
                    embedding_name,
                    val_f1,
                )

                results[(model_type, embedding_name)] = {
                    "model": model,
                    "val_probs": val_probs,
                    "metrics": metrics,
                }

        return results

    def _resolve_params(self, model_type: str, model_params: dict[str, Any]) -> dict[str, Any]:
        if self.sweep_mode and self.wandb_config is not None:
            if isinstance(self.wandb_config, Mapping):
                return dict(self.wandb_config)
            return dict(self.wandb_config.items())

        return dict(model_params.get(model_type, model_params.get(model_type.upper(), {})))

    @staticmethod
    def _probs_to_predictions(model, val_probs: np.ndarray, problem_spec: ProblemSpecification) -> np.ndarray:
        if problem_spec.problem_type == "multilabel":
            return (np.asarray(val_probs) >= 0.5).astype(int)

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
