from __future__ import annotations

import inspect
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from protein_embedding_classifier.core.training.torch_wrapper import TorchTrainingWrapper


class ModelFactory:
    def __init__(self):
        self._constructors = {
            "LR": LogisticRegression,
            "SVM": SVC,
            "RF": RandomForestClassifier,
            "KNN-2": KNeighborsClassifier,
            "MLP": TorchTrainingWrapper,
        }

    def create(
        self,
        model_type: str,
        params: dict[str, Any] | None = None,
        input_size: int | None = None,
        output_size: int | None = None,
    ):
        normalized_model_type = model_type.upper()
        merged_params = self._preprocess_params(normalized_model_type, params or {})

        if normalized_model_type == "XGB":
            try:
                from xgboost import XGBClassifier  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "XGB model requested but xgboost is not installed. "
                    "Install it with: poetry add xgboost"
                ) from exc

            filtered = self._filter_constructor_params(XGBClassifier, merged_params)
            return XGBClassifier(**filtered)

        if normalized_model_type not in self._constructors:
            raise ValueError(f"Unsupported model_type: {model_type}")

        constructor = self._constructors[normalized_model_type]

        if normalized_model_type == "SVM":
            merged_params.setdefault("probability", True)

        if normalized_model_type == "MLP":
            merged_params["input_size"] = input_size
            merged_params["output_size"] = output_size

        filtered = self._filter_constructor_params(constructor, merged_params)
        return constructor(**filtered)

    @staticmethod
    def _preprocess_params(model_type: str, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params)

        if model_type == "SVM" and isinstance(merged.get("kernel_config"), dict):
            kernel_config = merged.pop("kernel_config")
            merged.update(kernel_config)

        if model_type == "RF" and isinstance(merged.get("bootstrap_config"), dict):
            bootstrap_config = merged.pop("bootstrap_config")
            merged.update(bootstrap_config)

        if model_type == "KNN-2" and isinstance(merged.get("p_metric"), dict):
            p_metric = merged.pop("p_metric")
            merged.update(p_metric)

        if model_type == "MLP" and isinstance(merged.get("custom_layer_config"), dict):
            custom_layer_config = merged.pop("custom_layer_config")
            merged.update(custom_layer_config)

        return merged

    @staticmethod
    def _filter_constructor_params(constructor, params: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(constructor)
        allowed = set(signature.parameters.keys())
        return {name: value for name, value in params.items() if name in allowed}
