from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def decide(
    probs: np.ndarray,
    problem_type: str,
    threshold_config: dict[str, Any] | Mapping[str, Any] | None,
) -> np.ndarray:
    probs_array = np.asarray(probs)
    if probs_array.ndim != 2:
        raise ValueError(f"DecisionPolicy expects canonical 2D probs, got shape={tuple(probs_array.shape)}")

    if problem_type == "multiclass":
        return np.argmax(probs_array, axis=1)

    threshold = _resolve_threshold(threshold_config)

    if problem_type == "binary":
        if probs_array.shape[1] != 2:
            raise ValueError(
                f"Binary decision expects canonical shape (N,2), got {tuple(probs_array.shape)}"
            )
        return (probs_array[:, 1] >= threshold).astype(int)

    if problem_type == "multilabel":
        if probs_array.shape[1] < 1:
            raise ValueError(
                f"Multilabel decision expects canonical shape (N,C), got {tuple(probs_array.shape)}"
            )
        return (probs_array >= threshold).astype(int)

    raise ValueError(f"Unsupported problem_type={problem_type}")


def _resolve_threshold(threshold_config: dict[str, Any] | Mapping[str, Any] | None) -> float:
    if not isinstance(threshold_config, Mapping):
        return 0.5

    default_threshold = float(threshold_config.get("default", threshold_config.get("threshold", 0.5)))
    classifier = threshold_config.get("classifier_name")
    embedding_name = threshold_config.get("embedding_name")

    classifier_embedding = threshold_config.get("classifier_embedding", {})
    if isinstance(classifier_embedding, Mapping) and classifier and embedding_name:
        key = f"{classifier}::{embedding_name}"
        if key in classifier_embedding:
            return float(classifier_embedding[key])

    classifier_map = threshold_config.get("classifier", {})
    if isinstance(classifier_map, Mapping) and classifier in classifier_map:
        return float(classifier_map[classifier])

    return default_threshold
