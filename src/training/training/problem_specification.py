from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProblemSpecification:
    problem_type: str
    classes: tuple[Any, ...]
    output_size: int
    loss_name: str

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> ProblemSpecification:
        values = cls._normalize_singleton_iterables(np.asarray(labels, dtype=object))
        if values.size == 0:
            raise ValueError("Cannot infer ProblemSpecification from empty labels")

        if cls._is_multilabel(values):
            flat_labels: list[Any] = []
            for item in values:
                if isinstance(item, np.ndarray):
                    flat_labels.extend(item.tolist())
                elif isinstance(item, (list, tuple, set)):
                    flat_labels.extend(list(item))
                else:
                    flat_labels.append(item)
            classes = tuple(sorted(set(flat_labels)))
            return cls(
                problem_type="multilabel",
                classes=classes,
                output_size=len(classes),
                loss_name="BCEWithLogitsLoss",
            )

        unique_classes = tuple(np.unique(values).tolist())
        if len(unique_classes) == 2:
            return cls(
                problem_type="binary",
                classes=unique_classes,
                output_size=2,
                loss_name="BCEWithLogitsLoss",
            )

        return cls(
            problem_type="multiclass",
            classes=unique_classes,
            output_size=len(unique_classes),
            loss_name="CrossEntropyLoss",
        )

    @staticmethod
    def _is_multilabel(values: np.ndarray) -> bool:
        for item in values:
            if not isinstance(item, (list, tuple, set, np.ndarray)):
                continue

            if isinstance(item, np.ndarray):
                flattened = item.reshape(-1)
                if flattened.size > 1:
                    return True
                continue

            if len(item) > 1:
                return True

        return False

    @staticmethod
    def _normalize_singleton_iterables(values: np.ndarray) -> np.ndarray:
        normalized: list[Any] = []
        for item in values:
            if isinstance(item, np.ndarray):
                flattened = item.reshape(-1)
                if flattened.size == 1:
                    normalized.append(flattened[0])
                else:
                    normalized.append(item)
                continue

            if isinstance(item, (list, tuple, set)):
                if len(item) == 1:
                    normalized.append(next(iter(item)))
                else:
                    normalized.append(item)
                continue

            normalized.append(item)

        return np.asarray(normalized, dtype=object)
