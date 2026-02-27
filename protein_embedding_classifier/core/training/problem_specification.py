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
        values = np.asarray(labels, dtype=object)
        if values.size == 0:
            raise ValueError("Cannot infer ProblemSpecification from empty labels")

        if cls._is_multilabel(values):
            flat_labels: list[Any] = []
            for item in values:
                if isinstance(item, np.ndarray):
                    flat_labels.extend(item.tolist())
                else:
                    flat_labels.extend(list(item))
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
        return any(isinstance(item, (list, tuple, set, np.ndarray)) for item in values)
