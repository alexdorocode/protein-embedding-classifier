from __future__ import annotations

from typing import Any

import numpy as np


class ProbabilityAdapter:
    """Normalize raw classifier outputs into canonical probability tensors."""

    @staticmethod
    def to_canonical(
        raw_output: np.ndarray | list[Any],
        problem_type: str,
        classes: np.ndarray | list[Any] | tuple[Any, ...] | None = None,
        context: str = "",
    ) -> np.ndarray:
        probs = np.asarray(raw_output, dtype=np.float32)
        expected_classes = len(classes) if classes is not None else None
        label = f" [{context}]" if context else ""

        if probs.size == 0:
            raise ValueError(f"Empty probability output{label}")

        if probs.ndim == 0:
            raise ValueError(f"Invalid probability rank=0{label}")

        if problem_type == "binary":
            canonical = ProbabilityAdapter._binary_to_canonical(probs, label)
            ProbabilityAdapter._validate_class_count(canonical, expected_classes, label)
            ProbabilityAdapter._validate_range(canonical, label)
            return canonical

        if problem_type == "multiclass":
            canonical = ProbabilityAdapter._multiclass_to_canonical(probs, label)
            ProbabilityAdapter._validate_class_count(canonical, expected_classes, label)
            ProbabilityAdapter._validate_range(canonical, label)
            ProbabilityAdapter._validate_row_sum_to_one(canonical, label)
            return canonical

        if problem_type == "multilabel":
            canonical = ProbabilityAdapter._multilabel_to_canonical(probs, label)
            ProbabilityAdapter._validate_class_count(canonical, expected_classes, label)
            ProbabilityAdapter._validate_range(canonical, label)
            return canonical

        raise ValueError(f"Unsupported problem_type={problem_type}{label}")

    @staticmethod
    def _binary_to_canonical(probs: np.ndarray, label: str) -> np.ndarray:
        if probs.ndim == 1:
            positive = probs.reshape(-1, 1)
            return np.hstack([1.0 - positive, positive])

        if probs.ndim == 2 and probs.shape[1] == 1:
            positive = probs
            return np.hstack([1.0 - positive, positive])

        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs

        raise ValueError(
            f"Binary probabilities must be shaped (N,), (N,1), or (N,2), got {tuple(probs.shape)}{label}"
        )

    @staticmethod
    def _multiclass_to_canonical(probs: np.ndarray, label: str) -> np.ndarray:
        if probs.ndim != 2:
            raise ValueError(
                f"Multiclass probabilities must be 2D with shape (N,C), got {tuple(probs.shape)}{label}"
            )
        if probs.shape[1] < 2:
            raise ValueError(
                f"Multiclass probabilities must have C>=2, got {tuple(probs.shape)}{label}"
            )
        return probs

    @staticmethod
    def _multilabel_to_canonical(probs: np.ndarray, label: str) -> np.ndarray:
        if probs.ndim != 2:
            raise ValueError(
                f"Multilabel probabilities must be 2D with shape (N,C), got {tuple(probs.shape)}{label}"
            )
        if probs.shape[1] < 1:
            raise ValueError(
                f"Multilabel probabilities must have C>=1, got {tuple(probs.shape)}{label}"
            )
        return probs

    @staticmethod
    def _validate_class_count(probs: np.ndarray, expected_classes: int | None, label: str) -> None:
        if expected_classes is None:
            return
        if probs.shape[1] != int(expected_classes):
            raise ValueError(
                "Probability class dimension mismatch"
                f" expected={expected_classes} got={probs.shape[1]}{label}"
            )

    @staticmethod
    def _validate_row_sum_to_one(probs: np.ndarray, label: str, atol: float = 1e-3) -> None:
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, np.ones_like(row_sums), atol=atol):
            raise ValueError(f"Multiclass probabilities must sum to 1 across classes{label}")

    @staticmethod
    def _validate_range(probs: np.ndarray, label: str) -> None:
        if np.any(probs < -1e-6) or np.any(probs > 1.0 + 1e-6):
            raise ValueError(f"Probabilities must be in [0,1]{label}")
