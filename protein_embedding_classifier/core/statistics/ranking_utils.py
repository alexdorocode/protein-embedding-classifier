from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata, studentized_range


@dataclass(frozen=True)
class ScoreMatrix:
    run_ids: list[int]
    model_ids: list[str]
    values: np.ndarray

    @property
    def num_runs(self) -> int:
        if self.values.ndim != 2:
            return 0
        return int(self.values.shape[0])

    @property
    def num_models(self) -> int:
        if self.values.ndim != 2:
            return 0
        return int(self.values.shape[1])


def build_score_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_key: str = "seed",
    model_key: str = "model_config",
    score_key: str = "f1",
    drop_incomplete: bool = True,
) -> ScoreMatrix:
    score_buckets: dict[tuple[int, str], list[float]] = {}
    run_ids_set: set[int] = set()
    model_ids_set: set[str] = set()

    for row in rows:
        if not isinstance(row, Mapping):
            continue

        run_raw = row.get(run_key)
        model_raw = row.get(model_key)
        if run_raw is None or model_raw is None:
            continue

        try:
            run_id = int(run_raw)
        except Exception:
            continue

        model_id = str(model_raw).strip()
        if not model_id:
            continue

        score = _safe_float(row.get(score_key))
        if not np.isfinite(score):
            continue

        run_ids_set.add(run_id)
        model_ids_set.add(model_id)
        score_buckets.setdefault((run_id, model_id), []).append(float(score))

    run_ids = sorted(run_ids_set)
    model_ids = sorted(model_ids_set)
    if not run_ids or not model_ids:
        return ScoreMatrix(run_ids=[], model_ids=[], values=np.empty((0, 0), dtype=float))

    matrix = np.full((len(run_ids), len(model_ids)), np.nan, dtype=float)
    run_index = {run_id: idx for idx, run_id in enumerate(run_ids)}
    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}

    for (run_id, model_id), values in score_buckets.items():
        row_idx = run_index[run_id]
        col_idx = model_index[model_id]
        matrix[row_idx, col_idx] = float(np.mean(np.asarray(values, dtype=float)))

    if drop_incomplete:
        run_ids, model_ids, matrix = _drop_incomplete_rows_and_columns(
            run_ids=run_ids,
            model_ids=model_ids,
            matrix=matrix,
        )

    return ScoreMatrix(run_ids=run_ids, model_ids=model_ids, values=matrix)


def compute_rank_matrix(
    score_matrix: np.ndarray,
    *,
    higher_is_better: bool = True,
    tie_method: str = "average",
) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be a 2D array")
    if matrix.size == 0:
        return np.empty_like(matrix, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix contains non-finite values")

    ranks = np.empty_like(matrix, dtype=float)
    for row_idx in range(matrix.shape[0]):
        scores = matrix[row_idx, :]
        rank_input = -scores if higher_is_better else scores
        ranks[row_idx, :] = rankdata(rank_input, method=tie_method)
    return ranks


def compute_average_ranks(rank_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rank_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("rank_matrix must be a 2D array")
    if matrix.size == 0:
        return np.asarray([], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("rank_matrix contains non-finite values")
    return np.mean(matrix, axis=0)


def compute_critical_difference(num_models: int, num_runs: int, *, alpha: float = 0.05) -> float:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1)")
    if int(num_models) < 2 or int(num_runs) < 1:
        return float("nan")

    q_alpha = float(studentized_range.ppf(1.0 - float(alpha), int(num_models), np.inf) / np.sqrt(2.0))
    if not np.isfinite(q_alpha):
        return float("nan")

    scale = np.sqrt(int(num_models) * (int(num_models) + 1.0) / (6.0 * int(num_runs)))
    return float(q_alpha * scale)


def _drop_incomplete_rows_and_columns(
    run_ids: list[int],
    model_ids: list[str],
    matrix: np.ndarray,
) -> tuple[list[int], list[str], np.ndarray]:
    reduced_runs = list(run_ids)
    reduced_models = list(model_ids)
    reduced_matrix = np.asarray(matrix, dtype=float)

    changed = True
    while changed and reduced_matrix.size > 0:
        changed = False

        row_mask = np.isfinite(reduced_matrix).all(axis=1)
        if not row_mask.all():
            reduced_matrix = reduced_matrix[row_mask, :]
            reduced_runs = [run_id for run_id, keep in zip(reduced_runs, row_mask) if keep]
            changed = True

        if reduced_matrix.size == 0:
            break

        col_mask = np.isfinite(reduced_matrix).all(axis=0)
        if not col_mask.all():
            reduced_matrix = reduced_matrix[:, col_mask]
            reduced_models = [model_id for model_id, keep in zip(reduced_models, col_mask) if keep]
            changed = True

    if reduced_matrix.size == 0:
        reduced_matrix = np.empty((0, 0), dtype=float)

    return reduced_runs, reduced_models, reduced_matrix


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")
