from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import studentized_range

from .ranking_utils import compute_critical_difference


def run_nemenyi_posthoc(
    *,
    avg_ranks: Sequence[float] | np.ndarray,
    model_labels: Sequence[str],
    num_runs: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1)")

    labels = [str(label) for label in model_labels]
    ranks = np.asarray(avg_ranks, dtype=float)
    if ranks.ndim != 1:
        raise ValueError("avg_ranks must be a 1D array")
    if len(labels) != int(ranks.size):
        raise ValueError("model_labels length must match avg_ranks size")
    if int(ranks.size) < 2:
        raise ValueError("Nemenyi test requires at least 2 models")
    if int(num_runs) < 1:
        raise ValueError("Nemenyi test requires at least 1 run")
    if not np.isfinite(ranks).all():
        raise ValueError("avg_ranks contains non-finite values")

    num_models = int(ranks.size)
    standard_error = np.sqrt(num_models * (num_models + 1.0) / (6.0 * int(num_runs)))
    critical_difference = compute_critical_difference(num_models, int(num_runs), alpha=float(alpha))

    comparisons: list[dict[str, Any]] = []
    for idx_a in range(num_models):
        for idx_b in range(idx_a + 1, num_models):
            rank_diff = float(abs(ranks[idx_a] - ranks[idx_b]))
            q_stat = float(rank_diff / standard_error)
            p_value = float(studentized_range.sf(q_stat * np.sqrt(2.0), num_models, np.inf))
            comparisons.append(
                {
                    "model_a": labels[idx_a],
                    "model_b": labels[idx_b],
                    "p_value": p_value,
                    "rank_diff": rank_diff,
                    "critical_difference": float(critical_difference),
                    "significant": bool(np.isfinite(critical_difference) and rank_diff > critical_difference),
                }
            )

    comparisons.sort(
        key=lambda row: (
            row.get("p_value", float("inf")),
            str(row.get("model_a", "")),
            str(row.get("model_b", "")),
        )
    )

    return {
        "alpha": float(alpha),
        "num_models": num_models,
        "num_runs": int(num_runs),
        "critical_difference": float(critical_difference),
        "comparisons": comparisons,
    }
