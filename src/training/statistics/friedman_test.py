from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare


def run_friedman_test(score_matrix: np.ndarray, *, alpha: float = 0.05) -> dict[str, Any]:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1)")

    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be a 2D array")
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix contains non-finite values")

    num_runs, num_models = matrix.shape
    if int(num_runs) < 2:
        raise ValueError("Friedman test requires at least 2 runs")
    if int(num_models) < 3:
        raise ValueError("Friedman test requires at least 3 models")

    statistic, p_value = friedmanchisquare(*[matrix[:, idx] for idx in range(int(num_models))])
    p_value_float = float(p_value)

    return {
        "statistic": float(statistic),
        "p_value": p_value_float,
        "num_models": int(num_models),
        "num_runs": int(num_runs),
        "alpha": float(alpha),
        "significant": bool(np.isfinite(p_value_float) and p_value_float < float(alpha)),
    }
