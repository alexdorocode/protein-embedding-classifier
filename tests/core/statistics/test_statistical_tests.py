from __future__ import annotations

import numpy as np

from protein_embedding_classifier.core.statistics.friedman_test import run_friedman_test
from protein_embedding_classifier.core.statistics.nemenyi_test import run_nemenyi_posthoc
from protein_embedding_classifier.core.statistics.ranking_utils import (
    build_score_matrix,
    compute_average_ranks,
    compute_rank_matrix,
)


def _rows_three_seeds_four_models() -> list[dict[str, float | int | str]]:
    return [
        {"seed": 42, "model_config": "M1", "f1": 0.91},
        {"seed": 42, "model_config": "M2", "f1": 0.83},
        {"seed": 42, "model_config": "M3", "f1": 0.75},
        {"seed": 42, "model_config": "M4", "f1": 0.60},
        {"seed": 43, "model_config": "M1", "f1": 0.89},
        {"seed": 43, "model_config": "M2", "f1": 0.82},
        {"seed": 43, "model_config": "M3", "f1": 0.74},
        {"seed": 43, "model_config": "M4", "f1": 0.58},
        {"seed": 44, "model_config": "M1", "f1": 0.90},
        {"seed": 44, "model_config": "M2", "f1": 0.81},
        {"seed": 44, "model_config": "M3", "f1": 0.73},
        {"seed": 44, "model_config": "M4", "f1": 0.57},
    ]


def test_ranking_computation_correct():
    score_matrix = build_score_matrix(_rows_three_seeds_four_models())
    assert score_matrix.values.shape == (3, 4)

    rank_matrix = compute_rank_matrix(score_matrix.values, higher_is_better=True)
    avg_ranks = compute_average_ranks(rank_matrix)
    rank_by_model = {model: float(rank) for model, rank in zip(score_matrix.model_ids, avg_ranks)}

    assert rank_by_model["M1"] == 1.0
    assert rank_by_model["M2"] == 2.0
    assert rank_by_model["M3"] == 3.0
    assert rank_by_model["M4"] == 4.0


def test_friedman_test_runs_correctly():
    score_matrix = build_score_matrix(_rows_three_seeds_four_models())
    result = run_friedman_test(score_matrix.values, alpha=0.05)

    assert set(result.keys()) == {
        "statistic",
        "p_value",
        "num_models",
        "num_runs",
        "alpha",
        "significant",
    }
    assert result["num_models"] == 4
    assert result["num_runs"] == 3
    assert np.isfinite(float(result["statistic"]))
    assert np.isfinite(float(result["p_value"]))


def test_nemenyi_output_structure_correct():
    score_matrix = build_score_matrix(_rows_three_seeds_four_models())
    rank_matrix = compute_rank_matrix(score_matrix.values, higher_is_better=True)
    avg_ranks = compute_average_ranks(rank_matrix)

    result = run_nemenyi_posthoc(
        avg_ranks=avg_ranks,
        model_labels=score_matrix.model_ids,
        num_runs=score_matrix.num_runs,
        alpha=0.05,
    )

    assert np.isfinite(float(result["critical_difference"]))
    assert len(result["comparisons"]) == 6

    first = result["comparisons"][0]
    assert set(first.keys()) == {
        "model_a",
        "model_b",
        "p_value",
        "rank_diff",
        "critical_difference",
        "significant",
    }


def test_statistics_work_with_three_seeds_and_four_models():
    rows = _rows_three_seeds_four_models()
    score_matrix = build_score_matrix(rows)

    assert score_matrix.num_runs == 3
    assert score_matrix.num_models == 4

    friedman_result = run_friedman_test(score_matrix.values)
    assert friedman_result["num_runs"] == 3
    assert friedman_result["num_models"] == 4

    avg_ranks = compute_average_ranks(compute_rank_matrix(score_matrix.values, higher_is_better=True))
    nemenyi_result = run_nemenyi_posthoc(
        avg_ranks=avg_ranks,
        model_labels=score_matrix.model_ids,
        num_runs=score_matrix.num_runs,
    )
    assert len(nemenyi_result["comparisons"]) == 6
