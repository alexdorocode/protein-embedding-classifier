from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.training.pipeline import Pipeline


def _confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    return tn, fp, fn, tp


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    total = max(1, len(y_true))
    precision_den = tp + fp
    recall_den = tp + fn
    precision = float(tp / precision_den) if precision_den else 0.0
    recall = float(tp / recall_den) if recall_den else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": float((tp + tn) / total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_model_row(rows: list[dict[str, str]], model_type: str, embedding_name: str) -> dict[str, str]:
    for row in rows:
        if (
            row.get("model_type") == model_type
            and row.get("embedding_name") == embedding_name
            and row.get("strategy") == "single"
        ):
            return row
    raise AssertionError(f"Missing model row for {model_type}::{embedding_name}")


def _find_ensemble_row(rows: list[dict[str, str]], strategy: str) -> dict[str, str]:
    for row in rows:
        if (
            row.get("model_type") == "Ensemble"
            and row.get("embedding_name") == "all"
            and row.get("strategy") == strategy
        ):
            return row
    raise AssertionError(f"Missing ensemble row for strategy={strategy}")


def test_global_benchmark_generates_structured_execution_directories(monkeypatch, tmp_path):
    layout = {
        "root": tmp_path / "pec_data",
        "dataset": tmp_path / "pec_data" / "dataset",
        "sweep": tmp_path / "pec_data" / "sweep",
        "results": tmp_path / "pec_data" / "results",
        "logs": tmp_path / "pec_data" / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    executed: list[tuple[str, int, Path]] = []

    def _runner(step_name: str):
        def _run(self, conf):
            del conf
            seed_used = int(self.runtime_context.get("seed_used"))
            output_root = Path(str(self.runtime_context.get("output_root_override")))
            executed.append((step_name, seed_used, output_root))

            if step_name == "benchmark":
                run_dir = output_root / "sweep" / f"sweep_{seed_used}"
                (run_dir / "results").mkdir(parents=True, exist_ok=True)
                (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
                with (run_dir / "results" / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump({"models_evaluated": [], "ensembles": []}, handle)

        return _run

    monkeypatch.setattr(Pipeline, "_load_training_config", staticmethod(lambda conf: {}))
    monkeypatch.setattr(Pipeline, "_ensure_pec_data_layout", staticmethod(lambda reporting_conf: layout))
    monkeypatch.setattr(Pipeline, "_build_benchmark_variant_specs", staticmethod(lambda **kwargs: []))
    monkeypatch.setattr(
        Pipeline,
        "_expected_global_model_embedding_combinations",
        lambda self, pipeline_conf, training_global_conf: set(),
    )
    monkeypatch.setattr(Pipeline, "run_sweep_step", _runner("sweep"))
    monkeypatch.setattr(Pipeline, "run_ensemble_step", _runner("ensemble"))
    monkeypatch.setattr(Pipeline, "run_benchmark_step", _runner("benchmark"))

    pipeline = Pipeline(config_path="config/pipeline.yaml")
    pipeline.runtime_context = {"run_prefix": "exp"}

    pipeline.run_global_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "sweep": {},
            "ensemble": {},
            "benchmark": {},
            "experiment": {
                "main_seed": 42,
                "global_benchmark": {
                    "n_seeds": 3,
                },
            },
        }
    )

    global_benchmark_dir = layout["results"] / "global_benchmark"
    executions_dir = global_benchmark_dir / "executions"
    assert executions_dir.exists()

    execution_folders = sorted(path.name for path in executions_dir.iterdir() if path.is_dir())
    assert execution_folders == ["run_seed_42", "run_seed_43", "run_seed_44"]

    metadata_path = global_benchmark_dir / "metadata" / "experiment_seeds.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["main_seed"] == 42
    assert payload["generated_seeds"] == [42, 43, 44]
    assert payload["n_seeds"] == 3

    root_children = {path.name for path in global_benchmark_dir.iterdir()}
    assert root_children == {"executions", "aggregated", "predictions", "statistics", "metadata"}
    assert all(path.is_dir() for path in global_benchmark_dir.iterdir())

    seeds_used = sorted({seed for step, seed, _ in executed if step == "sweep"})
    assert seeds_used == [42, 43, 44]


def test_global_benchmark_complete_2x2x3_aggregation_with_non_nan_confusion(monkeypatch, tmp_path):
    layout = {
        "root": tmp_path / "pec_data",
        "dataset": tmp_path / "pec_data" / "dataset",
        "sweep": tmp_path / "pec_data" / "sweep",
        "results": tmp_path / "pec_data" / "results",
        "logs": tmp_path / "pec_data" / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    seeds = [100, 101, 102]
    classifiers = ["LR", "RF"]
    embeddings = ["ESM2", "ProtT5"]
    expected_model_combinations = {(classifier, embedding) for classifier in classifiers for embedding in embeddings}
    true_labels = [0, 0, 1, 1, 0, 1]

    seed_model_predictions: dict[int, dict[tuple[str, str], list[int]]] = {
        100: {
            ("LR", "ESM2"): [0, 0, 1, 1, 0, 1],
            ("LR", "ProtT5"): [0, 1, 1, 0, 0, 1],
            ("RF", "ESM2"): [0, 1, 1, 0, 0, 1],
            ("RF", "ProtT5"): [1, 1, 1, 0, 0, 1],
        },
        101: {
            ("LR", "ESM2"): [0, 0, 1, 0, 0, 1],
            ("LR", "ProtT5"): [0, 1, 1, 0, 1, 1],
            ("RF", "ESM2"): [0, 1, 1, 0, 1, 1],
            ("RF", "ProtT5"): [1, 1, 1, 1, 0, 1],
        },
        102: {
            ("LR", "ESM2"): [0, 0, 1, 1, 0, 0],
            ("LR", "ProtT5"): [0, 1, 0, 0, 1, 1],
            ("RF", "ESM2"): [0, 1, 0, 0, 1, 1],
            ("RF", "ProtT5"): [1, 1, 1, 0, 1, 1],
        },
    }

    seed_ensemble_predictions: dict[int, dict[str, list[int]]] = {
        100: {
            "uniform_soft_voting": [0, 0, 1, 1, 0, 1],
            "majority_global": [0, 1, 1, 0, 0, 1],
        },
        101: {
            "uniform_soft_voting": [0, 0, 1, 0, 0, 1],
            "majority_global": [1, 1, 1, 0, 1, 1],
        },
        102: {
            "uniform_soft_voting": [0, 0, 1, 1, 0, 0],
            "majority_global": [0, 1, 0, 0, 1, 1],
        },
    }

    def _run_sweep(self, conf):
        del conf
        seed_used = int(self.runtime_context.get("seed_used"))
        output_root = Path(str(self.runtime_context.get("output_root_override")))
        run_dir = output_root / "sweep" / f"sweep_{seed_used}"
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    def _run_ensemble(self, conf):
        del conf

    def _run_benchmark(self, conf):
        del conf
        seed_used = int(self.runtime_context.get("seed_used"))
        output_root = Path(str(self.runtime_context.get("output_root_override")))
        run_dir = output_root / "sweep" / f"sweep_{seed_used}"
        results_dir = run_dir / "results"
        predictions_dir = run_dir / "predictions"
        benchmark_predictions_dir = results_dir / "benchmark_predictions" / f"seed_{seed_used}"
        results_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        benchmark_predictions_dir.mkdir(parents=True, exist_ok=True)

        model_rows: list[dict[str, object]] = []
        for model_type, embedding_name in sorted(expected_model_combinations):
            metrics = _binary_metrics(true_labels, seed_model_predictions[seed_used][(model_type, embedding_name)])
            model_rows.append(
                {
                    "classifier_name": model_type,
                    "embedding_name": embedding_name,
                    "test_metrics": metrics,
                    "test_f1": float(metrics["f1"]),
                }
            )

        ensemble_rows: list[dict[str, object]] = []
        for strategy, predictions in sorted(seed_ensemble_predictions[seed_used].items()):
            metrics = _binary_metrics(true_labels, predictions)
            ensemble_rows.append(
                {
                    "variant": strategy,
                    "test_metrics": metrics,
                    "test_f1": float(metrics["f1"]),
                }
            )

        with (results_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
            json.dump({"models_evaluated": model_rows, "ensembles": ensemble_rows}, handle, indent=2)

        with (predictions_dir / f"predictions_seed_{seed_used}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "accession",
                    "true_label",
                    "predicted_label",
                    "prediction_probability",
                    "model_type",
                    "embedding_name",
                    "seed",
                ],
            )
            writer.writeheader()
            for model_type, embedding_name in sorted(expected_model_combinations):
                predictions = seed_model_predictions[seed_used][(model_type, embedding_name)]
                for index, (true_label, predicted_label) in enumerate(zip(true_labels, predictions), start=1):
                    writer.writerow(
                        {
                            "accession": f"ACC_{seed_used}_{index}",
                            "true_label": int(true_label),
                            "predicted_label": int(predicted_label),
                            "prediction_probability": 0.9 if int(predicted_label) == 1 else 0.1,
                            "model_type": model_type,
                            "embedding_name": embedding_name,
                            "seed": int(seed_used),
                        }
                    )

        for strategy, predictions in sorted(seed_ensemble_predictions[seed_used].items()):
            with (benchmark_predictions_dir / f"{strategy}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "accession",
                        "true_label",
                        "predicted_label",
                        "prediction_probability",
                        "strategy",
                        "seed",
                    ],
                )
                writer.writeheader()
                for index, (true_label, predicted_label) in enumerate(zip(true_labels, predictions), start=1):
                    writer.writerow(
                        {
                            "accession": f"ACC_{seed_used}_{index}",
                            "true_label": int(true_label),
                            "predicted_label": int(predicted_label),
                            "prediction_probability": 0.9 if int(predicted_label) == 1 else 0.1,
                            "strategy": strategy,
                            "seed": int(seed_used),
                        }
                    )

    monkeypatch.setattr(Pipeline, "_load_training_config", staticmethod(lambda conf: {}))
    monkeypatch.setattr(Pipeline, "_ensure_pec_data_layout", staticmethod(lambda reporting_conf: layout))
    monkeypatch.setattr(
        Pipeline,
        "_build_benchmark_variant_specs",
        staticmethod(
            lambda **kwargs: [
                {"variant": "uniform_soft_voting"},
                {"variant": "majority_global"},
            ]
        ),
    )
    monkeypatch.setattr(
        Pipeline,
        "_expected_global_model_embedding_combinations",
        lambda self, pipeline_conf, training_global_conf: set(expected_model_combinations),
    )
    monkeypatch.setattr(Pipeline, "run_sweep_step", _run_sweep)
    monkeypatch.setattr(Pipeline, "run_ensemble_step", _run_ensemble)
    monkeypatch.setattr(Pipeline, "run_benchmark_step", _run_benchmark)

    pipeline = Pipeline(config_path="config/pipeline.yaml")
    pipeline.runtime_context = {"run_prefix": "exp"}

    pipeline.run_global_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "sweep": {},
            "ensemble": {},
            "benchmark": {},
            "experiment": {
                "main_seed": 100,
                "global_benchmark": {
                    "n_seeds": 3,
                },
            },
        }
    )

    global_benchmark_dir = layout["results"] / "global_benchmark"
    model_csv = global_benchmark_dir / "aggregated" / "model_embedding_benchmark.csv"
    ensemble_csv = global_benchmark_dir / "aggregated" / "ensemble_strategy_benchmark.csv"
    ranking_csv = global_benchmark_dir / "aggregated" / "ranking_tables.csv"

    assert model_csv.exists()
    assert ensemble_csv.exists()
    assert ranking_csv.exists()

    model_rows = _read_csv_rows(model_csv)
    ensemble_rows = _read_csv_rows(ensemble_csv)

    observed_model_combinations = {
        (str(row.get("model_type")), str(row.get("embedding_name")))
        for row in model_rows
        if row.get("strategy") == "single"
    }
    assert observed_model_combinations == expected_model_combinations
    assert len(ensemble_rows) == 2

    for row in model_rows + ensemble_rows:
        for key in ("TP_mean", "TN_mean", "FP_mean", "FN_mean"):
            value = float(row.get(key, "nan"))
            assert np.isfinite(value), f"Expected finite confusion aggregate for {row.get('strategy')}:{key}"

    lr_esm2_row = _find_model_row(model_rows, "LR", "ESM2")
    expected_tp_mean = float(np.mean([_confusion_counts(true_labels, seed_model_predictions[seed][("LR", "ESM2")])[3] for seed in seeds]))
    expected_tn_mean = float(np.mean([_confusion_counts(true_labels, seed_model_predictions[seed][("LR", "ESM2")])[0] for seed in seeds]))
    expected_fp_mean = float(np.mean([_confusion_counts(true_labels, seed_model_predictions[seed][("LR", "ESM2")])[1] for seed in seeds]))
    expected_fn_mean = float(np.mean([_confusion_counts(true_labels, seed_model_predictions[seed][("LR", "ESM2")])[2] for seed in seeds]))

    assert float(lr_esm2_row["TP_mean"]) == pytest.approx(expected_tp_mean)
    assert float(lr_esm2_row["TN_mean"]) == pytest.approx(expected_tn_mean)
    assert float(lr_esm2_row["FP_mean"]) == pytest.approx(expected_fp_mean)
    assert float(lr_esm2_row["FN_mean"]) == pytest.approx(expected_fn_mean)

    uniform_row = _find_ensemble_row(ensemble_rows, "uniform_soft_voting")
    uniform_tp_mean = float(np.mean([_confusion_counts(true_labels, seed_ensemble_predictions[seed]["uniform_soft_voting"])[3] for seed in seeds]))
    assert float(uniform_row["TP_mean"]) == pytest.approx(uniform_tp_mean)

    lr_prott5_row = _find_model_row(model_rows, "LR", "ProtT5")
    rf_esm2_row = _find_model_row(model_rows, "RF", "ESM2")
    assert float(lr_prott5_row["rank_sum"]) == pytest.approx(float(rf_esm2_row["rank_sum"]))
    assert float(lr_prott5_row["rank_mean"]) == pytest.approx(float(rf_esm2_row["rank_mean"]))

    for seed in seeds:
        model_seed_dir = global_benchmark_dir / "predictions" / "model_predictions" / f"seed_{seed}"
        ensemble_seed_dir = global_benchmark_dir / "predictions" / "ensemble_predictions" / f"seed_{seed}"
        assert model_seed_dir.exists()
        assert ensemble_seed_dir.exists()
        assert len(list(model_seed_dir.glob("*.csv"))) == len(expected_model_combinations)
        assert len(list(ensemble_seed_dir.glob("*.csv"))) == len(seed_ensemble_predictions[seed])

    root_children = {path.name for path in global_benchmark_dir.iterdir()}
    assert root_children == {"executions", "aggregated", "predictions", "statistics", "metadata"}
    assert all(path.is_dir() for path in global_benchmark_dir.iterdir())
