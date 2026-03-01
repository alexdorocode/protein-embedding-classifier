from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.pipeline import Pipeline


class DummyPredictProbaModel:
    def __init__(self, bias: float):
        self.bias = float(bias)

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=np.float32)
        logits = arr[:, 0] + self.bias
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - positive, positive], axis=1)


def _write_model_artifact(run_dir: Path, classifier: str, embedding: str, bias: float) -> tuple[str, str, str]:
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model = DummyPredictProbaModel(bias=bias)

    model_rel = f"models/{classifier}_{embedding}.pkl"
    model_path = run_dir / model_rel
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    metadata_rel = f"models/{classifier}_{embedding}.metadata.json"
    metadata_path = run_dir / metadata_rel
    metadata = {
        "problem_type": "binary",
        "classes": [0, 1],
        "num_classes": 2,
        "normalization": "standard",
        "threshold_policy": {"default": 0.5},
        "classifier": classifier,
        "embedding_name": embedding,
        "serializer": "pickle",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_rel, metadata_rel, "pickle"


class _DatasetBundle:
    def __init__(self):
        self.train_ids = ["T1", "T2", "T3"]
        self.val_ids = ["V1", "V2", "V3", "V4"]
        self.test_ids = ["S1", "S2", "S3", "S4"]
        self.y_train = np.array([0, 1, 0])
        self.y_val = np.array([0, 1, 0, 1])
        self.y_test = np.array([0, 1, 0, 1])


def _setup_shared(monkeypatch, tmp_path):
    run_dir = tmp_path / "pec_data" / "sweep" / "run_20260301"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for classifier, embedding, bias in [
        ("LR", "ESM2", 0.2),
        ("RF", "ProtT5", -0.1),
        ("XGB", "ESM3", 0.05),
    ]:
        model_rel, meta_rel, serializer = _write_model_artifact(run_dir, classifier, embedding, bias)
        rows.append(
            {
                "embedding_name": embedding,
                "model_type": classifier,
                "validation_f1": 0.7,
                "config": "{}",
                "artifact_path": model_rel,
                "serializer": serializer,
                "metadata_path": meta_rel,
            }
        )

    best_csv = reports_dir / "best_classifier_per_embedding.csv"
    with best_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["embedding_name", "model_type", "validation_f1", "config", "artifact_path", "serializer", "metadata_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    dataset_bundle = _DatasetBundle()
    embedding_bundle = EmbeddingBundle(
        X_train={
            "ESM2": np.array([[0.1], [0.9], [0.2]], dtype=np.float32),
            "ProtT5": np.array([[0.0], [0.8], [0.4]], dtype=np.float32),
            "ESM3": np.array([[0.2], [1.1], [0.3]], dtype=np.float32),
        },
        X_val={
            "ESM2": np.array([[0.1], [0.8], [0.3], [1.0]], dtype=np.float32),
            "ProtT5": np.array([[0.2], [0.7], [0.2], [1.1]], dtype=np.float32),
            "ESM3": np.array([[0.0], [0.9], [0.4], [0.8]], dtype=np.float32),
        },
        X_test={
            "ESM2": np.array([[0.1], [0.7], [0.2], [1.2]], dtype=np.float32),
            "ProtT5": np.array([[0.1], [0.9], [0.3], [1.0]], dtype=np.float32),
            "ESM3": np.array([[0.2], [0.8], [0.1], [1.1]], dtype=np.float32),
        },
        y_train=dataset_bundle.y_train,
        y_val=dataset_bundle.y_val,
        y_test=dataset_bundle.y_test,
    )

    monkeypatch.setattr(Pipeline, "_build_dataset_bundle", staticmethod(lambda conf: dataset_bundle))
    monkeypatch.setattr(
        Pipeline,
        "_build_embedding_bundle_from_dataset",
        lambda self, dataset_bundle, dataset_conf, embeddings_conf: embedding_bundle,
    )
    monkeypatch.setattr(Pipeline, "_load_yaml", staticmethod(lambda path: {}))
    monkeypatch.setattr(Pipeline, "_load_training_config", staticmethod(lambda conf: {}))
    monkeypatch.setattr(
        Pipeline,
        "_ensure_pec_data_layout",
        staticmethod(
            lambda reporting_conf: {
                "root": tmp_path / "pec_data",
                "dataset": tmp_path / "pec_data" / "dataset",
                "sweep": tmp_path / "pec_data" / "sweep",
                "results": tmp_path / "pec_data" / "results",
                "logs": tmp_path / "pec_data" / "logs",
            }
        ),
    )
    return run_dir


def test_benchmark_multiseed_outputs(monkeypatch, tmp_path):
    run_dir = _setup_shared(monkeypatch, tmp_path)
    pipeline = Pipeline(config_path="config/pipeline.yaml")

    pipeline.run_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "benchmark": {
                "seeds": [42, 123],
                "include_majority": False,
                "include_trainable": True,
                "include_validation_weighted": True,
                "include_uniform": True,
            },
        }
    )

    assert (run_dir / "results" / "benchmark_multiseed_summary.csv").exists()
    assert (run_dir / "results" / "benchmark_multiseed_summary.json").exists()
    assert (run_dir / "results" / "benchmark_weights_analysis.json").exists()

    payload = json.loads((run_dir / "results" / "benchmark_multiseed_summary.json").read_text(encoding="utf-8"))
    assert len(payload.get("per_seed_results", [])) >= 4
    assert payload.get("aggregated_metrics")


def test_benchmark_weight_aggregation_helper():
    pipeline = Pipeline(config_path="config/pipeline.yaml")
    analysis = pipeline._build_benchmark_weights_analysis(
        [
            {
                "seed": 42,
                "ablation": "default",
                "weights": [0.2, 0.8],
                "models_used": ["LR::ESM2", "RF::ProtT5"],
                "validation_f1": 0.7,
                "test_f1": 0.68,
                "model_validation_scores": {"LR::ESM2": 0.65, "RF::ProtT5": 0.72},
            },
            {
                "seed": 123,
                "ablation": "default",
                "weights": [0.4, 0.6],
                "models_used": ["LR::ESM2", "RF::ProtT5"],
                "validation_f1": 0.71,
                "test_f1": 0.69,
                "model_validation_scores": {"LR::ESM2": 0.67, "RF::ProtT5": 0.73},
            },
        ]
    )
    assert analysis.get("aggregated")
    first = analysis["aggregated"][0]
    assert "mean_weight_per_model" in first
    assert "mean_entropy" in first


def test_benchmark_ablation_summary_export(monkeypatch, tmp_path):
    run_dir = _setup_shared(monkeypatch, tmp_path)
    pipeline = Pipeline(config_path="config/pipeline.yaml")

    pipeline.run_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "benchmark": {
                "seeds": [42],
                "include_majority": False,
                "include_trainable": False,
                "include_validation_weighted": True,
                "include_uniform": True,
                "ablations": [
                    {"embeddings": ["ESM2"]},
                    {"embeddings": ["ESM2", "ProtT5"]},
                ],
            },
        }
    )

    ablation_csv = run_dir / "results" / "benchmark_ablation_summary.csv"
    assert ablation_csv.exists()
    content = ablation_csv.read_text(encoding="utf-8")
    assert "Ablation" in content


def test_benchmark_overfitting_warning(caplog):
    pipeline = Pipeline(config_path="config/pipeline.yaml")
    caplog.set_level("WARNING")

    pipeline._build_overfitting_report(
        [
            {
                "ablation": "default",
                "variant": "uniform_soft_voting",
                "generalization_gap": 0.01,
                "test_f1": 0.71,
            },
            {
                "ablation": "default",
                "variant": "uniform_soft_voting",
                "generalization_gap": 0.02,
                "test_f1": 0.72,
            },
            {
                "ablation": "default",
                "variant": "trainable_weights_soft_voting",
                "generalization_gap": 0.05,
                "test_f1": 0.80,
            },
            {
                "ablation": "default",
                "variant": "trainable_weights_soft_voting",
                "generalization_gap": 0.08,
                "test_f1": 0.60,
            },
        ]
    )

    assert "Potential overfitting" in caplog.text
