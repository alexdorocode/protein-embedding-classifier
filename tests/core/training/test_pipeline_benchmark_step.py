from __future__ import annotations

import csv
import hashlib
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


class _DatasetBundle:
    def __init__(self):
        self.train_ids = ["T1", "T2", "T3"]
        self.val_ids = ["V1", "V2", "V3", "V4"]
        self.test_ids = ["S1", "S2", "S3", "S4"]
        self.y_train = np.array([0, 1, 0])
        self.y_val = np.array([0, 1, 0, 1])
        self.y_test = np.array([0, 1, 0, 1])


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


def test_pipeline_benchmark_step_generates_summary_reports(monkeypatch, tmp_path):
    run_dir = tmp_path / "pec_data" / "sweep" / "run_20260301"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows = []
    for classifier, embedding, bias, val_f1 in [
        ("LR", "ESM2", 0.25, 0.81),
        ("RF", "ProtT5", -0.10, 0.78),
        ("XGB", "ESM3", 0.05, 0.79),
    ]:
        model_rel, meta_rel, serializer = _write_model_artifact(run_dir, classifier, embedding, bias)
        artifact_rows.append(
            {
                "embedding_name": embedding,
                "model_type": classifier,
                "validation_f1": val_f1,
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
        writer.writerows(artifact_rows)

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

    pipeline = Pipeline(config_path="config/pipeline.yaml")

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

    pipeline.run_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "benchmark": {
                "metric": "f1_macro",
                "include_majority": True,
                "include_trainable": True,
                "include_validation_weighted": True,
                "include_uniform": True,
            },
        }
    )

    benchmark_csv = run_dir / "results" / "benchmark_summary.csv"
    benchmark_json = run_dir / "results" / "benchmark_summary.json"
    assert benchmark_csv.exists()
    assert benchmark_json.exists()

    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    assert "best_single" in payload
    assert "ensembles" in payload
    assert "ranking_by_test_metric" in payload
    assert len(payload["ensembles"]) >= 4

    comparison_rows = payload.get("comparison_table", [])
    assert comparison_rows
    assert "Delta vs Best Single (Test)" in comparison_rows[0]


def test_pipeline_benchmark_step_handles_multilabel_object_targets(monkeypatch, tmp_path):
    run_dir = tmp_path / "pec_data" / "sweep" / "run_20260301"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows = []
    for classifier, embedding, bias in [
        ("LR", "ESM2", 0.10),
        ("RF", "ProtT5", -0.05),
        ("XGB", "ESM3", 0.20),
    ]:
        model_rel, meta_rel, serializer = _write_model_artifact(run_dir, classifier, embedding, bias)
        metadata_path = run_dir / meta_rel
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["problem_type"] = "multilabel"
        metadata["classes"] = [False, True]
        metadata["num_classes"] = 2
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        artifact_rows.append(
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
        writer.writerows(artifact_rows)

    class _MultilabelDatasetBundle:
        def __init__(self):
            self.train_ids = ["T1", "T2", "T3"]
            self.val_ids = ["V1", "V2", "V3", "V4"]
            self.test_ids = ["S1", "S2", "S3", "S4"]
            self.y_train = np.array([[True], False, [False], True], dtype=object)
            self.y_val = np.array([[True], False, [False], True], dtype=object)
            self.y_test = np.array([[True], False, [False], True], dtype=object)

    dataset_bundle = _MultilabelDatasetBundle()
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

    pipeline = Pipeline(config_path="config/pipeline.yaml")

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

    pipeline.run_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "benchmark": {
                "include_majority": False,
                "include_trainable": True,
                "include_validation_weighted": True,
                "include_uniform": True,
            },
        }
    )

    benchmark_json = run_dir / "results" / "benchmark_summary.json"
    assert benchmark_json.exists()
    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    assert payload.get("best_single") is not None
    assert len(payload.get("ensembles", [])) >= 3


def test_pipeline_benchmark_includes_zero_shot_and_preserves_model_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "pec_data" / "sweep" / "run_20260301"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows = []
    for classifier, embedding, bias in [
        ("LR", "ESM2", 0.20),
        ("RF", "ProtT5", -0.10),
    ]:
        model_rel, meta_rel, serializer = _write_model_artifact(run_dir, classifier, embedding, bias)
        artifact_rows.append(
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
        writer.writerows(artifact_rows)

    class _DatasetBundleWithZero:
        def __init__(self):
            self.train_ids = ["T1", "T2", "T3"]
            self.val_ids = ["V1", "V2", "V3", "V4"]
            self.test_ids = ["S1", "S2", "S3", "S4"]
            self.zero_shot_ids = ["Z1", "Z2"]
            self.y_train = np.array([0, 1, 0])
            self.y_val = np.array([0, 1, 0, 1])
            self.y_test = np.array([0, 1, 0, 1])
            self.y_zero_shot = np.array([0, 1])

    dataset_bundle = _DatasetBundleWithZero()
    embedding_bundle = EmbeddingBundle(
        X_train={
            "ESM2": np.array([[0.1], [0.9], [0.2]], dtype=np.float32),
            "ProtT5": np.array([[0.0], [0.8], [0.4]], dtype=np.float32),
        },
        X_val={
            "ESM2": np.array([[0.1], [0.8], [0.3], [1.0]], dtype=np.float32),
            "ProtT5": np.array([[0.2], [0.7], [0.2], [1.1]], dtype=np.float32),
        },
        X_test={
            "ESM2": np.array([[0.1], [0.7], [0.2], [1.2]], dtype=np.float32),
            "ProtT5": np.array([[0.1], [0.9], [0.3], [1.0]], dtype=np.float32),
        },
        X_zero_shot={
            "ESM2": np.array([[0.05], [0.95]], dtype=np.float32),
            "ProtT5": np.array([[0.15], [0.85]], dtype=np.float32),
        },
        y_train=dataset_bundle.y_train,
        y_val=dataset_bundle.y_val,
        y_test=dataset_bundle.y_test,
        y_zero_shot=dataset_bundle.y_zero_shot,
    )

    tracked_model = run_dir / "models" / "LR_ESM2.pkl"
    before_hash = hashlib.sha256(tracked_model.read_bytes()).hexdigest()

    pipeline = Pipeline(config_path="config/pipeline.yaml")

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

    pipeline.run_benchmark_step(
        {
            "dataset": {},
            "embeddings": {},
            "benchmark": {
                "include_majority": False,
                "include_trainable": False,
                "include_validation_weighted": True,
                "include_uniform": True,
            },
        }
    )

    benchmark_csv = run_dir / "results" / "benchmark_summary.csv"
    with benchmark_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows
    assert "Zero-Shot F1" in rows[0]

    after_hash = hashlib.sha256(tracked_model.read_bytes()).hexdigest()
    assert before_hash == after_hash
