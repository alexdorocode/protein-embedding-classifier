from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from src.training.embedding_loading import EmbeddingBundle
from src.training.pipeline import Pipeline


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
        self.train_ids = ["T1", "T2"]
        self.val_ids = ["V1", "V2", "V3"]
        self.test_ids = ["S1", "S2", "S3"]
        self.y_train = np.array([0, 1])
        self.y_val = np.array([0, 1, 1])
        self.y_test = np.array([0, 1, 0])


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


def test_pipeline_ensemble_step_executes_from_persisted_artifacts(monkeypatch, tmp_path, caplog):
    run_dir = tmp_path / "pec_data" / "sweep" / "run_20260301"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    lr_model_rel, lr_meta_rel, lr_serializer = _write_model_artifact(run_dir, "LR", "ESM2", bias=0.3)
    rf_model_rel, rf_meta_rel, rf_serializer = _write_model_artifact(run_dir, "RF", "ProtT5", bias=-0.2)

    best_csv = reports_dir / "best_classifier_per_embedding.csv"
    with best_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["embedding_name", "model_type", "validation_f1", "config", "artifact_path", "serializer", "metadata_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "embedding_name": "ESM2",
                "model_type": "LR",
                "validation_f1": 0.81,
                "config": "{}",
                "artifact_path": lr_model_rel,
                "serializer": lr_serializer,
                "metadata_path": lr_meta_rel,
            }
        )
        writer.writerow(
            {
                "embedding_name": "ProtT5",
                "model_type": "RF",
                "validation_f1": 0.76,
                "config": "{}",
                "artifact_path": rf_model_rel,
                "serializer": rf_serializer,
                "metadata_path": rf_meta_rel,
            }
        )

    dataset_bundle = _DatasetBundle()
    embedding_bundle = EmbeddingBundle(
        X_train={
            "ESM2": np.array([[0.0], [1.0]], dtype=np.float32),
            "ProtT5": np.array([[0.2], [0.8]], dtype=np.float32),
        },
        X_val={
            "ESM2": np.array([[0.1], [0.9], [1.2]], dtype=np.float32),
            "ProtT5": np.array([[0.2], [0.7], [1.0]], dtype=np.float32),
        },
        X_test={
            "ESM2": np.array([[0.3], [0.6], [1.1]], dtype=np.float32),
            "ProtT5": np.array([[0.1], [0.5], [0.9]], dtype=np.float32),
        },
        y_train=dataset_bundle.y_train,
        y_val=dataset_bundle.y_val,
        y_test=dataset_bundle.y_test,
    )

    pipeline = Pipeline(config_path="config/pipeline.yaml")
    caplog.set_level("INFO")

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

    pipeline.run_ensemble_step(
        {
            "dataset": {},
            "embeddings": {},
            "ensemble": {
                "enabled": True,
                "mode": "global_soft",
                "selection": {
                    "embeddings": ["ESM2", "ProtT5"],
                    "classifiers": ["LR", "RF", "SVM"],
                },
                "weighting": {
                    "strategy": "uniform",
                    "metric": "f1_macro",
                },
            },
        }
    )

    ensemble_model = run_dir / "models" / "ensemble_model.pkl"
    ensemble_metadata = run_dir / "models" / "ensemble_model.metadata.json"

    assert ensemble_model.exists()
    assert ensemble_metadata.exists()

    metadata = json.loads(ensemble_metadata.read_text(encoding="utf-8"))
    assert metadata["mode"] == "global_soft"
    assert metadata["problem_type"] == "binary"
    assert metadata["num_classes"] == 2
    assert len(metadata["models_used"]) == 2
    assert "Ensemble selection requested=3 available=2" in caplog.text
