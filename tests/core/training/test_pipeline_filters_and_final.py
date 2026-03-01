from pathlib import Path

import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.pipeline import Pipeline


def test_resolve_selected_embeddings_with_group_and_name_filter():
    available = ["ESM2", "ProtT5", "GeOKG"]
    training_global_conf = {
        "embedding_groups": {
            "sentence": ["ESM2", "ProtT5"],
            "graph": ["GeOKG"],
        }
    }

    selected = Pipeline._resolve_selected_embeddings(
        available=available,
        filters={"embedding_group": "sentence", "embedding_name": "ProtT5"},
        training_global_conf=training_global_conf,
    )

    assert selected == ["ProtT5"]


def test_resolve_selected_classifiers_with_enabled_and_cli_filter():
    available = ["LR", "RF", "XGB"]
    training_global_conf = {
        "sweep": {
            "enabled_classifiers": ["LR", "XGB"],
        }
    }

    selected = Pipeline._resolve_selected_classifiers(
        available=available,
        filters={"classifier": "XGB"},
        training_global_conf=training_global_conf,
    )

    assert selected == ["XGB"]


def test_resolve_selected_classifiers_rt_alias_maps_to_rf():
    selected = Pipeline._resolve_selected_classifiers(
        available=["LR", "RF", "XGB"],
        filters={"classifier": "RT"},
        training_global_conf={},
    )

    assert selected == ["RF"]


def test_final_training_writes_result_and_model(monkeypatch, tmp_path):
    pipeline = Pipeline(config_path="config/pipeline.yaml")
    bundle = EmbeddingBundle(
        X_train={"ESM2": np.random.randn(6, 4).astype(np.float32)},
        X_val={"ESM2": np.random.randn(3, 4).astype(np.float32)},
        X_test={"ESM2": np.random.randn(3, 4).astype(np.float32)},
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0]),
        y_test=np.array([0, 1, 0]),
    )

    captured = {}

    def fake_train(self, embedding_bundle, training_config):
        captured["training_config"] = training_config
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        return {
            ("LR", embedding_name): {
                "model": {"type": "lr"},
                "metrics": {
                    "validation": {"f1": 0.71},
                    "test": {
                        "accuracy": 0.7,
                        "precision": 0.69,
                        "recall": 0.72,
                        "f1": 0.7,
                    },
                },
            }
        }

    from protein_embedding_classifier.core.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)
    monkeypatch.setattr(Pipeline, "_wandb_init_run", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(Pipeline, "_wandb_log", staticmethod(lambda payload: None))
    monkeypatch.setattr(Pipeline, "_wandb_finish_run", staticmethod(lambda run: None))

    rows = pipeline._run_final_training_for_classifier(
        classifier="LR",
        trial_results=[
            {
                "embedding_name": "ESM2",
                "config": {"C": 1.0, "normalize": "standard"},
                "validation_metrics": {"f1": 0.81},
            }
        ],
        embedding_bundle=bundle,
        train_conf={"feature_processing": {"normalize": "none"}},
        final_training_conf={
            "retrain_on_train_val": True,
            "evaluate_test": True,
            "save_model": True,
            "output_dir": str(tmp_path / "models"),
        },
        wandb_enabled=False,
        wandb_mode="offline",
        wandb_project="pec-test",
        wandb_entity=None,
    )

    assert len(rows) == 1
    assert rows[0]["embedding_name"] == "ESM2"
    assert rows[0]["test_metrics"]["f1"] == 0.7
    assert captured["training_config"]["feature_processing"]["normalize"] == "standard"

    model_path = tmp_path / "models" / "LR_ESM2.pkl"
    assert model_path.exists()

    csv_path = tmp_path / "final_test_results.csv"
    pipeline._write_final_test_results_csv(csv_path, rows)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "test_f1" in content


def test_run_sweep_step_returns_cleanly_when_all_classifiers_skipped(monkeypatch, caplog, tmp_path):
    pipeline = Pipeline(config_path="config/pipeline.yaml")

    bundle = EmbeddingBundle(
        X_train={"ESM3c": np.random.randn(6, 4).astype(np.float32)},
        X_val={"ESM3c": np.random.randn(3, 4).astype(np.float32)},
        X_test={"ESM3c": np.random.randn(3, 4).astype(np.float32)},
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0]),
        y_test=np.array([0, 1, 0]),
    )

    monkeypatch.setattr(Pipeline, "_build_embedding_bundle", lambda self, dataset_conf, embeddings_conf: bundle)
    monkeypatch.setattr(Pipeline, "_load_yaml", staticmethod(lambda path: {}))
    monkeypatch.setattr(Pipeline, "_load_training_config", staticmethod(lambda conf: {"final_training": {"enabled": False}}))
    monkeypatch.setattr(Pipeline, "_resolve_classifier_sweeps", staticmethod(lambda conf: {"MLP": "dummy.yaml"}))

    from protein_embedding_classifier.core.training.sweep_service import SweepService

    def fake_run(self, **kwargs):
        raise ValueError("MLP training is not supported for multilabel targets in this pipeline")

    monkeypatch.setattr(SweepService, "run", fake_run)

    conf = {
        "dataset": {},
        "embeddings": {},
        "train": {},
        "sweep": {
            "model_type": "MLP",
            "artifacts_dir": str(tmp_path),
        },
    }

    pipeline.run_sweep_step(conf)

    assert "No successful sweep results were produced" in caplog.text
