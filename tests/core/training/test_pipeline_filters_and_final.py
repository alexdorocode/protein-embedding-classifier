import csv
from pathlib import Path

import numpy as np

from src.training.embedding_loading import EmbeddingBundle
from src.training.pipeline import Pipeline


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

    from src.training.training.training_service import TrainingService

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


def test_write_full_sweep_results_csv_contains_required_metric_columns(tmp_path):
    output_path = tmp_path / "sweep_results_full.csv"

    trial_rows = [
        {
            "model_type": "LR",
            "embedding_name": "ESM2",
            "validation_metrics": {
                "accuracy": 0.82,
                "precision": 0.81,
                "recall": 0.83,
                "f1": 0.82,
            },
            "test_metrics": {
                "accuracy": 0.79,
                "precision": 0.78,
                "recall": 0.8,
                "f1": 0.79,
                "roc_auc": 0.86,
                "pr_auc": 0.84,
                "tp": 17,
                "tn": 20,
                "fp": 3,
                "fn": 2,
            },
            "seed_used": 314,
        }
    ]
    final_test_rows = [
        {
            "model_type": "SVM",
            "embedding_name": "ESM2",
            "validation_metrics": {
                "accuracy": 0.8,
                "precision": 0.79,
                "recall": 0.81,
                "f1": 0.8,
            },
            "test_metrics": {
                "accuracy": 0.76,
                "precision": 0.75,
                "recall": 0.77,
                "f1": 0.76,
                "roc_auc": 0.83,
                "pr_auc": 0.8,
                "confusion_matrix": [[9, 1], [2, 6]],
            },
            "seed_used": 315,
        }
    ]

    Pipeline._write_full_sweep_results_csv(
        output_path,
        trial_rows,
        final_test_rows=final_test_rows,
        seed_used=999,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "model_type",
            "embedding_name",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
            "TP",
            "TN",
            "FP",
            "FN",
            "seed_used",
        }
        assert required_columns.issubset(set(reader.fieldnames or []))
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["seed_used"] == "314"
    assert rows[0]["TP"] == "17"
    assert rows[0]["TN"] == "20"
    assert rows[0]["FP"] == "3"
    assert rows[0]["FN"] == "2"

    assert rows[1]["seed_used"] == "315"
    assert rows[1]["TP"] == "6"
    assert rows[1]["TN"] == "9"
    assert rows[1]["FP"] == "1"
    assert rows[1]["FN"] == "2"


def test_run_sweep_step_returns_cleanly_when_all_classifiers_skipped(monkeypatch, caplog, tmp_path):
    pipeline = Pipeline(config_path="config/pipeline.yaml")
    caplog.set_level("INFO")

    bundle = EmbeddingBundle(
        X_train={"ESM3c": np.random.randn(6, 4).astype(np.float32)},
        X_val={"ESM3c": np.random.randn(3, 4).astype(np.float32)},
        X_test={"ESM3c": np.random.randn(3, 4).astype(np.float32)},
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0]),
        y_test=np.array([0, 1, 0]),
    )

    class FakeDatasetBundle:
        train_ids = ["A", "B", "C", "D", "E", "F"]
        val_ids = ["G", "H", "I"]
        test_ids = ["J", "K", "L"]
        y_train = np.array([0, 1, 0, 1, 0, 1])
        y_val = np.array([0, 1, 0])
        y_test = np.array([0, 1, 0])

    monkeypatch.setattr(Pipeline, "_build_dataset_bundle", staticmethod(lambda conf: FakeDatasetBundle()))
    monkeypatch.setattr(
        Pipeline,
        "_build_embedding_bundle_from_dataset",
        lambda self, dataset_bundle, dataset_conf, embeddings_conf: bundle,
    )
    monkeypatch.setattr(Pipeline, "_load_yaml", staticmethod(lambda path: {}))
    monkeypatch.setattr(Pipeline, "_load_training_config", staticmethod(lambda conf: {"final_training": {"enabled": False}}))
    monkeypatch.setattr(Pipeline, "_resolve_classifier_sweeps", staticmethod(lambda conf: {"MLP": "dummy.yaml"}))

    from src.training.training.sweep_service import SweepService

    def fake_run(self, **kwargs):
        raise ImportError("MLP model requested but torch is not installed")

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

    assert "No successful sweep results were produced; all selected classifiers were skipped" in caplog.text


def test_final_training_continues_when_model_pickle_fails(monkeypatch, caplog, tmp_path):
    pipeline = Pipeline(config_path="config/pipeline.yaml")
    caplog.set_level("WARNING")

    bundle = EmbeddingBundle(
        X_train={"ESM2": np.random.randn(6, 4).astype(np.float32)},
        X_val={"ESM2": np.random.randn(3, 4).astype(np.float32)},
        X_test={"ESM2": np.random.randn(3, 4).astype(np.float32)},
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0]),
        y_test=np.array([0, 1, 0]),
    )

    class UnpicklableModel:
        def __getstate__(self):
            raise TypeError("cannot pickle")

    def fake_train(self, embedding_bundle, training_config):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        return {
            ("MLP", embedding_name): {
                "model": UnpicklableModel(),
                "metrics": {
                    "validation": {"f1": 0.6},
                    "test": {"f1": 0.55},
                },
            }
        }

    from src.training.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)
    monkeypatch.setattr(Pipeline, "_wandb_init_run", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(Pipeline, "_wandb_log", staticmethod(lambda payload: None))
    monkeypatch.setattr(Pipeline, "_wandb_finish_run", staticmethod(lambda run: None))

    rows = pipeline._run_final_training_for_classifier(
        classifier="MLP",
        trial_results=[
            {
                "embedding_name": "ESM2",
                "config": {"learning_rate": 1e-3},
                "validation_metrics": {"f1": 0.7},
            }
        ],
        embedding_bundle=bundle,
        train_conf={},
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
    assert "Skipping model artifact save" in caplog.text


def test_reporting_config_defaults_applied():
    conf = Pipeline._build_reporting_config({})

    assert conf["output_root"] == "../../pec_data"
    assert conf["run_prefix"] == "sweep"
    assert conf["dataset_name"] == "default_dataset"
    assert conf["prediction_split"] == "test"
    assert conf["thresholds"]["default"] == 0.5


def test_create_timestamped_run_dir_adds_suffix_on_collision(tmp_path):
    base = tmp_path / "sweep"
    base.mkdir(parents=True, exist_ok=True)

    first = Pipeline._create_timestamped_run_dir(base, "run")
    second = Pipeline._create_timestamped_run_dir(base, "run")

    assert first.exists()
    assert second.exists()
    assert first != second
