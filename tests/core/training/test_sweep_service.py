import sys
import types

import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.sweep_service import SweepService


def _bundle():
    return EmbeddingBundle(
        X_train={
            "ESM3c": np.random.randn(6, 4).astype(np.float32),
            "Ankh3": np.random.randn(6, 4).astype(np.float32),
        },
        X_val={
            "ESM3c": np.random.randn(3, 4).astype(np.float32),
            "Ankh3": np.random.randn(3, 4).astype(np.float32),
        },
        X_test={
            "ESM3c": np.random.randn(3, 4).astype(np.float32),
            "Ankh3": np.random.randn(3, 4).astype(np.float32),
        },
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0]),
        y_test=np.array([0, 1, 0]),
    )


def test_sweep_service_single_iteration_with_mocked_wandb(monkeypatch):
    wandb_calls = {"init": 0, "log": 0, "finish": 0}

    class FakeRun:
        def finish(self):
            wandb_calls["finish"] += 1

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: (wandb_calls.__setitem__("init", wandb_calls["init"] + 1) or FakeRun()),
        log=lambda payload: wandb_calls.__setitem__("log", wandb_calls["log"] + 1),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def fake_train(self, embedding_bundle, training_config=None):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        return {
            ("LR", embedding_name): {
                "model": object(),
                "val_probs": np.ones((3, 2), dtype=np.float32) * 0.5,
                "metrics": {"f1_score": 0.7},
            }
        }

    from protein_embedding_classifier.core.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)

    service = SweepService(model_type="LR")
    result = service.run(
        embedding_bundle=_bundle(),
        sweep_config={
            "metric": {"name": "f1_score", "goal": "maximize"},
            "parameters": {"C": {"values": [0.1, 1.0]}},
        },
        num_trials=1,
        artifacts_dir="artifacts",
    )

    assert result.best_metric == 0.7
    assert wandb_calls["init"] == 2
    assert wandb_calls["log"] == 2
    assert wandb_calls["finish"] == 2


def test_sweep_service_best_config_selection(monkeypatch):
    trial_configs = [{"C": 0.1}, {"C": 0.9}]

    def fake_build_trials(self, sweep_config, num_trials):
        return trial_configs

    monkeypatch.setattr(SweepService, "_build_trial_configs", fake_build_trials)

    def fake_train(self, embedding_bundle, training_config=None):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        score = float(self.wandb_config["C"])
        return {
            ("LR", embedding_name): {
                "model": object(),
                "val_probs": np.ones((3, 2), dtype=np.float32) * 0.5,
                "metrics": {"f1_score": score},
            }
        }

    from protein_embedding_classifier.core.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(init=lambda **kwargs: types.SimpleNamespace(finish=lambda: None), log=lambda payload: None),
    )

    service = SweepService(model_type="LR")
    result = service.run(
        embedding_bundle=_bundle(),
        sweep_config={"metric": {"name": "f1_score", "goal": "maximize"}, "parameters": {}},
        num_trials=2,
        artifacts_dir="artifacts",
    )

    assert result.best_config == {"C": 0.9}
    assert result.best_key[0] == "LR"


def test_sweep_service_metric_goal_minimize(monkeypatch, tmp_path):
    trial_configs = [{"objective": 0.9}, {"objective": 0.1}]

    def fake_build_trials(self, sweep_config, num_trials):
        return trial_configs

    monkeypatch.setattr(SweepService, "_build_trial_configs", fake_build_trials)

    def fake_train(self, embedding_bundle, training_config=None):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        objective = float(self.wandb_config["objective"])
        return {
            ("LR", embedding_name): {
                "model": object(),
                "val_probs": np.ones((3, 2), dtype=np.float32) * 0.5,
                "metrics": {"objective": objective},
            }
        }

    from protein_embedding_classifier.core.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)
    monkeypatch.setattr(SweepService, "_wandb_init", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(SweepService, "_wandb_log", staticmethod(lambda payload: None))
    monkeypatch.setattr(SweepService, "_wandb_finish", staticmethod(lambda run: None))
    monkeypatch.setattr(SweepService, "_wandb_config_update", staticmethod(lambda payload: None))

    service = SweepService(model_type="LR")
    result = service.run(
        embedding_bundle=_bundle(),
        sweep_config={"metric": {"name": "objective", "goal": "minimize"}, "parameters": {}},
        num_trials=2,
        artifacts_dir=str(tmp_path),
    )

    assert result.best_config == {"objective": 0.1}


def test_sweep_service_run_name_uniqueness():
    timestamp = "20260301020500000000"
    run_a = SweepService._build_run_name("LR", "ESM3c", 1, timestamp)
    run_b = SweepService._build_run_name("LR", "Ankh3", 1, timestamp)
    run_c = SweepService._build_run_name("LR", "ESM3c", 2, timestamp)

    assert run_a != run_b
    assert run_a != run_c
    assert run_b != run_c


def test_sweep_service_collects_trial_results(monkeypatch, tmp_path):
    monkeypatch.setattr(SweepService, "_wandb_init", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(SweepService, "_wandb_log", staticmethod(lambda payload: None))
    monkeypatch.setattr(SweepService, "_wandb_finish", staticmethod(lambda run: None))
    monkeypatch.setattr(SweepService, "_wandb_config_update", staticmethod(lambda payload: None))

    def fake_train(self, embedding_bundle, training_config=None):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        return {
            ("LR", embedding_name): {
                "model": object(),
                "val_probs": np.ones((3, 2), dtype=np.float32) * 0.5,
                "metrics": {"f1_score": 0.4},
            }
        }

    from protein_embedding_classifier.core.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)

    service = SweepService(model_type="LR")
    result = service.run(
        embedding_bundle=_bundle(),
        sweep_config={"metric": {"name": "f1_score", "goal": "maximize"}, "parameters": {}},
        num_trials=2,
        artifacts_dir=str(tmp_path),
    )

    assert len(result.trial_results) == 4
    first_row = result.trial_results[0]
    assert "model_type" in first_row
    assert "embedding_name" in first_row
    assert "trial_index" in first_row
    assert "config" in first_row
    assert "validation_metrics" in first_row


def test_sweep_service_exports_results_csv(tmp_path):
    service = SweepService(model_type="LR")
    rows = [
        {
            "model_type": "LR",
            "embedding_name": "ESM3c",
            "trial_index": 1,
            "config": {"C": 1.0},
            "validation_metrics": {"f1_score": 0.55, "accuracy": 0.66},
            "test_metrics": {"f1_score": 0.50},
            "selection_metric_name": "f1_score",
            "selection_metric_value": 0.55,
        }
    ]
    output_path = tmp_path / "sweep_results_full.csv"
    service._export_trial_results_csv(output_path, rows)

    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "model_type" in contents
    assert "embedding_name" in contents
    assert "val_f1_score" in contents
    assert "test_f1_score" in contents


def test_sweep_service_summary_table_builder():
    rows = [
        {
            "model_type": "LR",
            "embedding_name": "ESM3c",
            "validation_metrics": {"f1_score": 0.64},
        },
        {
            "model_type": "SVM",
            "embedding_name": "ESM3c",
            "validation_metrics": {"f1_score": 0.70},
        },
        {
            "model_type": "LR",
            "embedding_name": "Prost-T5",
            "validation_metrics": {"f1_score": 0.75},
        },
    ]

    table = SweepService.build_summary_table(rows, model_order=["LR", "SVM"])
    assert "Embedding | LR | SVM" in table
    assert "ESM3c" in table
    assert "0.6400" in table
    assert "0.7000" in table


def test_sweep_service_compute_metrics_handles_legacy_multilabel():
    y_true = np.array([
        ["GO:1", "GO:2"],
        ["GO:2"],
        ["GO:3"],
    ], dtype=object)
    val_probs = np.array(
        [
            [0.9, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.9],
        ],
        dtype=np.float32,
    )

    metrics = SweepService._compute_metrics(y_true=y_true, val_probs=val_probs)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert np.isfinite(metrics["f1_score"])
