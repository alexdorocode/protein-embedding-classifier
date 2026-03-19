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


def _fake_train_payload(embedding_name: str, score: float) -> dict:
    return {
        ("LR", embedding_name): {
            "model": object(),
            "val_probs": np.ones((3, 2), dtype=np.float32) * 0.5,
            "metrics": {
                "validation": {
                    "accuracy": 0.66,
                    "precision": 0.67,
                    "recall": 0.68,
                    "f1": score,
                    "roc_auc": 0.71,
                    "pr_auc": 0.69,
                },
                "test": None,
            },
        }
    }


def test_sweep_service_single_iteration_with_mocked_wandb(monkeypatch):
    wandb_calls = {"init": 0, "log": 0, "finish": 0}

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def finish(self):
            wandb_calls["finish"] += 1

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: (wandb_calls.__setitem__("init", wandb_calls["init"] + 1) or FakeRun()),
        log=lambda payload: wandb_calls.__setitem__("log", wandb_calls["log"] + 1),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def fake_train(self, embedding_bundle, training_config=None):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        return _fake_train_payload(embedding_name, score=0.7)

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
                "metrics": {
                    "validation": {"f1": objective},
                    "test": None,
                },
            }
        }

    from protein_embedding_classifier.core.training.training_service import TrainingService

    monkeypatch.setattr(TrainingService, "train", fake_train)
    monkeypatch.setattr(SweepService, "_wandb_init", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(SweepService, "_wandb_log", staticmethod(lambda payload, enabled: None))
    monkeypatch.setattr(SweepService, "_wandb_finish", staticmethod(lambda run: None))
    monkeypatch.setattr(SweepService, "_wandb_config_update", staticmethod(lambda payload, enabled: None))

    service = SweepService(model_type="LR")
    result = service.run(
        embedding_bundle=_bundle(),
        sweep_config={"metric": {"name": "f1", "goal": "minimize"}, "parameters": {}},
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
    monkeypatch.setattr(SweepService, "_wandb_log", staticmethod(lambda payload, enabled: None))
    monkeypatch.setattr(SweepService, "_wandb_finish", staticmethod(lambda run: None))
    monkeypatch.setattr(SweepService, "_wandb_config_update", staticmethod(lambda payload, enabled: None))

    def fake_train(self, embedding_bundle, training_config=None):
        embedding_name = next(iter(embedding_bundle.X_train.keys()))
        return _fake_train_payload(embedding_name, score=0.4)

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
            "validation_metrics": {"f1": 0.55, "accuracy": 0.66, "precision": 0.7, "recall": 0.6},
            "test_metrics": None,
        }
    ]
    output_path = tmp_path / "sweep_results_full.csv"
    service._export_trial_results_csv(output_path, rows)

    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "model_type" in contents
    assert "embedding_name" in contents
    assert "validation_f1" in contents
    assert "test_f1" in contents
    assert "test_roc_auc" in contents
    assert "test_pr_auc" in contents
    assert "TP" in contents
    assert "TN" in contents
    assert "FP" in contents
    assert "FN" in contents
    assert "seed_used" in contents


def test_sweep_logging_has_val_prefix_and_no_nan_test_metrics():
    validation = {"accuracy": 0.7, "f1": 0.65}
    clean_validation = SweepService._clean_metrics(validation)
    log_payload = {f"val_{key}": value for key, value in clean_validation.items()}

    assert "val_accuracy" in log_payload
    assert "val_f1" in log_payload
    assert "accuracy" not in log_payload
    assert "f1" not in log_payload

    clean_test = SweepService._clean_metrics({"accuracy": np.nan, "f1": np.nan})
    assert clean_test == {}


def test_binary_metrics_exist_in_validation_payload():
    metrics = {
        "accuracy": 0.72,
        "precision": 0.71,
        "recall": 0.73,
        "f1": 0.72,
        "roc_auc": 0.8,
        "pr_auc": 0.77,
    }
    cleaned = SweepService._clean_metrics(metrics)

    assert "roc_auc" in cleaned
    assert "pr_auc" in cleaned


def test_multilabel_metrics_do_not_require_roc_auc():
    metrics = {
        "micro_f1": 0.62,
        "macro_f1": 0.58,
        "f1": 0.58,
    }
    cleaned = SweepService._clean_metrics(metrics)

    assert "micro_f1" in cleaned
    assert "macro_f1" in cleaned
    assert "roc_auc" not in cleaned
