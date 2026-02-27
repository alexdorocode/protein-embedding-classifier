import sys
import types

import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.sweep_service import SweepService


def _bundle():
    return EmbeddingBundle(
        X_train={"ESM3c": np.random.randn(6, 4).astype(np.float32)},
        X_val={"ESM3c": np.random.randn(3, 4).astype(np.float32)},
        X_test={"ESM3c": np.random.randn(3, 4).astype(np.float32)},
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
        return {
            ("LR", "ESM3c"): {
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
    )

    assert result.best_metric == 0.7
    assert wandb_calls["init"] == 1
    assert wandb_calls["log"] == 1
    assert wandb_calls["finish"] == 1


def test_sweep_service_best_config_selection(monkeypatch):
    trial_configs = [{"C": 0.1}, {"C": 0.9}]

    def fake_build_trials(self, sweep_config, num_trials):
        return trial_configs

    monkeypatch.setattr(SweepService, "_build_trial_configs", fake_build_trials)

    def fake_train(self, embedding_bundle, training_config=None):
        score = float(self.wandb_config["C"])
        return {
            ("LR", "ESM3c"): {
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
    )

    assert result.best_config == {"C": 0.9}
    assert result.best_key == ("LR", "ESM3c")
