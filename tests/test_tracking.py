import numpy as np
import wandb

from src.core.tracking import WandBTracker


def test_wandbtracker_disabled_does_not_call_wandb(monkeypatch):
    calls = {"init": 0, "log": 0}

    def fake_init(*args, **kwargs):  # pragma: no cover - should not be called
        calls["init"] += 1

    def fake_log(*args, **kwargs):  # pragma: no cover - should not be called
        calls["log"] += 1

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "log", fake_log)

    tracker = WandBTracker(project="dummy", enabled=False)
    tracker.log({"accuracy": 0.9}, epoch=1)
    tracker.finish()

    assert calls["init"] == 0
    assert calls["log"] == 0


def test_wandbtracker_logs_and_finishes(monkeypatch):
    init_calls = []
    log_calls = []
    finished = {"value": False}

    class DummyRun:
        def finish(self):
            finished["value"] = True

    def fake_init(project, name=None, config=None, **kwargs):
        init_calls.append({
            "project": project,
            "name": name,
            "config": dict(config or {}),
            "kwargs": kwargs,
        })
        return DummyRun()

    def fake_log(payload):
        # store a shallow copy so later mutations don't affect assertions
        log_calls.append(dict(payload))

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "log", fake_log)

    tracker = WandBTracker(
        project="pec_develop",
        run_name="test-run",
        config={"foo": 1},
        enabled=True,
    )

    tracker.log({"loss": 0.123}, epoch=5)
    tracker.finish()

    # init called once with the expected project and name
    assert len(init_calls) == 1
    assert init_calls[0]["project"] == "pec_develop"
    assert init_calls[0]["name"] == "test-run"
    assert init_calls[0]["config"] == {"foo": 1}

    # log called once with merged metrics + context
    assert len(log_calls) == 1
    payload = log_calls[0]
    assert payload["loss"] == 0.123
    assert payload["epoch"] == 5

    # finish closes the underlying run
    assert finished["value"] is True
