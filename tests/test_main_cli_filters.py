import sys

from protein_embedding_classifier import main as main_module


def test_main_forwards_runtime_filters(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, config_path):
            captured["config_path"] = config_path

        def run(self, step=None, run_all=False, filters=None, runtime_context=None):
            captured["step"] = step
            captured["run_all"] = run_all
            captured["filters"] = filters
            captured["runtime_context"] = runtime_context

    monkeypatch.setattr(main_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--config",
            "config/pipeline.yaml",
            "--sweep",
            "--embedding_name",
            "ESM2",
            "--classifier",
            "LR",
            "--embedding_group",
            "sentence",
        ],
    )

    main_module.main()

    assert captured["config_path"] == "config/pipeline.yaml"
    assert captured["step"] == "sweep"
    assert captured["filters"] == {
        "embedding_name": "ESM2",
        "classifier": "LR",
        "embedding_group": "sentence",
    }
    assert "argv" in captured["runtime_context"]


def test_main_keeps_explicit_ensemble_step_with_evaluate_last_sweep(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, config_path):
            captured["config_path"] = config_path

        def run(self, step=None, run_all=False, filters=None, runtime_context=None):
            captured["step"] = step
            captured["run_all"] = run_all
            captured["filters"] = filters
            captured["runtime_context"] = runtime_context

    monkeypatch.setattr(main_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--step",
            "ensemble",
            "--evaluate-last-sweep",
        ],
    )

    main_module.main()

    assert captured["step"] == "ensemble"
    assert captured["runtime_context"]["evaluate_last_sweep"] is True
