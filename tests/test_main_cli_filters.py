import sys

from protein_embedding_classifier import main as main_module


def test_main_forwards_runtime_filters(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, config_path):
            captured["config_path"] = config_path

        def run(self, step=None, run_all=False, filters=None):
            captured["step"] = step
            captured["run_all"] = run_all
            captured["filters"] = filters

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
