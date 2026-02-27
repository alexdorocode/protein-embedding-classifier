import numpy as np
from sqlalchemy import text

from protein_embedding_classifier.runners.single_layer import SingleLayerRunner
from protein_embedding_classifier.core.db import load_db_config, create_engine_from_config, EmbeddingDB
from protein_embedding_classifier.tasks.enzyme_vs_not.task import EnzymeVsNotTask
from protein_embedding_classifier.core.tracking import WandBTracker
import wandb


def _pick_one_model_and_layer_with_data(engine):
    """Return (embedding_type_name, layer_index) that has at least one row.

    Same discovery query pattern as in test_single_layer_runner_integration.
    """

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT t.name, se.layer_index, COUNT(*) AS n
                FROM sequence_embeddings se
                JOIN sequence_embedding_type t
                  ON se.embedding_type_id = t.id
                GROUP BY t.name, se.layer_index
                HAVING COUNT(*) > 0
                ORDER BY t.name, se.layer_index
                """
            )
        ).fetchall()

    assert rows, "No embeddings found in database for any model/layer."

    name, layer_index, _ = rows[0]
    return str(name), int(layer_index)


class DummyClassifier:
    """Simple classifier that returns basic metrics about X and y."""

    def __init__(self):
        self.calls = []

    def fit_eval(self, X, y):
        self.calls.append((X, y))
        return {
            "n_samples": len(y),
            "dim": int(X.shape[1]) if isinstance(X, np.ndarray) and X.ndim == 2 else None,
        }


class TaskWithTracker(EnzymeVsNotTask):
    """EnzymeVsNotTask that also logs to a Tracker in its log method."""

    def __init__(self, engine, tracker):
        super().__init__(engine)
        self.tracker = tracker
        self.log_calls = []

    def log(self, metrics: dict, **context):
        # Record locally for assertions
        self.log_calls.append({"metrics": metrics, "context": context})
        # Forward to the tracker (which will go to wandb in real runs)
        self.tracker.log(metrics, **context)


def test_single_layer_runner_with_wandbtracker(monkeypatch):
    """End-to-end test: SingleLayerRunner + EnzymeVsNotTask + WandBTracker.

    This test hits the real BioData_UniProt DB (via config/db.yaml) to load
    embeddings and labels, but monkeypatches wandb so no network calls are
    made. It verifies that a wandb run is created and that metrics/context
    from SingleLayerRunner are logged once.
    """

    # Intercept wandb.init and wandb.log so we don't touch the real service.
    init_calls = []
    log_calls = []

    class DummyRun:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    def fake_init(project, name=None, config=None, **kwargs):
        init_calls.append({
            "project": project,
            "name": name,
            "config": dict(config or {}),
            "kwargs": kwargs,
        })
        return DummyRun()

    def fake_log(payload):
        log_calls.append(dict(payload))

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "log", fake_log)

    # Connect to BioData_UniProt and discover a populated (model, layer)
    cfg = load_db_config("config/db.yaml")
    engine = create_engine_from_config(cfg)

    embedding_type_name, layer_index = _pick_one_model_and_layer_with_data(engine)

    db = EmbeddingDB(engine)
    tracker = WandBTracker(
        project="pec_develop",  # your wandb project name
        run_name="pytest-single-layer",
        config={"embedding_type": embedding_type_name, "layer": layer_index},
        enabled=True,
    )

    task = TaskWithTracker(engine, tracker)
    clf = DummyClassifier()

    runner = SingleLayerRunner(db=db, task=task, classifier=clf)
    runner.run(embedding_type=embedding_type_name, layers=[layer_index])

    # Basic checks: classifier was run on real embeddings and labels
    assert len(clf.calls) == 1
    X, y = clf.calls[0]
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert y.shape[0] == X.shape[0]

    # Task.log called once with matching context
    assert len(task.log_calls) == 1
    log_entry = task.log_calls[0]
    assert log_entry["context"]["embedding_type"] == embedding_type_name
    assert log_entry["context"]["layer"] == layer_index

    # wandb.init called once with the expected project and run name
    assert len(init_calls) == 1
    assert init_calls[0]["project"] == "pec_develop"
    assert init_calls[0]["name"] == "pytest-single-layer"

    # wandb.log called at least once with metrics + context fields
    assert len(log_calls) >= 1
    combined = log_calls[0]
    assert combined["embedding_type"] == embedding_type_name
    assert combined["layer"] == layer_index
    assert "n_samples" in combined
    assert combined["n_samples"] == y.shape[0]
