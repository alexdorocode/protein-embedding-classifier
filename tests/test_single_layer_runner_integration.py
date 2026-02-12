import numpy as np
from sqlalchemy import text

from runners.single_layer import SingleLayerRunner
from src.core.db import load_db_config, create_engine_from_config, EmbeddingDB


class LoggingTask:
    """Minimal task implementation for integration testing.

    - get_labels: returns dummy labels but records accessions
    - log: records metrics and context
    """

    def __init__(self):
        self.label_calls = []
        self.log_calls = []

    def get_labels(self, accessions):
        self.label_calls.append(list(accessions))
        # Deterministic dummy labels; we only care that wiring works
        return [0] * len(accessions)

    def log(self, metrics, **kwargs):
        self.log_calls.append({"metrics": metrics, "kwargs": kwargs})


class DummyClassifier:
    """Simple classifier stub used for wiring tests.

    fit_eval just returns basic metrics about the input shapes.
    """

    def __init__(self):
        self.calls = []

    def fit_eval(self, X, y):
        self.calls.append((X, y))
        return {
            "n_samples": len(y),
            "dim": int(X.shape[1]) if isinstance(X, np.ndarray) and X.ndim == 2 else None,
        }


def _pick_one_model_and_layer_with_data(engine):
    """Return (embedding_type_name, layer_index) that has at least one row.

    This mirrors the discovery query used in test_embeddings_integration.
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


def test_single_layer_runner_integration_against_biodata_uniprot():
    """End-to-end wiring test for SingleLayerRunner using real BioData_UniProt.

    This uses config/db.yaml to connect, discovers one populated
    (embedding_type, layer) pair, and checks that SingleLayerRunner
    pulls embeddings, calls the classifier, and delegates logging.
    """

    # Connect to BioData_UniProt via the standard config
    cfg = load_db_config("config/db.yaml")
    engine = create_engine_from_config(cfg)

    # Discover a model name + layer that actually has data
    embedding_type_name, layer_index = _pick_one_model_and_layer_with_data(engine)

    # Wire up the DB adapter and runner dependencies
    db = EmbeddingDB(engine)
    task = LoggingTask()
    clf = DummyClassifier()

    runner = SingleLayerRunner(db=db, task=task, classifier=clf)

    runner.run(embedding_type=embedding_type_name, layers=[layer_index])

    # Task should have been asked for labels once, with at least one accession
    assert len(task.label_calls) == 1
    accessions = task.label_calls[0]
    assert len(accessions) > 0

    # Classifier should have been called once with a 2D numpy array
    assert len(clf.calls) == 1
    X, y = clf.calls[0]
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert len(y) == X.shape[0] == len(accessions)

    # Task.log should have been called once with metrics and matching context
    assert len(task.log_calls) == 1
    log_entry = task.log_calls[0]
    assert "metrics" in log_entry and "kwargs" in log_entry
    assert log_entry["metrics"]["n_samples"] == len(accessions)
    assert log_entry["kwargs"]["embedding_type"] == embedding_type_name
    assert log_entry["kwargs"]["layer"] == layer_index
