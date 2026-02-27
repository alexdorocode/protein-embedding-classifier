import numpy as np

from protein_embedding_classifier.core.db import load_db_config, create_engine_from_config
from protein_embedding_classifier.tasks.enzyme_vs_not.task import EnzymeVsNotTask
from protein_embedding_classifier.tasks.enzyme_vs_not.labels import load_labels


def test_enzyme_vs_not_task_returns_labels_for_known_accessions():
    """EnzymeVsNotTask.get_labels should align with labels.sql for real accessions.

    This is an integration-style test: it hits the real BioData_UniProt
    database defined in config/db.yaml, loads labels via labels.sql, then
    verifies that EnzymeVsNotTask returns the same labels for a subset of
    accessions.
    """

    cfg = load_db_config("config/db.yaml")
    engine = create_engine_from_config(cfg)

    # Load the label mapping directly from SQL
    labels = load_labels(engine)
    assert labels, "No labels returned by labels.sql; check the database contents."

    # Pick a small subset of accessions to test with
    sample_accessions = list(labels.keys())[:10]

    task = EnzymeVsNotTask(engine)
    y = task.get_labels(sample_accessions)

    # Basic type/shape checks
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.shape[0] == len(sample_accessions)
    assert y.dtype == np.int64

    # Values should match the labels mapping exactly
    expected = np.array([labels[acc] for acc in sample_accessions], dtype=np.int64)
    np.testing.assert_array_equal(y, expected)


def test_enzyme_vs_not_task_raises_when_all_labels_missing():
    """get_labels should raise RuntimeError if none of the accessions have labels."""

    # Use a dummy engine; we'll bypass DB access by injecting an empty label dict.
    class DummyEngine:
        def connect(self):  # pragma: no cover - not used
            raise RuntimeError("Should not be called in this test")

    class TaskWithEmptyLabels(EnzymeVsNotTask):
        def _ensure_labels_loaded(self):  # override to avoid DB
            self._labels = {}

    task = TaskWithEmptyLabels(DummyEngine())

    accessions = ["FAKE_ACC_1", "FAKE_ACC_2"]
    try:
        task.get_labels(accessions)
        assert False, "Expected RuntimeError when no labels match any accession"
    except RuntimeError as e:
        assert "No labels matched any accession" in str(e)
