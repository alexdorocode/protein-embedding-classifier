import numpy as np

from protein_embedding_classifier.core.db import load_db_config, create_engine_from_config
from protein_embedding_classifier.core.embeddings import EmbeddingStore
from protein_embedding_classifier.tasks.enzyme_vs_not.dataset import build_dataset
from protein_embedding_classifier.classifiers.registry import get_classifier


def test_enzyme_vs_not_logistic_end_to_end():
    """End-to-end smoke test for enzyme_vs_not task.

    This mirrors manual_test.py: it hits the real database, builds
    the dataset for one embedding configuration, trains a simple
    classifier, and verifies that basic invariants hold.
    """

    cfg = load_db_config("config/db.yaml")
    engine = create_engine_from_config(cfg)

    store = EmbeddingStore(engine, normalize=True)

    # Use the same configuration as manual_test.py
    X, y, acc = build_dataset(
        store,
        embedding_type="Prot-T5",
        layer=5,
    )

    # Basic dataset sanity checks
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0] == len(acc)
    assert X.shape[0] > 0

    clf = get_classifier("logistic", C=1.0)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    # Smoke-test accuracy: at least better than random guessing
    acc_train = (y_pred == y).mean()
    assert 0.5 <= acc_train <= 1.0
