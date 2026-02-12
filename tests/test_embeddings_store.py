import numpy as np
import pytest

from src.core.embeddings import EmbeddingStore


def test_get_uses_cache_and_db_once(monkeypatch):
    """EmbeddingStore.get should hit the DB only once per (model, layer)."""

    calls = {"resolve": 0, "load": 0}

    def fake_resolve(engine, embedding_type_name):  # noqa: ARG001
        calls["resolve"] += 1
        assert embedding_type_name == "ESM"
        return 42

    def fake_load(engine, embedding_type_id, layer_index):  # noqa: ARG001
        calls["load"] += 1
        assert embedding_type_id == 42
        assert layer_index == 5
        accessions = ["P1", "P2"]
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        return accessions, X

    monkeypatch.setattr("src.core.embeddings.resolve_embedding_type_id", fake_resolve)
    monkeypatch.setattr("src.core.embeddings.load_embeddings", fake_load)

    store = EmbeddingStore(engine=object(), normalize=False)

    acc1, X1 = store.get("ESM", layer=5)
    acc2, X2 = store.get("ESM", layer=5)

    # Same objects returned from cache
    assert acc1 is acc2
    assert X1 is X2

    # Underlying DB helpers called only once
    assert calls["resolve"] == 1
    assert calls["load"] == 1


def test_get_applies_l2_normalization(monkeypatch):
    """When normalize=True, rows should be L2-normalized."""

    def fake_resolve(engine, embedding_type_name):  # noqa: ARG001
        return 1

    def fake_load(engine, embedding_type_id, layer_index):  # noqa: ARG001
        accessions = ["A", "B"]
        X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        return accessions, X

    monkeypatch.setattr("src.core.embeddings.resolve_embedding_type_id", fake_resolve)
    monkeypatch.setattr("src.core.embeddings.load_embeddings", fake_load)

    store = EmbeddingStore(engine=object(), normalize=True)
    acc, X = store.get("Any", layer=0)

    assert acc == ["A", "B"]
    # First row should have norm 1
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms[0], 1.0)
    # Second row was zero vector; remains zero
    assert np.allclose(X[1], np.array([0.0, 0.0], dtype=np.float32))


def test_validate_raises_on_inconsistent_lengths():
    """_validate should reject mismatched accession/embedding counts."""

    X = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        EmbeddingStore._validate(["A", "B"], X)


def test_validate_raises_on_non_2d():
    """_validate should reject non-2D arrays."""

    X = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError):
        EmbeddingStore._validate(["A"], X)


def test_validate_raises_on_duplicate_accessions():
    """_validate should reject duplicate accessions."""

    X = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        EmbeddingStore._validate(["A", "A"], X)
