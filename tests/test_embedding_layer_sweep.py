import numpy as np
import pytest

from src.training.embeddings import EmbeddingLayerSweep


class FakeDB:
    """Simple in-memory stub for fetch_embeddings_by_layer."""

    def __init__(self):
        self.calls = []

    def fetch_embeddings_by_layer(self, embedding_type_id, layer_index, batch_size):  # noqa: ARG002
        """Yield a fixed number of (accession, embedding) pairs per layer.

        We also record the call arguments so the test can assert on them.
        """

        self.calls.append((embedding_type_id, layer_index, batch_size))

        # Three dummy proteins per layer, each with a 4-dim embedding
        for i in range(3):
            acc = f"ACC{layer_index}_{i}"
            emb = np.full((4,), layer_index + i, dtype=np.float32)
            yield acc, emb


def test_embedding_layer_sweep_runs_experiment_per_layer():
    """EmbeddingLayerSweep should iterate layers and call experiment_fn once per layer."""

    fake_db = FakeDB()
    sweep = EmbeddingLayerSweep(db=fake_db)

    layers = [0, 1, 3]

    def experiment_fn(layer_index, accessions, embeddings):
        # Basic sanity checks on what EmbeddingLayerSweep passes in
        assert layer_index in layers
        assert len(accessions) == 3
        assert embeddings.shape == (3, 4)

        # Return something layer-specific so we can assert on results
        return {
            "layer": layer_index,
            "n": len(accessions),
            "mean": float(embeddings.mean()),
        }

    results = sweep.run(
        embedding_type_id=42,
        layers=layers,
        experiment_fn=experiment_fn,
        batch_size=128,
    )

    # One DB call per layer with the correct arguments
    assert fake_db.calls == [(42, l, 128) for l in layers]

    # Results should have one entry per layer
    assert set(results.keys()) == set(layers)

    # Our FakeDB produces values such that the overall mean per layer is layer_index + 1
    for layer in layers:
        layer_result = results[layer]
        assert layer_result["layer"] == layer
        assert layer_result["n"] == 3
        assert layer_result["mean"] == pytest.approx(layer + 1.0)
