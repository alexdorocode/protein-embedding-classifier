import numpy as np
import pytest

from protein_embedding_classifier.core.embedding_loading import (
    EmbeddingBundle,
    LayerAggregationStrategy,
)
from protein_embedding_classifier.data.dataset_builder import DatasetBundle


def _sample_layered_embeddings():
    return {
        "ESM3c": {
            "P1": {
                0: np.array([1.0, 2.0], dtype=np.float32),
                1: np.array([3.0, 4.0], dtype=np.float32),
            },
            "P2": {
                0: np.array([5.0, 6.0], dtype=np.float32),
                1: np.array([7.0, 8.0], dtype=np.float32),
            },
        }
    }


def test_layer_aggregation_mean_max_concat_modes():
    layered = _sample_layered_embeddings()

    mean_result = LayerAggregationStrategy("mean").aggregate(layered)
    assert np.allclose(mean_result["ESM3c"]["P1"], np.array([2.0, 3.0], dtype=np.float32))

    max_result = LayerAggregationStrategy("max").aggregate(layered)
    assert np.allclose(max_result["ESM3c"]["P1"], np.array([3.0, 4.0], dtype=np.float32))

    mean_max_result = LayerAggregationStrategy("mean_max").aggregate(layered)
    assert np.allclose(mean_max_result["ESM3c"]["P1"], np.array([2.0, 3.0, 3.0, 4.0], dtype=np.float32))

    concat_result = LayerAggregationStrategy("concat").aggregate(layered)
    assert np.allclose(concat_result["ESM3c"]["P1"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_layer_aggregation_none_splits_multilayer_model_views():
    layered = _sample_layered_embeddings()
    result = LayerAggregationStrategy("none").aggregate(layered)

    assert "ESM3c__layer_0" in result
    assert "ESM3c__layer_1" in result
    assert np.allclose(result["ESM3c__layer_0"]["P1"], np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(result["ESM3c__layer_1"]["P1"], np.array([3.0, 4.0], dtype=np.float32))


def test_embedding_bundle_builds_ordered_matrices():
    dataset_bundle = DatasetBundle(
        train_ids=["P2", "P1"],
        val_ids=["P3"],
        test_ids=["P4"],
        y_train=np.array(["a", "b"], dtype=object),
        y_val=np.array(["c"], dtype=object),
        y_test=np.array(["d"], dtype=object),
    )

    raw_embeddings = {
        "ESM3c": {
            "P1": np.array([1.0, 1.0], dtype=np.float32),
            "P2": np.array([2.0, 2.0], dtype=np.float32),
            "P3": np.array([3.0, 3.0], dtype=np.float32),
            "P4": np.array([4.0, 4.0], dtype=np.float32),
        }
    }

    bundle = EmbeddingBundle.from_dataset(dataset_bundle, raw_embeddings)

    assert bundle.X_train["ESM3c"].shape == (2, 2)
    assert np.allclose(bundle.X_train["ESM3c"], np.array([[2.0, 2.0], [1.0, 1.0]], dtype=np.float32))
    assert np.allclose(bundle.X_val["ESM3c"], np.array([[3.0, 3.0]], dtype=np.float32))
    assert np.allclose(bundle.X_test["ESM3c"], np.array([[4.0, 4.0]], dtype=np.float32))


def test_embedding_bundle_raises_on_missing_embeddings():
    dataset_bundle = DatasetBundle(
        train_ids=["P1"],
        val_ids=["P2"],
        test_ids=["P3"],
        y_train=np.array(["a"], dtype=object),
        y_val=np.array(["b"], dtype=object),
        y_test=np.array(["c"], dtype=object),
    )

    raw_embeddings = {
        "GeOKG": {
            "P1": np.array([1.0], dtype=np.float32),
            "P2": np.array([2.0], dtype=np.float32),
        }
    }

    with pytest.raises(ValueError, match="Missing embeddings"):
        EmbeddingBundle.from_dataset(dataset_bundle, raw_embeddings)