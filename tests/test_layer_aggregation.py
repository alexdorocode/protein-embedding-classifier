import numpy as np
import pytest

from src.layer_aggregation.mean import MeanAggregation
from src.layer_aggregation.attention import AttentionAggregation


def _make_dummy_layers(n_layers: int = 3, n_samples: int = 5, dim: int = 4) -> np.ndarray:
    """Utility to create a simple (L, N, D) tensor with known structure."""
    X_layers = []
    for l in range(n_layers):
        # Each layer is filled with its index so we can reason about means
        X_layers.append(np.full((n_samples, dim), float(l), dtype=np.float32))
    return np.stack(X_layers, axis=0)


def test_mean_aggregation_reduces_over_layers():
    X_layers = _make_dummy_layers(n_layers=3, n_samples=2, dim=2)
    agg = MeanAggregation()

    X = agg.aggregate(X_layers)

    # Output shape should be (N, D)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2)

    # With layers filled with 0,1,2 the mean over layers is 1.0
    assert np.allclose(X, np.ones_like(X))
    assert agg.name == "mean"


def test_attention_aggregation_has_correct_shape_and_validates_input():
    X_layers = _make_dummy_layers(n_layers=4, n_samples=3, dim=6)
    agg = AttentionAggregation(n_layers=4)

    X = agg.aggregate(X_layers)

    # Output shape should be (N, D)
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 6)

    # Currently AttentionAggregation uses a mean over layers, so we can at least
    # assert that values lie between the min and max layer values.
    assert X.min() >= X_layers.min()
    assert X.max() <= X_layers.max()
    assert agg.name == "soft_attention"


def test_attention_aggregation_raises_on_wrong_rank():
    agg = AttentionAggregation()
    bad_input = np.zeros((5, 10), dtype=np.float32)  # missing layer dimension

    with pytest.raises(ValueError):
        agg.aggregate(bad_input)
