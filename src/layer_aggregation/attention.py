from __future__ import annotations

from typing import Optional

import numpy as np

from .base import LayerAggregator


class AttentionAggregation(LayerAggregator):
    """Simple placeholder attention-style aggregation over layers.

    For now this implements a uniform weighting over layers so that
    the interface is usable and testable. The contract is:

    - Input shape:  (L, N, D)
    - Output shape: (N, D)

    A more sophisticated attention mechanism can replace this later
    without affecting callers.
    """

    def __init__(self, n_layers: Optional[int] = None):
        self.n_layers = n_layers

    @property
    def name(self) -> str:
        return "soft_attention"

    def aggregate(self, X_layers: np.ndarray) -> np.ndarray:
        """Aggregate over the layer axis.

        Parameters
        ----------
        X_layers : np.ndarray
            Array of shape (L, N, D) where L is the number of layers.

        Returns
        -------
        np.ndarray
            Aggregated array of shape (N, D).
        """

        if X_layers.ndim != 3:
            raise ValueError(
                f"Expected X_layers with 3 dims (L, N, D), got shape {X_layers.shape}"
            )

        # For now, use a simple uniform weighting over layers. This keeps the
        # API stable while remaining a no-parameter baseline.
        return X_layers.mean(axis=0)
