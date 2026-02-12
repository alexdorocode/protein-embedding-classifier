# src/layer_aggregation/mean.py
import numpy as np
from .base import LayerAggregator


class MeanAggregation(LayerAggregator):
    @property
    def name(self) -> str:
        return "mean"

    def aggregate(self, X_layers: np.ndarray) -> np.ndarray:
        return X_layers.mean(axis=0)
