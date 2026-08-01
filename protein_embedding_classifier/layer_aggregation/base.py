# src/layer_aggregation/base.py
from abc import ABC, abstractmethod
import numpy as np


class LayerAggregator(ABC):
    @abstractmethod
    def aggregate(self, X_layers: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X_layers : np.ndarray
            Shape (n_layers, n_samples, dim)

        Returns
        -------
        X : np.ndarray
            Shape (n_samples, dim)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for logging / tags (e.g. 'mean', 'soft_attention')."""
        pass
