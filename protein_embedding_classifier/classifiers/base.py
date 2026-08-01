# src/classifiers/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseClassifier(ABC):
    """
    Base interface for all classifiers.

    Responsibilities:
    - Fit a model
    - Evaluate it
    - Return metrics in a standard dict

    MUST NOT:
    - Load data
    - Know about layers or embeddings
    - Log to wandb
    """

    @abstractmethod
    def fit_eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        pass

    def reset(self):
        """
        Optional hook for resetting internal state
        (useful for repeated runs / sweeps).
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

