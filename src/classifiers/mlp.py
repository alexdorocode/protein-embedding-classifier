from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from .base import BaseClassifier


class MLPClassifierWrapper(BaseClassifier):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
        activation: str = "relu",
        alpha: float = 1e-4,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self._build_model()

    def _build_model(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def reset(self):
        self._build_model()

    def fit_eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        self.model.fit(X, y)

        y_pred = self.model.predict(X)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }
