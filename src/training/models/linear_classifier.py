from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.training.models.base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self._build_model()

    def _build_model(self):
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
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
