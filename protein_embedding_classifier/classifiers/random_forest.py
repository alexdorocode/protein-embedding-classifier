from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from protein_embedding_classifier.classifiers.base import BaseClassifier


class RandomForestClassifierWrapper(BaseClassifier):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._build_model()

    def _build_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
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
