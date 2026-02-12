# src/runners/selected_layers.py
import numpy as np
from .base import BaseRunner


class SelectedLayersRunner(BaseRunner):
    def __init__(self, store, task, classifier, aggregator):
        super().__init__(store, task, classifier)
        self.store = store
        self.aggregator = aggregator

    def run(self, embedding_type: str, layers: list[int]):
        """
        layers: [2, 14, 24]
        """

        X_layers = []
        accessions_ref = None

        for layer in layers:
            accessions, X = self.store.get(embedding_type, layer)

            if accessions_ref is None:
                accessions_ref = accessions
            else:
                assert accessions == accessions_ref, \
                    "Accessions mismatch between layers"

            X_layers.append(X)

        # (L, N, D)
        X_layers = np.stack(X_layers, axis=0)

        # Aggregate
        X = self.aggregator.aggregate(X_layers)

        # Labels
        y = self.task.get_labels(accessions_ref)

        # Train + eval
        metrics = self.classifier.fit_eval(X, y)

        # Logging
        self.task.log(
            metrics,
            embedding_type=embedding_type,
            layers=layers,
            aggregation=self.aggregator.name,
        )

        return metrics
