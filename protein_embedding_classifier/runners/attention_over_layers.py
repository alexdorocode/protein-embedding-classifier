# src/runners/attention_over_layers.py
import numpy as np
from protein_embedding_classifier.runners.base import BaseRunner


class AttentionOverLayersRunner(BaseRunner):
    def run(self, embedding_type: str, layers: list[int] | None = None):
        """
        layers:
          - None  → usar totes les capes disponibles
          - list  → capes seleccionades
        """

        X_layers = []
        accessions_ref = None

        if layers is None:
            layers = self.store.get_available_layers(embedding_type)

        for layer in layers:
            accessions, X = self.store.get(embedding_type, layer)

            if accessions_ref is None:
                accessions_ref = accessions
            else:
                assert accessions == accessions_ref

            X_layers.append(X)

        X_layers = np.stack(X_layers, axis=0)

        y = self.task.get_labels(accessions_ref)

        # 🔥 aquí el classifier fa attention
        metrics = self.classifier.fit_eval(X_layers, y)

        self.task.log(
            metrics,
            embedding_type=embedding_type,
            layers=layers,
            aggregation=self.classifier.attention_name,
        )

        return metrics
