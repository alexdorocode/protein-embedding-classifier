from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from protein_embedding_classifier.runners.base import BaseRunner


class SingleLayerRunner(BaseRunner):
    """
    Run a classifier independently on each embedding layer.

    Responsibilities:
    - Iterate over layers
    - Load embeddings (one layer at a time)
    - Optionally apply aggregation
    - Fit + evaluate classifier
    - Delegate logging to task
    """

    def __init__(
        self,
        db,
        task,
        classifier,
        aggregator: Optional[callable] = None,
    ):
        super().__init__(db=db, task=task, classifier=classifier)
        self.aggregator = aggregator

    def run(
        self,
        embedding_type: str,
        layers: Iterable[int],
    ):
        """
        Parameters
        ----------
        embedding_type : str
            Human-readable embedding name (e.g. 'esm2_t33_650M')
        layers : iterable[int]
            Layer indices to evaluate
        """

        embedding_type_id = self.resolve_embedding_type_id(
            embedding_type
        )

        for layer in layers:
            accessions, X = self.load_single_layer(
                embedding_type_id=embedding_type_id,
                layer=layer,
            )

            if self.aggregator is not None:
                X = self.aggregator(X)

            y = self.get_labels(accessions)

            metrics = self.classifier.fit_eval(X, y)

            self.log(
                metrics,
                embedding_type=embedding_type,
                layer=layer,
                runner="single_layer",
            )
