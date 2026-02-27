from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import numpy as np


class BaseRunner(ABC):
    """
    Base class for all experiment runners.

    Responsibilities:
    - Hold references to db, task, classifier
    - Resolve embedding_type names
    - Provide shared helpers for loading data and logging

    Subclasses define:
    - how layers are selected
    - how embeddings are aggregated
    """

    def __init__(self, db, task, classifier):
        self.db = db
        self.task = task
        self.classifier = classifier

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_embedding_type_id(self, embedding_type: str) -> int:
        return self.db.resolve_embedding_type_id(embedding_type)

    def load_single_layer(
        self,
        embedding_type_id: int,
        layer: int,
    ) -> Tuple[list[str], np.ndarray]:
        return self.db.load_embeddings(
            embedding_type_id=embedding_type_id,
            layer_index=layer,
        )

    def get_labels(self, accessions: list[str]) -> np.ndarray:
        return self.task.get_labels(accessions)

    def log(
        self,
        metrics: dict,
        **metadata,
    ):
        self.task.log(metrics, **metadata)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
