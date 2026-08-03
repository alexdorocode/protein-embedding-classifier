"""
Embedding access layer.

Responsibilities:
- High-level access to embeddings by (model, layer)
- In-memory caching
- Optional normalization
- Dimensionality validation

This module MUST NOT:
- Do ML
- Know task labels
- Implement classifiers
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np

from src.training.db import (
    load_embeddings,
    resolve_embedding_type_id,
)

# src/core/embeddings.py

from typing import Callable, Iterable
import numpy as np


class EmbeddingStore:
    """
    In-memory embedding cache with DB backend.
    """

    def __init__(
        self,
        engine,
        normalize: bool = False,
    ):
        """
        Parameters
        ----------
        engine : sqlalchemy.Engine
            Database engine
        normalize : bool
            If True, L2-normalize embeddings on load
        """
        self.engine = engine
        self.normalize = normalize

        # cache[(embedding_type, layer)] = (accessions, X)
        self._cache: Dict[Tuple[str, int], Tuple[List[str], np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        embedding_type: str,
        layer: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get embeddings for (embedding_type, layer).

        Returns
        -------
        accessions : list[str]
        X : np.ndarray, shape (N, D)
        """

        key = (embedding_type, layer)

        if key not in self._cache:
            self._cache[key] = self._load_from_db(
                embedding_type=embedding_type,
                layer=layer,
            )

        return self._cache[key]

    def clear(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(
        self,
        embedding_type: str,
        layer: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Load embeddings from DB and apply post-processing.
        """

        embedding_type_id = resolve_embedding_type_id(
            self.engine,
            embedding_type,
        )

        accessions, X = load_embeddings(
            engine=self.engine,
            embedding_type_id=embedding_type_id,
            layer_index=layer,
        )

        self._validate(accessions, X)

        if self.normalize:
            X = self._l2_normalize(X)

        return accessions, X

    @staticmethod
    def _validate(
        accessions: List[str],
        X: np.ndarray,
    ) -> None:
        """
        Validate embedding matrix consistency.
        """

        if len(accessions) != X.shape[0]:
            raise ValueError(
                "Number of accessions does not match number of embeddings "
                f"({len(accessions)} != {X.shape[0]})"
            )

        if X.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape {X.shape}"
            )

        if len(set(accessions)) != len(accessions):
            raise ValueError("Duplicate accessions detected")

    @staticmethod
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        """
        L2-normalize rows of X.
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms


class EmbeddingLayerSweep:
    """
    Run one experiment function across multiple embedding layers.
    """

    def __init__(self, db):
        self.db = db

    def run(
        self,
        embedding_type_id: int,
        layers: Iterable[int],
        experiment_fn: Callable[[int, list[str], np.ndarray], dict],
        batch_size: int = 1000,
    ) -> Dict[int, dict]:
        results: Dict[int, dict] = {}

        for layer_index in layers:
            accessions = []
            embeddings = []

            for accession, embedding in self.db.fetch_embeddings_by_layer(
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                batch_size=batch_size,
            ):
                accessions.append(accession)
                embeddings.append(embedding)

            X = np.asarray(embeddings)
            results[layer_index] = experiment_fn(layer_index, accessions, X)

        return results
