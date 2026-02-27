from __future__ import annotations

from typing import Tuple, List

import numpy as np
from sqlalchemy import text

from protein_embedding_classifier.core.embeddings import EmbeddingStore
from protein_embedding_classifier.tasks.enzyme_vs_not.labels import load_labels
from protein_embedding_classifier.core.db import resolve_embedding_type_id


def build_dataset(
    store: EmbeddingStore,
    embedding_type: str,
    layer: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X, y for enzyme-vs-not task aligned by accession.
    """
    try:
        accessions, X = store.get(embedding_type, layer)
    except RuntimeError as e:
        if "No embeddings found" not in str(e):
            raise

        embedding_type_id = resolve_embedding_type_id(store.engine, embedding_type)
        with store.engine.connect() as conn:
            available_layers = conn.execute(
                text(
                    """
                    SELECT DISTINCT layer_index
                    FROM sequence_embeddings
                    WHERE embedding_type_id = :embedding_type_id
                    ORDER BY layer_index
                    """
                ),
                {"embedding_type_id": embedding_type_id},
            ).fetchall()

        if not available_layers:
            raise

        fallback_layer = available_layers[0].layer_index
        accessions, X = store.get(embedding_type, fallback_layer)

    labels = load_labels(store.engine)

    selected_accessions = []
    selected_embeddings = []
    selected_labels = []

    for idx, accession in enumerate(accessions):
        if accession in labels:
            selected_accessions.append(accession)
            selected_embeddings.append(X[idx])
            selected_labels.append(labels[accession])

    if not selected_accessions:
        raise RuntimeError("No labels matched any accession")

    return (
        np.asarray(selected_embeddings),
        np.asarray(selected_labels, dtype=np.int64),
        selected_accessions,
    )
