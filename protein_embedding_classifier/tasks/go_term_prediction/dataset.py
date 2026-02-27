from __future__ import annotations

from typing import Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text

from protein_embedding_classifier.core.embeddings import EmbeddingStore

LABELS_SQL_PATH = Path(__file__).with_name("labels.sql")


def load_labels(engine) -> Tuple[List[str], List[str]]:
    """
    Load labels from SQL.

    Returns
    -------
    accessions : list[str]
    go_terms : list[str]
    """
    with open(LABELS_SQL_PATH, "r") as f:
        sql = f.read()

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    accessions = []
    go_terms = []
    for r in rows:
        accessions.append(r.accession)
        go_terms.append(r.go_term)

    return accessions, go_terms


def build_dataset(
    store: EmbeddingStore,
    embedding_type: str,
    layer: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build X, Y for multi-label GO term prediction.

    Returns
    -------
    X : np.ndarray (N x D)
    Y : np.ndarray (N x M)  -> binary multi-label
    accessions : List[str]  -> N proteins
    go_terms : List[str]    -> M GO terms
    """

    accessions_emb, X = store.get(embedding_type, layer)

    accessions_labels, go_terms_list = load_labels(store.engine)

    # Convert to multi-label matrix (only for proteins that actually
    # have embeddings for this configuration, to keep the pivot small).
    df_labels = pd.DataFrame({"accession": accessions_labels, "go_term": go_terms_list})
    df_labels["value"] = 1

    # Filter early to avoid building an enormous dense pivot table
    df_labels = df_labels[df_labels["accession"].isin(accessions_emb)]

    Y_df = df_labels.pivot_table(
        index="accession",
        columns="go_term",
        values="value",
        fill_value=0,
    )

    # Only keep proteins with embeddings
    Y_df = Y_df.loc[Y_df.index.intersection(accessions_emb)]

    # Align X with Y
    X_out = []
    accessions_out = []
    for i, acc in enumerate(accessions_emb):
        if acc in Y_df.index:
            X_out.append(X[i])
            accessions_out.append(acc)

    X_out = np.asarray(X_out)
    Y_out = Y_df.loc[accessions_out].to_numpy(dtype=np.int64)

    go_terms = list(Y_df.columns)

    return X_out, Y_out, accessions_out, go_terms
