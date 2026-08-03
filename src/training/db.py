"""
Database utilities for plm-embedding-classifier.

Responsibilities of:
- Create PostgreSQL connection from YAML config
- Resolve embedding_type names to IDs
- Load embeddings for a given (embedding_type, layer)
- Return accessions + numpy array

This module MUST NOT:
- Do ML
- Do aggregation
- Do normalization


This file has the following classes and functions:
- load_db_config(path: str | Path) -> dict
- create_engine_from_config(cfg: dict) -> Engine
- resolve_embedding_type_id(engine: Engine, embedding_type_name: str, table: str = "sequence_embedding_type") -> int
- load_embeddings(engine: Engine, embedding_type_id: int, layer_index: int) -> Tuple[List[str], np.ndarray]
- BioDataDB: Interface for fetching embeddings in batches
- EmbeddingDB: Thin adapter around DB utilities

"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def load_db_config(path: str | Path) -> dict:
    """Load db.yaml configuration."""
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg["db"]


def create_engine_from_config(cfg: dict) -> Engine:
    """
    Create SQLAlchemy engine from db config.
    """
    user = cfg["user"]
    password = cfg["password"]
    host = cfg["host"]
    port = cfg["port"]
    name = cfg["name"]

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url)


# ---------------------------------------------------------------------
# Metadata resolution
# ---------------------------------------------------------------------

def resolve_embedding_type_id(
    engine: Engine,
    embedding_type_name: str,
    table: str = "sequence_embedding_type",
) -> int:
    """
    Resolve embedding_type name to its internal ID.
    """
    sql = text(f"""
        SELECT id
        FROM {table}
        WHERE name = :name
    """)

    with engine.connect() as conn:
        row = conn.execute(sql, {"name": embedding_type_name}).fetchone()

    if row is None:
        raise ValueError(f"Unknown embedding_type: {embedding_type_name}")

    return row.id


# ---------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------

EMBEDDING_SQL = """
SELECT
    a.code       AS accession,
    se.embedding AS embedding
FROM sequence_embeddings se
JOIN sequence s
    ON se.sequence_id = s.id
JOIN protein p
    ON p.sequence_id = s.id
JOIN accession a
    ON a.protein_id = p.id
WHERE se.embedding_type_id = :embedding_type_id
  AND se.layer_index      = :layer_index
    AND a."primary" = TRUE
"""


def coerce_embedding_vector(embedding_value: Any) -> np.ndarray:
    """Convert DB/CSV embedding payload to a float32 numpy vector."""
    if isinstance(embedding_value, (list, tuple)):
        return np.asarray(embedding_value, dtype=np.float32)

    if isinstance(embedding_value, (bytes, bytearray, memoryview)):
        return np.frombuffer(embedding_value, dtype=np.float32).copy()

    if isinstance(embedding_value, str):
        cleaned = embedding_value.strip().lstrip("[").rstrip("]")
        if cleaned:
            values = [float(token) for token in cleaned.split(",")]
            return np.asarray(values, dtype=np.float32)
        return np.empty((0,), dtype=np.float32)

    return np.asarray(embedding_value, dtype=np.float32)


def load_embeddings(
    engine: Engine,
    embedding_type_id: int,
    layer_index: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings for a given embedding_type_id and layer.

    Returns
    -------
    accessions : list[str]
        UniProt accession codes
    X : np.ndarray
        Shape (N, D) embedding matrix
    """

    with engine.connect() as conn:
        rows = conn.execute(
            text(EMBEDDING_SQL),
            {
                "embedding_type_id": embedding_type_id,
                "layer_index": layer_index,
            },
        ).fetchall()

    if len(rows) == 0:
        raise RuntimeError(
            f"No embeddings found for embedding_type_id={embedding_type_id}, "
            f"layer_index={layer_index}"
        )

    accessions: List[str] = []
    vectors: List[np.ndarray] = []

    for r in rows:
        accessions.append(r.accession)
        vec = coerce_embedding_vector(r.embedding)
        vectors.append(vec)

    X = np.vstack(vectors)

    return accessions, X


# ---------------------------------------------------------------------
# DB Class
# ---------------------------------------------------------------------

import psycopg2
import psycopg2.extras
import numpy as np


class BioDataDB:
    def __init__(self, host, dbname, user, password, port=5432):
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )

    def close(self):
        self.conn.close()

    def fetch_embeddings(
        self,
        embedding_type_id: int,
        layer_index: int,
        batch_size: int = 1000
    ):
        """
        Generator que retorna (accession, embedding)
        en batches per no petar la RAM.
        """

        query = """
        SELECT
            a.code       AS accession,
            se.embedding AS embedding
        FROM sequence_embeddings se
        JOIN sequence s
            ON se.sequence_id = s.id
        JOIN protein p
            ON p.sequence_id = s.id
        JOIN accession a
            ON a.protein_id = p.id
        WHERE se.embedding_type_id = %s
          AND se.layer_index      = %s
          AND a.primary = TRUE
        ORDER BY a.code
        """

        with self.conn.cursor(
            name="embedding_cursor",
            cursor_factory=psycopg2.extras.DictCursor
        ) as cursor:
            cursor.itersize = batch_size
            cursor.execute(query, (embedding_type_id, layer_index))

            for row in cursor:
                accession = row["accession"]
                embedding = np.array(row["embedding"], dtype=np.float32)
                yield accession, embedding


class EmbeddingDB:
    """
    Thin adapter around DB utilities.
    This is the ONLY interface runners should use.
    """

    def __init__(self, engine):
        self.engine = engine

    def resolve_embedding_type_id(self, embedding_type_name: str) -> int:
        return resolve_embedding_type_id(
            self.engine,
            embedding_type_name=embedding_type_name,
        )

    def load_embeddings(self, embedding_type_id: int, layer_index: int):
        accessions, X = load_embeddings(
            engine=self.engine,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
        )
        return accessions, X
