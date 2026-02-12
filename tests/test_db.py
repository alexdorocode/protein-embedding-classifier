import yaml
import numpy as np
from sqlalchemy import create_engine, text

from src.core.db import load_db_config, resolve_embedding_type_id, load_embeddings


def test_load_db_config(tmp_path):
    cfg = {"db": {"user": "u", "password": "p", "host": "h", "port": 1234, "name": "n"}}
    path = tmp_path / "db.yaml"
    path.write_text(yaml.safe_dump(cfg))

    out = load_db_config(path)
    assert out == cfg["db"]


def _setup_sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sequence_embedding_type (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE sequence (
                id INTEGER PRIMARY KEY
            );
        """))
        conn.execute(text("""
            CREATE TABLE protein (
                id INTEGER PRIMARY KEY,
                sequence_id INTEGER NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE accession (
                id INTEGER PRIMARY KEY,
                protein_id INTEGER NOT NULL,
                code TEXT NOT NULL,
                "primary" BOOLEAN NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE sequence_embeddings (
                id INTEGER PRIMARY KEY,
                sequence_id INTEGER NOT NULL,
                embedding_type_id INTEGER NOT NULL,
                layer_index INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );
        """))
    return engine


def test_resolve_embedding_type_id_and_load_embeddings():
    engine = _setup_sqlite_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO sequence_embedding_type (id, name) VALUES (1, 'ESM3c');"
        ))
        conn.execute(text(
            "INSERT INTO sequence (id) VALUES (10);"
        ))
        conn.execute(text(
            "INSERT INTO protein (id, sequence_id) VALUES (20, 10);"
        ))
        conn.execute(text(
            'INSERT INTO accession (id, protein_id, code, "primary") '
            "VALUES (30, 20, 'P12345', 1);"
        ))
        # simple 3-dim embedding stored as raw float32 bytes (to mimic pgvector storage)
        vec_bytes = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        conn.execute(text(
            "INSERT INTO sequence_embeddings "
            "(id, sequence_id, embedding_type_id, layer_index, embedding) "
            "VALUES (40, 10, 1, 5, :emb);"
        ), {"emb": vec_bytes})

    et_id = resolve_embedding_type_id(engine, "ESM3c")
    assert et_id == 1

    accs, X = load_embeddings(engine, et_id, layer_index=5)
    assert accs == ["P12345"]
    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 3)
    np.testing.assert_allclose(X[0], np.array([0.1, 0.2, 0.3], dtype=np.float32))
