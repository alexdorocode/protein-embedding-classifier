import numpy as np
from sqlalchemy import text

from src.training.db import load_db_config, create_engine_from_config
from src.training.embeddings import EmbeddingStore


def test_embedding_store_shapes_for_all_models():
    """Integration-style test: load one populated layer per model and check shapes.

    This hits the real database defined in config/db.yaml, so it assumes
    that at least one layer per embedding type has stored embeddings.
    """

    cfg = load_db_config("config/db.yaml")
    engine = create_engine_from_config(cfg)
    store = EmbeddingStore(engine, normalize=True)

    # Discover one example layer per embedding type that actually has data
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT t.name, se.layer_index, COUNT(*) AS n
                FROM sequence_embeddings se
                JOIN sequence_embedding_type t
                  ON se.embedding_type_id = t.id
                GROUP BY t.name, se.layer_index
                HAVING COUNT(*) > 0
                """
            )
        ).fetchall()

    assert rows, "No embeddings found in database for any model/layer."

    # Pick one layer per model
    layers_by_model = {}
    for name, layer_index, _ in rows:
        if name not in layers_by_model:
            layers_by_model[name] = int(layer_index)

    # For each model, fetch from EmbeddingStore and validate
    for model_name, layer in layers_by_model.items():
        accessions, X = store.get(model_name, layer=layer)
        print(f"{model_name} layer {layer}: N={len(accessions)}, shape={X.shape}")

        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert len(accessions) == X.shape[0]
