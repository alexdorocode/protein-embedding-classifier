from __future__ import annotations
from typing import Tuple, List
import numpy as np
from sqlalchemy import text
from src.core.embeddings import EmbeddingStore
from pathlib import Path

class DatasetTask:
    """Base class for a dataset/task."""

    LABELS_SQL_PATH: str  # ha de ser sobreescrit a la subclass
    LABEL_MAP: dict | None = None  # opcional: convertir labels a ints

    def __init__(self, store: EmbeddingStore):
        self.store = store

    def load_labels(self) -> dict[str, int]:
        """Load labels from SQL file and map to integers if LABEL_MAP is defined."""
        path = Path(self.LABELS_SQL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        with path.open() as f:
            sql = f.read()

        with self.store.engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()

        labels = {r.accession: r.label for r in rows}

        # Map labels to int if LABEL_MAP exists
        if self.LABEL_MAP:
            labels = {acc: self.LABEL_MAP[lbl] for acc, lbl in labels.items()}

        return labels

    def build_dataset(self, embedding_type: str, layer: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return X, y, accession aligned with embeddings."""
        accessions, X = self.store.get(embedding_type, layer)
        labels = self.load_labels()

        X_out, y_out, acc_out = [], [], []

        for acc, vec in zip(accessions, X):
            if acc not in labels:
                continue
            X_out.append(vec)
            y_out.append(labels[acc])
            acc_out.append(acc)

        if not X_out:
            raise RuntimeError("No samples after label matching")

        return np.asarray(X_out), np.asarray(y_out, dtype=np.int64), acc_out

class EnzymeVsNotTask(DatasetTask):
    LABELS_SQL_PATH = "tasks/enzyme_vs_not/labels.sql"
    LABEL_MAP = {"0": 0, "1": 1}  # opcional si ja tens 0/1 al SQL

class LocalizationTask(DatasetTask):
    LABELS_SQL_PATH = "tasks/localization/labels.sql"
    LABEL_MAP = {"nucleus": 0, "mitochondrion": 1, "cytoplasm": 2, "other": 3}

class GoSlimMFTask(DatasetTask):
    LABELS_SQL_PATH = "tasks/go_slim_mf/labels.sql"
    LABEL_MAP = {"enzyme": 0, "binding": 1, "catalytic": 2, "other": 3}
