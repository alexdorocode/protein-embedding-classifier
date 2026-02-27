from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable
import re

import pandas as pd
from sqlalchemy import text


class LabelLoader:
    def __init__(
        self,
        source: str,
        engine=None,
        file_path: str | None = None,
        db_query: str | None = None,
        db_query_file: str | None = None,
        accession_column: str = "accession",
        label_column: str = "label",
        artifacts_dir: str = "artifacts",
    ):
        self.source = source
        self.engine = engine
        self.file_path = file_path
        self.db_query = db_query
        self.db_query_file = db_query_file
        self.accession_column = accession_column
        self.label_column = label_column
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_file_labels(self) -> Dict[str, Any]:
        if not self.file_path:
            raise ValueError("LabelLoader file_path is required when source='file'")

        df = pd.read_csv(self.file_path)
        if self.accession_column not in df.columns or self.label_column not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{self.accession_column}' and '{self.label_column}'"
            )

        exploded_rows: list[tuple[str, Any]] = []
        for _, row in df.iterrows():
            accession_value = row[self.accession_column]
            label_value = row[self.label_column]

            if pd.isna(accession_value):
                continue

            accession_tokens = [
                token.strip() for token in re.split(r"[;,]", str(accession_value)) if token.strip()
            ]

            for accession in accession_tokens:
                exploded_rows.append((accession, label_value))

        exploded_df = pd.DataFrame(exploded_rows, columns=[self.accession_column, self.label_column])
        if exploded_df.empty:
            return {}

        duplicated = exploded_df[self.accession_column].duplicated(keep=False).any()
        grouped = exploded_df.groupby(self.accession_column)[self.label_column].apply(list)

        if duplicated:
            return {acc: values for acc, values in grouped.items()}

        return {acc: values[0] for acc, values in grouped.items()}

    def _load_db_labels(self) -> Dict[str, Any]:
        if self.engine is None:
            raise ValueError("LabelLoader engine is required when source='db'")

        query = self.db_query
        if not query and self.db_query_file:
            query_path = Path(self.db_query_file)
            if not query_path.is_absolute() and not query_path.exists():
                repo_root = Path(__file__).resolve().parents[2]
                query_path = repo_root / self.db_query_file
            query = query_path.read_text(encoding="utf-8")

        if not query:
            raise ValueError("Provide db_query or db_query_file when source='db'")

        with self.engine.connect() as conn:
            rows = conn.execute(text(query)).fetchall()

        labels: Dict[str, Any] = {}
        for row in rows:
            accession = getattr(row, self.accession_column)
            label = getattr(row, self.label_column)
            labels[accession] = label

        return labels

    def load(self, db_accessions: Iterable[str]) -> Dict[str, Any]:
        self.logger.info("Loading labels from %s", self.source)

        if self.source == "file":
            labels = self._load_file_labels()
        elif self.source == "db":
            labels = self._load_db_labels()
        else:
            raise ValueError("source must be 'db' or 'file'")

        db_accessions_set = set(db_accessions)
        aligned = {acc: value for acc, value in labels.items() if acc in db_accessions_set}

        missing = sorted(acc for acc in labels.keys() if acc not in db_accessions_set)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        missing_file = self.artifacts_dir / "missing_proteins.txt"
        missing_file.write_text("\n".join(missing), encoding="utf-8")

        self.logger.info("Total labels assigned: %d", len(aligned))
        self.logger.info("Missing proteins count: %d", len(missing))

        return aligned
