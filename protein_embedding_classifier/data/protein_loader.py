from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from sqlalchemy import text


class ProteinLoader:
    def __init__(self, engine, query: str | None = None):
        self.engine = engine
        self.query = query
        self.logger = logging.getLogger(self.__class__.__name__)

    def _detect_organism_column(self) -> str | None:
        candidates = ["organism", "organism_name", "species", "taxon_name"]
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'protein'
                    """
                )
            ).fetchall()
        existing = {row.column_name for row in rows}
        for name in candidates:
            if name in existing:
                return name
        return None

    def load(self) -> Tuple[List[str], Dict[str, Dict[str, str | None]]]:
        self.logger.info("Connecting with db")

        if self.query:
            query_text = self.query
        else:
            organism_column = self._detect_organism_column()
            if organism_column:
                query_text = (
                    "SELECT a.code AS accession, p." + organism_column + " AS organism "
                    "FROM protein p "
                    "JOIN accession a ON a.protein_id = p.id "
                    "WHERE a.primary = TRUE"
                )
            else:
                query_text = (
                    "SELECT a.code AS accession, NULL::text AS organism "
                    "FROM protein p "
                    "JOIN accession a ON a.protein_id = p.id "
                    "WHERE a.primary = TRUE"
                )

        with self.engine.connect() as conn:
            rows = conn.execute(text(query_text)).fetchall()

        accessions = [row.accession for row in rows]
        metadata = {
            row.accession: {"organism": getattr(row, "organism", None)}
            for row in rows
        }

        self.logger.info("%d proteins found", len(accessions))
        return accessions, metadata
