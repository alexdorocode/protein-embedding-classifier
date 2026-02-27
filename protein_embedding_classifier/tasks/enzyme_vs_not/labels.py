from sqlalchemy import text
from pathlib import Path

LABELS_SQL_PATH = Path(__file__).with_name("labels.sql")


def load_labels(engine) -> dict[str, int]:
    """Load enzyme-vs-not labels from the SQL file using the given engine.

    Returns
    -------
    dict[accession, label]
    """
    with open(LABELS_SQL_PATH, "r") as f:
        sql = f.read()

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    return {r.accession: int(r.label) for r in rows}
