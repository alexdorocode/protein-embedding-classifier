from sqlalchemy import text

LABELS_SQL_PATH = "tasks/enzyme_vs_not/labels.sql"


def load_labels(engine) -> dict[str, int]:
    """
    Load labels from SQL file.

    Returns
    -------
    dict[accession, label]
    """
    with open(LABELS_SQL_PATH, "r") as f:
        sql = f.read()

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    return {r.accession: int(r.label) for r in rows}
