import numpy as np

from protein_embedding_classifier.tasks.enzyme_vs_not.labels import load_labels


class EnzymeVsNotTask:
    """
    Binary classification task:
    enzyme (1) vs non-enzyme (0)
    """

    name = "enzyme_vs_not"

    def __init__(self, engine):
        self.engine = engine
        self._labels = None

    def _ensure_labels_loaded(self):
        if self._labels is None:
            self._labels = load_labels(self.engine)

    def get_labels(self, accessions: list[str]) -> np.ndarray:
        """
        Align labels with given accessions.
        """
        self._ensure_labels_loaded()

        y = []
        missing = 0

        for acc in accessions:
            if acc not in self._labels:
                missing += 1
                y.append(None)
            else:
                y.append(self._labels[acc])

        if missing == len(accessions):
            raise RuntimeError("No labels matched any accession")

        return np.asarray(y, dtype=np.int64)

    def log(self, metrics: dict, **context):
        """
        Log metrics for this task.
        For now: stdout. Later: DB / JSON / W&B.
        """
        msg = {
            "task": self.name,
            **context,
            **metrics,
        }
        print(msg)
