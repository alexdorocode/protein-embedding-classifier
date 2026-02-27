from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from protein_embedding_classifier.data.splits.base import SplitStrategy


class ZeroShotCSVSplit(SplitStrategy):
    def __init__(
        self,
        csv_path: str,
        accession_column: str = "accession",
        split_column: str = "split",
    ):
        self.csv_path = Path(csv_path)
        self.accession_column = accession_column
        self.split_column = split_column
        self.logger = logging.getLogger(self.__class__.__name__)

    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        _ = (labels, metadata)
        self.logger.info("Applying ZeroShotCSV strategy")
        self.logger.info("Dataset size: %d", len(accessions))

        df = pd.read_csv(self.csv_path)
        allowed = set(accessions)
        df = df[df[self.accession_column].isin(allowed)]

        train_ids = df[df[self.split_column] == "train"][self.accession_column].tolist()
        val_ids = df[df[self.split_column].isin(["val", "validation"])][self.accession_column].tolist()
        test_ids = df[df[self.split_column] == "test"][self.accession_column].tolist()

        self.logger.info("Dataset split:\nTrain: %d\nValidation: %d\nTest: %d", len(train_ids), len(val_ids), len(test_ids))
        return train_ids, val_ids, test_ids
