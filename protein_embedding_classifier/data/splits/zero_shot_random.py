from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

from src.dataset_builder.splits.base import SplitStrategy


class ZeroShotRandomSplit(SplitStrategy):
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        _ = (labels, metadata)
        self.logger.info("Applying ZeroShotRandom strategy")
        self.logger.info("Dataset size: %d", len(accessions))

        items = list(accessions)
        rng = random.Random(self.random_state)
        rng.shuffle(items)

        n_total = len(items)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)

        test_ids = items[:n_test]
        val_ids = items[n_test:n_test + n_val]
        train_ids = items[n_test + n_val:]

        self.logger.info("Dataset split:\nTrain: %d\nValidation: %d\nTest: %d", len(train_ids), len(val_ids), len(test_ids))
        return train_ids, val_ids, test_ids
