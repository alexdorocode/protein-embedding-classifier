from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import KFold

from src.dataset_builder.splits.base import SplitStrategy


class CrossValidationSplit(SplitStrategy):
    def __init__(self, n_splits: int = 5, fold_index: int = 0, random_state: int = 42):
        self.n_splits = n_splits
        self.fold_index = fold_index
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        _ = (labels, metadata)
        self.logger.info("Applying CrossValidation strategy")
        self.logger.info("Dataset size: %d", len(accessions))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(accessions))
        fold = self.fold_index % len(folds)
        train_idx, test_idx = folds[fold]

        train_ids = [accessions[i] for i in train_idx]
        test_ids = [accessions[i] for i in test_idx]

        val_cut = max(1, int(0.1 * len(train_ids))) if len(train_ids) > 1 else 0
        val_ids = train_ids[:val_cut]
        train_ids = train_ids[val_cut:]

        self.logger.info("Dataset split:\nTrain: %d\nValidation: %d\nTest: %d", len(train_ids), len(val_ids), len(test_ids))
        return train_ids, val_ids, test_ids
