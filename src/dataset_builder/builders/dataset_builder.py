from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from protein_embedding_classifier.data.label_loader import LabelLoader
from protein_embedding_classifier.data.protein_loader import ProteinLoader
from protein_embedding_classifier.data.splits.base import SplitStrategy


@dataclass
class DatasetBundle:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    zero_shot_ids: List[str] = field(default_factory=list)
    y_zero_shot: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=object))


class DatasetBuilder:
    def __init__(self, protein_loader: ProteinLoader, label_loader: LabelLoader, split_strategy: SplitStrategy):
        self.protein_loader = protein_loader
        self.label_loader = label_loader
        self.split_strategy = split_strategy
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self) -> DatasetBundle:
        accessions, metadata = self.protein_loader.load()
        labels = self.label_loader.load(accessions)

        aligned_ids = [acc for acc in accessions if acc in labels]
        aligned_metadata: Dict[str, Dict[str, Any]] = {
            acc: metadata.get(acc, {}) for acc in aligned_ids
        }

        train_ids, val_ids, test_ids = self.split_strategy.split(
            aligned_ids,
            labels,
            aligned_metadata,
        )

        zero_shot_getter = getattr(self.split_strategy, "get_zero_shot", None)
        zero_shot_ids = zero_shot_getter() if callable(zero_shot_getter) else []

        y_train = np.asarray([labels[acc] for acc in train_ids], dtype=object)
        y_val = np.asarray([labels[acc] for acc in val_ids], dtype=object)
        y_test = np.asarray([labels[acc] for acc in test_ids], dtype=object)
        y_zero_shot = np.asarray([labels[acc] for acc in zero_shot_ids], dtype=object)

        self.logger.info(
            "DatasetBuilder produced bundle with %d train, %d val, %d test, %d zero-shot",
            len(train_ids),
            len(val_ids),
            len(test_ids),
            len(zero_shot_ids),
        )

        return DatasetBundle(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            zero_shot_ids=zero_shot_ids,
            y_zero_shot=y_zero_shot,
        )
