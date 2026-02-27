from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import KFold

from protein_embedding_classifier.data.splits.base import SplitStrategy
from protein_embedding_classifier.data.splits.zero_shot_organism import ZeroShotOrganismSplit


class IndependentValidationTrainTestSplit(SplitStrategy):
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.logger = logging.getLogger(self.__class__.__name__)
        self._organism_matcher = ZeroShotOrganismSplit()

    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        _ = labels
        self.logger.info("Applying independent validation/train_test strategy")
        self.logger.info("Dataset size: %d", len(accessions))

        val_ids, remaining_ids = self._select_validation(accessions, metadata)
        train_ids, test_ids = self._split_train_test(remaining_ids, metadata)

        self.logger.info("Dataset split:\nTrain: %d\nValidation: %d\nTest: %d", len(train_ids), len(val_ids), len(test_ids))
        return train_ids, val_ids, test_ids

    def _select_validation(
        self,
        accessions: List[str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        validation_conf = self.conf.get("validation", {})
        strategy = validation_conf.get("strategy", "random")

        if strategy == "random":
            random_conf = validation_conf.get("random", {})
            val_size = random_conf.get("val_size", 0.1)
            random_state = random_conf.get("random_state", 42)
            items = list(accessions)
            rng = random.Random(random_state)
            rng.shuffle(items)
            n_val = int(len(items) * val_size)
            val_ids = items[:n_val]
            remaining_ids = items[n_val:]
            return val_ids, remaining_ids

        if strategy == "organism":
            organism_conf = validation_conf.get("organism", {})
            val_organisms = organism_conf.get("val_organisms", [])
            train_test_conf = self.conf.get("train_test", {})
            reserved_train_test_organisms = train_test_conf.get("organisms", self.conf.get("train_test_organisms", []))
            all_others = any(
                self._organism_matcher._normalize(group) in {"all others", "all_others"}
                for group in val_organisms
            )
            organism_groups = [
                group
                for group in val_organisms
                if self._organism_matcher._normalize(group) not in {"all others", "all_others"}
            ]

            val_ids: List[str] = []
            remaining_ids: List[str] = []
            for accession in accessions:
                organism = metadata.get(accession, {}).get("organism")
                in_group = any(
                    self._organism_matcher._matches_group(organism, group_name)
                    for group_name in organism_groups
                )
                in_reserved_train_test = any(
                    self._organism_matcher._matches_group(organism, group_name)
                    for group_name in reserved_train_test_organisms
                )
                if in_group:
                    val_ids.append(accession)
                elif all_others and not in_reserved_train_test:
                    val_ids.append(accession)
                else:
                    remaining_ids.append(accession)
            return val_ids, remaining_ids

        if strategy == "csv":
            csv_conf = validation_conf.get("csv", {})
            csv_path = Path(csv_conf["csv_path"])
            accession_column = csv_conf.get("accession_column", "accession")
            split_column = csv_conf.get("split_column", "split")
            validation_values = set(csv_conf.get("validation_values", ["val", "validation"]))

            df = pd.read_csv(csv_path)
            allowed = set(accessions)
            df = df[df[accession_column].isin(allowed)]

            val_ids = df[df[split_column].isin(validation_values)][accession_column].tolist()
            val_set = set(val_ids)
            remaining_ids = [accession for accession in accessions if accession not in val_set]
            return val_ids, remaining_ids

        raise ValueError(f"Unknown validation strategy: {strategy}")

    def _split_train_test(
        self,
        accessions: List[str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        train_test_conf = self.conf.get("train_test", {})
        strategy = train_test_conf.get("strategy", "random")

        scoped_organisms = train_test_conf.get("organisms", self.conf.get("train_test_organisms", []))
        if scoped_organisms:
            in_scope: List[str] = []
            out_scope: List[str] = []
            for accession in accessions:
                organism = metadata.get(accession, {}).get("organism")
                matched = any(
                    self._organism_matcher._matches_group(organism, group_name)
                    for group_name in scoped_organisms
                )
                if matched:
                    in_scope.append(accession)
                else:
                    out_scope.append(accession)

            scoped_train, scoped_test = self._apply_train_test_strategy(in_scope, strategy, train_test_conf)
            train_ids = out_scope + scoped_train
            test_ids = scoped_test
            return train_ids, test_ids

        return self._apply_train_test_strategy(accessions, strategy, train_test_conf)

    @staticmethod
    def _apply_train_test_strategy(
        accessions: List[str],
        strategy: str,
        train_test_conf: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        if strategy == "random":
            random_conf = train_test_conf.get("random", {})
            test_size = random_conf.get("test_size", 0.2)
            random_state = random_conf.get("random_state", 42)
            items = list(accessions)
            rng = random.Random(random_state)
            rng.shuffle(items)
            n_test = int(len(items) * test_size)
            test_ids = items[:n_test]
            train_ids = items[n_test:]
            return train_ids, test_ids

        if strategy == "cross_validation":
            cv_conf = train_test_conf.get("cross_validation", {})
            n_splits = cv_conf.get("n_splits", 5)
            fold_index = cv_conf.get("fold_index", 0)
            random_state = cv_conf.get("random_state", 42)

            if len(accessions) < 2:
                return list(accessions), []

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = list(kf.split(accessions))
            fold = fold_index % len(folds)
            train_idx, test_idx = folds[fold]
            train_ids = [accessions[i] for i in train_idx]
            test_ids = [accessions[i] for i in test_idx]
            return train_ids, test_ids

        raise ValueError(f"Unknown train_test strategy: {strategy}")