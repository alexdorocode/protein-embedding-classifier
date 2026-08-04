from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import KFold

from src.dataset_builder.splits.base import SplitStrategy
from src.dataset_builder.splits.zero_shot_organism import ZeroShotOrganismSplit


class IndependentValidationTrainTestSplit(SplitStrategy):
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.logger = logging.getLogger(self.__class__.__name__)
        self._organism_matcher = ZeroShotOrganismSplit()
        self._csv_cache: dict[str, pd.DataFrame] = {}
        self._train_ids: List[str] = []
        self._val_ids: List[str] = []
        self._test_ids: List[str] = []
        self._zero_shot_ids: List[str] = []

    def get_train(self) -> List[str]:
        return list(self._train_ids)

    def get_validation(self) -> List[str]:
        return list(self._val_ids)

    def get_test(self) -> List[str]:
        return list(self._test_ids)

    def get_zero_shot(self) -> List[str]:
        return list(self._zero_shot_ids)

    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        self.logger.info("Applying independent validation/train_test strategy")
        self.logger.info("Dataset size: %d", len(accessions))

        accession_set = set(accessions)
        val_ids, remaining_ids = self._select_validation(accessions, metadata)
        train_ids, test_ids = self._split_train_test(remaining_ids, metadata)
        zero_shot_ids = self._select_zero_shot(accessions, metadata)

        zero_set = set(zero_shot_ids)
        if zero_set:
            train_ids = [accession for accession in train_ids if accession not in zero_set]
            val_ids = [accession for accession in val_ids if accession not in zero_set]
            test_ids = [accession for accession in test_ids if accession not in zero_set]

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        zero_set = set(zero_shot_ids)

        overlap_train_zero = train_set.intersection(zero_set)
        overlap_val_zero = val_set.intersection(zero_set)
        overlap_test_zero = test_set.intersection(zero_set)
        if overlap_train_zero or overlap_val_zero or overlap_test_zero:
            raise ValueError(
                "Zero-shot leakage detected. Overlaps: "
                f"train={len(overlap_train_zero)} validation={len(overlap_val_zero)} test={len(overlap_test_zero)}"
            )

        covered = train_set.union(val_set).union(test_set).union(zero_set)
        if covered != accession_set:
            missing = accession_set.difference(covered)
            extra = covered.difference(accession_set)
            raise ValueError(
                "Primary split coverage mismatch after applying zero-shot holdout. "
                f"missing={len(missing)} extra={len(extra)}"
            )

        self._train_ids = list(train_ids)
        self._val_ids = list(val_ids)
        self._test_ids = list(test_ids)
        self._zero_shot_ids = list(zero_shot_ids)

        self._log_split_summary(labels=labels, metadata=metadata)

        if not self._zero_shot_ids:
            self.logger.warning("Zero-shot split is empty; zero-shot scoring will be skipped")
        elif len(self._zero_shot_ids) < 10:
            self.logger.warning(
                "Zero-shot split has very small sample size (n=%d); metrics may be statistically unstable",
                len(self._zero_shot_ids),
            )

        self.logger.info("Zero-shot verified as isolated holdout.")

        return train_ids, val_ids, test_ids

    def _read_csv_once(self, csv_path: Path) -> pd.DataFrame:
        key = str(csv_path.resolve())
        cached = self._csv_cache.get(key)
        if cached is not None:
            return cached
        df = pd.read_csv(csv_path)
        self._csv_cache[key] = df
        return df

    @staticmethod
    def _validate_csv_columns(df: pd.DataFrame, accession_column: str, split_column: str, csv_path: Path) -> None:
        if accession_column not in df.columns:
            raise ValueError(f"CSV split file missing accession column '{accession_column}': {csv_path}")
        if split_column not in df.columns:
            raise ValueError(f"CSV split file missing split column '{split_column}': {csv_path}")

    @staticmethod
    def _raise_if_duplicate_accessions_across_splits(
        df: pd.DataFrame,
        accession_column: str,
        split_column: str,
        allowed_values: set[str],
        csv_path: Path,
    ) -> None:
        scoped = df[df[split_column].astype(str).isin(allowed_values)][[accession_column, split_column]].dropna()
        if scoped.empty:
            return
        split_counts = scoped.groupby(accession_column)[split_column].nunique()
        duplicated = split_counts[split_counts > 1]
        if not duplicated.empty:
            examples = duplicated.index.tolist()[:10]
            raise ValueError(
                "Duplicate accession assignments across split values in CSV "
                f"{csv_path}. Examples: {examples}"
            )

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
            validation_values = {str(v) for v in csv_conf.get("validation_values", ["val", "validation"])}

            df = self._read_csv_once(csv_path)
            self._validate_csv_columns(df, accession_column, split_column, csv_path)
            self._raise_if_duplicate_accessions_across_splits(
                df=df,
                accession_column=accession_column,
                split_column=split_column,
                allowed_values=validation_values,
                csv_path=csv_path,
            )
            allowed = set(accessions)
            df = df[df[accession_column].isin(allowed)]

            if df.empty:
                raise ValueError("Validation CSV has no overlap with dataset accessions")

            val_ids = df[df[split_column].isin(validation_values)][accession_column].tolist()
            if not val_ids:
                raise ValueError("Validation CSV selection produced zero validation samples after overlap filtering")
            val_set = set(val_ids)
            remaining_ids = [accession for accession in accessions if accession not in val_set]

            self.logger.info(
                "Validation split (csv): size=%d (%.2f%% of dataset)",
                len(val_ids),
                100.0 * (len(val_ids) / max(1, len(accessions))),
            )
            return val_ids, remaining_ids

        raise ValueError(f"Unknown validation strategy: {strategy}")

    def _select_zero_shot(
        self,
        accessions: List[str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        zero_conf = self.conf.get("zero_shot", {})
        strategy = zero_conf.get("strategy")
        if not strategy:
            return []

        if strategy == "csv":
            csv_conf = zero_conf.get("csv", {})
            csv_path = Path(csv_conf["csv_path"])
            accession_column = csv_conf.get("accession_column", "accession")
            split_column = csv_conf.get("split_column", "split")
            zero_values = {str(v) for v in csv_conf.get("validation_values", ["zero_shot", "zs"])}

            df = self._read_csv_once(csv_path)
            self._validate_csv_columns(df, accession_column, split_column, csv_path)
            self._raise_if_duplicate_accessions_across_splits(
                df=df,
                accession_column=accession_column,
                split_column=split_column,
                allowed_values=zero_values,
                csv_path=csv_path,
            )
            allowed = set(accessions)
            df = df[df[accession_column].isin(allowed)]
            if df.empty:
                return []

            zero_ids = df[df[split_column].isin(zero_values)][accession_column].tolist()
            return list(dict.fromkeys(zero_ids))

        if strategy == "organism":
            organism_conf = zero_conf.get("organism", {})
            groups = organism_conf.get("val_organisms", organism_conf.get("test_organisms", []))
            if not groups:
                return []
            selected: List[str] = []
            for accession in accessions:
                organism = metadata.get(accession, {}).get("organism")
                if any(self._organism_matcher._matches_group(organism, group) for group in groups):
                    selected.append(accession)
            return selected

        if strategy == "random":
            random_conf = zero_conf.get("random", {})
            test_size = float(random_conf.get("val_size", random_conf.get("test_size", 0.1)))
            random_state = int(random_conf.get("random_state", 42))
            items = list(accessions)
            rng = random.Random(random_state)
            rng.shuffle(items)
            n_zero = int(len(items) * test_size)
            return items[:n_zero]

        raise ValueError(f"Unknown zero_shot strategy: {strategy}")

    def _log_split_summary(self, labels: Dict[str, Any], metadata: Dict[str, Dict[str, Any]]) -> None:
        split_map = {
            "Train": self._train_ids,
            "Validation": self._val_ids,
            "Test": self._test_ids,
            "Zero-shot": self._zero_shot_ids,
        }
        self.logger.info(
            "Split Summary:\n- Train size: %d\n- Validation size: %d\n- Test size: %d\n- Zero-shot size: %d",
            len(self._train_ids),
            len(self._val_ids),
            len(self._test_ids),
            len(self._zero_shot_ids),
        )

        for split_name, split_ids in split_map.items():
            class_counts: dict[str, int] = {}
            for accession in split_ids:
                key = str(labels.get(accession, "<missing_label>"))
                class_counts[key] = class_counts.get(key, 0) + 1
            self.logger.info("%s class distribution: %s", split_name, class_counts)

        zero_org_counts: dict[str, int] = {}
        for accession in self._zero_shot_ids:
            organism = str(metadata.get(accession, {}).get("organism", "<unknown>"))
            zero_org_counts[organism] = zero_org_counts.get(organism, 0) + 1
        if zero_org_counts:
            self.logger.info("Zero-shot organism distribution: %s", zero_org_counts)

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