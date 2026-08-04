from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Tuple

from src.dataset_builder.splits.base import SplitStrategy


class ZeroShotOrganismSplit(SplitStrategy):
    def __init__(
        self,
        test_organisms: List[str] | None = None,
        train_test_organisms: List[str] | None = None,
        val_organisms: List[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.test_organisms = set(test_organisms or [])
        self.train_test_organisms = set(train_test_organisms or [])
        self.val_organisms = set(val_organisms or [])
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _normalize(text_value: str | None) -> str:
        if not text_value:
            return ""
        normalized = text_value.lower().strip()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _matches_group(self, organism: str | None, group_name: str) -> bool:
        organism_norm = self._normalize(organism)
        group_norm = self._normalize(group_name)

        aliases = {
            "human": ["human", "homo sapiens", "h sapiens", "humans"],
            "homo sapiens": ["human", "homo sapiens", "h sapiens", "humans"],
            "mouse": ["mouse", "mus musculus", "m musculus", "mice"],
            "mus musculus": ["mouse", "mus musculus", "m musculus", "mice"],
            "ecoli": ["ecoli", "e coli", "escherichia coli", "eacoli"],
            "eacoli": ["ecoli", "e coli", "escherichia coli", "eacoli"],
            "escherichia coli": ["ecoli", "e coli", "escherichia coli", "eacoli"],
            "yeast": ["yeast", "saccharomyces cerevisiae", "baker s yeast"],
            "saccharomyces cerevisiae": ["yeast", "saccharomyces cerevisiae", "baker s yeast"],
        }

        tokens = aliases.get(group_norm, [group_norm])
        return any(token in organism_norm for token in tokens if token)

    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        _ = labels
        self.logger.info("Applying ZeroShotOrganism strategy")
        self.logger.info("Dataset size: %d", len(accessions))

        test_groups = [g for g in self.test_organisms if self._normalize(g) != "all others" and self._normalize(g) != "all_others"]
        train_test_groups = [g for g in self.train_test_organisms if self._normalize(g) != "all others" and self._normalize(g) != "all_others"]
        val_groups = [g for g in self.val_organisms if self._normalize(g) != "all others" and self._normalize(g) != "all_others"]
        test_all_others = any(self._normalize(g) in {"all others", "all_others"} for g in self.test_organisms)
        val_all_others = any(self._normalize(g) in {"all others", "all_others"} for g in self.val_organisms)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        train_test_candidates: List[str] = []

        for accession in accessions:
            organism = metadata.get(accession, {}).get("organism")

            in_test = any(self._matches_group(organism, group_name) for group_name in test_groups)
            in_train_test = any(self._matches_group(organism, group_name) for group_name in train_test_groups)
            in_val = any(self._matches_group(organism, group_name) for group_name in val_groups)

            if in_test:
                test_ids.append(accession)
            elif in_train_test:
                train_test_candidates.append(accession)
            elif in_val:
                val_ids.append(accession)
            elif val_all_others:
                val_ids.append(accession)
            elif test_all_others:
                test_ids.append(accession)
            else:
                train_ids.append(accession)

        if train_test_candidates:
            shuffled_candidates = list(train_test_candidates)
            random.Random(self.random_state).shuffle(shuffled_candidates)
            split_idx = int(len(shuffled_candidates) * self.test_size)
            test_from_candidates = shuffled_candidates[:split_idx]
            train_from_candidates = shuffled_candidates[split_idx:]
            test_ids.extend(test_from_candidates)
            train_ids.extend(train_from_candidates)

        self.logger.info("Dataset split:\nTrain: %d\nValidation: %d\nTest: %d", len(train_ids), len(val_ids), len(test_ids))
        return train_ids, val_ids, test_ids
