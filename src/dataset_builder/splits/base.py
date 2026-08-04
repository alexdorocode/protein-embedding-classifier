from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class SplitStrategy(ABC):
    @abstractmethod
    def split(
        self,
        accessions: List[str],
        labels: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        pass
