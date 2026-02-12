from __future__ import annotations

from typing import Type, Dict

from .base import BaseClassifier
from .linear import LogisticRegressionClassifier
from .mlp import MLPClassifierWrapper
from .random_forest import RandomForestClassifierWrapper


CLASSIFIERS: Dict[str, Type[BaseClassifier]] = {
    "lr": LogisticRegressionClassifier,
    "mlp": MLPClassifierWrapper,
    "rf": RandomForestClassifierWrapper,
}


def get_classifier(name: str, **kwargs) -> BaseClassifier:
    if name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(CLASSIFIERS)}"
        )
    return CLASSIFIERS[name](**kwargs)
