from __future__ import annotations

from typing import Type, Dict

from src.training.models.base import BaseClassifier
from src.training.models.linear import LogisticRegressionClassifier
from src.training.models.mlp import MLPClassifierWrapper
from src.training.models.random_forest import RandomForestClassifierWrapper


CLASSIFIERS: Dict[str, Type[BaseClassifier]] = {
    "lr": LogisticRegressionClassifier,
    "mlp": MLPClassifierWrapper,
    "rf": RandomForestClassifierWrapper,
}


def get_classifier(name: str, **kwargs) -> BaseClassifier:
    aliases = {
        "logistic": "lr",
        "random_forest": "rf",
    }
    normalized_name = aliases.get(name, name)

    if normalized_name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(CLASSIFIERS)}"
        )
    return CLASSIFIERS[normalized_name](**kwargs)
