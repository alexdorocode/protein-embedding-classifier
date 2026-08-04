from __future__ import annotations

from typing import Type, Dict

from protein_embedding_classifier.classifiers.base import BaseClassifier
from protein_embedding_classifier.classifiers.linear import LogisticRegressionClassifier
from protein_embedding_classifier.classifiers.mlp_protein_classifier import MLPProteinClassifier
from protein_embedding_classifier.classifiers.random_forest import RandomForestClassifierWrapper


CLASSIFIERS: Dict[str, Type[BaseClassifier]] = {
    "lr": LogisticRegressionClassifier,
    "mlp": MLPProteinClassifier,
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
