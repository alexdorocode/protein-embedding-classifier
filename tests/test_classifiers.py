import numpy as np
import pytest

from protein_embedding_classifier.classifiers.linear import LogisticRegressionClassifier
from protein_embedding_classifier.classifiers.mlp import MLPClassifierWrapper
from protein_embedding_classifier.classifiers.random_forest import RandomForestClassifierWrapper
from protein_embedding_classifier.classifiers.registry import get_classifier, CLASSIFIERS


def _make_toy_binary_data(n_samples: int = 50, dim: int = 4):
    rng = np.random.RandomState(0)
    X_pos = rng.normal(loc=1.0, scale=0.5, size=(n_samples // 2, dim))
    X_neg = rng.normal(loc=-1.0, scale=0.5, size=(n_samples // 2, dim))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    return X, y


@pytest.mark.parametrize("cls, kwargs", [
    (LogisticRegressionClassifier, {"C": 1.0, "max_iter": 200}),
    (MLPClassifierWrapper, {"hidden_layer_sizes": (8,), "max_iter": 200}),
    (RandomForestClassifierWrapper, {"n_estimators": 20, "max_depth": 5}),
])
def test_classifiers_fit_and_report_metrics(cls, kwargs):
    X, y = _make_toy_binary_data()

    clf = cls(**kwargs)
    metrics = clf.fit_eval(X, y)

    # Basic metric sanity checks
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_reset_rebuilds_internal_model():
    X, y = _make_toy_binary_data(n_samples=20)

    clf = LogisticRegressionClassifier(C=0.1, max_iter=50)
    model_id_before = id(clf.model)
    clf.fit_eval(X, y)

    clf.reset()
    model_id_after = id(clf.model)

    assert model_id_after != model_id_before


def test_get_classifier_registry_and_errors():
    # Known names
    clf_lr = get_classifier("lr", C=0.5)
    assert isinstance(clf_lr, LogisticRegressionClassifier)

    clf_mlp = get_classifier("mlp", hidden_layer_sizes=(16,))
    assert isinstance(clf_mlp, MLPClassifierWrapper)

    clf_rf = get_classifier("rf", n_estimators=10)
    assert isinstance(clf_rf, RandomForestClassifierWrapper)

    # Registry should expose all of them
    assert set(CLASSIFIERS.keys()) == {"lr", "mlp", "rf"}

    # Unknown name should raise a clear error
    with pytest.raises(ValueError):
        get_classifier("unknown")
