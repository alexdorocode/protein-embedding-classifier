import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.training_service import TrainingService


class DummyModel:
    def fit(self, X, y):
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        probs = np.full((X.shape[0], n_classes), 1.0 / n_classes, dtype=np.float32)
        return probs


class DummyModelFactory:
    def __init__(self):
        self.calls = []

    def create(self, model_type, params=None, input_size=None, output_size=None):
        self.calls.append((model_type, input_size, output_size))
        return DummyModel()


def _make_bundle():
    return EmbeddingBundle(
        X_train={
            "ESM3c": np.random.randn(10, 8).astype(np.float32),
            "GeOKG": np.random.randn(10, 4).astype(np.float32),
        },
        X_val={
            "ESM3c": np.random.randn(4, 8).astype(np.float32),
            "GeOKG": np.random.randn(4, 4).astype(np.float32),
        },
        X_test={
            "ESM3c": np.random.randn(4, 8).astype(np.float32),
            "GeOKG": np.random.randn(4, 4).astype(np.float32),
        },
        y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0, 1]),
        y_test=np.array([0, 1, 0, 1]),
    )


def test_training_service_returns_expected_dictionary_structure():
    factory = DummyModelFactory()
    service = TrainingService(model_factory=factory)

    results = service.train(
        embedding_bundle=_make_bundle(),
        training_config={"model_types": ["LR"]},
    )

    assert ("LR", "ESM3c") in results
    payload = results[("LR", "ESM3c")]
    assert "model" in payload
    assert "val_probs" in payload
    assert "metrics" in payload


def test_training_service_trains_per_embedding():
    factory = DummyModelFactory()
    service = TrainingService(model_factory=factory)

    results = service.train(
        embedding_bundle=_make_bundle(),
        training_config={"model_types": ["LR"]},
    )

    assert len(results) == 2
    assert len(factory.calls) == 2


def test_training_service_metrics_exist():
    service = TrainingService(model_factory=DummyModelFactory())

    results = service.train(
        embedding_bundle=_make_bundle(),
        training_config={"model_types": ["LR"]},
    )

    for payload in results.values():
        assert "f1_score" in payload["metrics"]


def test_training_service_handles_multilabel_targets():
    bundle = EmbeddingBundle(
        X_train={"ESM3c": np.random.randn(8, 6).astype(np.float32)},
        X_val={"ESM3c": np.random.randn(4, 6).astype(np.float32)},
        X_test={"ESM3c": np.random.randn(4, 6).astype(np.float32)},
        y_train=np.array([
            ["GO:1", "GO:2"],
            ["GO:2"],
            ["GO:1"],
            ["GO:3"],
            ["GO:1", "GO:3"],
            ["GO:2", "GO:3"],
            ["GO:1"],
            ["GO:2"],
        ], dtype=object),
        y_val=np.array([
            ["GO:1"],
            ["GO:2", "GO:3"],
            ["GO:3"],
            ["GO:1", "GO:2"],
        ], dtype=object),
        y_test=np.array([
            ["GO:1"],
            ["GO:2"],
            ["GO:3"],
            ["GO:1", "GO:3"],
        ], dtype=object),
    )

    service = TrainingService()
    results = service.train(
        embedding_bundle=bundle,
        training_config={"model_types": ["LR"]},
    )

    payload = results[("LR", "ESM3c")]
    assert payload["val_probs"].shape == (4, 3)
    assert "f1_score" in payload["metrics"]
