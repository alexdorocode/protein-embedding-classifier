import numpy as np
from sklearn.preprocessing import StandardScaler

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.training_service import TrainingService


class DummyModel:
    def __init__(self):
        self.fit_x = None
        self.fit_y = None
        self.predict_x = None

    def fit(self, X, y):
        self.fit_x = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        self.predict_x = np.asarray(X).copy()
        n_classes = len(self.classes_)
        probs = np.full((X.shape[0], n_classes), 1.0 / n_classes, dtype=np.float32)
        return probs


class DummyModelFactory:
    def __init__(self):
        self.calls = []
        self.models = []

    def create(self, model_type, params=None, input_size=None, output_size=None):
        self.calls.append((model_type, input_size, output_size))
        model = DummyModel()
        self.models.append(model)
        return model


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
    assert "validation" in payload["metrics"]
    assert "test" in payload["metrics"]


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
        assert "f1" in payload["metrics"]["validation"]


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
    assert "macro_f1" in payload["metrics"]["validation"]


def test_training_service_allows_mlp_with_multilabel_when_model_supports_it(monkeypatch):
    class FakeMultilabelMLP:
        def fit(self, X_train, y_train, X_val, y_val):
            self._output_dim = y_train.shape[1]

        def predict_proba(self, X):
            return np.full((X.shape[0], self._output_dim), 0.5, dtype=np.float32)

    class FakeModelFactory:
        def create(self, model_type, params=None, input_size=None, output_size=None):
            assert model_type == "MLP"
            return FakeMultilabelMLP()

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

    service = TrainingService(model_factory=FakeModelFactory())
    results = service.train(
        embedding_bundle=bundle,
        training_config={
            "model_types": ["MLP"],
            "model_params": {"MLP": {}},
            "evaluate_test": True,
        },
    )

    payload = results[("MLP", "ESM3c")]
    assert payload["val_probs"].shape == (4, 3)
    assert "macro_f1" in payload["metrics"]["validation"]
    assert payload["metrics"]["test"] is not None


def test_training_service_l2_normalization_rowwise_unit_norm():
    factory = DummyModelFactory()
    service = TrainingService(model_factory=factory)

    bundle = EmbeddingBundle(
        X_train={"ESM3c": np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)},
        X_val={"ESM3c": np.array([[8.0, 15.0], [7.0, 24.0]], dtype=np.float32)},
        X_test={"ESM3c": np.array([[9.0, 12.0], [20.0, 21.0]], dtype=np.float32)},
        y_train=np.array([0, 1]),
        y_val=np.array([0, 1]),
        y_test=np.array([0, 1]),
    )

    service.train(
        embedding_bundle=bundle,
        training_config={"model_types": ["LR"], "feature_processing": {"normalize": "l2"}},
    )

    model = factory.models[0]
    assert model.fit_x is not None
    assert model.predict_x is not None

    train_row_norms = np.linalg.norm(model.fit_x, axis=1)
    val_row_norms = np.linalg.norm(model.predict_x, axis=1)
    assert np.allclose(train_row_norms, np.ones_like(train_row_norms), atol=1e-6)
    assert np.allclose(val_row_norms, np.ones_like(val_row_norms), atol=1e-6)


def test_training_service_standard_scaler_fit_on_train_and_applies_to_val_test():
    factory = DummyModelFactory()
    service = TrainingService(model_factory=factory)

    x_train = np.array(
        [
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [4.0, 13.0],
        ],
        dtype=np.float32,
    )
    x_val = np.array([[5.0, 14.0], [6.0, 15.0]], dtype=np.float32)
    x_test = np.array([[7.0, 16.0], [8.0, 17.0]], dtype=np.float32)

    bundle = EmbeddingBundle(
        X_train={"ESM3c": x_train},
        X_val={"ESM3c": x_val},
        X_test={"ESM3c": x_test},
        y_train=np.array([0, 1, 0, 1]),
        y_val=np.array([0, 1]),
        y_test=np.array([0, 1]),
    )

    service.train(
        embedding_bundle=bundle,
        training_config={"model_types": ["LR"], "feature_processing": {"normalize": "standard"}},
    )

    model = factory.models[0]
    assert model.fit_x is not None
    assert model.predict_x is not None

    assert np.allclose(model.fit_x.mean(axis=0), np.zeros(2), atol=1e-7)
    assert np.allclose(model.fit_x.std(axis=0), np.ones(2), atol=1e-7)

    expected_scaler = StandardScaler().fit(x_train)
    expected_val = expected_scaler.transform(x_val)
    expected_test = expected_scaler.transform(x_test)

    assert np.allclose(model.predict_x, expected_val, atol=1e-7)
    transformed_test = expected_scaler.transform(x_test)
    assert np.allclose(transformed_test, expected_test, atol=1e-7)


def test_training_service_standard_scaler_no_leakage_from_val_distribution():
    factory = DummyModelFactory()
    service = TrainingService(model_factory=factory)

    x_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ],
        dtype=np.float32,
    )
    x_val = np.array([[1000.0, 2000.0], [1100.0, 2200.0]], dtype=np.float32)

    bundle = EmbeddingBundle(
        X_train={"ESM3c": x_train},
        X_val={"ESM3c": x_val},
        X_test={"ESM3c": np.array([[1200.0, 2400.0]], dtype=np.float32)},
        y_train=np.array([0, 1, 0, 1]),
        y_val=np.array([0, 1]),
        y_test=np.array([0]),
    )

    service.train(
        embedding_bundle=bundle,
        training_config={"model_types": ["LR"], "feature_processing": {"normalize": "standard"}},
    )

    model = factory.models[0]
    expected_scaler = StandardScaler().fit(x_train)
    expected_val = expected_scaler.transform(x_val)

    assert np.allclose(model.predict_x, expected_val, atol=1e-7)


def test_training_service_predict_proba_contract_with_normalization_modes():
    for normalization_mode in ["none", "l2", "standard"]:
        factory = DummyModelFactory()
        service = TrainingService(model_factory=factory)
        results = service.train(
            embedding_bundle=_make_bundle(),
            training_config={
                "model_types": ["LR"],
                "feature_processing": {"normalize": normalization_mode},
            },
        )

        payload = results[("LR", "ESM3c")]
        val_probs = payload["val_probs"]
        assert val_probs.shape[0] == _make_bundle().X_val["ESM3c"].shape[0]
        assert np.allclose(val_probs.sum(axis=1), np.ones(val_probs.shape[0]), atol=1e-6)
