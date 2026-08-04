import numpy as np
import pytest

from src.training.embedding_loading import EmbeddingBundle
from src.training.training.training_service import TrainingService


def _binary_bundle(n_train: int = 40, n_val: int = 16, n_features: int = 12) -> EmbeddingBundle:
    rng = np.random.default_rng(7)

    x_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    x_val = rng.normal(size=(n_val, n_features)).astype(np.float32)
    x_test = rng.normal(size=(n_val, n_features)).astype(np.float32)

    y_train = np.array([0] * (n_train // 2) + [1] * (n_train - n_train // 2))
    y_val = np.array([0] * (n_val // 2) + [1] * (n_val - n_val // 2))
    y_test = np.array([0] * (n_val // 2) + [1] * (n_val - n_val // 2))

    rng.shuffle(y_train)
    rng.shuffle(y_val)
    rng.shuffle(y_test)

    return EmbeddingBundle(
        X_train={"ESM3c": x_train},
        X_val={"ESM3c": x_val},
        X_test={"ESM3c": x_test},
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def _binary_bundle_multi_embedding(n_train: int = 40, n_val: int = 16) -> EmbeddingBundle:
    rng = np.random.default_rng(11)

    embedding_dims = {
        "ESM3c": 12,
        "Ankh3-Large": 16,
        "Prost-T5": 10,
        "GeOKG": 20,
    }

    x_train = {
        embedding_name: rng.normal(size=(n_train, dim)).astype(np.float32)
        for embedding_name, dim in embedding_dims.items()
    }
    x_val = {
        embedding_name: rng.normal(size=(n_val, dim)).astype(np.float32)
        for embedding_name, dim in embedding_dims.items()
    }
    x_test = {
        embedding_name: rng.normal(size=(n_val, dim)).astype(np.float32)
        for embedding_name, dim in embedding_dims.items()
    }

    y_train = np.array([0] * (n_train // 2) + [1] * (n_train - n_train // 2))
    y_val = np.array([0] * (n_val // 2) + [1] * (n_val - n_val // 2))
    y_test = np.array([0] * (n_val // 2) + [1] * (n_val - n_val // 2))

    rng.shuffle(y_train)
    rng.shuffle(y_val)
    rng.shuffle(y_test)

    return EmbeddingBundle(
        X_train=x_train,
        X_val=x_val,
        X_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


@pytest.mark.parametrize(
    ("model_type", "model_params"),
    [
        ("LR", {"max_iter": 100}),
        ("SVM", {"kernel_config": {"kernel": "linear"}, "C": 0.5}),
        ("RF", {"n_estimators": 20, "max_depth": 4}),
        ("KNN-2", {"n_neighbors": 3, "p_metric": {"p": 2, "metric": "minkowski"}}),
        ("XGB", {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.2}),
        (
            "MLP",
            {
                "num_epochs": 2,
                "batch_size": 8,
                "early_stopping_patience": 1,
                "learning_rate": 1e-3,
            },
        ),
    ],
)
def test_each_classifier_trains(model_type: str, model_params: dict):
    if model_type == "XGB":
        pytest.importorskip("xgboost")

    if model_type == "MLP":
        pytest.importorskip("torch")
        try:
            from src.training.models.mlp_protein_classifier import MLPProteinClassifier  # noqa: F401
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"MLP dependencies unavailable: {exc}")

    service = TrainingService()
    bundle = _binary_bundle()

    results = service.train(
        embedding_bundle=bundle,
        training_config={
            "model_types": [model_type],
            "model_params": {model_type: model_params},
            "metrics_average": "macro",
        },
    )

    key = (model_type, "ESM3c")
    assert key in results

    payload = results[key]
    assert "model" in payload
    assert "val_probs" in payload
    assert "metrics" in payload
    assert "validation" in payload["metrics"]
    assert "f1" in payload["metrics"]["validation"]
    assert payload["val_probs"].shape[0] == bundle.X_val["ESM3c"].shape[0]


@pytest.mark.parametrize(
    ("model_type", "model_params"),
    [
        ("LR", {"max_iter": 100}),
        ("SVM", {"kernel_config": {"kernel": "linear"}, "C": 0.5}),
        ("RF", {"n_estimators": 20, "max_depth": 4}),
        ("KNN-2", {"n_neighbors": 3, "p_metric": {"p": 2, "metric": "minkowski"}}),
        ("XGB", {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.2}),
        (
            "MLP",
            {
                "num_epochs": 2,
                "batch_size": 8,
                "early_stopping_patience": 1,
                "learning_rate": 1e-3,
            },
        ),
    ],
)
def test_each_classifier_trains_across_all_embedding_views(model_type: str, model_params: dict):
    if model_type == "XGB":
        pytest.importorskip("xgboost")

    if model_type == "MLP":
        pytest.importorskip("torch")
        try:
            from src.training.models.mlp_protein_classifier import MLPProteinClassifier  # noqa: F401
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"MLP dependencies unavailable: {exc}")

    service = TrainingService()
    bundle = _binary_bundle_multi_embedding()

    results = service.train(
        embedding_bundle=bundle,
        training_config={
            "model_types": [model_type],
            "model_params": {model_type: model_params},
            "metrics_average": "macro",
        },
    )

    for embedding_name in bundle.X_train:
        key = (model_type, embedding_name)
        assert key in results

        payload = results[key]
        assert "model" in payload
        assert "val_probs" in payload
        assert "metrics" in payload
        assert "validation" in payload["metrics"]
        assert "f1" in payload["metrics"]["validation"]
        assert payload["val_probs"].shape[0] == bundle.X_val[embedding_name].shape[0]
