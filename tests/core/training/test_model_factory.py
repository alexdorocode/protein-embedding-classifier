import sys
import types

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from protein_embedding_classifier.core.training.model_factory import ModelFactory
from protein_embedding_classifier.core.training.torch_wrapper import TorchTrainingWrapper


def test_model_factory_lr():
    factory = ModelFactory()
    model = factory.create("LR", params={"C": 0.5, "max_iter": 50})
    assert isinstance(model, LogisticRegression)


def test_model_factory_svm_with_kernel_config_flatten():
    factory = ModelFactory()
    model = factory.create(
        "SVM",
        params={"kernel_config": {"kernel": "rbf", "gamma": "scale"}, "C": 1.0},
    )
    assert isinstance(model, SVC)
    assert model.kernel == "rbf"
    assert model.probability is True


def test_model_factory_rf_with_bootstrap_config_flatten():
    factory = ModelFactory()
    model = factory.create(
        "RF",
        params={"bootstrap_config": {"bootstrap": False}, "n_estimators": 10},
    )
    assert isinstance(model, RandomForestClassifier)
    assert model.bootstrap is False


def test_model_factory_knn2_with_p_metric_flatten():
    factory = ModelFactory()
    model = factory.create(
        "KNN-2",
        params={"p_metric": {"p": 2, "metric": "minkowski"}, "n_neighbors": 3},
    )
    assert isinstance(model, KNeighborsClassifier)
    assert model.p == 2


def test_model_factory_xgb(monkeypatch):
    class FakeXGBClassifier:
        def __init__(self, max_depth=None, n_estimators=None, **kwargs):
            self.kwargs = kwargs
            self.max_depth = max_depth
            self.n_estimators = n_estimators

    fake_module = types.SimpleNamespace(XGBClassifier=FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)

    factory = ModelFactory()
    model = factory.create("XGB", params={"max_depth": 3, "n_estimators": 5})
    assert isinstance(model, FakeXGBClassifier)
    assert model.max_depth == 3
    assert model.n_estimators == 5


def test_model_factory_mlp_with_custom_layer_config():
    factory = ModelFactory()
    model = factory.create(
        "MLP",
        params={
            "custom_layer_config": {"custom_hidden_layers": [16, 8], "num_hidden_layers": 2},
            "learning_rate": 1e-3,
        },
        input_size=32,
        output_size=4,
    )

    assert isinstance(model, TorchTrainingWrapper)
    assert model.input_size == 32
    assert model.output_size == 4
    assert model.custom_hidden_layers == [16, 8]
