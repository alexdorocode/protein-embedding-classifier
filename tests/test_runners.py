import numpy as np

from protein_embedding_classifier.runners.single_layer import SingleLayerRunner
from protein_embedding_classifier.runners.selected_layers import SelectedLayersRunner
from protein_embedding_classifier.runners.attention_over_layers import AttentionOverLayersRunner


class FakeDB:
    def __init__(self):
        self.resolve_calls = []
        self.load_calls = []

    def resolve_embedding_type_id(self, embedding_type: str) -> int:
        self.resolve_calls.append(embedding_type)
        return 7

    def load_embeddings(self, embedding_type_id: int, layer_index: int):  # noqa: ARG002
        self.load_calls.append((embedding_type_id, layer_index))
        accessions = ["P1", "P2"]
        X = np.full((2, 3), float(layer_index), dtype=np.float32)
        return accessions, X


class FakeStore:
    def __init__(self, available_layers=None):
        self.available_layers = list(available_layers or [0, 1, 2])
        self.get_calls = []

    def get_available_layers(self, embedding_type: str):  # noqa: ARG002
        return list(self.available_layers)

    def get(self, embedding_type: str, layer: int):  # noqa: ARG002
        self.get_calls.append((embedding_type, layer))
        accessions = ["P1", "P2"]
        X = np.full((2, 4), float(layer), dtype=np.float32)
        return accessions, X


class FakeTask:
    def __init__(self):
        self.label_calls = []
        self.log_calls = []

    def get_labels(self, accessions):
        self.label_calls.append(list(accessions))
        return [0] * len(accessions)

    def log(self, metrics, **kwargs):
        self.log_calls.append({"metrics": metrics, "kwargs": kwargs})


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def fit_eval(self, X, y):
        self.calls.append((X, y))
        return {"n_samples": len(y), "shape": tuple(X.shape)}


class FakeAttentionClassifier(FakeClassifier):
    attention_name = "layer_attention"


class FakeAggregator:
    def __init__(self, name="agg_mean"):
        self._name = name
        self.calls = []

    @property
    def name(self):
        return self._name

    def aggregate(self, X_layers):
        self.calls.append(X_layers)
        # simple reduce over layer axis
        return X_layers.mean(axis=0)


def test_single_layer_runner_unit_wiring():
    db = FakeDB()
    task = FakeTask()
    clf = FakeClassifier()

    runner = SingleLayerRunner(db=db, task=task, classifier=clf)

    layers = [0, 2]
    embedding_type = "dummy_model"

    runner.run(embedding_type=embedding_type, layers=layers)

    # DB resolve called once, load_embeddings once per layer
    assert db.resolve_calls == [embedding_type]
    assert db.load_calls == [(7, l) for l in layers]

    # Labels + classifier + logging once per layer
    assert len(task.label_calls) == len(layers)
    assert len(clf.calls) == len(layers)
    assert len(task.log_calls) == len(layers)


def test_selected_layers_runner_aggregates_and_logs():
    store = FakeStore()
    task = FakeTask()
    clf = FakeClassifier()
    agg = FakeAggregator(name="mean")

    runner = SelectedLayersRunner(store=store, task=task, classifier=clf, aggregator=agg)

    layers = [0, 2]
    metrics = runner.run(embedding_type="dummy_model", layers=layers)

    # Store.get called once per requested layer
    assert store.get_calls == [("dummy_model", l) for l in layers]

    # Aggregator sees stacked (L, N, D) input
    assert len(agg.calls) == 1
    X_layers = agg.calls[0]
    assert X_layers.shape == (len(layers), 2, 4)

    # Classifier called once on aggregated (N, D)
    assert len(clf.calls) == 1
    X, y = clf.calls[0]
    assert X.shape == (2, 4)
    assert y == [0, 0]

    # Task.log called once with embedding_type, layers, and aggregation name
    assert len(task.log_calls) == 1
    log_entry = task.log_calls[0]
    assert log_entry["kwargs"]["embedding_type"] == "dummy_model"
    assert log_entry["kwargs"]["layers"] == layers
    assert log_entry["kwargs"]["aggregation"] == "mean"

    # Metrics are returned from classifier
    assert metrics["n_samples"] == 2


def test_attention_over_layers_runner_with_explicit_layers():
    store = FakeStore(available_layers=[0, 1, 2])
    task = FakeTask()
    clf = FakeAttentionClassifier()

    runner = AttentionOverLayersRunner(db=store, task=task, classifier=clf)
    # Attach store attribute used in run
    runner.store = store

    layers = [1, 2]
    metrics = runner.run(embedding_type="dummy_model", layers=layers)

    # Store.get called for specified layers only
    assert store.get_calls == [("dummy_model", l) for l in layers]

    # Classifier sees full (L, N, D) tensor
    assert len(clf.calls) == 1
    X_layers, y = clf.calls[0]
    assert X_layers.shape == (len(layers), 2, 4)
    assert y == [0, 0]

    # Task.log called once with attention_name
    assert len(task.log_calls) == 1
    log_entry = task.log_calls[0]
    assert log_entry["kwargs"]["embedding_type"] == "dummy_model"
    assert log_entry["kwargs"]["layers"] == layers
    assert log_entry["kwargs"]["aggregation"] == "layer_attention"

    assert metrics["n_samples"] == 2


def test_attention_over_layers_runner_uses_all_available_layers_when_none():
    store = FakeStore(available_layers=[0, 3, 5])
    task = FakeTask()
    clf = FakeAttentionClassifier()

    runner = AttentionOverLayersRunner(db=store, task=task, classifier=clf)
    runner.store = store

    metrics = runner.run(embedding_type="dummy_model", layers=None)

    # When layers=None, runner should query all available layers from the store
    assert store.get_calls == [("dummy_model", l) for l in [0, 3, 5]]
    assert metrics["n_samples"] == 2
