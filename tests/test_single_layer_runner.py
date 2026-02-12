import numpy as np

from runners.single_layer import SingleLayerRunner


class FakeDB:
    def __init__(self):
        self.resolve_calls = []
        self.load_calls = []

    def resolve_embedding_type_id(self, embedding_type: str) -> int:
        self.resolve_calls.append(embedding_type)
        # Return a fixed ID so tests can assert on it indirectly via calls
        return 42

    def load_embeddings(self, embedding_type_id: int, layer_index: int):  # noqa: ARG002
        self.load_calls.append((embedding_type_id, layer_index))
        # Simple deterministic data depending on layer_index
        accessions = [f"P{layer_index}A", f"P{layer_index}B"]
        X = np.full((2, 3), float(layer_index), dtype=np.float32)
        return accessions, X


class FakeTask:
    def __init__(self):
        self.label_calls = []
        self.log_calls = []

    def get_labels(self, accessions):
        self.label_calls.append(list(accessions))
        # Deterministic labels, just for wiring test
        return [0] * len(accessions)

    def log(self, metrics, **kwargs):
        # Record everything so assertions can inspect
        self.log_calls.append({"metrics": metrics, "kwargs": kwargs})


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def fit_eval(self, X, y):
        self.calls.append((X, y))
        # Return a simple metrics dict that depends on y length
        return {"n_samples": len(y)}


def test_single_layer_runner_calls_dependencies_per_layer_without_aggregator():
    fake_db = FakeDB()
    fake_task = FakeTask()
    fake_clf = FakeClassifier()

    runner = SingleLayerRunner(db=fake_db, task=fake_task, classifier=fake_clf)

    layers = [0, 2, 5]
    embedding_type = "esm2_t33_650M"

    runner.run(embedding_type=embedding_type, layers=layers)

    # DB should be asked once for the embedding type id
    assert fake_db.resolve_calls == [embedding_type]

    # And once per layer for embeddings, with resolved id
    assert fake_db.load_calls == [(42, layer) for layer in layers]

    # Task should get labels once per layer, with the correct accessions
    assert len(fake_task.label_calls) == len(layers)
    for layer, accs in zip(layers, fake_task.label_calls):
        assert accs == [f"P{layer}A", f"P{layer}B"]

    # Classifier should be called once per layer with arrays of shape (2,3)
    assert len(fake_clf.calls) == len(layers)
    for X, y in fake_clf.calls:
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 3)
        assert y == [0, 0]

    # Task.log should be called once per layer with metrics and proper kwargs
    assert len(fake_task.log_calls) == len(layers)
    for call, layer in zip(fake_task.log_calls, layers):
        assert call["metrics"] == {"n_samples": 2}
        assert call["kwargs"]["embedding_type"] == embedding_type
        assert call["kwargs"]["layer"] == layer


def test_single_layer_runner_uses_aggregator_if_provided():
    fake_db = FakeDB()
    fake_task = FakeTask()
    fake_clf = FakeClassifier()

    def aggregator(X):
        # mark that aggregation happened by changing the shape/value
        return X.mean(axis=0, keepdims=True)

    runner = SingleLayerRunner(
        db=fake_db,
        task=fake_task,
        classifier=fake_clf,
        aggregator=aggregator,
    )

    layers = [1]
    runner.run(embedding_type="ESM3c", layers=layers)

    # Aggregator should have reduced to a single row
    assert len(fake_clf.calls) == 1
    X, y = fake_clf.calls[0]
    assert X.shape == (1, 3)
    # y still has one label per original accession
    assert y == [0, 0]
