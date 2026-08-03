import types

import numpy as np

from src.training.training.torch_wrapper import TorchTrainingWrapper


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def to(self, _device):
        return self

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.data)


class FakeLoss:
    def __init__(self, value=0.5):
        self.value = float(value)

    def backward(self):
        return None

    def item(self):
        return self.value


class FakeOptimizer:
    def zero_grad(self):
        return None

    def step(self):
        return None


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeMLPClassifier:
    def __init__(self, *args, **kwargs):
        self.call_count = 0

    def to(self, _device):
        return self

    def train(self):
        return None

    def eval(self):
        return None

    def state_dict(self):
        return {"state": 1}

    def load_state_dict(self, _state):
        return None

    def parameters(self):
        return []

    def __call__(self, x_batch):
        n_samples = x_batch.shape[0]
        self.call_count += 1
        if self.call_count % 2 == 0:
            return FakeTensor(np.random.randn(n_samples, 1))
        return FakeTensor(np.random.randn(n_samples, 2))


class FakeTensorDataset:
    def __init__(self, *tensors):
        self.tensors = tensors


class FakeDataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset

    def __iter__(self):
        yield self.dataset.tensors

    def __len__(self):
        return 1


def _fake_torch_module():
    fake_torch = types.SimpleNamespace()
    fake_torch.float32 = "float32"
    fake_torch.long = "long"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.tensor = lambda data, dtype=None, device=None: FakeTensor(data)
    fake_torch.sigmoid = lambda t: FakeTensor(1.0 / (1.0 + np.exp(-t.data)))
    fake_torch.softmax = lambda t, dim=1: FakeTensor(np.exp(t.data) / np.exp(t.data).sum(axis=dim, keepdims=True))
    fake_torch.optim = types.SimpleNamespace(
        Adam=lambda params, lr: FakeOptimizer(),
        RMSprop=lambda params, lr: FakeOptimizer(),
        SGD=lambda params, lr: FakeOptimizer(),
        Adagrad=lambda params, lr: FakeOptimizer(),
    )
    fake_torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(TensorDataset=FakeTensorDataset, DataLoader=FakeDataLoader)
    )
    fake_torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(
            one_hot=lambda y, num_classes: FakeTensor(np.eye(num_classes)[y.data.astype(int)])
        )
    )

    fake_nn = types.SimpleNamespace(
        CrossEntropyLoss=lambda: (lambda logits, y: FakeLoss(0.4)),
        BCEWithLogitsLoss=lambda: (lambda logits, y: FakeLoss(0.5)),
    )
    return fake_torch, fake_nn


def test_torch_wrapper_binary_forward_pass_predict_proba_shape():
    wrapper = TorchTrainingWrapper()
    fake_torch, _ = _fake_torch_module()
    wrapper._torch = fake_torch
    wrapper.device = "cpu"

    class BinaryModel:
        def eval(self):
            return None

        def __call__(self, x):
            return FakeTensor(np.zeros((x.shape[0], 1)))

    wrapper.model = BinaryModel()
    probs = wrapper.predict_proba(np.zeros((5, 3), dtype=np.float32))

    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_torch_wrapper_early_stopping_triggers(monkeypatch):
    wrapper = TorchTrainingWrapper(num_epochs=20, early_stopping_patience=1)
    fake_torch, fake_nn = _fake_torch_module()

    monkeypatch.setattr(TorchTrainingWrapper, "_get_torch_modules", staticmethod(lambda: (fake_torch, fake_nn)))

    fake_mlp_module = types.SimpleNamespace(MLPProteinClassifier=FakeMLPClassifier)
    import sys

    monkeypatch.setitem(sys.modules, "protein_embedding_classifier.classifiers.mlp_protein_classifier", fake_mlp_module)

    eval_calls = {"count": 0}

    def fake_eval_loss(self, criterion, x_val_tensor, y_val_tensor, num_classes):
        eval_calls["count"] += 1
        return float(eval_calls["count"])

    monkeypatch.setattr(TorchTrainingWrapper, "_evaluate_val_loss", fake_eval_loss)

    X_train = np.random.randn(8, 4).astype(np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X_val = np.random.randn(4, 4).astype(np.float32)
    y_val = np.array([0, 1, 0, 1])

    wrapper.fit(X_train, y_train, X_val, y_val)

    assert eval_calls["count"] < wrapper.num_epochs


def test_torch_wrapper_predict_proba_multiclass_shape():
    wrapper = TorchTrainingWrapper()
    fake_torch, _ = _fake_torch_module()
    wrapper._torch = fake_torch
    wrapper.device = "cpu"

    class MultiClassModel:
        def eval(self):
            return None

        def __call__(self, x):
            return FakeTensor(np.ones((x.shape[0], 3)))

    wrapper.model = MultiClassModel()
    probs = wrapper.predict_proba(np.zeros((6, 5), dtype=np.float32))

    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
