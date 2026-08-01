import numpy as np
import pytest

from protein_embedding_classifier.core.decision.decision_policy import decide
from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.probability.probability_adapter import ProbabilityAdapter
from protein_embedding_classifier.core.training.training_service import TrainingService


def test_probability_adapter_binary_shapes_to_n_by_2():
    one_dim = ProbabilityAdapter.to_canonical(np.array([0.1, 0.8], dtype=np.float32), "binary", classes=[0, 1])
    two_dim = ProbabilityAdapter.to_canonical(np.array([[0.1], [0.8]], dtype=np.float32), "binary", classes=[0, 1])

    assert one_dim.shape == (2, 2)
    assert two_dim.shape == (2, 2)
    assert np.allclose(one_dim[:, 0] + one_dim[:, 1], np.ones(2))


def test_probability_adapter_rejects_non_normalized_multiclass():
    with pytest.raises(ValueError, match="sum to 1"):
        ProbabilityAdapter.to_canonical(
            np.array([[0.6, 0.6], [0.2, 0.2]], dtype=np.float32),
            "multiclass",
            classes=[0, 1],
        )


def test_decision_policy_uses_classifier_embedding_threshold():
    probs = np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float32)
    preds = decide(
        probs=probs,
        problem_type="binary",
        threshold_config={
            "default": 0.5,
            "classifier_embedding": {"LR::ESM3c": 0.7},
            "classifier_name": "LR",
            "embedding_name": "ESM3c",
        },
    )

    assert preds.tolist() == [0, 0]


def test_training_service_raises_if_model_lacks_predict_proba():
    class NoProbaModel:
        def fit(self, X, y):
            return None

    class NoProbaFactory:
        def create(self, model_type, params=None, input_size=None, output_size=None):
            return NoProbaModel()

    bundle = EmbeddingBundle(
        X_train={"ESM3c": np.random.randn(8, 4).astype(np.float32)},
        X_val={"ESM3c": np.random.randn(4, 4).astype(np.float32)},
        X_test={"ESM3c": np.random.randn(4, 4).astype(np.float32)},
        y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        y_val=np.array([0, 1, 0, 1]),
        y_test=np.array([0, 1, 0, 1]),
    )

    service = TrainingService(model_factory=NoProbaFactory())

    with pytest.raises(ValueError, match="predict_proba"):
        service.train(
            embedding_bundle=bundle,
            training_config={"model_types": ["LR"]},
        )
