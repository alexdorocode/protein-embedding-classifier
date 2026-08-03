from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.training.ensemble.soft_voting_service import (
    EnsembleConfig,
    EnsembleMode,
    EnsembleSelectionConfig,
    ModelArtifact,
    SoftVotingContractError,
    SoftVotingService,
    WeightingConfig,
    WeightingStrategyType,
    validate_soft_voting_contract,
)


class _FixedProbabilityModel:
    def __init__(self, probs: np.ndarray):
        self._probs = np.asarray(probs, dtype=np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = np.asarray(X).shape[0]
        if n_samples != self._probs.shape[0]:
            raise ValueError("Input sample size mismatch")
        return self._probs


@dataclass
class _DummyWeightTrainer:
    learned_weights: np.ndarray
    calls: int = 0

    def fit(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        model_identifiers,
        problem_type,
        classes,
        metric,
        params,
    ) -> np.ndarray:
        self.calls += 1
        assert validation_probabilities.ndim == 3
        assert validation_labels.ndim == 1
        assert len(model_identifiers) == validation_probabilities.shape[0]
        assert problem_type in {"binary", "multiclass", "multilabel"}
        assert len(classes) > 0
        return np.asarray(self.learned_weights, dtype=np.float64)


def _artifact(
    *,
    probs: np.ndarray,
    classifier: str,
    embedding: str,
    problem_type: str = "binary",
    classes: list[int] | None = None,
    normalization: str = "standard",
) -> ModelArtifact:
    classes = classes or [0, 1]
    return ModelArtifact(
        model=_FixedProbabilityModel(probs=probs),
        problem_type=problem_type,
        classes=list(classes),
        num_classes=len(classes),
        normalization=normalization,
        threshold_policy={"default": 0.5},
        classifier_name=classifier,
        embedding_name=embedding,
        metadata={"classifier_name": classifier, "embedding_name": embedding},
    )


def test_validate_soft_voting_contract_requires_two_models():
    artifacts = [
        _artifact(
            probs=np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
            classifier="LR",
            embedding="ESM2",
        )
    ]

    with pytest.raises(SoftVotingContractError, match="at least 2 models"):
        validate_soft_voting_contract(artifacts)


def test_validate_soft_voting_contract_rejects_mismatched_class_order():
    artifacts = [
        _artifact(
            probs=np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
            classifier="LR",
            embedding="ESM2",
            classes=[0, 1],
        ),
        _artifact(
            probs=np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32),
            classifier="RF",
            embedding="ESM2",
            classes=[1, 0],
        ),
    ]

    with pytest.raises(SoftVotingContractError, match="class ordering"):
        validate_soft_voting_contract(artifacts)


def test_soft_voting_service_global_soft_returns_contract_payload():
    x_test = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    artifacts = [
        _artifact(
            probs=np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=np.float32),
            classifier="LR",
            embedding="ESM2",
        ),
        _artifact(
            probs=np.array([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4]], dtype=np.float32),
            classifier="RF",
            embedding="ProtT5",
        ),
    ]
    config = EnsembleConfig(
        enabled=True,
        mode=EnsembleMode.GLOBAL_SOFT,
        selection=EnsembleSelectionConfig(),
        weighting=WeightingConfig(strategy=WeightingStrategyType.UNIFORM),
    )
    service = SoftVotingService(model_artifacts=artifacts, config=config)

    service.fit_with_validation(X_val=x_test, y_val=np.array([0, 1, 0]))
    output = service.predict(x_test)

    assert set(output.keys()) == {"probabilities", "labels", "metadata"}
    assert output["probabilities"].shape == (3, 2)
    assert output["labels"].shape == (3,)
    assert output["metadata"]["problem_type"] == "binary"
    assert output["metadata"]["num_classes"] == 2
    assert output["metadata"]["ensemble"]["mode"] == "global_soft"
    assert len(output["metadata"]["models"]) == 2


def test_validation_score_based_strategy_prefers_better_model():
    x_val = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y_val = np.array([0, 1, 0, 1])
    artifacts = [
        _artifact(
            probs=np.array([[0.95, 0.05], [0.05, 0.95], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
            classifier="LR",
            embedding="ESM2",
        ),
        _artifact(
            probs=np.array([[0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
            classifier="RF",
            embedding="ESM2",
        ),
    ]
    config = EnsembleConfig(
        enabled=True,
        mode=EnsembleMode.GLOBAL_SOFT,
        weighting=WeightingConfig(strategy=WeightingStrategyType.VALIDATION_SCORE_BASED, metric="f1_macro"),
    )
    service = SoftVotingService(model_artifacts=artifacts, config=config)

    service.fit_with_validation(X_val=x_val, y_val=y_val)
    output = service.predict(x_val)
    weights = np.asarray(output["metadata"]["ensemble"]["weights"], dtype=np.float64)

    assert weights.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_trainable_weights_strategy_delegates_to_weight_trainer():
    x_val = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y_val = np.array([0, 1, 1])
    trainer = _DummyWeightTrainer(learned_weights=np.array([0.1, 0.9], dtype=np.float64))
    artifacts = [
        _artifact(
            probs=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=np.float32),
            classifier="LR",
            embedding="ESM2",
        ),
        _artifact(
            probs=np.array([[0.7, 0.3], [0.1, 0.9], [0.4, 0.6]], dtype=np.float32),
            classifier="XGB",
            embedding="ProtT5",
        ),
    ]
    config = EnsembleConfig(
        enabled=True,
        mode=EnsembleMode.GLOBAL_SOFT,
        weighting=WeightingConfig(strategy=WeightingStrategyType.TRAINABLE_WEIGHTS, metric="f1_macro"),
    )
    service = SoftVotingService(model_artifacts=artifacts, config=config, weight_trainer=trainer)

    service.fit_with_validation(X_val=x_val, y_val=y_val)
    output = service.predict(x_val)

    assert trainer.calls == 1
    assert np.allclose(np.asarray(output["metadata"]["ensemble"]["weights"]), np.array([0.1, 0.9]))


def test_majority_mode_requires_majority_service():
    x_test = np.array([[1.0], [2.0]], dtype=np.float32)
    artifacts = [
        _artifact(
            probs=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
            classifier="LR",
            embedding="ESM2",
        ),
        _artifact(
            probs=np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float32),
            classifier="RF",
            embedding="ProtT5",
        ),
    ]
    config = EnsembleConfig(
        enabled=True,
        mode=EnsembleMode.MAJORITY_GLOBAL,
        weighting=WeightingConfig(strategy=WeightingStrategyType.UNIFORM),
    )
    service = SoftVotingService(model_artifacts=artifacts, config=config)
    service.fit_with_validation(X_val=x_test, y_val=np.array([0, 1]))

    with pytest.raises(SoftVotingContractError, match="majority_voting_service"):
        service.predict(x_test)
