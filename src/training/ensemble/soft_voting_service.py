from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypedDict

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.training.decision.decision_policy import decide
from src.training.probability.probability_adapter import ProbabilityAdapter


class SoftVotingContractError(RuntimeError):
    """Raised when SoftVotingService contract preconditions or runtime invariants fail."""


class EnsembleMode(str, Enum):
    GLOBAL_SOFT = "global_soft"
    MAJORITY_GLOBAL = "majority_global"
    MAJORITY_BY_EMBEDDING = "majority_by_embedding"
    MAJORITY_BY_CLASSIFIER = "majority_by_classifier"


class WeightingStrategyType(str, Enum):
    UNIFORM = "uniform"
    VALIDATION_SCORE_BASED = "validation_score_based"
    TRAINABLE_WEIGHTS = "trainable_weights"


class SoftVotingOutput(TypedDict):
    probabilities: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelArtifact:
    """Minimal model artifact contract for ensemble inference.

    Invariants:
    - ``model`` exposes ``predict_proba(X)`` and must be inference-ready.
    - ``classes`` order must match model output columns.
    - ``num_classes`` must equal ``len(classes)``.
    - ``problem_type`` must match across all artifacts in an ensemble execution.
    - ``metadata`` is preserved and propagated into ensemble metadata output.
    """

    model: Any
    problem_type: str
    classes: list[Any]
    num_classes: int
    normalization: str
    threshold_policy: dict[str, Any]
    classifier_name: str
    embedding_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, model: Any, metadata: Mapping[str, Any]) -> "ModelArtifact":
        classes = list(metadata.get("classes", []))
        classifier_name = str(metadata.get("classifier_name", metadata.get("classifier", "")))
        return cls(
            model=model,
            problem_type=str(metadata.get("problem_type", "binary")),
            classes=classes,
            num_classes=int(metadata.get("num_classes", len(classes))),
            normalization=str(metadata.get("normalization", "none")),
            threshold_policy=dict(metadata.get("threshold_policy", {}))
            if isinstance(metadata.get("threshold_policy", {}), Mapping)
            else {},
            classifier_name=classifier_name,
            embedding_name=str(metadata.get("embedding_name", "")),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class EnsembleSelectionConfig:
    embeddings: list[str] | None = None
    classifiers: list[str] | None = None


@dataclass(frozen=True)
class WeightingConfig:
    strategy: WeightingStrategyType = WeightingStrategyType.UNIFORM
    metric: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration contract for ensemble execution.

    Example YAML shape:
    ensemble:
      enabled: true
      mode: global_soft
      selection:
        embeddings: ["ESM2", "ProtT5"]
        classifiers: ["LR", "XGB"]
      weighting:
        strategy: validation_score_based
        metric: f1_macro
        params:
          normalize_scores: true
      prechecks:
        enforce_same_normalization: false
        min_models: 2
    """

    enabled: bool = True
    mode: EnsembleMode = EnsembleMode.GLOBAL_SOFT
    selection: EnsembleSelectionConfig = field(default_factory=EnsembleSelectionConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    enforce_same_normalization: bool = False
    min_models: int = 2


class WeightTrainer(Protocol):
    """Learns ensemble weights from validation probabilities.

    Contract:
    - Input shape for validation probabilities must be ``(n_models, n_samples, n_classes)``.
    - Returned weights must have shape ``(n_models,)``.
    - Weights must be finite and non-negative.
    """

    def fit(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        model_identifiers: Sequence[str],
        problem_type: str,
        classes: Sequence[Any],
        metric: str | None,
        params: Mapping[str, Any],
    ) -> np.ndarray:
        ...


class WeightingStrategy(ABC):
    """Strategy interface for ensemble weighting.

    Extension point:
    - New strategies subclass this class and implement ``fit_from_validation``.
    - ``SoftVotingService`` consumes this interface only and avoids branching logic.
    """

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None
        self._model_ids: list[str] = []

    def initialize(self, model_identifiers: Sequence[str]) -> None:
        self._model_ids = list(model_identifiers)
        if not self._model_ids:
            raise SoftVotingContractError("Weighting strategy received no model identifiers")

    @abstractmethod
    def fit_from_validation(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        artifacts: Sequence[ModelArtifact],
    ) -> None:
        """Estimate strategy weights using validation probabilities and labels."""

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise SoftVotingContractError("Ensemble weights are not initialized")
        return np.asarray(self._weights, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "strategy": self.__class__.__name__,
            "weights": self.get_weights().tolist(),
            "model_identifiers": list(self._model_ids),
        }

    @staticmethod
    def _normalize_weights(raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 1:
            raise SoftVotingContractError(f"Weights must be 1D, got shape={tuple(arr.shape)}")
        if arr.size == 0:
            raise SoftVotingContractError("Weights cannot be empty")
        if not np.all(np.isfinite(arr)):
            raise SoftVotingContractError("Weights must be finite")
        if np.any(arr < 0):
            raise SoftVotingContractError("Weights must be non-negative")
        total = float(arr.sum())
        if total <= 0:
            raise SoftVotingContractError("Sum of weights must be > 0")
        return arr / total


class UniformWeightingStrategy(WeightingStrategy):
    """Assign equal weights to all models."""

    def fit_from_validation(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        artifacts: Sequence[ModelArtifact],
    ) -> None:
        del validation_probabilities
        del validation_labels
        del artifacts
        n_models = len(self._model_ids)
        self._weights = np.ones(n_models, dtype=np.float64) / float(n_models)


class ValidationScoreWeightingStrategy(WeightingStrategy):
    """Weight models according to validation F1 and normalize scores to probabilities."""

    def __init__(self, metric: str = "f1") -> None:
        super().__init__()
        self.metric = metric

    def fit_from_validation(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        artifacts: Sequence[ModelArtifact],
    ) -> None:
        if validation_probabilities.ndim != 3:
            raise SoftVotingContractError(
                "Validation probabilities must have shape (n_models, n_samples, n_classes)"
            )
        if not artifacts:
            raise SoftVotingContractError("Validation-score strategy requires artifact metadata")

        problem_type = str(artifacts[0].problem_type)
        classes = list(artifacts[0].classes)
        scores: list[float] = []
        y_true = np.asarray(validation_labels)

        if problem_type == "multilabel":
            y_true_bin = self._to_multilabel_matrix(y_true, classes)
        else:
            y_true_bin = None

        for model_probs in validation_probabilities:
            if problem_type == "multilabel":
                y_pred_bin = (np.asarray(model_probs) >= 0.5).astype(int)
                score = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
            else:
                y_pred = np.argmax(model_probs, axis=1)
                score = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            scores.append(max(score, 0.0))
        self._weights = self._normalize_weights(np.asarray(scores, dtype=np.float64))

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["metric"] = self.metric
        return payload

    @staticmethod
    def _to_multilabel_matrix(values: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
        values_array = np.asarray(values, dtype=object)
        if values_array.ndim == 2 and np.issubdtype(values_array.dtype, np.number):
            return values_array.astype(int)

        converted: list[list[Any]] = []
        for item in values_array:
            if isinstance(item, np.ndarray):
                converted.append(item.tolist())
            elif isinstance(item, (list, tuple, set)):
                converted.append(list(item))
            elif item is None:
                converted.append([])
            else:
                converted.append([item])

        class_list = list(classes)
        mlb = MultiLabelBinarizer(classes=class_list if class_list else None)
        if class_list:
            mlb.fit([class_list])
        else:
            mlb.fit(converted)
        return mlb.transform(converted)


class TrainableWeightsStrategy(WeightingStrategy):
    """Delegates weight learning to a ``WeightTrainer`` component."""

    def __init__(
        self,
        trainer: WeightTrainer,
        problem_type: str,
        classes: Sequence[Any],
        metric: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.problem_type = problem_type
        self.classes = list(classes)
        self.metric = metric
        self.params = dict(params or {})

    def fit_from_validation(
        self,
        validation_probabilities: np.ndarray,
        validation_labels: np.ndarray,
        artifacts: Sequence[ModelArtifact],
    ) -> None:
        del artifacts
        learned = self.trainer.fit(
            validation_probabilities=validation_probabilities,
            validation_labels=np.asarray(validation_labels),
            model_identifiers=list(self._model_ids),
            problem_type=self.problem_type,
            classes=self.classes,
            metric=self.metric,
            params=self.params,
        )
        self._weights = self._normalize_weights(np.asarray(learned, dtype=np.float64))

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["metric"] = self.metric
        payload["params"] = dict(self.params)
        return payload


class MajorityVotingService(Protocol):
    """External extension point for majority-based ensemble modes."""

    def predict(
        self,
        mode: EnsembleMode,
        model_predictions: Mapping[str, np.ndarray],
        artifacts: Sequence[ModelArtifact],
    ) -> np.ndarray:
        ...


class SimpleMajorityVotingService:
    """Reference majority-voting implementation used by orchestration layers.

    Notes:
    - Accepts per-model (or pre-grouped) prediction vectors.
    - Resolves each sample by the most common label across voters.
    - Deterministic tie-breaker: smallest numeric label.
    """

    def predict(
        self,
        mode: EnsembleMode,
        model_predictions: Mapping[str, np.ndarray],
        artifacts: Sequence[ModelArtifact],
    ) -> np.ndarray:
        del mode
        del artifacts
        if not model_predictions:
            raise SoftVotingContractError("Majority voting requires at least one prediction source")

        vectors = [np.asarray(value) for value in model_predictions.values()]
        first_shape = vectors[0].shape
        if len(first_shape) != 1:
            raise SoftVotingContractError(
                "Majority voting expects 1D label vectors from each prediction source"
            )
        for vector in vectors[1:]:
            if vector.shape != first_shape:
                raise SoftVotingContractError(
                    "All majority voting prediction vectors must share the same shape"
                )

        stacked = np.stack(vectors, axis=0)
        if not np.issubdtype(stacked.dtype, np.integer):
            unique = np.unique(stacked)
            mapping = {value: idx for idx, value in enumerate(unique.tolist())}
            inverse = {idx: value for value, idx in mapping.items()}
            encoded = np.vectorize(lambda val: mapping[val])(stacked)
            voted = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, encoded)
            return np.asarray([inverse[int(item)] for item in voted], dtype=object)

        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, stacked)


def validate_soft_voting_contract(
    model_artifacts: Sequence[ModelArtifact],
    *,
    min_models: int = 2,
    enforce_same_normalization: bool = False,
) -> tuple[str, list[Any], int]:
    """Validate preconditions required by the soft voting contract.

    Mandatory validations:
    - At least two models (or ``min_models``) are provided.
    - All model artifacts share identical ``problem_type``.
    - All model artifacts share identical ordered ``classes``.
    - All model artifacts share identical ``num_classes``.
    - Every model exposes callable ``predict_proba``.
    - Optional normalization consistency check.

    Raises:
    - ``SoftVotingContractError`` with explicit reason on first violation.
    """

    if len(model_artifacts) < int(min_models):
        raise SoftVotingContractError(
            f"Soft voting requires at least {min_models} models, got {len(model_artifacts)}"
        )

    first = model_artifacts[0]
    shared_problem_type = str(first.problem_type)
    shared_classes = list(first.classes)
    shared_num_classes = int(first.num_classes)
    shared_normalization = str(first.normalization)

    if shared_num_classes != len(shared_classes):
        raise SoftVotingContractError(
            "Invalid artifact metadata: num_classes must equal len(classes) for first model"
        )

    for artifact in model_artifacts:
        if str(artifact.problem_type) != shared_problem_type:
            raise SoftVotingContractError(
                "Inconsistent problem_type across model artifacts"
            )
        if list(artifact.classes) != shared_classes:
            raise SoftVotingContractError(
                "Inconsistent class ordering across model artifacts"
            )
        if int(artifact.num_classes) != shared_num_classes:
            raise SoftVotingContractError(
                "Inconsistent num_classes across model artifacts"
            )
        predict_proba = getattr(artifact.model, "predict_proba", None)
        if predict_proba is None or not callable(predict_proba):
            raise SoftVotingContractError(
                f"Model does not expose callable predict_proba: {artifact.classifier_name}/{artifact.embedding_name}"
            )
        if int(artifact.num_classes) != len(list(artifact.classes)):
            raise SoftVotingContractError(
                "Invalid artifact metadata: num_classes must equal len(classes)"
            )
        if enforce_same_normalization and str(artifact.normalization) != shared_normalization:
            raise SoftVotingContractError(
                "Inconsistent normalization across model artifacts while enforce_same_normalization=true"
            )

    return shared_problem_type, shared_classes, shared_num_classes


def create_weighting_strategy(
    weighting_config: WeightingConfig,
    *,
    problem_type: str,
    classes: Sequence[Any],
    weight_trainer: WeightTrainer | None = None,
) -> WeightingStrategy:
    """Factory for pluggable weighting strategies."""

    strategy_type = WeightingStrategyType(weighting_config.strategy)
    if strategy_type == WeightingStrategyType.UNIFORM:
        return UniformWeightingStrategy()
    if strategy_type == WeightingStrategyType.VALIDATION_SCORE_BASED:
        return ValidationScoreWeightingStrategy(metric=str(weighting_config.metric or "f1"))
    if strategy_type == WeightingStrategyType.TRAINABLE_WEIGHTS:
        if weight_trainer is None:
            raise SoftVotingContractError(
                "trainable_weights strategy requires a WeightTrainer instance"
            )
        return TrainableWeightsStrategy(
            trainer=weight_trainer,
            problem_type=problem_type,
            classes=classes,
            metric=weighting_config.metric,
            params=weighting_config.params,
        )
    raise SoftVotingContractError(f"Unsupported weighting strategy: {strategy_type}")


class SoftVotingService:
    """Production-grade contract service for ensemble soft voting.

    Architectural invariants:
    - SoftVotingService MUST NOT train base classifiers.
    - SoftVotingService MUST NOT implement threshold logic directly.
    - Decision conversion is delegated to ``DecisionPolicy`` (`decide`).
    - Probability consistency is enforced via ``ProbabilityAdapter``.
    - Weight training is delegated to a ``WeightTrainer`` through strategy plug-ins.

    Public flow:
    1) Construct service with validated model artifacts and config.
    2) Optionally call ``fit_with_validation`` for non-uniform weighting.
    3) Call ``predict`` on validation/test features.

    Returned payload contract:
    {
        "probabilities": np.ndarray,
        "labels": np.ndarray,
        "metadata": dict,
    }
    """

    def __init__(
        self,
        model_artifacts: Sequence[ModelArtifact],
        config: EnsembleConfig,
        *,
        probability_adapter: type[ProbabilityAdapter] = ProbabilityAdapter,
        decision_policy: Any = decide,
        weighting_strategy: WeightingStrategy | None = None,
        weight_trainer: WeightTrainer | None = None,
        majority_voting_service: MajorityVotingService | None = None,
    ) -> None:
        self._config = config
        selected = self._apply_selection(model_artifacts, config.selection)
        self._problem_type, self._classes, self._num_classes = validate_soft_voting_contract(
            selected,
            min_models=config.min_models,
            enforce_same_normalization=config.enforce_same_normalization,
        )
        self._artifacts = list(selected)
        self._probability_adapter = probability_adapter
        self._decision_policy = decision_policy
        self._majority_voting_service = majority_voting_service
        self._weighting_strategy = weighting_strategy or create_weighting_strategy(
            config.weighting,
            problem_type=self._problem_type,
            classes=self._classes,
            weight_trainer=weight_trainer,
        )
        self._weighting_strategy.initialize(self._model_identifiers)

    @property
    def _model_identifiers(self) -> list[str]:
        return [f"{a.classifier_name}::{a.embedding_name}" for a in self._artifacts]

    def collect_validation_probabilities(self, X_val: Any) -> np.ndarray:
        """Collect validated per-model probabilities on a validation set.

        Returns:
        - Shape ``(n_models, n_samples, n_classes)``.
        """

        model_probabilities: list[np.ndarray] = []
        for artifact in self._artifacts:
            raw = np.asarray(artifact.model.predict_proba(X_val))
            canonical = self._probability_adapter.to_canonical(
                raw_output=raw,
                problem_type=self._problem_type,
                classes=artifact.classes,
                context=f"ensemble/validation/{artifact.classifier_name}/{artifact.embedding_name}",
            )
            model_probabilities.append(canonical)

        stacked = np.stack(model_probabilities, axis=0)
        if stacked.ndim != 3 or stacked.shape[2] != self._num_classes:
            raise SoftVotingContractError(
                "Validation probabilities must stack to shape (n_models, n_samples, n_classes)"
            )
        return stacked

    def fit_with_validation(self, X_val: Any, y_val: np.ndarray) -> None:
        """Estimate strategy weights using validation probabilities.

        Separation guarantee:
        - This method only estimates ensemble weights.
        - It does not fit/retrain base models.
        """

        val_probs = self.collect_validation_probabilities(X_val)
        self._weighting_strategy.fit_from_validation(
            validation_probabilities=val_probs,
            validation_labels=np.asarray(y_val),
            artifacts=self._artifacts,
        )
        weights = self._weighting_strategy.get_weights()
        if weights.shape != (len(self._artifacts),):
            raise SoftVotingContractError(
                f"Weight shape mismatch expected={(len(self._artifacts),)} got={tuple(weights.shape)}"
            )

    def predict(self, X: Any) -> SoftVotingOutput:
        """Run ensemble inference over validation/test features.

        High-level algorithm:
        1) Validate and collect per-model probabilities.
        2) Route by ensemble mode:
           - global_soft: weighted probability average.
           - majority_*: delegate to majority service for label vote fusion.
        3) Validate final probabilities with ProbabilityAdapter.
        4) Convert probabilities to labels via DecisionPolicy.
        5) Return probabilities, labels, metadata.
        """

        if not self._config.enabled:
            raise SoftVotingContractError("Ensemble disabled by configuration")

        per_model_probs = self._collect_inference_probabilities(X)
        mode = self._config.mode

        if mode == EnsembleMode.GLOBAL_SOFT:
            ensemble_probs = self._aggregate_soft_probabilities(per_model_probs)
            labels = self._decision_policy(
                probs=ensemble_probs,
                problem_type=self._problem_type,
                threshold_config={"default": 0.5},
            )
            return {
                "probabilities": ensemble_probs,
                "labels": np.asarray(labels),
                "metadata": self._build_metadata(weights=self._weighting_strategy.get_weights(), mode=mode),
            }

        labels = self._predict_majority_labels(per_model_probs, mode=mode)
        majority_probs = self._probability_adapter.to_canonical(
            raw_output=self._aggregate_soft_probabilities(per_model_probs),
            problem_type=self._problem_type,
            classes=self._classes,
            context=f"ensemble/{mode.value}/canonical",
        )
        return {
            "probabilities": majority_probs,
            "labels": np.asarray(labels),
            "metadata": self._build_metadata(weights=self._weighting_strategy.get_weights(), mode=mode),
        }

    def _collect_inference_probabilities(self, X: Any) -> np.ndarray:
        model_probabilities: list[np.ndarray] = []
        for artifact in self._artifacts:
            raw = np.asarray(artifact.model.predict_proba(X))
            canonical = self._probability_adapter.to_canonical(
                raw_output=raw,
                problem_type=self._problem_type,
                classes=artifact.classes,
                context=f"ensemble/inference/{artifact.classifier_name}/{artifact.embedding_name}",
            )
            model_probabilities.append(canonical)
        probs = np.stack(model_probabilities, axis=0)
        if probs.ndim != 3:
            raise SoftVotingContractError(f"Expected 3D stacked probabilities, got shape={tuple(probs.shape)}")
        return probs

    def _aggregate_soft_probabilities(self, per_model_probabilities: np.ndarray) -> np.ndarray:
        weights = self._weighting_strategy.get_weights()
        if weights.shape != (per_model_probabilities.shape[0],):
            raise SoftVotingContractError(
                "Weight vector length must match number of models in probability tensor"
            )

        weighted = np.tensordot(weights, per_model_probabilities, axes=(0, 0))
        canonical = self._probability_adapter.to_canonical(
            raw_output=weighted,
            problem_type=self._problem_type,
            classes=self._classes,
            context="ensemble/aggregated",
        )
        return canonical

    def _predict_majority_labels(self, per_model_probabilities: np.ndarray, *, mode: EnsembleMode) -> np.ndarray:
        if self._majority_voting_service is None:
            raise SoftVotingContractError(
                f"Mode '{mode.value}' requires an injected majority_voting_service"
            )

        predictions: dict[str, np.ndarray] = {}
        for index, artifact in enumerate(self._artifacts):
            probs = per_model_probabilities[index]
            predictions[f"{artifact.classifier_name}::{artifact.embedding_name}"] = np.asarray(
                self._decision_policy(
                    probs=probs,
                    problem_type=self._problem_type,
                    threshold_config=artifact.threshold_policy,
                )
            )

        if mode == EnsembleMode.MAJORITY_BY_EMBEDDING:
            predictions = self._group_predictions(predictions, group_by="embedding")
        elif mode == EnsembleMode.MAJORITY_BY_CLASSIFIER:
            predictions = self._group_predictions(predictions, group_by="classifier")

        return np.asarray(
            self._majority_voting_service.predict(
                mode=mode,
                model_predictions=predictions,
                artifacts=self._artifacts,
            )
        )

    def _group_predictions(self, predictions: Mapping[str, np.ndarray], *, group_by: str) -> dict[str, np.ndarray]:
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for artifact in self._artifacts:
            key = artifact.embedding_name if group_by == "embedding" else artifact.classifier_name
            predictions_key = f"{artifact.classifier_name}::{artifact.embedding_name}"
            buckets[key].append(np.asarray(predictions[predictions_key]))
        grouped: dict[str, np.ndarray] = {}
        for key, members in buckets.items():
            stacked = np.stack(members, axis=0)
            grouped[key] = np.asarray(np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, stacked))
        return grouped

    @staticmethod
    def _apply_selection(
        model_artifacts: Sequence[ModelArtifact],
        selection: EnsembleSelectionConfig,
    ) -> list[ModelArtifact]:
        selected = list(model_artifacts)
        if selection.embeddings:
            allowed = set(selection.embeddings)
            selected = [artifact for artifact in selected if artifact.embedding_name in allowed]
        if selection.classifiers:
            allowed = set(selection.classifiers)
            selected = [artifact for artifact in selected if artifact.classifier_name in allowed]
        if not selected:
            raise SoftVotingContractError("No models selected for ensemble after applying selection filters")
        return selected

    def _build_metadata(self, *, weights: np.ndarray, mode: EnsembleMode) -> dict[str, Any]:
        return {
            "problem_type": self._problem_type,
            "classes": list(self._classes),
            "num_classes": self._num_classes,
            "ensemble": {
                "enabled": self._config.enabled,
                "mode": mode.value,
                "model_count": len(self._artifacts),
                "selection": {
                    "embeddings": list(self._config.selection.embeddings or []),
                    "classifiers": list(self._config.selection.classifiers or []),
                },
                "weighting": self._weighting_strategy.describe(),
                "weights": np.asarray(weights, dtype=np.float64).tolist(),
            },
            "models": [
                {
                    "classifier_name": artifact.classifier_name,
                    "embedding_name": artifact.embedding_name,
                    "problem_type": artifact.problem_type,
                    "classes": list(artifact.classes),
                    "num_classes": artifact.num_classes,
                    "normalization": artifact.normalization,
                    "threshold_policy": dict(artifact.threshold_policy),
                    "metadata": dict(artifact.metadata),
                }
                for artifact in self._artifacts
            ],
            "normalization": [artifact.normalization for artifact in self._artifacts],
            "threshold_policy": [dict(artifact.threshold_policy) for artifact in self._artifacts],
        }