from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from protein_embedding_classifier.core.embedding_loading import EmbeddingBundle
from protein_embedding_classifier.core.training.model_factory import ModelFactory
from protein_embedding_classifier.core.training.training_service import TrainingService


@dataclass(frozen=True)
class SweepResult:
    best_config: dict[str, Any]
    best_metric: float
    best_key: tuple[str, str]


class SweepService:
    def __init__(
        self,
        model_type: str,
        model_factory: ModelFactory | None = None,
        rng_seed: int = 42,
    ):
        self.model_type = model_type
        self.model_factory = model_factory or ModelFactory()
        self.rng = np.random.default_rng(rng_seed)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        embedding_bundle: EmbeddingBundle,
        sweep_config: Mapping[str, Any],
        num_trials: int = 10,
        training_config: dict[str, Any] | None = None,
        wandb_project: str = "protein-embedding-classifier",
    ) -> SweepResult:
        trial_configs = self._build_trial_configs(sweep_config, num_trials)
        if not trial_configs:
            raise ValueError("Sweep config produced no trials")

        best_metric = float("-inf")
        best_config: dict[str, Any] = {}
        best_key: tuple[str, str] | None = None

        metric_name = str(sweep_config.get("metric", {}).get("name", "f1_score"))

        for trial_index, params in enumerate(trial_configs, start=1):
            run = self._wandb_init(wandb_project=wandb_project, config=params, trial_index=trial_index)
            try:
                service = TrainingService(
                    model_factory=self.model_factory,
                    sweep_mode=True,
                    wandb_config=params,
                )

                effective_training_config = dict(training_config or {})
                effective_training_config["model_types"] = [self.model_type]

                results = service.train(
                    embedding_bundle=embedding_bundle,
                    training_config=effective_training_config,
                )

                for result_key, payload in results.items():
                    metric_value = float(payload.get("metrics", {}).get(metric_name, float("nan")))
                    self._wandb_log(
                        {
                            metric_name: metric_value,
                            "trial_index": trial_index,
                            "model_type": result_key[0],
                            "embedding_name": result_key[1],
                        }
                    )

                    if np.isfinite(metric_value) and metric_value > best_metric:
                        best_metric = metric_value
                        best_config = dict(params)
                        best_key = result_key
            finally:
                self._wandb_finish(run)

        if best_key is None:
            raise RuntimeError("Sweep finished but no valid metric was produced")

        self.logger.info(
            "Sweep completed model_type=%s best_embedding=%s %s=%.6f",
            best_key[0],
            best_key[1],
            metric_name,
            best_metric,
        )
        return SweepResult(best_config=best_config, best_metric=best_metric, best_key=best_key)

    def _build_trial_configs(self, sweep_config: Mapping[str, Any], num_trials: int) -> list[dict[str, Any]]:
        parameters = sweep_config.get("parameters", {})
        if not isinstance(parameters, Mapping):
            return []

        return [self._sample_trial(parameters) for _ in range(int(max(1, num_trials)))]

    def _sample_trial(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        sampled: dict[str, Any] = {}
        for key, spec in parameters.items():
            if not isinstance(spec, Mapping):
                continue

            if "values" in spec:
                values = list(spec["values"])
                if not values:
                    continue
                idx = int(self.rng.integers(0, len(values)))
                sampled[key] = values[idx]
                continue

            if "min" in spec and "max" in spec:
                min_value = spec["min"]
                max_value = spec["max"]
                distribution = str(spec.get("distribution", "uniform"))

                if distribution == "int_uniform":
                    sampled[key] = int(self.rng.integers(int(min_value), int(max_value) + 1))
                elif distribution in {"log_uniform", "log_uniform_values"}:
                    low = np.log(float(min_value))
                    high = np.log(float(max_value))
                    sampled[key] = float(np.exp(self.rng.uniform(low, high)))
                else:
                    sampled[key] = float(self.rng.uniform(float(min_value), float(max_value)))

        return sampled

    @staticmethod
    def _wandb_init(wandb_project: str, config: dict[str, Any], trial_index: int):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.init(project=wandb_project, config=config, name=f"sweep-trial-{trial_index}")

    @staticmethod
    def _wandb_log(payload: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return
        wandb.log(payload)

    @staticmethod
    def _wandb_finish(run) -> None:
        if run is None:
            return
        run.finish()
