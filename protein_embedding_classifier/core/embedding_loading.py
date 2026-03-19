from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from protein_embedding_classifier.core.db import coerce_embedding_vector
from protein_embedding_classifier.data.dataset_builder import DatasetBundle


LayeredSequenceEmbeddings = dict[str, dict[str, dict[int, np.ndarray]]]
AggregatedEmbeddings = dict[str, dict[str, np.ndarray]]


class SequenceEmbeddingLoader:
    def __init__(self, engine: Engine, chunk_size: int = 5000):
        self.engine = engine
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def load(self, config: dict[str, Any], accessions: list[str]) -> LayeredSequenceEmbeddings:
        sequence_conf = config.get("sequencePE", {})
        models_conf = sequence_conf.get("models", {})
        if not sequence_conf.get("enabled", False):
            self.logger.info("sequencePE disabled; skipping sequence embedding load")
            return {}

        enabled_models = {
            model_name: model_conf
            for model_name, model_conf in models_conf.items()
            if model_conf.get("enabled", False)
        }
        if not enabled_models:
            self.logger.info("No enabled sequence models found")
            return {}

        self.logger.info("Loading sequence embeddings for %d model(s)", len(enabled_models))
        embeddings: LayeredSequenceEmbeddings = {}
        for model_name, model_conf in enabled_models.items():
            layer_indices = model_conf.get("layer_index", [0])
            model_embeddings = self._load_model_layers(model_name=model_name, layer_indices=layer_indices, accessions=accessions)
            embeddings[model_name] = model_embeddings
            self.logger.info(
                "Loaded sequence model=%s accessions=%d layers=%s",
                model_name,
                len(model_embeddings),
                sorted({layer for layer_map in model_embeddings.values() for layer in layer_map.keys()}),
            )

        return embeddings

    def _load_model_layers(self, model_name: str, layer_indices: list[int], accessions: list[str]) -> dict[str, dict[int, np.ndarray]]:
        query = text(
            """
            SELECT
                a.code AS accession,
                se.layer_index AS layer_index,
                se.embedding AS embedding
            FROM sequence_embeddings se
            JOIN sequence_embedding_type st
              ON st.id = se.embedding_type_id
            JOIN sequence s
              ON s.id = se.sequence_id
            JOIN protein p
              ON p.sequence_id = s.id
            JOIN accession a
              ON a.protein_id = p.id
            WHERE st.name = :model_name
              AND se.layer_index IN :layer_indices
              AND a.code IN :accessions
              AND a."primary" = TRUE
            """
        ).bindparams(bindparam("layer_indices", expanding=True), bindparam("accessions", expanding=True))

        model_embeddings: dict[str, dict[int, np.ndarray]] = {}
        unique_layers = sorted(set(int(layer_index) for layer_index in layer_indices))

        if not accessions:
            return model_embeddings

        with self.engine.connect() as conn:
            for start in range(0, len(accessions), self.chunk_size):
                accession_chunk = accessions[start:start + self.chunk_size]
                rows = conn.execute(
                    query,
                    {
                        "model_name": model_name,
                        "layer_indices": unique_layers,
                        "accessions": accession_chunk,
                    },
                ).fetchall()

                for row in rows:
                    accession = row.accession
                    layer_index = int(row.layer_index)
                    vector = coerce_embedding_vector(row.embedding)
                    model_embeddings.setdefault(accession, {})[layer_index] = vector

        return model_embeddings


class LayerAggregationStrategy:
    def __init__(self, mode: str):
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)

    def aggregate(self, sequence_embeddings: LayeredSequenceEmbeddings) -> AggregatedEmbeddings:
        if self.mode not in {"none", "mean", "max", "mean_max", "concat"}:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")

        aggregated: AggregatedEmbeddings = {}
        for model_name, accession_map in sequence_embeddings.items():
            if self.mode == "none":
                self._aggregate_none_mode(model_name, accession_map, aggregated)
                continue

            model_output: dict[str, np.ndarray] = {}
            for accession, layer_map in accession_map.items():
                layers = self._ordered_layers(layer_map)
                model_output[accession] = self._aggregate_layers(layers)
            aggregated[model_name] = model_output

        self.logger.info("Aggregation mode=%s produced %d model view(s)", self.mode, len(aggregated))
        return aggregated

    def _aggregate_none_mode(
        self,
        model_name: str,
        accession_map: dict[str, dict[int, np.ndarray]],
        output: AggregatedEmbeddings,
    ) -> None:
        all_layers = sorted({layer for layer_map in accession_map.values() for layer in layer_map.keys()})
        if len(all_layers) <= 1:
            only_layer = all_layers[0] if all_layers else 0
            output[model_name] = {
                accession: layer_map[only_layer]
                for accession, layer_map in accession_map.items()
                if only_layer in layer_map
            }
            return

        for layer_index in all_layers:
            layered_model_name = f"{model_name}__layer_{layer_index}"
            output[layered_model_name] = {
                accession: layer_map[layer_index]
                for accession, layer_map in accession_map.items()
                if layer_index in layer_map
            }

    def _aggregate_layers(self, ordered_layers: list[np.ndarray]) -> np.ndarray:
        layer_matrix = np.stack(ordered_layers, axis=0)
        if self.mode == "mean":
            return layer_matrix.mean(axis=0)
        if self.mode == "max":
            return layer_matrix.max(axis=0)
        if self.mode == "mean_max":
            return np.concatenate([layer_matrix.mean(axis=0), layer_matrix.max(axis=0)], axis=0)
        if self.mode == "concat":
            return np.concatenate(ordered_layers, axis=0)
        raise ValueError(f"Unknown aggregation mode: {self.mode}")

    @staticmethod
    def _ordered_layers(layer_map: dict[int, np.ndarray]) -> list[np.ndarray]:
        ordered_indices = sorted(layer_map.keys())
        return [layer_map[layer_index] for layer_index in ordered_indices]


import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class GOEmbeddingLoader:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load(self, config: dict[str, Any], accessions: list[str]) -> Dict[str, Dict[str, np.ndarray]]:
        go_conf = config.get("GOPE", {})
        if not go_conf.get("enabled", False):
            self.logger.info("GOPE disabled; skipping GO embedding load")
            return {}

        geokg_conf = go_conf.get("models", {}).get("GeOKG", {})
        if not geokg_conf.get("enabled", False):
            self.logger.info("GO model GeOKG disabled")
            return {}

        file_info = go_conf.get("file_info", {})
        folder = Path(file_info["folder"])
        accession_column = file_info.get("accession_column", "accession")
        embedding_column = file_info.get("embedding_column", "embedding")

        file_names = self._resolve_go_file_names(file_info)

        accession_set = set(accessions)

        # Store embeddings per ontology
        ontology_embeddings: dict[str, dict[str, np.ndarray]] = {}
        ontology_dims: dict[str, int] = {}

        for file_name in file_names:
            file_path = folder / file_name
            df = pd.read_csv(file_path)

            if accession_column not in df.columns or embedding_column not in df.columns:
                raise ValueError(
                    f"GO file {file_path} missing required columns "
                    f"{accession_column}/{embedding_column}"
                )

            # Infer ontology name from filename
            ontology_name = self._infer_ontology_name(file_name)

            filtered_df = df[df[accession_column].isin(accession_set)]

            emb_dict: dict[str, np.ndarray] = {}

            for record in filtered_df[[accession_column, embedding_column]].to_dict("records"):
                accession = str(record[accession_column])
                vector = coerce_embedding_vector(record[embedding_column])
                emb_dict[accession] = vector

            if emb_dict:
                ontology_dims[ontology_name] = len(next(iter(emb_dict.values())))
            else:
                ontology_dims[ontology_name] = 0

            ontology_embeddings[ontology_name] = emb_dict

            self.logger.info(
                "Loaded GO embeddings from %s (ontology=%s) matched=%d",
                file_path,
                ontology_name,
                len(filtered_df),
            )

        # Concatenate BP + MF + CC
        concatenated_embeddings: dict[str, np.ndarray] = {}

        for acc in accessions:
            vectors = []

            for ontology_name in sorted(ontology_embeddings.keys()):
                emb_dict = ontology_embeddings[ontology_name]
                dim = ontology_dims[ontology_name]

                if acc in emb_dict:
                    vectors.append(emb_dict[acc])
                else:
                    # pad missing ontology with zeros
                    vectors.append(np.zeros(dim, dtype=np.float32))

            if vectors:
                concatenated_embeddings[acc] = np.concatenate(vectors)

        final_dim = sum(ontology_dims.values())
        self.logger.info(
            "Concatenated GO embeddings: total_dim=%d ontologies=%s",
            final_dim,
            list(ontology_embeddings.keys()),
        )

        return {"GeOKG": concatenated_embeddings}

    @staticmethod
    def _resolve_go_file_names(file_info: dict[str, Any]) -> list[str]:
        if "file_names" in file_info:
            return list(file_info["file_names"])
        return [
            value
            for key, value in file_info.items()
            if key.startswith("file_name") and isinstance(value, str)
        ]

    @staticmethod
    def _infer_ontology_name(file_name: str) -> str:
        file_name = file_name.lower()
        if "_p_" in file_name or "bp" in file_name:
            return "BP"
        if "_f_" in file_name or "mf" in file_name:
            return "MF"
        if "_c_" in file_name or "cc" in file_name:
            return "CC"
        return file_name

class EmbeddingService:
    def __init__(self, embeddings_config: dict[str, Any], engine: Engine, accessions: list[str]):
        self.embeddings_config = embeddings_config
        self.engine = engine
        self.accessions = list(accessions)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.sequence_loader = SequenceEmbeddingLoader(engine=engine)
        self.go_loader = GOEmbeddingLoader()

    def load_embeddings(self) -> AggregatedEmbeddings:
        all_embeddings: AggregatedEmbeddings = {}

        sequence_raw = self.sequence_loader.load(self.embeddings_config, self.accessions)
        if sequence_raw:
            aggregation_mode = self.embeddings_config.get("aggregation", {}).get("mode", "none")
            aggregator = LayerAggregationStrategy(mode=aggregation_mode)
            sequence_aggregated = aggregator.aggregate(sequence_raw)
            all_embeddings.update(sequence_aggregated)

        go_embeddings = self.go_loader.load(self.embeddings_config, self.accessions)
        if go_embeddings:
            all_embeddings.update(go_embeddings)

        self.logger.info("Loaded embedding views: %s", sorted(all_embeddings.keys()))
        return all_embeddings


@dataclass
class EmbeddingBundle:
    X_train: dict[str, np.ndarray]
    X_val: dict[str, np.ndarray]
    X_test: dict[str, np.ndarray]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_zero_shot: dict[str, np.ndarray] = field(default_factory=dict)
    y_zero_shot: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=object))

    @classmethod
    def from_dataset(
        cls,
        dataset_bundle: DatasetBundle,
        raw_embeddings: AggregatedEmbeddings,
        allowed_missing_model_artifacts: dict[str, str] | None = None,
    ) -> EmbeddingBundle:
        logger = logging.getLogger(cls.__name__)
        allowed_missing_model_artifacts = allowed_missing_model_artifacts or {}
        train_ids = dataset_bundle.train_ids
        val_ids = dataset_bundle.val_ids
        test_ids = dataset_bundle.test_ids
        zero_shot_ids = list(getattr(dataset_bundle, "zero_shot_ids", []))

        x_train: dict[str, np.ndarray] = {}
        x_val: dict[str, np.ndarray] = {}
        x_test: dict[str, np.ndarray] = {}
        x_zero_shot: dict[str, np.ndarray] = {}

        for model_name, model_embeddings in raw_embeddings.items():
            allow_missing = model_name in allowed_missing_model_artifacts
            report_path = allowed_missing_model_artifacts.get(model_name)
            if allow_missing and report_path:
                missing_all = sorted(
                    {
                        accession
                        for accession in (train_ids + val_ids + test_ids + zero_shot_ids)
                        if accession not in model_embeddings
                    }
                )
                cls._write_missing_artifact(report_path, missing_all, logger, model_name)

            x_train[model_name] = cls._build_matrix(
                model_name,
                "train",
                train_ids,
                model_embeddings,
                logger,
                allow_missing=allow_missing,
            )
            x_val[model_name] = cls._build_matrix(
                model_name,
                "val",
                val_ids,
                model_embeddings,
                logger,
                allow_missing=allow_missing,
            )
            x_test[model_name] = cls._build_matrix(
                model_name,
                "test",
                test_ids,
                model_embeddings,
                logger,
                allow_missing=allow_missing,
            )
            x_zero_shot[model_name] = cls._build_matrix(
                model_name,
                "zero_shot",
                zero_shot_ids,
                model_embeddings,
                logger,
                allow_missing=allow_missing,
            )

        return cls(
            X_train=x_train,
            X_val=x_val,
            X_test=x_test,
            y_train=dataset_bundle.y_train,
            y_val=dataset_bundle.y_val,
            y_test=dataset_bundle.y_test,
            X_zero_shot=x_zero_shot,
            y_zero_shot=np.asarray(getattr(dataset_bundle, "y_zero_shot", np.asarray([], dtype=object))),
        )

    @staticmethod
    def _build_matrix(
        model_name: str,
        split_name: str,
        split_ids: list[str],
        model_embeddings: dict[str, np.ndarray],
        logger: logging.Logger,
        allow_missing: bool,
    ) -> np.ndarray:
        missing = [accession for accession in split_ids if accession not in model_embeddings]
        if missing:
            if not allow_missing:
                logger.error(
                    "Missing embeddings model=%s split=%s missing=%d",
                    model_name,
                    split_name,
                    len(missing),
                )
                raise ValueError(
                    f"Missing embeddings for model={model_name} split={split_name}. "
                    f"Example missing accessions: {missing[:10]}"
                )
            logger.warning(
                "Missing embeddings tolerated model=%s split=%s missing=%d",
                model_name,
                split_name,
                len(missing),
            )

        example = next(iter(model_embeddings.values()), None)
        if allow_missing and example is None and split_ids:
            raise ValueError(
                f"Model {model_name} has no embeddings loaded; cannot fill missing values."
            )

        zero_vector = None if example is None else np.zeros_like(example, dtype=np.float32)
        vectors = [model_embeddings.get(accession, zero_vector) for accession in split_ids]
        if not vectors:
            dim = int(example.shape[0]) if example is not None else 0
            return np.empty((0, dim), dtype=np.float32)

        return np.vstack(vectors)

    @staticmethod
    def _write_missing_artifact(
        path: str,
        missing_accessions: list[str],
        logger: logging.Logger,
        model_name: str,
    ) -> None:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            for accession in missing_accessions:
                handle.write(f"{accession}\n")
        logger.info(
            "Missing embeddings artifact written model=%s path=%s count=%d",
            model_name,
            artifact_path,
            len(missing_accessions),
        )