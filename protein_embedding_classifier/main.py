from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from protein_embedding_classifier.core.db import create_engine_from_config, load_db_config
from protein_embedding_classifier.data.dataset_builder import DatasetBuilder
from protein_embedding_classifier.data.label_loader import LabelLoader
from protein_embedding_classifier.data.protein_loader import ProteinLoader
from protein_embedding_classifier.data.splits.cross_validation import CrossValidationSplit
from protein_embedding_classifier.data.splits.zero_shot_csv import ZeroShotCSVSplit
from protein_embedding_classifier.data.splits.zero_shot_organism import ZeroShotOrganismSplit
from protein_embedding_classifier.data.splits.zero_shot_random import ZeroShotRandomSplit
from protein_embedding_classifier.logging_config import configure_logging


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_split_strategy(conf: Dict[str, Any]):
    strategy_name = conf.get("strategy", "zero_shot_random")

    if strategy_name == "cross_validation":
        return CrossValidationSplit(
            n_splits=conf.get("n_splits", 5),
            fold_index=conf.get("fold_index", 0),
            random_state=conf.get("random_state", 42),
        )

    if strategy_name == "zero_shot_csv":
        return ZeroShotCSVSplit(
            csv_path=conf["csv_path"],
            accession_column=conf.get("accession_column", "accession"),
            split_column=conf.get("split_column", "split"),
        )

    if strategy_name == "zero_shot_organism":
        return ZeroShotOrganismSplit(
            test_organisms=conf.get("test_organisms", []),
            val_organisms=conf.get("val_organisms", []),
        )

    if strategy_name == "zero_shot_random":
        return ZeroShotRandomSplit(
            test_size=conf.get("test_size", 0.2),
            val_size=conf.get("val_size", 0.1),
            random_state=conf.get("random_state", 42),
        )

    raise ValueError(f"Unknown split strategy: {strategy_name}")


def run_dataset_step(conf: Dict[str, Any]) -> None:
    pipeline_logger = logging.getLogger("Pipeline")

    db_conf_path = conf.get("db_config_path", "config/db.yaml")
    db_conf = load_db_config(db_conf_path)
    engine = create_engine_from_config(db_conf)

    protein_conf = conf.get("protein_loader", {})
    label_conf = conf.get("label_loader", {})
    split_conf = conf.get("split", {})

    protein_loader = ProteinLoader(engine=engine, query=protein_conf.get("query"))
    label_loader = LabelLoader(
        source=label_conf.get("source", "db"),
        engine=engine,
        file_path=label_conf.get("file_path"),
        db_query=label_conf.get("db_query"),
        db_query_file=label_conf.get("db_query_file"),
        accession_column=label_conf.get("accession_column", "accession"),
        label_column=label_conf.get("label_column", "label"),
        artifacts_dir=label_conf.get("artifacts_dir", "artifacts"),
    )
    split_strategy = _build_split_strategy(split_conf)

    builder = DatasetBuilder(
        protein_loader=protein_loader,
        label_loader=label_loader,
        split_strategy=split_strategy,
    )

    bundle = builder.build()
    pipeline_logger.info(
        "Dataset bundle ready: train=%d val=%d test=%d",
        len(bundle.train_ids),
        len(bundle.val_ids),
        len(bundle.test_ids),
    )


def run_step(step: str, conf: Dict[str, Any]) -> None:
    logger = logging.getLogger("Pipeline")
    logger.info("Starting step: %s", step)

    if step == "dataset":
        run_dataset_step(conf.get("dataset", conf))
    elif step in {"embeddings", "train", "sweep", "ensemble", "evaluate"}:
        logger.info("Step '%s' is configured but intentionally not implemented in this refactor.", step)
    else:
        raise ValueError(f"Unknown step: {step}")

    logger.info("Finished step: %s", step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protein Embedding Classifier Pipeline")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument(
        "--step",
        choices=["dataset", "embeddings", "train", "sweep", "ensemble", "evaluate"],
        help="Execute a single pipeline step",
    )
    parser.add_argument("--all", action="store_true", help="Execute all pipeline steps sequentially")

    args = parser.parse_args()

    configure_logging()
    conf = _load_yaml(args.config)

    if args.all:
        for step in ["dataset", "embeddings", "train", "sweep", "ensemble", "evaluate"]:
            run_step(step, conf)
        return

    step = args.step or "dataset"
    run_step(step, conf)


if __name__ == "__main__":
    main()
