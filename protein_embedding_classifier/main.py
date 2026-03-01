from __future__ import annotations

import argparse
from protein_embedding_classifier.core.pipeline import Pipeline
from protein_embedding_classifier.logging_config import configure_logging

PIPELINE_STEPS = [
    "dataset",
    "embeddings",
    "train",
    "sweep",
    "ensemble",
    "evaluate",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Protein Embedding Classifier Pipeline")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument(
        "--step",
        choices=PIPELINE_STEPS,
        help="Execute a single pipeline step",
    )
    parser.add_argument("--train", action="store_true", help="Shortcut for --step train")
    parser.add_argument("--sweep", action="store_true", help="Shortcut for --step sweep")
    parser.add_argument("--all", action="store_true", help="Execute all pipeline steps sequentially")
    parser.add_argument("--embedding_name", help="Run only the specified embedding view")
    parser.add_argument("--classifier", help="Run only the specified classifier")
    parser.add_argument("--embedding_group", help="Run embeddings from a configured embedding group")

    args = parser.parse_args()

    selected_steps = [flag for flag in (args.train, args.sweep) if flag]
    if len(selected_steps) > 1:
        parser.error("Use at most one of --train or --sweep")

    selected_step = args.step
    if args.train:
        selected_step = "train"
    elif args.sweep:
        selected_step = "sweep"

    configure_logging()
    pipeline = Pipeline(config_path=args.config)
    pipeline.run(
        step=selected_step,
        run_all=args.all,
        filters={
            "embedding_name": args.embedding_name,
            "classifier": args.classifier,
            "embedding_group": args.embedding_group,
        },
    )


if __name__ == "__main__":
    main()
