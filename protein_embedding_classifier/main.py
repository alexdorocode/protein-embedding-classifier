from __future__ import annotations

import argparse
import sys
from protein_embedding_classifier.core.pipeline import Pipeline
from protein_embedding_classifier.logging_config import configure_logging

PIPELINE_STEPS = [
    "dataset",
    "embeddings",
    "train",
    "sweep",
    "ensemble",
    "benchmark",
    "evaluate",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Protein Embedding Classifier Pipeline")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument(
        "--step",
        choices=[*PIPELINE_STEPS, "all"],
        help="Execute a single pipeline step",
    )
    parser.add_argument("--train", action="store_true", help="Shortcut for --step train")
    parser.add_argument("--sweep", action="store_true", help="Shortcut for --step sweep")
    parser.add_argument("--all", action="store_true", help="Execute all pipeline steps sequentially")
    parser.add_argument("--embedding_name", help="Run only the specified embedding view")
    parser.add_argument("--classifier", help="Run only the specified classifier")
    parser.add_argument("--embedding_group", help="Run embeddings from a configured embedding group")
    parser.add_argument("--run-prefix", help="Prefix for timestamped sweep run folders")
    parser.add_argument(
        "--evaluate-last-sweep",
        action="store_true",
        help="Load the latest sweep artifacts and evaluate saved best models",
    )

    args = parser.parse_args()

    selected_steps = [flag for flag in (args.train, args.sweep) if flag]
    if len(selected_steps) > 1:
        parser.error("Use at most one of --train or --sweep")

    selected_step = args.step
    run_all = bool(args.all)
    if args.train:
        selected_step = "train"
    elif args.sweep:
        selected_step = "sweep"

    if args.evaluate_last_sweep and selected_step is None and not run_all:
        selected_step = "evaluate"

    if selected_step == "all":
        run_all = True
        selected_step = None

    configure_logging()
    pipeline = Pipeline(config_path=args.config)
    pipeline.run(
        step=selected_step,
        run_all=run_all,
        filters={
            "embedding_name": args.embedding_name,
            "classifier": args.classifier,
            "embedding_group": args.embedding_group,
        },
        runtime_context={
            "run_prefix": args.run_prefix,
            "evaluate_last_sweep": bool(args.evaluate_last_sweep),
            "argv": list(sys.argv),
        },
    )


if __name__ == "__main__":
    main()
