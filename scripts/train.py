#!/usr/bin/env python3
"""
Train Script

Command-line interface for training models.

Usage:
    python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp --config config/training/training_config.yaml

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train a model on protein embedding data'
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset run ID to use for training'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model name to train (mlp, random_forest, linear, etc.)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--base-path', '-b',
        type=str,
        default='dataset_designer_runs',
        help='Base path for dataset runs'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='protein-embedding-classifier',
        help='W&B project name'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='W&B experiment name'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting training...")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {args.base_path}/{args.dataset}")
        from src.dataset_builder import load_run
        run_data = load_run(args.dataset, base_path=args.base_path)
        dataset = run_data.tp_ntp_pairs
        
        logger.info(f"Loaded {len(dataset)} samples")
        logger.info(f"Species: {run_data.metadata.get('species', 'unknown')}")
        
        # Load configuration
        config = {}
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        
        # Set up W&B if enabled
        if args.wandb:
            import wandb
            logger.info("Initializing Weights & Biases")
            wandb.init(
                project=args.project,
                name=args.experiment,
                config={
                    'dataset': args.dataset,
                    'model': args.model,
                    **config
                }
            )
        
        # Train model
        logger.info(f"Training {args.model} model...")
        from src.training import train_model
        model = train_model(
            model_name=args.model,
            data=dataset,
            config=config
        )
        
        logger.info("Training completed successfully")
        
        # Save model
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{args.dataset}_{args.model}.pkl"
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to: {model_path}")
        
        # Log to W&B if enabled
        if args.wandb:
            import wandb
            wandb.log({'model_path': str(model_path)})
            wandb.finish()
        
        logger.info("Training script completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        if args.wandb:
            import wandb
            wandb.finish()
        return 1


if __name__ == '__main__':
    sys.exit(main())
