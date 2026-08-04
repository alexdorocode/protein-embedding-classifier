#!/usr/bin/env python3
"""
Benchmark Script

Command-line interface for running benchmark comparisons.

Usage:
    python -m scripts.benchmark --config config/benchmark.yaml --output benchmark_results.json

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

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
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description='Run benchmark comparison of models'
    )
    
    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to benchmark configuration YAML file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='benchmark_results.json',
        help='Path to save benchmark results'
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting benchmark...")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        logger.info(f"Benchmark name: {config.get('name', 'unnamed')}")
        logger.info(f"Models: {config.get('models', [])}")
        logger.info(f"Datasets: {config.get('datasets', [])}")
        
        # Set up W&B if enabled
        if args.wandb:
            import wandb
            logger.info("Initializing Weights & Biases")
            wandb.init(
                project=args.project,
                name=config.get('name'),
                config=config
            )
        
        # Run benchmark
        logger.info("Running benchmark...")
        from src.orchestration import run_benchmark
        results = run_benchmark(config)
        
        logger.info("Benchmark completed successfully")
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Log to W&B if enabled
        if args.wandb:
            import wandb
            wandb.log({'benchmark_complete': True})
            wandb.save(str(output_path))
            wandb.finish()
        
        # Print summary
        logger.info("\nBenchmark Summary:")
        logger.info(f"- Models tested: {len(results.get('models', []))}")
        logger.info(f"- Datasets used: {len(results.get('datasets', []))}")
        logger.info(f"- Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
        if args.wandb:
            import wandb
            wandb.finish()
        return 1


if __name__ == '__main__':
    sys.exit(main())
