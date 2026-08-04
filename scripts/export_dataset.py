#!/usr/bin/env python3
"""
Export Dataset Script

Command-line interface for exporting datasets.

Usage:
    python -m scripts.export_dataset --run 20260803_0258_7672b947 --output exported_dataset.csv --format csv

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


def main():
    """Main entry point for dataset export script."""
    parser = argparse.ArgumentParser(
        description='Export a dataset from a run'
    )
    
    # Required arguments
    parser.add_argument(
        '--run', '-r',
        type=str,
        required=True,
        help='Run ID to export'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: <run_id>_dataset.<format>)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='csv',
        choices=['csv', 'json', 'pickle'],
        help='Output format'
    )
    parser.add_argument(
        '--base-path', '-b',
        type=str,
        default='dataset_designer_runs',
        help='Base path for dataset runs'
    )
    parser.add_argument(
        '--include-metadata', '-m',
        action='store_true',
        help='Include run metadata in export'
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
    
    logger.info("Starting dataset export...")
    logger.info(f"Run ID: {args.run}")
    
    try:
        # Load run
        logger.info(f"Loading run from {args.base_path}/{args.run}")
        from src.dataset_builder import load_run
        run_data = load_run(args.run, base_path=args.base_path)
        
        logger.info(f"Loaded run: {run_data.run_id}")
        logger.info(f"Species: {run_data.metadata.get('species', 'unknown')}")
        logger.info(f"Number of pairs: {len(run_data.tp_ntp_pairs)}")
        
        # Prepare export data
        if args.include_metadata:
            export_data = {
                'metadata': run_data.metadata,
                'dataset': run_data.tp_ntp_pairs,
                'assignments': run_data.assignments
            }
        else:
            export_data = run_data.tp_ntp_pairs
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"{args.run}_dataset.{args.format}")
        
        # Save data
        logger.info(f"Exporting to {output_path}")
        
        if args.format == 'csv':
            if isinstance(export_data, dict):
                # Save dataset part
                export_data['dataset'].to_csv(output_path, index=False)
            else:
                export_data.to_csv(output_path, index=False)
        
        elif args.format == 'json':
            import json
            with open(output_path, 'w') as f:
                if isinstance(export_data, dict):
                    # Convert DataFrames to lists for JSON serialization
                    json_data = {
                        'metadata': export_data.get('metadata'),
                        'dataset': export_data.get('dataset').to_dict(orient='records'),
                        'assignments': export_data.get('assignments').to_dict(orient='records')
                    }
                    json.dump(json_data, f, indent=2, default=str)
                else:
                    json.dump(export_data.to_dict(orient='records'), f, indent=2, default=str)
        
        elif args.format == 'pickle':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f)
        
        logger.info(f"Dataset exported to: {output_path}")
        logger.info("Export script completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
