#!/usr/bin/env python3
"""
Predict Script

Command-line interface for making predictions with trained models.

Usage:
    python -m scripts.predict --model path/to/model.pkl --data path/to/data.csv --output predictions.csv

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> any:
    """Load a trained model from file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_data(data_path: str) -> any:
    """Load data for prediction."""
    import pandas as pd
    
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Try as CSV first
        try:
            return pd.read_csv(data_path)
        except Exception:
            # Try as pickle
            with open(data_path, 'rb') as f:
                return pickle.load(f)


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description='Make predictions with a trained model'
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to input data file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for prediction'
    )
    parser.add_argument(
        '--probabilities', '-p',
        action='store_true',
        help='Output probability predictions instead of class predictions'
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
    
    logger.info("Starting prediction...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model}")
        model = load_model(args.model)
        logger.info(f"Model type: {type(model).__name__}")
        
        # Load data
        logger.info(f"Loading data from {args.data}")
        data = load_data(args.data)
        logger.info(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        
        # Make predictions
        logger.info("Making predictions...")
        from src.prediction import predict, batch_predict
        
        if args.batch_size > 1:
            # Convert data to list for batch prediction
            if hasattr(data, 'values'):
                data_list = data.values.tolist()
            elif isinstance(data, list):
                data_list = data
            else:
                data_list = [data]
            
            if args.probabilities:
                predictions = batch_predict(
                    model=model,
                    data_list=data_list,
                    batch_size=args.batch_size
                )
            else:
                predictions = batch_predict(
                    model=model,
                    data_list=data_list,
                    batch_size=args.batch_size
                )
        else:
            if args.probabilities:
                predictions = predict(model=model, data=data)
            else:
                predictions = predict(model=model, data=data)
        
        logger.info(f"Predictions shape: {len(predictions) if isinstance(predictions, list) else 'N/A'}")
        
        # Save predictions
        output_path = Path(args.output)
        
        if isinstance(predictions, list):
            import pandas as pd
            df = pd.DataFrame(predictions, columns=['prediction'])
            df.to_csv(output_path, index=False)
        elif hasattr(predictions, 'to_csv'):
            predictions.to_csv(output_path, index=False)
        else:
            import numpy as np
            np.savetxt(output_path, predictions, delimiter=',')
        
        logger.info(f"Predictions saved to: {output_path}")
        logger.info("Prediction script completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
