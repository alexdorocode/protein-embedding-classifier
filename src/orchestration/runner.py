"""
Experiment Runner Module

Handles running of individual experiments.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs individual experiments.
    """
    
    def __init__(self):
        """Initialize the experiment runner."""
        self.results = {}
    
    def run(self, experiment_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run an experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Optional configuration override
            
        Returns:
            Experiment results
        """
        from .experiment_definitions import ExperimentDefinitions
        from src.dataset_builder import load_run
        from src.training import train_model
        from src.prediction import predict
        from src.explainability import explain
        from src.output import save_results
        
        # Get experiment configuration
        definitions = ExperimentDefinitions()
        exp_config = definitions.get_experiment(experiment_name)
        
        # Override with provided config
        if config:
            for key, value in config.items():
                setattr(exp_config, key, value)
        
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Description: {exp_config.description}")
        
        results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'dataset': exp_config.dataset,
                'model': exp_config.model,
                'training': exp_config.training,
                'evaluation': exp_config.evaluation,
            }
        }
        
        try:
            # Load dataset
            logger.info(f"Loading dataset: {exp_config.dataset}")
            run_data = load_run(exp_config.dataset, base_path='dataset_designer_runs')
            dataset = run_data.tp_ntp_pairs
            results['dataset_info'] = {
                'run_id': run_data.run_id,
                'species': run_data.metadata.get('species', 'unknown'),
                'num_pairs': len(dataset)
            }
            
            # Train model
            logger.info(f"Training model: {exp_config.model}")
            model = train_model(
                model_name=exp_config.model,
                data=dataset,
                config=exp_config.training
            )
            results['model'] = model
            
            # Evaluate model
            logger.info("Evaluating model")
            # For now, just store training config
            results['evaluation'] = exp_config.evaluation
            
            # Generate explanations (optional)
            if exp_config.explainability:
                logger.info("Generating explanations")
                explanations = explain(
                    model=model,
                    data=dataset.head(10),  # Explain first 10 samples
                    method='feature_importance'
                )
                results['explanations'] = explanations
            
            # Save results
            logger.info("Saving results")
            save_results(results, f"experiment_{experiment_name}", format='json')
            
            logger.info(f"Experiment {experiment_name} completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {str(e)}")
            results['error'] = str(e)
            raise
        
        return results
