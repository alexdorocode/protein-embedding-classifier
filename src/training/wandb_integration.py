"""
Weights & Biases Integration Module

Handles integration with Weights & Biases for experiment tracking.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

import wandb
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WandbIntegration:
    """
    Handles W&B integration for experiment tracking.
    """
    
    def __init__(self, project: str = "protein-embedding-classifier", 
                 entity: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize W&B integration.
        
        Args:
            project: W&B project name
            entity: W&B entity (team/user)
            config: W&B configuration
        """
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.run = None
    
    def init_run(self, run_name: str = None, run_config: Optional[Dict] = None) -> None:
        """
        Initialize a W&B run.
        
        Args:
            run_name: Name for this run
            run_config: Configuration for this run
        """
        config = self.config.copy()
        if run_config:
            config.update(run_config)
        
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config,
            reinit=True
        )
        logger.info(f"W&B run initialized: {self.run.name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
        """
        if self.run:
            wandb.log(metrics, step=step)
            logger.debug(f"Logged metrics: {metrics}")
    
    def log_artifact(self, path: str, name: str, artifact_type: str = "dataset") -> None:
        """
        Log an artifact to W&B.
        
        Args:
            path: Path to the artifact file/directory
            name: Name for the artifact
            artifact_type: Type of artifact
        """
        if self.run:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_dir(path)
            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact: {name}")
    
    def log_model(self, model: any, model_name: str, model_path: str = None) -> None:
        """
        Log a model to W&B.
        
        Args:
            model: Model to log
            model_name: Name for the model
            model_path: Path to save the model
        """
        if self.run:
            if model_path:
                # Save model first
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Log as artifact
                self.log_artifact(model_path, model_name, "model")
            
            # Also log model metadata
            wandb.log({"model_name": model_name})
            logger.info(f"Logged model: {model_name}")
    
    def finish_run(self) -> None:
        """Finish the current W&B run."""
        if self.run:
            wandb.finish()
            logger.info("W&B run finished")
            self.run = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish_run()
