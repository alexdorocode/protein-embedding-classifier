"""
Abstract Base Classes and Contracts

This module defines abstract base classes and interfaces that future PEC modules
should implement to ensure compatibility and extensibility.

These contracts support:
- Future classifier families
- Future result aggregators
- Future embedding loading
- Future pooling strategies
- Experiment manifest compatibility
- Artifact traceability
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from pathlib import Path
import json


# =============================================================================
# Pre-Embedding Dataset Layer Contracts
# =============================================================================

class UniverseReaderProtocol(Protocol):
    """Protocol for universe readers."""
    
    def read_csv(
        self,
        file_path: Path,
        candidate_separator: str = ";",
        target_label_default: str = "positive",
    ) -> List[Any]:
        """Read a CSV file and return UniverseRecord instances."""
        ...


class UniverseNormalizerProtocol(Protocol):
    """Protocol for universe normalizers."""
    
    def normalize(
        self,
        records: List[Any],
        universe_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> tuple[List[Any], Any]:
        """Normalize universe records and return (records, manifest)."""
        ...


class DatasetPolicyValidatorProtocol(Protocol):
    """Protocol for policy validators."""
    
    def validate(self, policy: Any) -> tuple[bool, List[str]]:
        """Validate a dataset policy."""
        ...


class DatasetVariantGeneratorProtocol(Protocol):
    """Protocol for variant generators."""
    
    def generate(
        self,
        universe: List[Any],
        policy: Any,
        seed: int,
        variant_id: Optional[str] = None,
    ) -> Any:
        """Generate a dataset variant."""
        ...


class SplitStrategyProtocol(Protocol):
    """Protocol for split strategies."""
    
    def split(
        self,
        assignments: List[Any],
        seed: int,
        variant_id: str,
    ) -> tuple[Any, Any, Any, Any]:
        """Split assignments into train, val, test, manifest."""
        ...


class LineageBuilderProtocol(Protocol):
    """Protocol for lineage builders."""
    
    def build(
        self,
        lineage_id: Optional[str] = None,
        source_artifacts: Optional[List[Any]] = None,
        transforms: Optional[List[Any]] = None,
        runtime: Optional[Any] = None,
        random_seed: Optional[int] = None,
    ) -> Any:
        """Build a lineage manifest."""
        ...


class BundleExporterProtocol(Protocol):
    """Protocol for bundle exporters."""
    
    def export(
        self,
        variant: Any,
        split_artifacts: Dict[str, Any],
        split_manifest: Any,
        universe_manifest: Any,
        policy: Any,
        lineage_manifest: Any,
        bundle_path: Path,
    ) -> Any:
        """Export a dataset bundle."""
        ...


# =============================================================================
# Future Layer Contracts (Forward Compatibility)
# =============================================================================

@runtime_checkable
class EmbeddingLoaderProtocol(Protocol):
    """
    Protocol for future embedding loading modules.
    
    Future embedding loaders should implement this protocol to ensure
    compatibility with the pre-embedding dataset layer.
    """
    
    @abstractmethod
    def load(self, protein_ids: List[str]) -> Dict[str, Any]:
        """
        Load embeddings for a list of protein IDs.
        
        Args:
            protein_ids: List of protein IDs to load embeddings for
            
        Returns:
            Dictionary mapping protein_id to embedding
        """
        ...
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        ...


@runtime_checkable
class ClassifierProtocol(Protocol):
    """
    Protocol for future classifier modules.
    
    All classifiers should implement this protocol to ensure they can be
    used interchangeably in the PEC pipeline.
    """
    
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs: Any) -> None:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional training parameters
        """
        ...
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict labels for input data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        ...
    
    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """
        Predict probabilities for input data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        ...
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the classifier to disk."""
        ...
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the classifier from disk."""
        ...
    
    @property
    @abstractmethod
    def class_name(self) -> str:
        """Get the classifier name."""
        ...


@runtime_checkable
class AggregatorProtocol(Protocol):
    """
    Protocol for future aggregation modules.
    
    Aggregators combine predictions from multiple classifiers.
    """
    
    @abstractmethod
    def aggregate(self, predictions: List[Any], **kwargs: Any) -> Any:
        """
        Aggregate predictions from multiple classifiers.
        
        Args:
            predictions: List of prediction arrays
            **kwargs: Additional aggregation parameters
            
        Returns:
            Aggregated predictions
        """
        ...
    
    @abstractmethod
    def get_weights(self) -> Optional[List[float]]:
        """Get the weights used for aggregation."""
        ...


@runtime_checkable
class ExperimentManifestProtocol(Protocol):
    """
    Protocol for experiment manifests.
    
    Ensures that experiment manifests can be validated and tracked.
    """
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate the experiment manifest."""
        ...
    
    @abstractmethod
    def get_dataset_variant_id(self) -> str:
        """Get the dataset variant ID."""
        ...
    
    @abstractmethod
    def get_classifier_ids(self) -> List[str]:
        """Get the list of classifier IDs."""
        ...
    
    @abstractmethod
    def get_artifact_path(self, artifact_type: str) -> Optional[Path]:
        """Get the path to a specific artifact."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================

class PECModule(ABC):
    """
    Abstract base class for all PEC modules.
    
    Provides common functionality for:
    - Configuration management
    - Logging
    - Serialization
    - Validation
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate the module configuration."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"name": self.name}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PECStep(ABC):
    """
    Abstract base class for PEC pipeline steps.
    
    All pipeline steps should inherit from this class to ensure
    they can be executed both independently and as part of the pipeline.
    """
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the pipeline step.
        
        Args:
            inputs: Dictionary of input artifacts
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of output artifacts
        """
        ...
    
    @abstractmethod
    def get_inputs(self) -> List[str]:
        """Get the list of required input keys."""
        ...
    
    @abstractmethod
    def get_outputs(self) -> List[str]:
        """Get the list of produced output keys."""
        ...
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate that all required inputs are present."""
        required = self.get_inputs()
        missing = [k for k in required if k not in inputs]
        return missing


class TraceableArtifact(ABC):
    """
    Abstract base class for traceable artifacts.
    
    All artifacts should implement this to ensure provenance tracking.
    """
    
    @abstractmethod
    def get_artifact_id(self) -> str:
        """Get the unique artifact identifier."""
        ...
    
    @abstractmethod
    def get_lineage(self) -> Any:
        """Get the lineage manifest for this artifact."""
        ...
    
    @abstractmethod
    def get_source_artifacts(self) -> List[str]:
        """Get the list of source artifact IDs."""
        ...
    
    def is_traceable(self) -> bool:
        """Check if the artifact has complete lineage."""
        try:
            lineage = self.get_lineage()
            return lineage is not None
        except Exception:
            return False
