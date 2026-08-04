"""
Training Module

This module handles all model training operations for the Protein Embedding Classifier.

Currently re-exports from protein_embedding_classifier.classifiers and protein_embedding_classifier.core for backward compatibility.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

# Re-export from existing modules
from protein_embedding_classifier.classifiers.registry import get_classifier
from protein_embedding_classifier.classifiers.base import BaseClassifier
from protein_embedding_classifier.core.embeddings import EmbeddingStore

# Public API
__all__ = [
    'get_classifier',
    'BaseClassifier',
    'EmbeddingStore',
]
