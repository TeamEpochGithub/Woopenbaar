"""Training module for retrieval models.

This module provides configuration classes and validation utilities for training
various retrieval models including embedders, rerankers and RAG models. It handles
training data validation and configuration management to ensure robust model training.

The module exposes configuration classes for different model types as well as
validation utilities to verify training data format and quality.
"""

from .train_data_validation import TrainingDataValidator, validate_training_files
from .training_config import (  # Configuration for training embedding models; Configuration for training RAG models; Configuration for training reranking models
    EmbedderTrainingConfig,
    RAGTrainingConfig,
    RerankerTrainingConfig,
)

# Expose key classes and functions at package level
__all__ = [
    # Training configuration classes for different model types
    "EmbedderTrainingConfig",  # Config for embedding model training
    "RerankerTrainingConfig",  # Config for reranker model training
    "RAGTrainingConfig",  # Config for RAG model training
    # Utilities for validating training data
    "TrainingDataValidator",  # Class for validating training data
    "validate_training_files",  # Function to validate training file formats
]
