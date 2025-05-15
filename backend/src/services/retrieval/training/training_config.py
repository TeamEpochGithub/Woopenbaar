"""Training configuration module.

This module provides dataclasses for configuring the training of various retrieval models
including embedders, rerankers and RAG models. The configuration classes specify hyperparameters
and training settings for each model type.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbedderTrainingConfig:
    """Configuration for training embedding models.

    Attributes:
        batch_size: Total batch size across all devices
        mini_batch_size: Batch size per device for distributed training
        epochs: Number of training epochs
        learning_rate: Initial learning rate for optimizer
        warmup_ratio: Fraction of training steps for LR warmup
        eval_split: Fraction of data to use for evaluation
        bf16: Whether to use bfloat16 precision
        distributed: Whether to use distributed training
    """

    batch_size: int
    mini_batch_size: int  # Used for gradient accumulation in distributed training
    epochs: int
    learning_rate: float
    warmup_ratio: float  # Fraction of steps for linear warmup
    eval_split: float  # Fraction of data used for evaluation
    bf16: bool  # Whether to use bfloat16 mixed precision
    distributed: bool  # Whether to use DistributedDataParallel


@dataclass
class RerankerTrainingConfig:
    """Configuration for training reranking models.

    Attributes:
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate for optimizer
        warmup_ratio: Fraction of training steps for LR warmup
        eval_split: Fraction of data to use for evaluation
        bf16: Whether to use bfloat16 precision
        max_seq_length: Maximum sequence length for input text
    """

    batch_size: int
    epochs: int
    learning_rate: float
    warmup_ratio: float  # Fraction of steps for linear warmup
    eval_split: float  # Fraction of data used for evaluation
    bf16: bool  # Whether to use bfloat16 mixed precision
    max_seq_length: int  # Maximum input sequence length


@dataclass
class RAGTrainingConfig:
    """Configuration for training a complete RAG model.

    Contains configurations for the embedder and reranker components, as well as
    training infrastructure settings.

    Attributes:
        embedder: Configuration for training the embedding model
        chunk_reranker: Configuration for training the chunk reranker
        document_reranker: Configuration for training the document reranker
        output_dir: Directory to save model checkpoints and logs
        device: Device to use for training (e.g. 'cuda', 'cpu')
    """

    embedder: Optional[EmbedderTrainingConfig]  # May train without embedder
    chunk_reranker: Optional[RerankerTrainingConfig]  # May train without chunk reranker
    document_reranker: Optional[
        RerankerTrainingConfig
    ]  # May train without doc reranker
    output_dir: str  # Directory for saving checkpoints and logs
    device: str  # Device for training (cuda/cpu)
