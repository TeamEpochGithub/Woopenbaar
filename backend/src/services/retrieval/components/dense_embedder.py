"""Text embedding module.

This module provides embedder classes for generating text embeddings.
The module includes an abstract base class and a SentenceTransformer implementation.
"""

import abc
import json
import logging
import os
from typing import Any, List, Tuple, TypedDict, Union, cast

import numpy as np
from datasets import Dataset, DatasetDict  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    evaluation,
    losses,
)
from torch import Tensor

from ..training.training_config import EmbedderTrainingConfig

logger = logging.getLogger(__name__)


class TrainingPair(TypedDict):
    """Structure for training pair data.

    Each training pair consists of a query and a matching document chunk.

    Attributes:
        query: The query text
        chunk: The corresponding document chunk text
    """

    query: str
    chunk: str


class BaseDenseEmbedder(abc.ABC):
    """Base class for text embedders.

    Defines the interface for text embedding systems.

    Attributes:
        device (str): Device to run the model on ('cuda', 'cpu', etc.)
    """

    def __init__(self, device: str) -> None:
        """Initialize the embedder.

        Args:
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        self.device = device

    @abc.abstractmethod
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Union[Tensor, np.ndarray[Any, np.dtype[Any]]]:
        """Generate embeddings for a list of sentences.

        Args:
            sentences: List of text strings to embed
            batch_size: Number of sentences to process in each batch
            show_progress_bar: Whether to display a progress bar during encoding
            convert_to_numpy: Whether to convert the output to a numpy array
            normalize_embeddings: Whether to L2-normalize the embeddings

        Returns:
            Tensor or numpy array of embeddings with shape (len(sentences), embedding_dim)
        """

    @abc.abstractmethod
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embeddings.

        Returns:
            Embedding dimension as an integer
        """

    def train(self, train_data_path: str, config: Any, output_dir: str) -> None:
        """Train the embedder model.

        Default implementation logs a warning. Subclasses should override
        this method if they support training.

        Args:
            train_data_path: Path to training data
            config: Training configuration
            output_dir: Output directory for trained model
        """
        logger.warning(
            f"{self.__class__.__name__} does not implement training functionality"
        )

    def to(self, device: str) -> "BaseDenseEmbedder":
        """Move model to specified device.

        Default implementation logs a warning. Subclasses should override
        this method if they support device movement.

        Args:
            device: Device to move model to ('cuda', 'cpu', etc.)

        Returns:
            Self for method chaining
        """
        logger.warning(f"{self.__class__.__name__} does not implement device movement")
        return self


class SentenceTransformerEmbedder(BaseDenseEmbedder):
    """Wrapper for fine-tunable dense embedding model with modern training API.

    This class provides a high-level interface for working with SentenceTransformer models,
    including training, inference, and model management functionality.

    Attributes:
        device (str): Device to run the model on ('cuda', 'cpu', etc.)
        model (SentenceTransformer): The underlying SentenceTransformer model
    """

    def __init__(self, model_name_or_path: str, device: str) -> None:
        """Initialize the model from a model name (HuggingFace) or a local path.

        Args:
            model_name_or_path: Either a model name from HuggingFace or a path to a saved model
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        super().__init__(device)
        # Convert Path object to string if needed
        self.model = SentenceTransformer(model_name_or_path, device=device)  # type: ignore

    def encode(  # type: ignore
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Union[Tensor, np.ndarray[Any, np.dtype[Any]]]:
        """Generate embeddings for a list of sentences.

        Args:
            sentences: List of text strings to embed
            batch_size: Number of sentences to process in each batch
            show_progress_bar: Whether to display a progress bar during encoding
            convert_to_numpy: Whether to convert the output to a numpy array
            normalize_embeddings: Whether to L2-normalize the embeddings

        Returns:
            Tensor or numpy array of embeddings with shape (len(sentences), embedding_dim)
        """
        return self.model.encode(  # type: ignore
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            device=self.device,
        )

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embeddings.

        Returns:
            Embedding dimension as an integer
        """
        dimension = self.model.get_sentence_embedding_dimension()
        assert dimension is not None, "Embedding dimension should not be None"
        return dimension

    def train(
        self,
        train_data_path: str,
        config: EmbedderTrainingConfig,
        output_dir: str,
    ) -> None:
        """Train the embedder model on provided data.

        Args:
            train_data_path: Path to JSON file containing training data
            config: Configuration object containing training parameters
            output_dir: Directory to save trained model and logs

        The training data should be in JSON format with a list of dictionaries,
        each containing 'query' and 'chunk' keys. The format is enforced using
        the TrainingPair TypedDict.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logging_dir = os.path.join(output_dir, "logs")

        logger.info(f"Starting embedder training with data from {train_data_path}")

        # Load data from JSON file
        with open(train_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Extract training pairs - expect a list format only
        pairs: List[TrainingPair] = []

        if isinstance(raw_data, dict) and "pairs" in raw_data:
            # If data is wrapped in a dictionary with 'pairs' key
            if isinstance(raw_data["pairs"], list):
                pairs = cast(List[TrainingPair], raw_data["pairs"])
            else:
                logger.error(
                    "Invalid data format. The 'pairs' key must contain a list of dictionaries"
                )
                return
        elif isinstance(raw_data, list):
            # If data is directly a list of pairs
            pairs = cast(List[TrainingPair], raw_data)
        else:
            logger.error(
                "Invalid data format. Expected a list of dictionaries or a dictionary with 'pairs' key"
            )
            return

        if not pairs:
            logger.warning("No training pairs found")
            return

        # Validate each pair has the required fields
        for i, pair in enumerate(pairs):
            if "query" not in pair or "chunk" not in pair:
                logger.error(
                    f"Invalid pair at index {i}. Each pair must have 'query' and 'chunk' keys"
                )
                return

        logger.info(f"Loaded {len(pairs)} training pairs")

        # Prepare training data - convert to tuples for training
        train_data: List[Tuple[str, str]] = [
            (pair["query"], pair["chunk"]) for pair in pairs
        ]

        # Split data into train and eval sets if eval_split > 0
        if config.eval_split > 0:
            # Shuffle the data
            np.random.shuffle(train_data)
            split_idx = int(len(train_data) * (1 - config.eval_split))
            train_pairs = train_data[:split_idx]
            eval_pairs = train_data[split_idx:]

            # Create evaluation data
            eval_queries = {str(i): query for i, (query, _) in enumerate(eval_pairs)}
            eval_corpus = {str(i): chunk for i, (_, chunk) in enumerate(eval_pairs)}
            eval_relevant = {str(i): {str(i)} for i in range(len(eval_pairs))}

            # Create evaluator
            evaluator = evaluation.InformationRetrievalEvaluator(  # type: ignore
                eval_queries,
                eval_corpus,
                eval_relevant,
                show_progress_bar=True,
                corpus_chunk_size=1000,
                precision_recall_at_k=[1, 5],
                accuracy_at_k=[1, 5],
                ndcg_at_k=[10],
                mrr_at_k=[10],
                map_at_k=[10],
            )

            # Update train_data to only include training pairs
            train_data = train_pairs
            logger.info(
                f"Split data: {len(train_pairs)} training pairs, {len(eval_pairs)} evaluation pairs"
            )
        else:
            evaluator = None
            logger.info("Using all data for training (no evaluation split)")

        # Convert training data to dataset format
        train_dataset = Dataset.from_dict(  # type: ignore
            {
                "anchor": [q for q, _ in train_data],
                "positive": [c for _, c in train_data],
            }
        )

        # Configure loss - use MultipleNegativesRankingLoss
        loss = losses.CachedMultipleNegativesRankingLoss(
            self.model, mini_batch_size=config.mini_batch_size, show_progress_bar=False
        )

        # Create dataset dict
        dataset = DatasetDict({"train": train_dataset})  # type: ignore

        # Training arguments
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            bf16=config.bf16,
            eval_strategy="epoch" if evaluator else "no",
            metric_for_best_model="eval_cosine_recall@5" if evaluator else None,
            greater_is_better=True,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            dataloader_drop_last=True,
            warmup_ratio=config.warmup_ratio,
            max_steps=-1,
            ddp_find_unused_parameters=False,
            local_rank=-1,
            logging_strategy="epoch",
            logging_first_step=False,
            logging_dir=logging_dir,
        )

        logger.info(
            f"Training config: epochs={config.epochs}, batch_size={config.batch_size}, learning_rate={config.learning_rate}"
        )

        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset["train"],
            loss=loss,
            evaluator=evaluator,
        )

        # Run initial evaluation before training
        if evaluator:
            logger.info("Running initial evaluation before training")
            initial_eval = trainer.evaluate()
            logger.info(f"Initial evaluation results: {initial_eval}")

        logger.info("Beginning sentence transformer training")
        trainer.train()  # type: ignore

        # Save final model
        logger.info(f"Training complete, saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        logger.info("Model saved successfully")

    def save(self, path: str) -> None:
        """Save model weights and parameters to path.

        Args:
            path: Directory path where model should be saved
        """
        logger.info(f"Saving sentence transformer model to {path}")
        self.model.save(path)  # type: ignore

    def load(self, path: str) -> None:
        """Load model weights and parameters from path.

        Args:
            path: Path to directory containing saved model
        """
        logger.info(f"Loading sentence transformer model from {path}")
        self.model = SentenceTransformer(path, device=self.device)  # type: ignore

    def to(self, device: str) -> "SentenceTransformerEmbedder":
        """Move model to specified device.

        Args:
            device: Device to move model to ('cuda', 'cpu', etc.)

        Returns:
            Self for method chaining
        """
        logger.info(f"Moving sentence transformer model to {device}")
        self.device = device
        self.model.to(device)
        return self
