"""Cross-encoder reranking module.

This module provides reranking classes for document chunks based on relevance.
The module includes an abstract base class and a cross-encoder based implementation.
"""

import abc
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset  # type: ignore
from sentence_transformers import CrossEncoder  # type: ignore
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainingArguments  # type: ignore
from transformers import default_data_collator  # type: ignore

from backend.src.data_classes import DocumentChunk

from ..training.training_config import RerankerTrainingConfig

logger = logging.getLogger(__name__)


class BaseChunkReranker(abc.ABC):
    """Base class for chunk rerankers.

    Defines the interface for reranking systems that work with document chunks or full documents.

    Attributes:
        device (str): Device to run model on ('cuda', 'cpu', etc.)
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize the chunk reranker.

        Args:
            device: Device to run model on ('cuda', 'cpu', etc.)
        """
        logger.info(f"Initializing chunk reranker on device: {device}")
        self.device = device

    @abc.abstractmethod
    def rerank(
        self, query: str, items: List[DocumentChunk], k: int
    ) -> List[DocumentChunk]:
        """Rerank items based on query relevance.

        Args:
            query: Query string to rank items against
            items: List of items to rerank
            k: Number of top items to return

        Returns:
            List of top-k reranked items
        """

    def train(self, train_data_path: str, config: Any, output_dir: str) -> None:
        """Train the reranker model.

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


class CrossEncoderReranker(BaseChunkReranker):
    """Cross-encoder based reranker.

    Uses a cross-encoder model to rerank items based on their text content.
    Works with any type that has a 'content' attribute or method.

    Attributes:
        device (str): Device to run model on ('cuda', 'cpu', etc.)
        max_length (int): Maximum sequence length for tokenization
        chunk_strategy (str): Strategy for handling long texts ('max', 'mean', 'first')
        model (CrossEncoder): The underlying cross-encoder model
        tokenizer (AutoTokenizer): Tokenizer for the model
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_length: int = 512,
        chunk_strategy: str = "max",
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name_or_path: HuggingFace model name or path to local model
            device: Device to run model on ('cuda', 'cpu', etc.)
            max_length: Maximum sequence length for tokenization
            chunk_strategy: Strategy for handling long texts ('max', 'mean', 'first')
        """
        super().__init__(device)
        self.max_length = max_length
        self.chunk_strategy = chunk_strategy
        self.model = CrossEncoder(model_name_or_path, device=device)  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # type: ignore
        # Explicitly move model to device
        self.model.model = self.model.model.to(self.device)  # type: ignore

    def rerank(
        self, query: str, items: List[DocumentChunk], k: int
    ) -> List[DocumentChunk]:
        """Rerank items using the cross-encoder without truncation.

        Args:
            query: Query string to rank items against
            items: List of items to rerank
            k: Number of top items to return

        Returns:
            List of top-k reranked items

        Note:
            Items must have a 'content' attribute or method that returns text content.
            Falls back to original order if reranking fails.
        """
        if not items:
            return []

        logger.debug(f"Reranking {len(items)} items with cross-encoder")

        try:
            # Get content from items (supports both attribute and method)
            contents: List[str] = []
            for item in items:
                if hasattr(item, "content"):
                    content = item.content
                    contents.append(content)
                else:
                    logger.error(f"Item {item} has no content attribute/method")
                    return items[:k]

            # Score all items
            scores: List[float] = []
            for content in contents:
                score = self._score_long_content(query, content)
                scores.append(score)

            # Create (item, score) pairs and sort by score
            item_scores = list(zip(items, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top-k items
            reranked_items = [item for item, _ in item_scores[:k]]
            logger.debug(
                f"Reranking complete, returning top {len(reranked_items)} items"
            )

            return reranked_items
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            # Fall back to original items (limited to k) if reranking fails
            return items[:k]

    def _score_long_content(self, query: str, content: str) -> float:
        """Score a potentially long content by splitting into chunks and aggregating scores.

        Args:
            query: Query string to score content against
            content: Content text to score

        Returns:
            Float score indicating relevance of content to query

        Note:
            Handles long content by splitting into chunks that fit within max_length.
            Uses chunk_strategy to aggregate scores from multiple chunks.
        """
        # If content is short enough, just score directly
        query_tokens = self.tokenizer.tokenize(query)  # type: ignore
        content_tokens = self.tokenizer.tokenize(content)  # type: ignore

        if (
            len(query_tokens) + len(content_tokens) <= self.max_length - 3  # type: ignore
        ):  # Account for special tokens
            return float(self.model.predict([(query, content)]))  # type: ignore

        # Content is too long, need to chunk
        logger.debug(
            f"Content too long ({len(content_tokens)} tokens), chunking for scoring"  # type: ignore
        )

        # Split content into chunks that fit within max_length along with the query
        max_content_tokens = (
            self.max_length - len(query_tokens) - 3  # type: ignore
        )  # Account for special tokens

        # Create chunks based on tokens
        chunks: List[str] = []
        current_chunk: List[Any] = []
        current_length = 0

        for token in content_tokens:  # type: ignore
            if current_length + 1 > max_content_tokens:
                # Current chunk is full, convert to text and start a new chunk
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))  # type: ignore
                current_chunk = [token]
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))  # type: ignore

        # Score each chunk with the query
        chunk_pairs = [(query, chunk) for chunk in chunks]
        chunk_scores = self.model.predict(chunk_pairs, show_progress_bar=False)  # type: ignore

        # Aggregate scores based on strategy
        if self.chunk_strategy == "max":
            final_score = float(np.max(chunk_scores))  # type: ignore
        elif self.chunk_strategy == "mean":
            final_score = float(np.mean(chunk_scores))  # type: ignore
        elif self.chunk_strategy == "first":
            final_score = float(chunk_scores[0])  # type: ignore
        else:
            # Default to max
            final_score = float(np.max(chunk_scores))  # type: ignore

        return final_score

    def train(
        self, train_data_path: str, config: RerankerTrainingConfig, output_dir: str
    ) -> None:
        """Train the cross-encoder model on provided data.

        Args:
            train_data_path: Path to JSON file containing training data
            config: Configuration object containing training parameters
            output_dir: Directory to save trained model and logs

        Note:
            Training data should be a JSON file containing a list of dictionaries with
            'query', 'text', and 'label' fields.
        """
        print(f"\nLoading cross-encoder training data from {train_data_path}")

        # Load training data
        with open(train_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print("WARNING: No training data found")
            return

        print(f"Loaded {len(data)} training examples")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Use the current model's configuration
        model_name = self.model.model.config._name_or_path  # type: ignore
        tokenizer = self.tokenizer  # type: ignore
        model = AutoModelForSequenceClassification.from_pretrained(model_name)  # type: ignore

        # Define compute_metrics function for evaluation
        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            logits, labels = eval_pred
            # Convert logits to probabilities using sigmoid (binary classification)
            scores = 1 / (1 + np.exp(-logits))
            predictions = (scores > 0.5).astype(int)

            # Convert to 1D arrays
            predictions = predictions.reshape(-1)
            labels = labels.reshape(-1).astype(int)

            # Calculate confusion matrix components
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            total = len(labels)
            accuracy = (tp + tn) / total * 100

            print("\nEvaluation Results:")
            print(f"Accuracy: {accuracy:.2f}%")
            print("Confusion Matrix:")
            print(f"TP: {tp} | FP: {fp}")
            print(f"FN: {fn} | TN: {tn}")

            return {
                "accuracy": float(accuracy),
                "tp": float(tp),
                "fp": float(fp),
                "tn": float(tn),
                "fn": float(fn),
            }

        # Split into train and eval sets
        import random

        random.shuffle(data)
        split_idx = int(len(data) * (1 - config.eval_split))
        train_data = data[:split_idx]
        eval_data = data[split_idx:] if config.eval_split > 0 else []  # type: ignore

        print("\nSplit sizes:")
        print(f"Training set: {len(train_data)} examples")
        print(f"Evaluation set: {len(eval_data)} examples")  # type: ignore

        # Create datasets

        train_dataset = Dataset.from_list(train_data)  # type: ignore
        eval_dataset = Dataset.from_list(eval_data) if eval_data else None  # type: ignore

        # Tokenize datasets
        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            tokenized = tokenizer(  # type: ignore
                examples["query"],
                examples["text"],
                truncation=True,
                max_length=config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Add labels separately
            tokenized["labels"] = [float(label) for label in examples["label"]]

            # Remove original columns to avoid duplicating data
            return tokenized  # type: ignore

        # Apply tokenization with batched processing
        train_dataset = train_dataset.map(  # type: ignore
            tokenize_function, batched=True, remove_columns=["query", "text", "label"]
        )

        if eval_dataset:
            eval_dataset = eval_dataset.map(  # type: ignore
                tokenize_function,
                batched=True,
                remove_columns=["query", "text", "label"],
            )

        # Set up training arguments
        training_args = TrainingArguments(  # type: ignore
            output_dir=output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_ratio=config.warmup_ratio,
            learning_rate=config.learning_rate,
            bf16=config.bf16,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="accuracy" if eval_dataset else None,
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
        )

        # Initialize trainer
        trainer = Trainer(  # type: ignore
            model=model,  # type: ignore
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # type: ignore
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
        )

        # Run initial evaluation before training
        if eval_dataset:
            print("\nRunning initial evaluation before training:")
            initial_eval = trainer.evaluate()  # type: ignore
            print(f"Initial evaluation results: {initial_eval}")

        print("\nStarting cross-encoder training...")
        trainer.train()  # type: ignore

        # Save model
        print(f"\nSaving trained model to {output_dir}")
        model.save_pretrained(output_dir)  # type: ignore
        tokenizer.save_pretrained(output_dir)  # type: ignore

        # Update the cross-encoder with the newly trained model
        self.model = CrossEncoder(output_dir, device=self.device)  # type: ignore
        print("Cross-encoder training complete!")

    def save(self, path: str) -> None:
        """Save model weights and parameters to path.

        Args:
            path: Directory path where model should be saved
        """
        self.model.save(path)  # type: ignore

    def load(self, path: str) -> None:
        """Load model weights and parameters from path.

        Args:
            path: Path to directory containing saved model
        """
        logger.info(f"Loading cross-encoder model from {path}")
        self.model = CrossEncoder(path, device=self.device)  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(path)  # type: ignore
