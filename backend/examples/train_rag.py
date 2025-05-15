"""Example script that demonstrates how to initialize and train a RAG system from scratch."""

import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.conf.config import Config
from backend.src.services.factory import create_retrieval_service
from backend.src.services.retrieval.training.training_config import (
    EmbedderTrainingConfig,
    RAGTrainingConfig,
    RerankerTrainingConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize and train a RAG system.

    The function performs the following steps:
    1. Initializes all RAG components with default configurations
    2. Creates a RetrievalService instance
    3. Indexes the document collection
    4. Configures training parameters for embedder and rerankers
    5. Trains the components using pre-generated data
    6. Saves the trained models
    """
    # Initialize RAG components using the same setup as app.py
    logger.info("Initializing RAG...")

    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Force reprocessing for training to ensure fresh index
    # Config.FORCE_REPROCESS = True
    retrieval_service = create_retrieval_service()

    # Use training data
    training_data_path = os.path.join(Config.DATA_DIR, "train")

    # Configure training with more stable parameters
    config = RAGTrainingConfig(
        embedder=EmbedderTrainingConfig(
            batch_size=128,
            mini_batch_size=64,
            epochs=200,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            eval_split=0.1,
            bf16=True,
            distributed=True,
        ),
        chunk_reranker=RerankerTrainingConfig(
            batch_size=64,
            epochs=200,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            eval_split=0.1,
            bf16=True,
            max_seq_length=512,
        ),
        document_reranker=None,
        device=Config.DEVICE,  # Use consistent device configuration
        output_dir=str(Config.MODELS_DIR),
    )

    # Train using generated data
    logger.info("Training RAG components...")
    retrieval_service.train(config, data_path=training_data_path)

    # Save trained service
    logger.info("Saving trained RAG service...")
    retrieval_service.save(str(cache_dir))

    logger.info("RAG training complete!")


if __name__ == "__main__":
    main()
