"""Example script that demonstrates how to generate synthetic training data and train a RAG system."""

import logging
import os
import sys
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.conf.config import Config
from backend.src.services.factory import create_llm_service, create_retrieval_service
from backend.src.services.retrieval.training.training_config import (
    EmbedderTrainingConfig,
    RAGTrainingConfig,
    RerankerTrainingConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Initialize and a RAG system, generate training data and train the RAG components.

    The function performs the following steps:
    1. Loads a pre-initialized RAG service from cache
    2. Sets up an LLM service for synthetic data generation
    3. Generates training data samples for different components
    4. Configures training parameters for embedder and rerankers
    5. Trains the components and saves the updated models
    """
    # Load or initialize the RAG service using the same setup as app.py
    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use existing cache if available, otherwise build from scratch
    retrieval_service = create_retrieval_service()

    # Initialize LLM for data generation
    logger.info("Initializing LLM for data generation...")
    llm_service = create_llm_service()

    # Set up training data directory
    data_dir = Path(Config.DATA_DIR) / "train"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    logger.info("Generating training data...")
    retrieval_service.generate_training_data(
        llm_service=llm_service,
        output_dir=str(data_dir),
        embedder_samples=3000,
        chunk_reranker_samples=1000,
        doc_reranker_samples=1,
        source_name="vws",
    )

    # Configure training parameters
    embedder_config = EmbedderTrainingConfig(
        batch_size=128,
        mini_batch_size=64,
        epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_split=0.1,
        bf16=True,
        distributed=True,
    )

    reranker_config = RerankerTrainingConfig(
        batch_size=32,
        epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_split=0.1,
        bf16=True,
        max_seq_length=512,
    )

    # Create RAG training config
    training_config = RAGTrainingConfig(
        embedder=embedder_config,
        chunk_reranker=reranker_config,
        document_reranker=reranker_config,
        output_dir=str(Config.MODELS_DIR),
        device=Config.DEVICE,  # Use consistent device configuration
    )

    # Train RAG components
    logger.info("Training RAG components...")
    retrieval_service.train(config=training_config, data_path=str(data_dir))

    # Save the trained RAG service
    logger.info("Saving trained RAG service...")
    retrieval_service.save(str(Config.CACHE_DIR))

    logger.info("RAG generation and training complete!")


if __name__ == "__main__":
    main()
