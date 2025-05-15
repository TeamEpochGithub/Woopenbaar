"""
Distributed document processing implementation.

This module enables parallel processing of documents across multiple machines:
1. Divides work into chunks
2. Processes chunks independently
3. Synchronizes results across machines

Required Environment:
    - SSH access to base machine (if distributed)
    - CUDA-capable device
    - Logging directory structure

Configuration:
    - chunk_id: Identifier for current processing chunk
    - total_chunks: Total number of parallel processes
    - device: CUDA device specification
    - base_machine: SSH connection string
    - sync_interval: Document count between syncs
"""

# main.py from publicatie_rapport plus documents to standard data format
import argparse
from pathlib import Path

import yaml

from standard_data_format.src.document_processing_pipeline import (
    DocumentProcessingPipeline,
)
from standard_data_format.utils.logger import setup_logger

# Create logger
logger = setup_logger()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.debug(f"Raw config loaded from {config_path}")
        logger.debug(f"PDF converter section: {config.get('pdf_converter', {})}")

        # Get PDF converter settings
        pdf_config = config.get("pdf_converter", {})

        # Merge configurations with explicit PDF converter settings
        merged_config = {
            **config["marker"],
            **config["paths"],
            "allow_regular_conversion": pdf_config.get(
                "allow_regular_conversion", False
            ),
            "allow_ocr": pdf_config.get("allow_ocr", False),
            "allow_llm": pdf_config.get("allow_llm", True),
            "use_llm": True,
            "use_local_model": True,
            "pdf_converter": pdf_config,  # Include the full pdf_converter section
        }

        logger.debug(f"Merged config: {merged_config}")
        logger.debug(
            f"allow_regular_conversion: {merged_config.get('allow_regular_conversion')}"
        )
        logger.debug(f"allow_ocr: {merged_config.get('allow_ocr')}")
        logger.debug(f"allow_llm: {merged_config.get('allow_llm')}")

        return merged_config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise


def main():
    """
    Entry point for distributed document processing.

    Command Line Arguments:
        --chunk: Chunk ID to process (0-based)
        --total-chunks: Total number of chunks (default: 6)
        --device: CUDA device to use (default: "cuda:0")
        --base-machine: SSH string for base machine
        --sync-interval: Documents between syncs (default: 10)
        --verbose: Show detailed output
        --custom-metadata: Pre-divided metadata file path
        --config: Path to configuration file (local_config.yaml or gemini_config.yaml)

    Processing Flow:
        1. Parse command line arguments
        2. Configure logging for this chunk
        3. Initialize processing pipeline
        4. Process assigned chunk of documents
        5. Combine metadata from all chunks
    """
    parser = argparse.ArgumentParser(
        description="Process documents in distributed mode"
    )
    parser.add_argument("--chunk", type=int, help="Chunk ID to process (0-based)")
    parser.add_argument(
        "--total-chunks", type=int, default=6, help="Total number of chunks"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--base-machine", type=str, help="Base machine SSH string (e.g., user@hostname)"
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=10,
        help="Number of documents to process before syncing",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output for this process"
    )
    parser.add_argument(
        "--custom-metadata", type=str, help="Path to a pre-divided metadata file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (local_config.yaml or gemini_config.yaml)",
    )
    args = parser.parse_args()

    # Configure logging with chunk information
    global logger
    logger = setup_logger(
        chunk_id=args.chunk, verbose=args.verbose, log_dir=Path("logs")
    )

    logger.info("Starting document processing pipeline...")

    # Load configuration from file
    config = load_config(args.config)

    # Override config values with command line arguments
    config["device"] = args.device
    config["sync_interval"] = args.sync_interval
    if args.custom_metadata:
        config["metadata_path"] = args.custom_metadata
    config["base_machine"] = args.base_machine

    # Initialize pipeline with chunk information
    pipeline = DocumentProcessingPipeline(
        config=config,
        test=False,
        base_machine=args.base_machine,
        sync_interval=args.sync_interval,
        chunk_id=args.chunk,
        metadata_path=config["metadata_path"],
    )

    pipeline.process_documents_distributed(
        chunk_id=args.chunk, total_chunks=args.total_chunks
    )


if __name__ == "__main__":
    main()
