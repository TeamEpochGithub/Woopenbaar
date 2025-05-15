"""Flask application for a RAG-powered chat system with document processing capabilities."""

import argparse
import logging
import os
import sys
from typing import Optional

import torch
from flask import Flask
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.conf.config import Config
from backend.src.api import setup_api
from backend.src.services import (
    BaseLLMService,
    create_chat_history_service,
    create_llm_service,
    create_query_processing_service,
    create_retrieval_service,
    create_safety_service,
)

# Logging is configured in backend/__init__.py
logger = logging.getLogger(__name__)


def create_app(llm_service: Optional[BaseLLMService] = None) -> Flask:
    """Create and configure the Flask application with RAG and LLM services."""
    logger.info("Starting application setup...")

    # Create Flask app
    app = Flask(__name__)
    CORS(app)

    # Create services using factory methods
    if llm_service is None:
        llm_service = create_llm_service()
        if not llm_service:
            raise ValueError(f"Failed to create {Config.LLM_SERVICE} LLM service")

    retrieval_service = create_retrieval_service()
    logger.info(
        f"Retrieval service created with {len(retrieval_service.get_sources())} data sources"
    )

    # Create safety service first
    logger.info("Creating safety service")
    safety_service = create_safety_service()
    logger.info("Safety service created")

    # Create query processing service
    logger.info("Creating query processing service")
    query_processing_service = create_query_processing_service(
        llm_service, retrieval_service, safety_service=safety_service
    )
    logger.info("Query processing service created")

    # Create history service
    logger.info("Creating chat history service")
    history_service = create_chat_history_service()
    logger.info("Chat history service created")

    # Set up API routes
    logger.info("Setting up API routes")
    setup_api(
        app,
        retrieval_service,
        query_processing_service,
        history_service,
        llm_service,
        safety_service,
    )
    logger.info("API routes configured")

    logger.info("Application setup complete")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Raggle with arguments (--llm, --reprocess, --dtype, --vllm-port, --enable-internet, --websearch-api-key)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["local", "vllm", "gemini", "deepseek"],
        default="local",
        help="LLM service to use (default: local)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of documents even if cache exists",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type for LLM (use float16 for GPUs with compute capability < 8.0)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8001,
        help="Port for the vLLM server (default: 8001)",
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default="localhost",
        help="Hostname or IP of the vLLM server (default: localhost)",
    )
    parser.add_argument(
        "--websearch-api-key",
        type=str,
        help="API key for web search service",
    )

    args = parser.parse_args()

    # Set configuration from command line arguments
    Config.FORCE_REPROCESS = args.reprocess
    Config.LLM_SERVICE = args.llm

    # Set the LLM dtype from command line argument
    if not hasattr(Config, "LLM_DTYPE"):
        setattr(Config, "LLM_DTYPE", args.dtype)
    else:
        Config.LLM_DTYPE = args.dtype

    # Update Config with vLLM port
    if not hasattr(Config, "VLLM_PORT"):
        setattr(Config, "VLLM_PORT", args.vllm_port)
    else:
        Config.VLLM_PORT = args.vllm_port

    WEBSEARCH_API_KEY: Optional[str] = args.websearch_api_key
    if args.websearch_api_key:
        if not hasattr(Config, "WEBSEARCH_API_KEY"):
            setattr(Config, "WEBSEARCH_API_KEY", WEBSEARCH_API_KEY)
        else:
            Config.WEBSEARCH_API_KEY = WEBSEARCH_API_KEY  # type: ignore

    # Set default number of internet search results
    INTERNET_SEARCH_RESULTS: int = 5
    if not hasattr(Config, "INTERNET_SEARCH_RESULTS"):
        setattr(Config, "INTERNET_SEARCH_RESULTS", INTERNET_SEARCH_RESULTS)
    else:
        Config.INTERNET_SEARCH_RESULTS = INTERNET_SEARCH_RESULTS  # type: ignore

    logger.info(f"Using LLM service: {Config.LLM_SERVICE}")
    logger.info(f"Using LLM data type: {Config.LLM_DTYPE}")
    logger.info(
        f"Internet search enabled: {getattr(Config, 'ENABLE_INTERNET_SEARCH', False)}"
    )

    # Initialize LLM service
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        llm_service = create_llm_service()
        if not llm_service:
            raise ValueError(f"Failed to create {Config.LLM_SERVICE} LLM service")
        logger.info(
            f"{Config.LLM_SERVICE.capitalize()} LLM service initialized successfully!"
        )
    except Exception as e:
        logger.error(f"Failed to initialize {Config.LLM_SERVICE} LLM service: {str(e)}")
        sys.exit(1)

    # Create app and run
    app = create_app(llm_service)
    app.run(host="0.0.0.0", port=Config.FLASK_PORT)
