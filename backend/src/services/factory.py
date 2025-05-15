"""Service factory module for centralized service instantiation.

This module provides factory methods for creating service instances,
keeping service initialization logic in one place.
"""

import logging
from pathlib import Path
from typing import Optional

from backend.conf.config import Config
from backend.src.services.llm import (
    BaseLLMService,
    DeepseekLLMService,
    GeminiLLMService,
    LocalLLMService,
)

# Import the new query processing service
from backend.src.services.query_processing.query_processing_service import (
    QueryProcessingService,
)
from backend.src.services.retrieval import RetrievalService
from backend.src.services.retrieval.components import (
    BaseChunkRetriever,
    BaseDocumentRetriever,
    BaseTokenizer,
    BertTokenizer,
    BM25DocumentRetriever,
    CrossEncoderReranker,
    DateDocumentReranker,
    FaissChunkRetriever,
    SentenceTransformerEmbedder,
    create_preprocessor,
)
from backend.src.services.safety import SafetyService
from backend.src.services.store import ChatHistoryService

logger = logging.getLogger(__name__)


def create_document_retriever(tokenizer: BaseTokenizer) -> BaseDocumentRetriever:
    """Create a document retriever configured with a tokenizer.

    Args:
        tokenizer: Tokenizer for text processing

    Returns:
        Initialized document retriever
    """
    return BM25DocumentRetriever(tokenizer=tokenizer)


def create_chunk_retriever(
    embedding_model: SentenceTransformerEmbedder,
) -> BaseChunkRetriever:
    """Create a chunk retriever configured with an embedding model.

    Args:
        embedding_model: Embedding model for generating embeddings

    Returns:
        Initialized chunk retriever
    """
    return FaissChunkRetriever(embedding_model=embedding_model)


def create_retrieval_service() -> RetrievalService:
    """Create and configure a RetrievalService instance.

    This factory method handles the creation of all components needed for the retrieval service,
    including embedders, rerankers, tokenizers, and retrievers. It also handles document processing
    and indexing if needed.

    Returns:
        Configured RetrievalService instance
    """
    # Ensure cache directory exists
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rag_cache_file = Config.CACHE_DIR / "rag.pkl"
    if rag_cache_file.exists() and not Config.FORCE_REPROCESS:
        logger.info(f"Loading RAG from cache: {rag_cache_file}")
        return RetrievalService.from_cache(str(Config.CACHE_DIR))

    logger.info(
        "Cache not found or reprocessing forced, initializing retrieval service components"
    )

    # Initialize RAG retrieval components
    logger.info(f"Initializing dense embedder: {Config.DENSE_EMBEDDER_MODEL}")
    embedder = SentenceTransformerEmbedder(
        model_name_or_path=Config.DENSE_EMBEDDER_MODEL,
        device=Config.DEVICE,
    )

    logger.info(f"Initializing chunk reranker: {Config.CHUNK_RERANKER_MODEL}")
    chunk_reranker = CrossEncoderReranker(
        model_name_or_path=Config.CHUNK_RERANKER_MODEL,
        device=Config.DEVICE,
    )
    document_reranker = DateDocumentReranker(
        mode=Config.DOCUMENT_RERANKER_MODE,
        device=Config.DEVICE,
    )

    # Extract tokenizer from reranker
    logger.info(f"Initializing tokenizer with max_length={Config.TOKENIZER_MAX_LENGTH}")
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path=Config.CHUNK_RERANKER_MODEL,
        max_length=Config.TOKENIZER_MAX_LENGTH,
    )

    # Create initialized retrievers
    logger.info("Creating document retriever")
    document_retriever = create_document_retriever(tokenizer)

    logger.info("Creating chunk retriever")
    chunk_retriever = create_chunk_retriever(embedder)

    # Create RAG service with initialized components
    logger.info("Initializing retrieval service")
    retrieval_service = RetrievalService(
        tokenizer=tokenizer,
        embedding_model=embedder,
        document_retriever=document_retriever,
        chunk_retriever=chunk_retriever,
        document_reranker=document_reranker,
        chunk_reranker=chunk_reranker,
    )

    # Process documents from multiple data sources
    documents_path = Path(Config.DOCUMENTS_PATH)

    if documents_path.exists():

        # Create preprocessing pipeline using the factory method
        logger.info(
            f"Creating document preprocessor (semantic_chunker={Config.USE_SEMANTIC_CHUNKER})"
        )
        preprocessor = create_preprocessor(
            use_semantic_chunker=Config.USE_SEMANTIC_CHUNKER
        )

        # Check if there are data source subdirectories
        data_sources = [item for item in documents_path.iterdir() if item.is_dir()]
        if len(data_sources) == 0:
            logger.error(f"No data sources found in {documents_path}")
            raise ValueError(f"No data sources found in {documents_path}")

        logger.info(f"Found {len(data_sources)} data sources to process")

        # Process each data source directory
        for source_dir in data_sources:
            source_name = source_dir.name
            source_description = ""

            # Try to read description.txt if it exists
            description_file = source_dir / "description.txt"
            if description_file.exists():
                with open(description_file, "r", encoding="utf-8") as f:
                    source_description = f.read().strip()
            else:
                logger.warning(f"No description file found for source '{source_name}'")

            # Check for documents directory
            documents_dir = source_dir / "docs"
            if not documents_dir.exists() or not documents_dir.is_dir():
                logger.warning(
                    f"No 'docs' directory found in data source '{source_name}', skipping"
                )
                continue

            # Load and preprocess documents for this source
            chunked_documents = preprocessor.load_and_preprocess_mp(documents_dir, 8)

            # Index documents for this source
            logger.info(
                f"Indexing {len(chunked_documents)} documents for source '{source_name}'"
            )
            retrieval_service.index(chunked_documents, source_name, source_description)

        # Save RAG service for future use
        logger.info(f"Saving retrieval service to cache: {Config.CACHE_DIR}")
        retrieval_service.save(str(Config.CACHE_DIR))

    else:
        error_msg = f"Could not find the documents path: {documents_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return retrieval_service


def create_chat_history_service() -> ChatHistoryService:
    """Create and configure a ChatHistoryService instance.

    Returns:
        Configured ChatHistoryService instance
    """
    return ChatHistoryService()


def create_safety_service() -> SafetyService:
    """Create and configure a SafetyService instance.

    Returns:
        Configured SafetyService instance
    """
    return SafetyService()


def create_llm_service() -> BaseLLMService:
    """Create and initialize the LLM service based on configuration."""
    try:
        if Config.LLM_SERVICE == "local":
            # For local mode, we run the server internally and use LocalLLMService
            return LocalLLMService()
        elif Config.LLM_SERVICE == "vllm":
            # For vllm mode, we connect to an external server
            from backend.src.services.llm.vllm_client_service import VLLMClientService

            vllm_url = f"http://{Config.VLLM_HOST}:{Config.VLLM_PORT}"
            return VLLMClientService(api_base_url=vllm_url)
        elif Config.LLM_SERVICE == "gemini":
            return GeminiLLMService()
        elif Config.LLM_SERVICE == "deepseek":
            return DeepseekLLMService()
        else:
            raise ValueError(f"Unsupported LLM service: {Config.LLM_SERVICE}")
    except Exception as e:
        logger.error(f"Failed to create {Config.LLM_SERVICE} LLM service: {e}")
        raise e


def create_query_processing_service(
    llm_service: Optional[BaseLLMService] = None,
    retrieval_service: Optional[RetrievalService] = None,
    safety_service: Optional[SafetyService] = None,
) -> QueryProcessingService:
    """Create and configure a QueryProcessingService instance using the refactored components.

    Args:
        llm_service: LLM service for language model interactions
        retrieval_service: Service for document retrieval
        safety_service: Service for content safety validation

    Returns:
        Configured QueryProcessingService instance
    """
    # Create services if not provided
    if llm_service is None:
        logger.info("No LLM service provided, creating new one")
        llm_service = create_llm_service()

    if retrieval_service is None:
        logger.info("No retrieval service provided, creating new one")
        retrieval_service = create_retrieval_service()

    if safety_service is None:
        logger.info("No safety service provided, creating new one")
        safety_service = create_safety_service()

    # Create the query processing service with all required services
    logger.info("Initializing QueryProcessingService with components")
    query_processing_service = QueryProcessingService(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        safety_service=safety_service,
    )

    return query_processing_service
