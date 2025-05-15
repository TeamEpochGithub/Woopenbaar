"""Retrieval Augmented Generation (RAG) service for document search and retrieval.

This module implements a complete RAG pipeline that combines document and chunk-level
retrieval with optional reranking capabilities. The RetrievalService class manages
indexing and retrieval of documents and their chunks, supporting both simple retrieval
and advanced reranking.

The service can be configured with various components:
- Document and chunk retrievers for initial candidate selection
- Document and chunk rerankers for improved relevance ranking
- Embedding model for semantic search
- Tokenizer for text processing

The service also supports:
- Caching/persistence of the indexed data
- Training data generation for components
- Training of retrieval and reranking components
- Multiple data sources for compartmentalized document retrieval
"""

import copy
import json
import logging
import os
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from backend.conf.config import Config
from backend.src.data_classes import (
    ChunkedDocument,
    ChunkID,
    DocumentChunk,
    FilterOptions,
)
from backend.src.services.llm import BaseLLMService

from .components import (
    BaseChunkRetriever,
    BaseDenseEmbedder,
    BaseDocumentRetriever,
    BaseTokenizer,
    CrossEncoderReranker,
    DateDocumentReranker,
)
from .training.data_generator import TrainingDataGenerator
from .training.training_config import RAGTrainingConfig

logger = logging.getLogger(__name__)


class RetrievalService:
    """Retrieval Augmented Generation (RAG) system for document retrieval.

    This class implements a complete RAG pipeline that combines document and chunk-level
    retrieval with optional reranking capabilities. It manages the indexing and retrieval
    of documents and their chunks, supporting both simple retrieval and advanced reranking.

    Attributes:
        indexed (bool): Whether documents have been indexed
        data_sources (Set[str]): Set of available data source names
        source_descriptions (Dict[str, str]): Mapping of data source names to their descriptions
        documents_by_source (Dict[str, Dict[UUID, ChunkedDocument]]): Mapping of data sources to their documents
        chunks_by_source (Dict[str, Dict[ChunkID, DocumentChunk]]): Mapping of data sources to their chunks
        document_retrievers (Dict[str, BaseDocumentRetriever]): Mapping of data sources to their document retrievers
        chunk_retrievers (Dict[str, BaseChunkRetriever]): Mapping of data sources to their chunk retrievers
        embedding_model (BaseDenseEmbedder): Model for generating embeddings
        document_reranker (Reranker): Component for reranking retrieved documents
        chunk_reranker (Reranker): Component for reranking retrieved chunks
        tokenizer (BaseTokenizer): Tokenizer for text processing
        cache_path Optional[str]: Path to the cache directory
    """

    # === Initialization ===

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        embedding_model: BaseDenseEmbedder,
        document_retriever: BaseDocumentRetriever,
        chunk_retriever: BaseChunkRetriever,
        document_reranker: Optional[DateDocumentReranker] = None,
        chunk_reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        """Initialize RAG service with component dependencies.

        Args:
            tokenizer: Tokenizer for processing text
            embedding_model: SentenceTransformer model for generating embeddings
            document_retriever: Unindexed document retriever to be cloned for each source
            chunk_retriever: Unindexed chunk retriever to be cloned for each source
            document_reranker: Optional component for reranking documents
            chunk_reranker: Optional component for reranking chunks
        """
        self.indexed = False
        self.data_sources: Set[str] = set()
        self.source_descriptions: Dict[str, str] = {}
        self.documents_by_source: Dict[str, Dict[UUID, ChunkedDocument]] = {}
        self.chunks_by_source: Dict[str, Dict[ChunkID, DocumentChunk]] = {}

        # Global lookup maps to efficiently retrieve documents/chunks by ID across all sources
        self.all_documents: Dict[UUID, Tuple[str, ChunkedDocument]] = {}
        self.all_chunks: Dict[ChunkID, Tuple[str, DocumentChunk]] = {}

        # Create dictionaries to hold source-specific retrievers
        self.document_retrievers: Dict[str, BaseDocumentRetriever] = {}
        self.chunk_retrievers: Dict[str, BaseChunkRetriever] = {}

        # Store component dependencies
        self.tokenizer: BaseTokenizer = tokenizer
        self.embedding_model: BaseDenseEmbedder = embedding_model
        self.document_retriever: BaseDocumentRetriever = document_retriever
        self.chunk_retriever: BaseChunkRetriever = chunk_retriever
        self.chunk_reranker: Optional[CrossEncoderReranker] = chunk_reranker
        self.cache_path: Optional[str] = None

        logger.info(
            f"RetrievalService initialized with: "
            f"document_retriever={type(document_retriever).__name__}, "
            f"chunk_retriever={type(chunk_retriever).__name__}"
        )

        if not document_reranker:
            logger.info("No document reranker provided")

        if not chunk_reranker:
            logger.info("No chunk reranker provided")

    @classmethod
    def from_cache(cls, cache_path: str) -> "RetrievalService":
        """Load a cached RetrievalService instance from disk.

        Args:
            cache_path: Directory path containing the cached RAG service

        Returns:
            RetrievalService: Loaded instance from cache
        """
        logger.info(f"Loading RetrievalService from cache at {cache_path}")
        cache_file = os.path.join(cache_path, "rag.pkl")

        try:
            with open(cache_file, "rb") as f:
                instance = pickle.load(f)

            return instance
        except Exception as e:
            logger.error(f"Failed to load from cache: {str(e)}")
            raise

    # === Public Methods ===

    def index(
        self,
        chunked_documents: List[ChunkedDocument],
        source_name: str,
        source_description: str,
    ):
        """Index documents and their chunks for retrieval.

        Processes and indexes the provided documents and their chunks for both document-level
        and chunk-level retrieval. Updates internal state and indexes in retrieval components.

        Args:
            chunked_documents: List of ChunkedDocument objects
            source_name: Name of the data source
            source_description: Description of the data source
        """
        if not chunked_documents:
            logger.warning(f"No documents provided for indexing source '{source_name}'")
            return

        # Initialize source if not exists
        if source_name not in self.data_sources:
            logger.info(f"Creating new data source: '{source_name}'")
            self.data_sources.add(source_name)
            self.source_descriptions[source_name] = source_description
            self.documents_by_source[source_name] = {}
            self.chunks_by_source[source_name] = {}

            # Create new retrievers for this source
            self._create_retrievers_for_source(source_name)
        else:
            logger.info(f"Updating existing data source: '{source_name}'")

        # Count existing documents before adding new ones
        existing_docs = len(self.documents_by_source[source_name])
        existing_chunks = len(self.chunks_by_source[source_name])

        # Add documents to source
        for doc in chunked_documents:
            # Add to source-specific dictionary
            self.documents_by_source[source_name][doc.uuid] = doc

            # Add to global lookup for cross-source retrieval
            self.all_documents[doc.uuid] = (source_name, doc)

            # Add chunks to source-specific dictionary
            self.chunks_by_source[source_name].update(doc.chunks)

            # Add chunks to global lookup for cross-source retrieval
            for chunk_id, chunk in doc.chunks.items():
                self.all_chunks[chunk_id] = (source_name, chunk)

        # Log document and chunk counts after adding new ones
        new_docs = len(self.documents_by_source[source_name]) - existing_docs
        new_chunks = len(self.chunks_by_source[source_name]) - existing_chunks
        logger.debug(f"Added {new_docs} documents and {new_chunks} chunks")

        # Index documents for retrieval
        logger.info(
            f"Indexing documents with {type(self.document_retrievers[source_name]).__name__}"
        )
        self.document_retrievers[source_name].index_documents(chunked_documents)

        # Index chunks for retrieval
        logger.info(
            f"Indexing chunks with {type(self.chunk_retrievers[source_name]).__name__}"
        )
        self.chunk_retrievers[source_name].index_chunks(
            self.chunks_by_source[source_name]
        )

        self.indexed = True
        logger.info(
            f"Successfully indexed {len(chunked_documents)} documents with {len(self.chunks_by_source[source_name])} chunks for source '{source_name}'"
        )

    def _create_retrievers_for_source(self, source_name: str) -> None:
        """Create document and chunk retrievers for a specific data source.

        Creates fresh document and chunk retrievers for each data source
        by directly copying the prototype retrievers.

        Args:
            source_name: Name of the data source to create retrievers for
        """
        logger.info(f"Creating retrievers for data source: '{source_name}'")

        # Use deepcopy for document retriever to ensure a fresh BM25 instance
        self.document_retrievers[source_name] = copy.deepcopy(self.document_retriever)

        # Use deepcopy for chunk retriever to ensure a fresh FAISS index
        self.chunk_retrievers[source_name] = copy.deepcopy(self.chunk_retriever)

    def find(
        self,
        query: str,
        source_name: str,
        initial_documents_k: int = Config.INITIAL_DOCUMENTS_K,
        final_documents_k: int = Config.FINAL_DOCUMENTS_K,
        initial_chunks_k: int = Config.INITIAL_CHUNKS_K,
        final_chunks_k: int = Config.FINAL_CHUNKS_K,
        filters: Optional[FilterOptions] = None,
        prioritize_earlier: bool = False,
    ) -> tuple[List[DocumentChunk], List[ChunkedDocument]]:
        """Retrieve relevant document chunks using the full RAG pipeline.

        Implements a two-stage retrieval process:
        1. Document-level retrieval and optional reranking
        2. Chunk-level retrieval and optional reranking within selected documents

        Args:
            query: User search query
            initial_documents_k: Number of documents to retrieve before reranking
            final_documents_k: Number of documents to keep after reranking
            initial_chunks_k: Number of chunks to retrieve before reranking
            final_chunks_k: Number of chunks to keep after reranking
            filters: Optional metadata filters to apply to documents
            source_name: Name of the data source to search in
            prioritize_earlier: Whether to prioritize earlier documents

        Returns:
            A tuple containing:
            - List of relevant document chunks ordered by relevance
            - List of unique source documents that contained those chunks

        Raises:
            RuntimeError: If documents haven't been indexed
            ValueError: If the specified source doesn't exist
        """
        logger.info(f"Starting retrieval for query '{query}' in source '{source_name}'")
        logger.debug(
            f"Retrieval parameters: initial_docs_k={initial_documents_k}, final_docs_k={final_documents_k}, "
            + f"initial_chunks_k={initial_chunks_k}, final_chunks_k={final_chunks_k}"
        )

        if not self.indexed:
            logger.error("Attempted retrieval before indexing documents")
            raise RuntimeError("Documents not indexed. Call index() first.")

        if source_name not in self.data_sources:
            logger.error(f"Invalid data source requested: '{source_name}'")
            raise ValueError(f"Data source '{source_name}' does not exist")

        # Document retrieval stage
        logger.info(
            f"Starting document retrieval stage: retrieving top-{initial_documents_k} documents"
        )
        retrieved_docs = self.document_retrievers[source_name].retrieve(
            [query], initial_documents_k
        )

        if not retrieved_docs:
            logger.warning("Document retriever returned no results")
            return [], []

        docs = retrieved_docs[0]  # Take first (and only) result from batch
        logger.info(f"Initial document retrieval returned {len(docs)} documents")

        # Apply document filters if provided
        if filters:
            logger.info(f"Applying filters: {filters}")
            pre_filter_count = len(docs)
            docs = self._apply_filters(docs, filters)
            logger.info(
                f"After filtering: {len(docs)}/{pre_filter_count} documents remain"
            )

            # If no documents left after filtering, return empty result
            if not docs:
                logger.warning("No documents passed filtering criteria")
                return [], []
        else:
            logger.debug("No filters applied")

        if docs:
            # we only rerank documents when we want to prioritize earlier ones,
            # otherwise they are already sorted by BM25 'score'
            if prioritize_earlier:
                docs = DateDocumentReranker.rerank(query, docs)
            logger.info(f"Document reranking complete, {len(docs)} documents")
        else:
            logger.error("No documents to rerank!")

        logger.info(f"Selected {len(docs)} documents for chunk retrieval")

        if not docs:
            logger.warning("No documents selected, returning empty result")
            return [], []

        # Get chunk IDs from selected documents
        candidate_chunk_ids: List[ChunkID] = []
        for doc in docs:
            chunk_ids = list(
                self.documents_by_source[source_name][doc.uuid].chunks.keys()
            )
            candidate_chunk_ids.extend(chunk_ids)

        logger.info(
            f"Collected {len(candidate_chunk_ids)} candidate chunks from {len(docs)} documents"
        )

        # Chunk retrieval stage
        logger.info(
            f"Starting chunk retrieval from {len(candidate_chunk_ids)} candidate chunks"
        )
        retrieved_chunks = self.chunk_retrievers[source_name].retrieve(
            [query], candidate_chunk_ids, initial_chunks_k
        )

        if not retrieved_chunks:
            logger.warning("Chunk retriever returned no results")
            return [], []

        chunks = retrieved_chunks[0]  # Take first (and only) result from batch
        logger.info(f"Initial chunk retrieval returned {len(chunks)} chunks")

        # Chunk reranking if available
        if self.chunk_reranker and chunks:
            try:
                logger.info(f"Starting chunk reranking for {len(chunks)} chunks")
                chunk_rerank_start = time.time()
                reranked_chunks = self.chunk_reranker.rerank(
                    query, chunks, final_chunks_k
                )
                chunk_rerank_time = time.time() - chunk_rerank_start
                chunks = reranked_chunks
                logger.info(
                    f"Chunk reranking completed in {chunk_rerank_time:.2f}s, {len(chunks)} chunks"
                )
            except Exception as e:
                logger.error(f"Error in chunk reranking: {str(e)}")
                logger.warning(
                    f"Falling back to top-{final_chunks_k} chunks without reranking"
                )
                chunks = chunks[:final_chunks_k]
        else:
            if not self.chunk_reranker:
                logger.debug("No chunk reranker available, using top-k selection")
            chunks = chunks[:final_chunks_k]

        logger.info(f"Selected {len(chunks)} final chunks")

        # Get unique source documents from the final chunks
        unique_source_docs: Set[ChunkedDocument] = set()
        for chunk in chunks:
            doc_uuid = chunk.parent_document.uuid
            if doc_uuid in self.documents_by_source[source_name]:
                unique_source_docs.add(self.documents_by_source[source_name][doc_uuid])

        logger.info(
            f"Returning {len(chunks)} chunks from {len(unique_source_docs)} unique documents"
        )
        return chunks, list(unique_source_docs)

    def get_sources(self) -> List[Dict[str, str]]:
        """Get a list of available data sources.

        Returns:
            List of dictionaries with 'name' and 'description' for each data source
        """
        sources: List[Dict[str, str]] = []
        for source_name in self.data_sources:
            sources.append(
                {
                    "name": source_name,
                    "description": self.source_descriptions.get(source_name, ""),
                }
            )
        return sources

    def get_source_description(self, source_name: str) -> str:
        """Get the description of a specific data source.

        Args:
            source_name: Name of the data source

        Returns:
            Description of the data source

        Raises:
            ValueError: If the source does not exist
        """
        if source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' does not exist")

        return self.source_descriptions.get(source_name, "")

    def _apply_filters(
        self, docs: List[ChunkedDocument], filters: FilterOptions
    ) -> List[ChunkedDocument]:
        """Apply metadata filters to documents.

        Args:
            docs: List of documents to filter
            filters: Filter criteria to apply

        Returns:
            Filtered list of documents
        """
        logger.debug(f"Applying filters to {len(docs)} documents")
        filtered_docs = docs

        # Apply period filter if provided
        if filters.period:
            logger.debug(
                f"Applying date filter: from {filters.period.start_date} to {filters.period.end_date}"
            )
            period_filtered: List[ChunkedDocument] = []
            for doc in filtered_docs:
                # Skip documents with no date when applying date filter
                if not doc.create_date:
                    logger.debug(
                        f"Document {doc.uuid} has no date, skipping from date filter"
                    )
                    continue

                # Apply start date filter if provided
                try:
                    if (
                        filters.period.start_date
                        and doc.create_date < filters.period.start_date
                    ):
                        logger.debug(
                            f"Document {doc.uuid} date {doc.create_date} before start date {filters.period.start_date}, filtering out"
                        )
                        continue
                except Exception as e:
                    logger.error(f"Error applying start date filter: {e}")
                    logger.error(
                        f"Document date: {doc.create_date}, Start date: {filters.period.start_date}"
                    )
                    continue

                # Apply end date filter if provided
                try:
                    if (
                        filters.period.end_date
                        and doc.create_date > filters.period.end_date
                    ):
                        logger.debug(
                            f"Document {doc.uuid} date {doc.create_date} after end date {filters.period.end_date}, filtering out"
                        )
                        continue
                except Exception as e:
                    logger.error(f"Error applying end date filter: {e}")
                    logger.error(
                        f"Document date: {doc.create_date}, End date: {filters.period.end_date}"
                    )
                    continue

                period_filtered.append(doc)

            logger.info(
                f"Date filter: {len(period_filtered)}/{len(filtered_docs)} documents passed"
            )
            filtered_docs = period_filtered

        # Apply document type filter if provided
        if filters.doc_types:
            logger.debug(f"Applying document type filter: {filters.doc_types}")
            type_filtered: List[ChunkedDocument] = []
            for doc in filtered_docs:
                # Check if document has a type field
                if doc.type in filters.doc_types:
                    type_filtered.append(doc)
                else:
                    logger.debug(
                        f"Document {doc.uuid} type '{doc.type}' not in allowed types, filtering out"
                    )

            logger.info(
                f"Type filter: {len(type_filtered)}/{len(filtered_docs)} documents passed"
            )
            filtered_docs = type_filtered

        logger.debug(
            f"Filter application complete: {len(filtered_docs)}/{len(docs)} documents remain after all filters"
        )
        return filtered_docs

    def get_random_documents(
        self, k: int, source_name: Optional[str] = None
    ) -> List[ChunkedDocument]:
        """Get k random chunked documents.

        Args:
            k: Number of random documents to return
            source_name: Optional name of the data source to get documents from.
                         If None, selects from all available sources.

        Returns:
            List of randomly selected ChunkedDocument objects

        Raises:
            RuntimeError: If no documents have been indexed
            ValueError: If the specified source doesn't exist
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index() first.")

        if source_name is not None and source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' does not exist")

        if source_name is not None:
            # Get documents from specific source
            docs = list(self.documents_by_source[source_name].values())
        else:
            # Get documents from all sources
            docs = [doc for _, doc in self.all_documents.values()]

        if not docs:
            return []

        return random.sample(docs, min(k, len(docs)))

    def get_documents_by_uuids(
        self, uuids: List[str], source_name: Optional[str] = None
    ) -> List[ChunkedDocument]:
        """Get multiple documents by their UUIDs.

        Args:
            uuids: List of UUID strings of documents to retrieve
            source_name: Optional name of the data source to get documents from.
                         If None, retrieves from any source.

        Returns:
            List of ChunkedDocument objects with matching UUIDs

        Raises:
            KeyError: If any document with UUID not found
            ValueError: If the specified source doesn't exist
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index() first.")

        if source_name is not None and source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' does not exist")

        result: List[ChunkedDocument] = []
        for uuid_str in uuids:
            uuid_obj = UUID(uuid_str)

            if source_name is not None:
                # Get document from specific source
                result.append(self.documents_by_source[source_name][uuid_obj])
            else:
                # Get document from any source
                if uuid_obj not in self.all_documents:
                    raise KeyError(f"Document with UUID {uuid_str} not found")

                _, doc = self.all_documents[uuid_obj]
                result.append(doc)

        return result

    def get_random_chunks(
        self, k: int, source_name: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Get k random document chunks.

        Args:
            k: Number of random chunks to return
            source_name: Optional name of the data source to get chunks from.
                         If None, selects from all available sources.

        Returns:
            List of randomly selected DocumentChunk objects

        Raises:
            RuntimeError: If no documents have been indexed
            ValueError: If the specified source doesn't exist
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index() first.")

        if source_name is not None and source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' does not exist")

        if source_name is not None:
            # Get chunks from specific source
            chunks = list(self.chunks_by_source[source_name].values())
        else:
            # Get chunks from all sources
            chunks = [chunk for _, chunk in self.all_chunks.values()]

        if not chunks:
            return []

        return random.sample(chunks, min(k, len(chunks)))

    def get_chunks_by_uuids(
        self, chunk_ids: List[str], source_name: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Get multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve
            source_name: Optional name of the data source to get chunks from.
                         If None, retrieves from any source.

        Returns:
            List of DocumentChunk objects with matching chunk_ids

        Raises:
            KeyError: If any chunk with chunk_id not found
            ValueError: If the specified source doesn't exist or if a chunk ID isn't valid
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index() first.")

        if source_name is not None and source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' does not exist")

        result: List[DocumentChunk] = []
        for chunk_id_str in chunk_ids:
            try:
                # Properly convert string to int then to ChunkID
                chunk_id_int = int(chunk_id_str)
                chunk_id_obj = ChunkID(chunk_id_int)
            except ValueError:
                raise ValueError(
                    f"Invalid chunk ID format: {chunk_id_str}. Must be a valid integer."
                )

            if source_name is not None:
                # Get chunk from specific source
                if chunk_id_obj not in self.chunks_by_source[source_name]:
                    raise KeyError(
                        f"Chunk with ID {chunk_id_str} not found in source '{source_name}'"
                    )
                result.append(self.chunks_by_source[source_name][chunk_id_obj])
            else:
                # Get chunk from any source
                if chunk_id_obj not in self.all_chunks:
                    raise KeyError(f"Chunk with ID {chunk_id_str} not found")

                _, chunk = self.all_chunks[chunk_id_obj]
                result.append(chunk)

        return result

    def get_document_source(self, document_uuid: str) -> str:
        """Get the source name for a document by its UUID.

        Args:
            document_uuid: UUID string of the document

        Returns:
            Source name of the document

        Raises:
            KeyError: If document with UUID not found
        """
        uuid_obj = UUID(document_uuid)
        if uuid_obj not in self.all_documents:
            raise KeyError(f"Document with UUID {document_uuid} not found")

        source_name, _ = self.all_documents[uuid_obj]
        return source_name

    def get_chunk_source(self, chunk_id: str) -> str:
        """Get the source name for a chunk by its ID.

        Args:
            chunk_id: ID string of the chunk

        Returns:
            Source name of the chunk

        Raises:
            KeyError: If chunk with ID not found
            ValueError: If the chunk ID isn't valid
        """
        try:
            # Properly convert string to int then to ChunkID
            chunk_id_int = int(chunk_id)
            chunk_id_obj = ChunkID(chunk_id_int)
        except ValueError:
            raise ValueError(
                f"Invalid chunk ID format: {chunk_id}. Must be a valid integer."
            )

        if chunk_id_obj not in self.all_chunks:
            raise KeyError(f"Chunk with ID {chunk_id} not found")

        source_name, _ = self.all_chunks[chunk_id_obj]
        return source_name

    # === Caching ===

    def save(self, cache_dir: str) -> None:
        """Save RetrievalService instance to cache.

        Args:
            cache_dir: Directory path where the cache file will be saved
        """
        try:
            # Ensure the directory exists
            os.makedirs(cache_dir, exist_ok=True)
            logger.debug(f"Created cache directory: {cache_dir}")

            filename = "rag.pkl"
            cache_path = os.path.join(cache_dir, filename)

            # Calculate sizes to be saved
            doc_count = sum(len(docs) for docs in self.documents_by_source.values())
            chunk_count = sum(len(chunks) for chunks in self.chunks_by_source.values())
            source_count = len(self.data_sources)
            logger.debug(
                f"Saving data: {source_count} sources, {doc_count} documents, {chunk_count} chunks"
            )

            # Save the file
            with open(cache_path, "wb") as f:
                pickle.dump(self, f)

            filesize = os.path.getsize(cache_path) / (1024 * 1024)  # Size in MB
            logger.info(f"RAG cached successfully to {cache_path} ({filesize:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to cache RAG: {e}")
            # Log stack trace for debugging
            import traceback

            logger.error(traceback.format_exc())

    def _load_from_cache(self) -> None:
        """Load a previously cached RetrievalService instance from disk.

        Raises:
            Exception: If loading from cache fails
        """
        if self.cache_path is None:
            raise ValueError("Cache path not set")

        try:
            with open(self.cache_path, "rb") as f:
                cached_instance = pickle.load(f)
                self.__dict__.update(cached_instance.__dict__)
        except Exception as e:
            logger.error(f"Failed to load RAG from cache: {e}")

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)

    # === Training ===

    def generate_training_data(
        self,
        llm_service: BaseLLMService,
        output_dir: str,
        embedder_samples: int = 100,
        chunk_reranker_samples: int = 1000,
        doc_reranker_samples: int = 1000,
        source_name: str = "default",
    ) -> None:
        """Generate training data from indexed documents using LLM.

        Args:
            llm_service: LLM service for generating queries
            output_dir: Directory where to save the generated training data files
            embedder_samples: Number of training samples for embedder
            chunk_reranker_samples: Number of training samples for chunk reranker
            doc_reranker_samples: Number of training samples for document reranker
            source_name: Name of the data source to use for training data generation

        Raises:
            RuntimeError: If documents not indexed
            ValueError: If LLM service not provided or source doesn't exist
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index() first.")

        if source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' does not exist")

        # Generate training data using indexed documents and chunks
        generator = TrainingDataGenerator(
            documents=self.documents_by_source[source_name],
            chunks=self.chunks_by_source[source_name],
            document_retriever=self.document_retrievers[source_name],
            llm_service=llm_service,
        )

        # Generate different types of training data
        chunk_query_pairs = generator.generate_chunk_query_pairs(embedder_samples)
        reranker_chunk_pairs = generator.generate_reranker_chunk_pairs(
            chunk_reranker_samples
        )
        reranker_doc_pairs = generator.generate_reranker_document_pairs(
            doc_reranker_samples
        )

        # Save each type of training data to separate files
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save to separate JSON files
        with open(
            os.path.join(output_dir, "chunk_query_pairs.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(chunk_query_pairs, f, ensure_ascii=False, indent=2)

        with open(
            os.path.join(output_dir, "reranker_chunk_pairs.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(reranker_chunk_pairs, f, ensure_ascii=False, indent=2)

        with open(
            os.path.join(output_dir, "reranker_document_pairs.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(reranker_doc_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Training data generated and saved to {output_dir}")

    def train(self, config: RAGTrainingConfig, data_path: str) -> None:
        """Train RAG components using provided training data.

        Args:
            config: Configuration for training different components
            data_path: Path to directory containing training data files

        Raises:
            FileNotFoundError: If training data directory not found
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data directory not found at {data_path}")

        # Train embedder if config provided
        if config.embedder is not None:
            embedder_data_path = os.path.join(data_path, "chunk_query_pairs.json")
            if os.path.exists(embedder_data_path):
                embedder_output_dir = os.path.join(config.output_dir, "embedder")
                logger.info(f"Training embedder with data from {embedder_data_path}")
                self.embedding_model.train(
                    train_data_path=embedder_data_path,
                    config=config.embedder,
                    output_dir=embedder_output_dir,
                )
                logger.info(
                    f"Embedder training completed. Model saved to {embedder_output_dir}"
                )
            else:
                logger.warning(
                    f"Embedder training data not found at {embedder_data_path}"
                )

        # Train chunk reranker if config provided
        if config.chunk_reranker is not None and self.chunk_reranker is not None:
            chunk_reranker_data_path = os.path.join(
                data_path, "reranker_chunk_pairs.json"
            )
            if os.path.exists(chunk_reranker_data_path):
                chunk_reranker_output_dir = os.path.join(
                    config.output_dir, "chunk_reranker"
                )
                logger.info(
                    f"Training chunk reranker with data from {chunk_reranker_data_path}"
                )
                self.chunk_reranker.train(
                    train_data_path=chunk_reranker_data_path,
                    config=config.chunk_reranker,
                    output_dir=chunk_reranker_output_dir,
                )
                logger.info(
                    f"Chunk reranker training completed. Model saved to {chunk_reranker_output_dir}"
                )
            else:
                logger.warning(
                    f"Chunk reranker training data not found at {chunk_reranker_data_path}"
                )

        logger.info("RAG components training completed")
