"""Chunk retrieval module.

This module provides chunk retrieval classes that can search and retrieve document chunks.
The module includes an abstract base class and a FAISS-based implementation.
"""

import abc
import logging
from typing import Any, Dict, List

import faiss  # type: ignore
import numpy as np

from backend.src.data_classes import ChunkID, DocumentChunk
from backend.src.services.retrieval.components.dense_embedder import (
    SentenceTransformerEmbedder,
)

logger = logging.getLogger(__name__)


class BaseChunkRetriever(abc.ABC):
    """Base class for chunk retrievers.

    Defines the interface for chunk retrieval systems.

    Attributes:
        chunks: Dictionary mapping chunk IDs to document chunks
        indexed (bool): Whether chunks have been indexed
    """

    def __init__(self) -> None:
        """Initialize the chunk retriever."""
        self.chunks: Dict[ChunkID, DocumentChunk] = {}
        self._indexed: bool = False

    @property
    def indexed(self) -> bool:
        """Check if chunks have been indexed.

        Returns:
            True if chunks have been indexed, False otherwise
        """
        return self._indexed

    @abc.abstractmethod
    def index_chunks(self, chunks: Dict[ChunkID, DocumentChunk]) -> None:
        """Index chunks for retrieval.

        Args:
            chunks: Dictionary mapping chunk IDs to document chunks
        """

    @abc.abstractmethod
    def retrieve(
        self, queries: List[str], candidate_chunk_ids: List[ChunkID], k: int
    ) -> List[List[DocumentChunk]]:
        """Retrieve top-k chunks for each query from candidates.

        Args:
            queries: List of query strings
            candidate_chunk_ids: List of chunk IDs to search among
            k: Number of chunks to retrieve per query

        Returns:
            List of lists containing top-k chunks for each query

        Raises:
            RuntimeError: If chunks have not been indexed
        """


class FaissChunkRetriever(BaseChunkRetriever):
    """A dense retriever using FAISS for efficient similarity search.

    This class implements a chunk retrieval system using Facebook AI Similarity Search (FAISS)
    for efficient similarity search over dense vector embeddings. It uses HNSW (Hierarchical
    Navigable Small World) indexing for fast approximate nearest neighbor search.

    Attributes:
        embedding_model: Model used to generate dense embeddings
        dimension (int): Dimension of the embedding vectors
        index: Main FAISS index for similarity search
        chunks (Dict[ChunkID, DocumentChunk]): Mapping of chunk IDs to document chunks
        indexed (bool): Whether chunks have been indexed
    """

    def __init__(self, embedding_model: SentenceTransformerEmbedder):
        """Initialize the FAISS retriever.

        Args:
            embedding_model: Initialized SentenceTransformerEmbedder model to use for embeddings.
                Must implement encode() and get_sentence_embedding_dimension() methods.
        """
        super().__init__()
        logger.info("Initializing FAISS chunk retriever")
        self.embedding_model = embedding_model
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Use a hybrid index that works well for both small and large datasets
        # IndexHNSWFlat is efficient for both small and large datasets
        self.index: Any = faiss.IndexIDMap2(  # type: ignore
            faiss.IndexHNSWFlat(
                self.dimension, 32
            )  # 32 neighbors per node  # type: ignore
        )

        logger.info(
            f"FAISS chunk retriever initialized with dimension={self.dimension}, index=HNSW"
        )

    def __deepcopy__(self, memo: Any) -> "FaissChunkRetriever":
        """Create a deep copy of the retriever with a fresh FAISS index.

        This ensures each data source has its own independent FAISS index.

        Args:
            memo: Memoization dictionary for deepcopy

        Returns:
            A new FaissChunkRetriever with the same embedding model but a fresh FAISS index
        """
        import copy

        # Create a new instance with the same embedding model
        result = FaissChunkRetriever(self.embedding_model)
        # Copy any other instance attributes EXCEPT the index
        for k, v in self.__dict__.items():
            if (
                k != "index" and k != "embedding_model"
            ):  # embedding_model already set in __init__
                result.__dict__[k] = copy.deepcopy(v, memo)
        return result

    def index_chunks(self, chunks: Dict[ChunkID, DocumentChunk]) -> None:
        """Index the provided chunks for similarity search.

        Computes embeddings for all chunks and adds them to the FAISS index.

        Args:
            chunks: Dictionary mapping chunk IDs to document chunks to be indexed
        """
        self.chunks = chunks
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return

        chunk_list = list(chunks.values())
        logger.info(f"Indexing {len(chunk_list)} chunks")
        all_embeddings = self._compute_embeddings(
            [chunk.content for chunk in chunk_list]
        )

        # No training needed for HNSW index
        chunk_ids = np.array([int(chunk.uuid) for chunk in chunk_list], dtype=np.int64)
        self.index.add_with_ids(all_embeddings, chunk_ids)  # type: ignore
        self._indexed = True

    def _compute_embeddings(
        self, texts: List[str]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute embeddings for a list of texts.

        Uses the embedding model to generate dense vector embeddings for the input texts.

        Args:
            texts: List of text strings to compute embeddings for

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dimension)
        """
        logger.debug("Computing embeddings")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def retrieve(
        self, queries: List[str], candidate_chunk_ids: List[ChunkID], k: int
    ) -> List[List[DocumentChunk]]:
        """Retrieve top-k relevant chunks for each query from candidates using FAISS.

        For each query, finds the k most similar chunks from among the candidate chunks
        using approximate nearest neighbor search.

        Args:
            queries: List of query strings to find relevant chunks for
            candidate_chunk_ids: List of chunk IDs to search among
            k: Number of chunks to retrieve per query

        Returns:
            List of lists of chunks, where each inner list contains the k most relevant
            chunks for the corresponding query

        Raises:
            RuntimeError: If chunks have not been indexed yet
        """
        if not self.indexed:
            logger.error("Attempting to retrieve from unindexed FAISS")
            raise RuntimeError("Chunks not indexed. Call index_chunks() first.")

        if not queries:
            logger.warning("No queries provided for retrieval")
            return []

        if not candidate_chunk_ids:
            logger.warning("No candidate chunk IDs provided for retrieval")
            return [[] for _ in queries]

        try:
            query_embeddings = self._compute_embeddings(queries)

            # Create a subset index for searching within candidate chunks
            subset_index: Any = faiss.IndexIDMap2(  # type: ignore
                faiss.IndexHNSWFlat(self.dimension, 32)  # type: ignore
            )

            # Filter candidate IDs to include only those that exist in chunks
            valid_chunk_ids = [
                cid for cid in candidate_chunk_ids if ChunkID(cid) in self.chunks
            ]

            if len(valid_chunk_ids) == 0:
                logger.warning("No valid chunk IDs found in candidates")
                return [[] for _ in queries]

            # Process all embeddings in a single batch
            contents = [self.chunks[ChunkID(cid)].content for cid in valid_chunk_ids]
            candidate_embeddings = self._compute_embeddings(contents)
            candidate_ids = np.array(
                [int(cid) for cid in valid_chunk_ids], dtype=np.int64
            )

            subset_index.add_with_ids(candidate_embeddings, candidate_ids)  # type: ignore

            logger.debug(f"Searching FAISS index for top-{k} chunks")

            # Calculate minimum k value to prevent out-of-bounds
            actual_k = min(k, len(valid_chunk_ids))
            if actual_k < k:
                logger.warning(
                    f"Requested k={k} but only {actual_k} valid chunks available"
                )

            distances, indices = subset_index.search(  # type: ignore
                query_embeddings, actual_k
            )  # type: ignore

            all_results: List[List[DocumentChunk]] = []

            # Handle potential numpy array issues
            for i in range(len(indices)):  # Safe iteration over array dimensions
                results: List[DocumentChunk] = []
                # Make sure we're indexing an array that exists
                if i < len(indices):
                    query_indices = indices[i]
                    for j in range(len(query_indices)):  # Safe iteration over indices
                        idx = int(query_indices[j])  # Convert to Python int
                        if idx != -1 and ChunkID(idx) in self.chunks:
                            results.append(self.chunks[ChunkID(idx)])
                all_results.append(results)

            logger.info(
                f"Retrieved chunks for {len(all_results)} queries, total: {sum(len(r) for r in all_results)} chunks"
            )
            return all_results

        except Exception as e:
            logger.error(f"Error during chunk retrieval: {str(e)}")
            return [[] for _ in queries]
