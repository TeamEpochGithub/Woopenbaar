"""Document retrieval module.

This module provides document retrieval classes that can search and retrieve entire documents.
The module includes an abstract base class and a BM25-based implementation.
"""

import abc
import logging
from typing import Any, List, Optional

import bm25s  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore

from backend.src.data_classes import ChunkedDocument

logger = logging.getLogger(__name__)


class BaseDocumentRetriever(abc.ABC):
    """Base class for document retrievers.

    Defines the interface for document retrieval systems.

    Attributes:
        documents: List of indexed documents
        indexed (bool): Whether documents have been indexed
    """

    def __init__(self) -> None:
        """Initialize the document retriever."""
        self.documents: Optional[List[ChunkedDocument]] = None
        self._indexed: bool = False

    @property
    def indexed(self) -> bool:
        """Check if documents have been indexed.

        Returns:
            True if documents have been indexed, False otherwise
        """
        return self._indexed

    @abc.abstractmethod
    def index_documents(self, documents: List[ChunkedDocument]) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of documents to index
        """

    @abc.abstractmethod
    def retrieve(self, queries: List[str], k: int) -> List[List[ChunkedDocument]]:
        """Retrieve top-k documents for each query.

        Args:
            queries: List of query strings
            k: Number of documents to retrieve per query

        Returns:
            List of lists containing top-k documents for each query

        Raises:
            RuntimeError: If documents have not been indexed
        """


class BM25DocumentRetriever(BaseDocumentRetriever):
    """BM25-based document retriever.

    Uses the BM25 ranking algorithm to index and retrieve documents based on text similarity.
    Requires a tokenizer for text processing and maintains an in-memory index.

    Attributes:
        tokenizer: Component for text tokenization
        retriever: Underlying BM25 implementation
        documents: List of indexed documents
        indexed (bool): Whether documents have been indexed
    """

    def __init__(self, tokenizer) -> None:  # type: ignore
        """Initialize BM25 retriever with tokenizer.

        Args:
            tokenizer: Component for text tokenization
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.retriever = bm25s.BM25()  # type: ignore

    def __deepcopy__(self, memo: Any) -> "BM25DocumentRetriever":
        """Create a deep copy of the retriever with a fresh BM25 instance.

        This is critical when copying retrievers for different data sources to
        ensure each source has its own independent BM25 index.

        Args:
            memo: Memoization dictionary for deepcopy

        Returns:
            A new BM25DocumentRetriever with the same tokenizer but a fresh BM25 instance
        """
        import copy

        # Create a new instance with the same tokenizer
        result = BM25DocumentRetriever(self.tokenizer)  # type: ignore
        # Copy any other instance attributes EXCEPT the retriever
        # (which we want to keep fresh)
        for k, v in self.__dict__.items():
            if (
                k != "retriever" and k != "tokenizer"
            ):  # tokenizer already set in __init__
                result.__dict__[k] = copy.deepcopy(v, memo)
        return result

    def index_documents(self, documents: List[ChunkedDocument]) -> None:
        """Index documents using BM25.

        Processes and indexes the provided documents using BM25 algorithm.
        Documents must be indexed before retrieval is possible.

        Args:
            documents: List of documents to index
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            self.documents = []
            self._indexed = True
            return

        logger.info(f"Indexing {len(documents)} documents with BM25")
        self.documents = documents

        contents = [doc.content for doc in documents]
        logger.debug(f"Extracted {len(contents)} document contents for tokenization")

        tokenized_docs = self.tokenizer.tokenize(contents)  # type: ignore
        logger.debug(f"Tokenized {len(tokenized_docs)} documents")  # type: ignore

        self.retriever.index(tokenized_docs, show_progress=True)  # type: ignore
        self._indexed = True
        logger.info(f"BM25 document indexing complete for {len(documents)} documents")

    def retrieve(self, queries: List[str], k: int) -> List[List[ChunkedDocument]]:
        """Retrieve top-k documents for each query.

        Args:
            queries: List of query strings
            k: Number of documents to retrieve per query

        Returns:
            List of lists containing top-k documents for each query

        Raises:
            RuntimeError: If documents have not been indexed
        """
        if not self.indexed:
            logger.error("Attempting to retrieve from unindexed BM25")
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        query_tokens = self.tokenizer.tokenize(queries)  # type: ignore

        all_indices = self.retriever.retrieve(query_tokens, k=k, return_as="documents")  # type: ignore

        results: List[List[ChunkedDocument]] = []
        for indices in all_indices:  # type: ignore
            retrieved = [self.documents[idx] for idx in indices]  # type: ignore
            results.append(retrieved)  # type: ignore

        return results

    def get_document_vectors(self, documents: List[str]) -> csr_matrix:  # type: ignore
        """Get BM25 vector representations of documents.

        Args:
            documents: List of document strings to vectorize

        Returns:
            Sparse matrix of BM25 vectors

        Raises:
            RuntimeError: If documents have not been indexed
        """
        if not self.indexed:
            logger.error("Attempting to get vectors from unindexed BM25")
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        logger.info(f"Generating BM25 vectors for {len(documents)} documents")
        tokenized_docs = self.tokenizer.tokenize(documents)  # type: ignore
        vectors = self.retriever.get_document_vectors(tokenized_docs)  # type: ignore
        return vectors  # type: ignore
