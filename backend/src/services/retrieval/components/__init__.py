"""Retrieval components package.

This package provides various components used in the retrieval system:

- BaseTokenizer: Abstract base class for tokenizers
- BertTokenizer: BERT-based text tokenization
- BaseEmbedder: Abstract base class for text embedders
- SentenceTransformerEmbedder: Text embedding using sentence transformers
- BaseDocumentRetriever: Abstract base class for document retrievers
- BM25DocumentRetriever: BM25 algorithm for document retrieval
- BaseChunkRetriever: Abstract base class for chunk retrievers
- FaissChunkRetriever: FAISS-based vector similarity search for chunks
- BaseChunkReranker: Abstract base class for chunk rerankers
- CrossEncoderReranker: Cross-encoder model for reranking results
- BaseDocumentReranker: Abstract base class for document rerankers
- DateDocumentReranker: Document reranking based on dates
- WebSearchComponent: Component for performing web searches

These components can be composed to build retrieval pipelines with different
characteristics and trade-offs between speed and accuracy.
"""

from .chunk_reranker import BaseChunkReranker, CrossEncoderReranker
from .chunk_retriever import BaseChunkRetriever, FaissChunkRetriever
from .dense_embedder import BaseDenseEmbedder, SentenceTransformerEmbedder
from .document_reranker import BaseDocumentReranker, DateDocumentReranker
from .document_retriever import BaseDocumentRetriever, BM25DocumentRetriever

# Finally import standalone utilities
from .preprocessing import create_preprocessor

# Then import implementations that depend on base classes
# First import base classes and utilities
from .tokenizer import BaseTokenizer, BertTokenizer

__all__ = [
    "BaseTokenizer",
    "BertTokenizer",
    "BaseDenseEmbedder",
    "SentenceTransformerEmbedder",
    "BaseDocumentRetriever",
    "BM25DocumentRetriever",
    "BaseChunkRetriever",
    "FaissChunkRetriever",
    "BaseChunkReranker",
    "CrossEncoderReranker",
    "BaseDocumentReranker",
    "DateDocumentReranker",
    "create_preprocessor",
]
