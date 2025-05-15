"""Data classes module for document processing and chunking.

This module provides the core data structures for handling documents and their chunks:

Classes:
    - ChunkedDocument: Represents a document split into chunks
    - DocumentChunk: Individual chunk of a document
    - RawDocument: Original, unprocessed document
    - ContextPiece: A piece of context from a document chunk
    - QueryResult: Result of a query processing operation
Types:
    - ChunkID: Identifier for document chunks

The classes in this module form the foundation for document processing operations
throughout the application.
"""

from backend.src.data_classes.chunked_document import ChunkedDocument
from backend.src.data_classes.context_piece import ContextPiece

# Then import classes that depend on the base types
# First import the types and base classes that don't have dependencies
from backend.src.data_classes.document_chunk import ChunkID, DocumentChunk
from backend.src.data_classes.filters import FilterOptions, PeriodFilter

# Finally import classes that depend on multiple other classes
from backend.src.data_classes.query_result import QueryResult
from backend.src.data_classes.raw_document import RawDocument

__all__ = [
    "ChunkID",
    "RawDocument",
    "FilterOptions",
    "PeriodFilter",
    "DocumentChunk",
    "ChunkedDocument",
    "ContextPiece",
    "QueryResult",
]
