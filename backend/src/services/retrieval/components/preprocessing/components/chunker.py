"""Document chunking module.

This module provides functionality to split documents into chunks using different strategies:
1. Semantic chunking - using embeddings to find natural boundaries based on content similarity
2. Naive chunking - splitting content into evenly sized chunks at sentence boundaries

The module defines a base DocumentChunkerBase abstract class and two implementations.
"""

import abc
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from backend.conf.config import Config
from backend.src.data_classes import (
    ChunkedDocument,
    ChunkID,
    DocumentChunk,
    RawDocument,
)

logger = logging.getLogger(__name__)


class DocumentChunkerBase(abc.ABC):
    """Base abstract class for document chunkers.

    All document chunking strategies should implement this interface.
    """

    @abc.abstractmethod
    def chunk_document(self, document: RawDocument) -> ChunkedDocument:
        """Split a document into chunks.

        Args:
            document: RawDocument object containing the text to chunk

        Returns:
            ChunkedDocument containing the original document data and its chunks
        """


class SemanticDocumentChunker(DocumentChunkerBase):
    """Splits documents into semantic chunks based on content similarity.

    Uses sentence embeddings and adaptive thresholds to identify natural chunk boundaries
    while preserving context through overlapping windows. Supports different chunking
    parameters based on document size.

    Attributes:
        small_context_size: Number of surrounding sentences to include for small docs
        medium_context_size: Number of surrounding sentences to include for medium docs
        large_context_size: Number of surrounding sentences to include for large docs
        small_similarity_threshold: Similarity threshold for splitting small docs
        medium_similarity_threshold: Similarity threshold for splitting medium docs
        large_similarity_threshold: Similarity threshold for splitting large docs
        embedding_model: Model for computing text embeddings
        _sentence_pattern: Regex pattern for splitting text into sentences
    """

    def __init__(
        self,
        embedding_model: Any,
        small_context_size: Optional[int] = None,
        medium_context_size: Optional[int] = None,
        large_context_size: Optional[int] = None,
        small_similarity_threshold: Optional[float] = None,
        medium_similarity_threshold: Optional[float] = None,
        large_similarity_threshold: Optional[float] = None,
    ):
        """Initialize the document chunker with configurable parameters.

        Args:
            embedding_model: Model for computing text embeddings
            small_context_size: Context size for small documents (<1800 chars)
            medium_context_size: Context size for medium documents (1800-30000 chars)
            large_context_size: Context size for large documents (>30000 chars)
            small_similarity_threshold: Similarity threshold for small documents
            medium_similarity_threshold: Similarity threshold for medium documents
            large_similarity_threshold: Similarity threshold for large documents
        """
        # Use provided values or defaults from Config
        self.small_context_size: int = small_context_size or Config.SMALL_CONTEXT_SIZE
        self.medium_context_size: int = (
            medium_context_size or Config.MEDIUM_CONTEXT_SIZE
        )
        self.large_context_size: int = large_context_size or Config.LARGE_CONTEXT_SIZE
        self.small_similarity_threshold: float = (
            small_similarity_threshold or Config.SMALL_SIMILARITY_THRES
        )
        self.medium_similarity_threshold: float = (
            medium_similarity_threshold or Config.MEDIUM_SIMILARITY_THRES
        )
        self.large_similarity_threshold: float = (
            large_similarity_threshold or Config.LARGE_SIMILARITY_THRES
        )
        self.embedding_model: Any = embedding_model

        # Regex pattern for splitting text into sentences
        # Handles common sentence endings while avoiding false splits on abbreviations
        self._sentence_pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        )

    def chunk_document(self, document: RawDocument) -> ChunkedDocument:
        """Split a document into semantic chunks based on content similarity.

        The chunking process:
        1. Determines appropriate context size and similarity threshold based on doc length
        2. Splits text into sentences
        3. Creates overlapping sentence groups with context windows
        4. Computes embeddings for each group
        5. Identifies chunk boundaries where similarity drops below threshold
        6. Creates DocumentChunk objects with unique IDs

        Args:
            document: RawDocument object containing the text to chunk

        Returns:
            ChunkedDocument containing the original document data and its chunks
        """
        content = document.content
        content_length = len(content)

        # Adjust context size and similarity threshold based on content length
        if content_length < 1800:
            context_size = self.small_context_size
            similarity_threshold = self.small_similarity_threshold
        elif content_length > 30000:
            context_size = self.large_context_size
            similarity_threshold = self.large_similarity_threshold
        else:
            context_size = self.medium_context_size
            similarity_threshold = self.medium_similarity_threshold

        # Split text into sentences
        sentences = self._sentence_pattern.split(content)

        # Create sentence groups with overlapping context windows
        # Each group includes preceding and following sentences up to context_size
        sentence_groups = [
            " ".join(sentences[max(0, i - context_size) : i + context_size + 1])
            for i in range(len(sentences))
        ]

        # Compute normalized embeddings for sentence groups
        embeddings = self.embedding_model.encode(
            sentences=sentence_groups,
            batch_size=128,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Find chunk boundaries based on similarity threshold
        norms = np.linalg.norm(embeddings, axis=1)
        chunk_boundaries = [0]  # Always start with first sentence

        # Vectorized similarity calculation to replace the loop
        if len(embeddings) > 1:
            # Calculate all dot products between adjacent embeddings at once
            dot_products = np.sum(embeddings[1:] * embeddings[:-1], axis=1)

            # Calculate norm products for normalization
            norm_products = norms[1:] * norms[:-1]

            # Avoid division by zero
            norm_products = np.maximum(1e-8, norm_products)

            # Calculate all similarities at once
            similarities = dot_products / norm_products

            # Find indices where similarity drops below threshold
            new_boundaries = np.where(similarities < similarity_threshold)[0] + 1
            chunk_boundaries.extend(new_boundaries.tolist())

        chunk_boundaries.append(len(sentences))  # Always end with last sentence

        # Create chunks text
        chunks_text: List[str] = []
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            chunk_text = " ".join(sentences[start:end])
            chunks_text.append(chunk_text)

        # Create ChunkedDocument
        chunked_doc = ChunkedDocument(
            uuid=document.uuid,
            vws_id=document.vws_id,
            create_date=document.create_date,
            type=document.type,
            link=document.link,
            attachment_links=document.attachment_links,
            content=document.content,
            chunks={},  # Start with empty chunks dict
        )

        # Generate chunk IDs and create DocumentChunk objects
        chunks = self._create_document_chunks(chunks_text, document, chunked_doc)

        # Set the chunks on the document
        chunked_doc.chunks = chunks

        # Return the complete ChunkedDocument
        return chunked_doc

    def _create_document_chunks(
        self,
        chunks_text: List[str],
        document: RawDocument,
        parent_document: ChunkedDocument,
    ) -> Dict[ChunkID, DocumentChunk]:
        """Create DocumentChunk objects from chunk texts.

        Args:
            chunks_text: List of chunk texts
            document: Original RawDocument
            parent_document: The ChunkedDocument that will contain these chunks

        Returns:
            Dictionary mapping chunk IDs to DocumentChunk objects
        """
        chunks: Dict[ChunkID, DocumentChunk] = {}
        # Use lower 46 bits of UUID for document ID to leave room for chunk index
        doc_id_int = int(document.uuid.int & ((1 << 46) - 1))

        for i, chunk_text in enumerate(chunks_text):
            # Generate unique chunk ID: doc_id in upper bits, chunk index in lower 17 bits
            chunk_id = ChunkID((doc_id_int << 17) | i)

            # Create DocumentChunk with parent document set immediately
            chunks[chunk_id] = DocumentChunk(
                uuid=chunk_id,
                content_date=document.create_date,
                type=document.type,
                link=document.link,
                content=chunk_text,
                parent_document=parent_document,  # Set parent document right away
            )

        return chunks


class NaiveDocumentChunker(DocumentChunkerBase):
    """Splits documents into evenly sized chunks.

    Ensures chunks end at sentence boundaries and include configurable overlap.

    Attributes:
        chunk_size: Target size for each chunk in characters
        overlap_percentage: Percentage of chunk size to overlap with next chunk
        _sentence_pattern: Regex pattern for splitting text into sentences
    """

    def __init__(self, overlap_percentage: float = 0.05):
        """Initialize the document chunker with configurable parameters.

        Args:
            overlap_percentage: Percentage of chunk size to use as overlap (default: 5%)
        """
        self.overlap_percentage = overlap_percentage

        # Regular expression to match sentence boundaries by looking for spaces following a period, question mark, or exclamation mark,
        # but excluding abbreviations like 'e.g.', 'U.S.', or titles like 'Dr.' using negative lookbehinds.
        self._sentence_pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        )

    def chunk_document(self, document: RawDocument) -> ChunkedDocument:
        """Split a document into evenly sized chunks.

        Args:
            document: RawDocument object containing the text to chunk

        Returns:
            ChunkedDocument containing the original document data and its chunks
        """
        content = document.content

        # Calculate target chunk size based on content length and Config.CHUNK_SIZE
        num_chunks = max(1, round(len(content) / Config.IDEAL_CHUNK_SIZE))
        target_chunk_size = len(content) / num_chunks

        # Split text into sentences
        sentences = self._sentence_pattern.split(content)

        # Calculate overlap size in characters
        overlap_size = int(target_chunk_size * self.overlap_percentage)

        # Create chunks text first
        chunks_text = self._create_chunks(sentences, target_chunk_size, overlap_size)

        # Create ChunkedDocument
        chunked_doc = ChunkedDocument(
            uuid=document.uuid,
            vws_id=document.vws_id,
            create_date=document.create_date,
            type=document.type,
            link=document.link,
            attachment_links=document.attachment_links,
            content=document.content,
            chunks={},  # Start with empty chunks dict
        )

        # Generate chunk IDs and create DocumentChunk objects with proper parent document
        chunks = self._create_document_chunks(chunks_text, document, chunked_doc)

        # Set the chunks on the document
        chunked_doc.chunks = chunks

        # Return the complete ChunkedDocument
        return chunked_doc

    def _create_chunks(
        self, sentences: List[str], target_size: float, overlap_size: int
    ) -> List[str]:
        """Create evenly sized chunks, ending at sentence boundaries.

        Args:
            sentences: List of sentences to chunk
            target_size: Target size for each chunk in characters
            overlap_size: Number of characters to overlap between chunks

        Returns:
            List of chunk texts
        """
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length: int = 0
        overlap_sentences: List[str] = []

        for sentence in sentences:
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space

            # If we've reached target size and have at least one sentence
            if current_length >= target_size and len(current_chunk) > 0:
                # Create chunk text
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Prepare overlap for next chunk
                overlap_sentences = []
                overlap_length = 0

                # Work backwards to find sentences for overlap
                for s in reversed(current_chunk):
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                    if overlap_length >= overlap_size:
                        break

                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences.copy()
                current_length = overlap_length

        # Add any remaining content as final chunk
        if current_chunk and (not chunks or current_chunk != overlap_sentences):
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _create_document_chunks(
        self,
        chunks_text: List[str],
        document: RawDocument,
        parent_document: ChunkedDocument,
    ) -> Dict[ChunkID, DocumentChunk]:
        """Create DocumentChunk objects from chunk texts.

        Args:
            chunks_text: List of chunk texts
            document: Original RawDocument
            parent_document: The ChunkedDocument that will contain these chunks

        Returns:
            Dictionary mapping chunk IDs to DocumentChunk objects
        """
        chunks: Dict[ChunkID, DocumentChunk] = {}
        # Use lower 46 bits of UUID for document ID to leave room for chunk index
        doc_id_int = int(document.uuid.int & ((1 << 46) - 1))

        for i, chunk_text in enumerate(chunks_text):
            # Generate unique chunk ID: doc_id in upper bits, chunk index in lower 17 bits
            chunk_id = ChunkID((doc_id_int << 17) | i)

            # Create DocumentChunk with parent document set immediately
            chunks[chunk_id] = DocumentChunk(
                uuid=chunk_id,
                content_date=document.create_date,
                type=document.type,
                link=document.link,
                content=chunk_text,
                parent_document=parent_document,  # Set parent document right away
            )

        return chunks
