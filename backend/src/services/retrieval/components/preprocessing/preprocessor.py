"""Document preprocessing module.

This module provides the main preprocessing pipeline for transforming raw document data
into structured, chunked documents with enriched metadata. It orchestrates multiple
preprocessing steps including parsing, cleaning, chunking, and metadata extraction.

The Preprocessor class coordinates these steps while providing error handling and logging
to ensure robust document processing at scale.
"""

import abc
import logging
import multiprocessing
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm

from backend.src.data_classes import ChunkedDocument, RawDocument

from .components.abbreviation_replacer import AbbreviationProcessorBase
from .components.chunker import DocumentChunkerBase
from .components.cleaner import ContentCleanerBase, EmptyContentError
from .components.extractor import ExtractorBase
from .components.parser import DocumentParserBase

logger = logging.getLogger(__name__)

# Global variable for multiprocessing workers.
global_preprocessor: Optional["Preprocessor"] = None


def init_worker(preprocessor_instance: "Preprocessor") -> None:
    """Initialize the global preprocessor for multiprocessing workers.

    Args:
        preprocessor_instance: Preprocessor instance to be used by workers
    """
    global global_preprocessor
    global_preprocessor = preprocessor_instance


def worker(raw_doc: RawDocument) -> Optional[ChunkedDocument]:
    """Process a single document in a worker process.

    Args:
        raw_doc: Raw document to process

    Returns:
        Processed chunked document or None if document should be excluded
    """
    if global_preprocessor is None:
        raise ValueError("Preprocessor not initialized")
    return global_preprocessor.preprocess_document(raw_doc)


class PreprocessorBase(abc.ABC):
    """Abstract base class defining the interface for a Preprocessor."""

    @abc.abstractmethod
    def preprocess_document(self, raw_doc: RawDocument) -> Optional[ChunkedDocument]:
        """Process a single raw document through the entire preprocessing pipeline.

        Args:
            raw_doc: Raw document to process

        Returns:
            Processed chunked document or None if document should be excluded
        """


class Preprocessor:
    """Main preprocessing module that orchestrates parsing, cleaning and chunking of documents.

    This class coordinates the full preprocessing pipeline by combining multiple specialized
    components to transform raw document data into structured, enriched documents. The pipeline
    includes parsing, content cleaning, abbreviation expansion, chunking, and metadata extraction.

    Attributes:
        parser: Component for loading and parsing document files
        cleaner: Component for cleaning and normalizing document content
        chunker: Component for splitting documents into semantic chunks
        abbreviation_processor: Component for expanding abbreviations in text
        extractor: Component for extracting metadata from document content
        _mp_pool: Reusable multiprocessing pool to avoid repeated initialization
    """

    def __init__(
        self,
        parser: DocumentParserBase,
        cleaner: ContentCleanerBase,
        chunker: DocumentChunkerBase,
        abbreviation_replacer: AbbreviationProcessorBase,
        extractor: ExtractorBase,
    ):
        """Initialize preprocessing components.

        Creates a preprocessing pipeline by combining specialized components for each
        preprocessing step. Components can be injected for customization and testing.

        Args:
            parser: Parser for loading documents from files
            cleaner: Content cleaner for document text
            chunker: Chunker to split documents into chunks
            abbreviation_replacer: Replacer for expanding abbreviations
            extractor: Extractor for metadata inference
        """
        self.parser = parser
        self.cleaner = cleaner
        self.chunker = chunker
        self.abbreviation_processor = abbreviation_replacer
        self.extractor = extractor
        self._mp_pool: Optional[Pool] = None

    def _get_or_create_pool(self, num_workers: int) -> Pool:
        """Get an existing multiprocessing pool or create a new one if none exists.

        This ensures we reuse the same pool across multiple processing calls, preventing
        repeated initialization of tokenizers which leads to warnings.

        Args:
            num_workers: Number of worker processes for the pool

        Returns:
            A multiprocessing pool
        """
        if self._mp_pool is None:
            logger.info(f"Creating multiprocessing pool with {num_workers} workers")
            self._mp_pool = multiprocessing.Pool(
                processes=num_workers, initializer=init_worker, initargs=(self,)
            )
        return self._mp_pool

    def __del__(self) -> None:
        """Cleanup multiprocessing pool when the preprocessor is destroyed."""
        if self._mp_pool is not None:
            logger.info("Closing multiprocessing pool")
            self._mp_pool.close()
            self._mp_pool.join()

    def preprocess_document(self, raw_doc: RawDocument) -> Optional[ChunkedDocument]:
        """Process a single raw document through the entire preprocessing pipeline.

        Args:
            raw_doc: Raw document to process

        Returns:
            Processed and chunked document with enriched metadata, or None if document should be excluded
        """
        try:
            # 1. Clean and normalize content
            raw_doc.content = self.cleaner.clean_content(raw_doc.content, raw_doc.type)

            # 2. Expand abbreviations & Abbreviate expansions
            raw_doc.content = self.abbreviation_processor.process_text(
                raw_doc.content, mode="both"
            )

            # 3. Split document into chunks and get ChunkedDocument
            chunked_document = self.chunker.chunk_document(raw_doc)

            # 4. Extract metadata and enrich document
            chunked_document = self.extractor.enrich_chunked_document(chunked_document)

            return chunked_document

        except EmptyContentError as e:
            logger.info(
                f"Excluding document due to empty content: {raw_doc.link} - {str(e)}"
            )
            return None
        except Exception as e:
            logger.error(f"Error processing document {raw_doc.link}: {str(e)}")
            raise

    def load_and_preprocess_documents_mp(
        self, raw_documents: Sequence[RawDocument], num_workers: int = 1
    ) -> List[ChunkedDocument]:
        """Process a list of raw documents in parallel using multiprocessing.

        Args:
            raw_documents: Sequence of raw documents to process
            num_workers: Number of worker processes to use

        Returns:
            List of processed chunked documents (excluding empty documents)

        Raises:
            Exception: If an error occurs during parallel processing
        """
        if not raw_documents:
            logger.warning("No documents provided for preprocessing")
            return []

        try:
            pool = self._get_or_create_pool(num_workers)
            results: List[ChunkedDocument] = []
            total: int = len(raw_documents)
            with tqdm(total=total, desc="Processing documents") as pbar:
                for result in pool.imap_unordered(worker, raw_documents):
                    if result is not None:  # Only include non-empty documents
                        results.append(result)
                    pbar.update(1)
            return results
        except Exception as e:
            logger.error(f"Error during parallel document processing: {e}")
            raise

    def load_and_preprocess_mp(
        self, documents_path: Path, num_workers: int = 1
    ) -> List[ChunkedDocument]:
        """Load and process documents from a directory in parallel.

        Args:
            documents_path: Path to directory containing documents
            num_workers: Number of worker processes to use

        Returns:
            List of processed chunked documents (excluding empty documents)
        """
        # Load raw documents
        raw_documents: List[RawDocument] = self.parser.parse_directory(documents_path)

        if not raw_documents:
            logger.warning("No documents found to process")
            return []

        # Process documents with the reusable pool
        return self.load_and_preprocess_documents_mp(raw_documents, num_workers)
