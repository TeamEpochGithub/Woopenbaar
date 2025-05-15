"""Preprocessing package for document processing.

This package provides components for preprocessing documents through the components
subpackage, along with the main Preprocessor class that orchestrates the pipeline.

The components subpackage contains:
- AbbreviationReplacer: Expands common abbreviations in text
- DocumentChunker: Splits documents into smaller chunks
- ContentCleaner: Cleans and normalizes document content
- Extractor: Extracts metadata and structured information
- DocumentParser: Parses documents from different formats

These components can be used individually or composed into a full preprocessing
pipeline to prepare documents for downstream tasks like retrieval and analysis.
"""

from backend.conf.config import Config

from ..dense_embedder import SentenceTransformerEmbedder
from .components import AbbreviationProcessor, ContentCleaner, DocumentParser, Extractor
from .components.chunker import NaiveDocumentChunker, SemanticDocumentChunker
from .preprocessor import Preprocessor

__all__ = [
    "create_preprocessor",
]


def create_preprocessor(use_semantic_chunker: bool = False) -> Preprocessor:
    """Create and configure a Preprocessor instance with all required components.

    Args:
        use_semantic_chunker: If True, uses the semantic DocumentChunker instead of
                              the naive chunker. Default is False.

    Returns:
        Configured Preprocessor instance ready for document processing
    """
    # Initialize preprocessing components
    doc_parser = DocumentParser()

    # Select the appropriate chunker based on the parameter
    if use_semantic_chunker:
        embedding_model = SentenceTransformerEmbedder(
            Config.DENSE_EMBEDDER_MODEL, Config.DEVICE
        )
        chunker = SemanticDocumentChunker(embedding_model)
    else:
        chunker = NaiveDocumentChunker()

    abbreviation_replacer = AbbreviationProcessor(
        abbreviation_file=str(Config.ABBREVIATIONS_PATH)
    )
    extractor = Extractor()

    # Create and return preprocessing pipeline
    return Preprocessor(
        parser=doc_parser,
        cleaner=ContentCleaner(),
        chunker=chunker,
        abbreviation_replacer=abbreviation_replacer,
        extractor=extractor,
    )
