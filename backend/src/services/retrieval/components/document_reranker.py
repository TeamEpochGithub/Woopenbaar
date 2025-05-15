"""Document reranking module.

This module provides document reranking classes that prioritize documents based on dates.
The module includes an abstract base class and a date-based implementation.
"""

import abc
import logging
from datetime import datetime
from typing import Any, List

from backend.src.data_classes import ChunkedDocument

logger = logging.getLogger(__name__)


class BaseDocumentReranker(abc.ABC):
    """Base class for document rerankers.

    Defines the interface for document reranking systems.

    Attributes:
        device (str): Device to run model on ('cuda', 'cpu', etc.)
        mode (str): Operation mode for reranking
    """

    def __init__(self, device: str = "cuda", mode: str = "None") -> None:
        """Initialize the document reranker.

        Args:
            device: Device to run model on ('cuda', 'cpu', etc.)
            mode: Operation mode for reranking
        """
        logger.info(f"Initializing document reranker with mode: {mode}")
        self.device = device
        self.mode = mode

    @abc.abstractmethod
    def rerank(
        self, query: str, documents: List[ChunkedDocument]
    ) -> List[ChunkedDocument]:
        """Rerank documents based on custom criteria.

        Args:
            query: User query string
            documents: List of documents to rerank

        Returns:
            List of reranked documents
        """

    def train(self, train_data_path: str, config: Any, output_dir: str) -> None:
        """Train the reranker model.

        Default implementation logs a warning. Subclasses should override
        this method if they support training.

        Args:
            train_data_path: Path to training data
            config: Training configuration
            output_dir: Output directory for trained model
        """
        logger.warning(
            f"{self.__class__.__name__} does not implement training functionality"
        )


class DateDocumentReranker(BaseDocumentReranker):
    """Reranks documents based on their creation date.

    Provides two modes: ranking and weighting. Ranking orders documents by date,
    while weighting assigns scores based on recency.

    Attributes:
        device (str): Device to run model on ('cuda', 'cpu', etc.)
        mode (str): Operation mode ('ranked', 'weighed', or None)
    """

    def __init__(self, device: str = "cuda", mode: str = "None") -> None:
        """Initialize date-based document reranker.

        Args:
            device: Device to run model on ('cuda', 'cpu', etc.)
            mode: States whether the documents will be ranked or weighed.
                  Ranking would be more static than weighing
        """
        super().__init__(device, mode)
        logger.info(f"Loading Document Date Reranker in mode: {mode}")

    @classmethod
    def ranking(cls, documents: List[ChunkedDocument]) -> List[ChunkedDocument]:
        """Sorts documents from oldest to newest based on 'create_date'.
        Documents with no date are placed at the end.

        Args:
            documents: List of ChunkedDocument objects.

        Returns:
            List of documents sorted by date.
        """

        def get_sort_key(doc: ChunkedDocument):
            return doc.create_date or datetime.max

        return sorted(documents, key=get_sort_key)

    @classmethod
    def rerank(
        cls, query: str, documents: List[ChunkedDocument]
    ) -> List[ChunkedDocument]:
        """Rerank documents based on their creation date, oldest documents first.

        Args:
            query: User query (unused)
            documents: List of documents

        Returns:
            List of documents processed according to the selected mode
        """

        return cls.ranking(documents)
