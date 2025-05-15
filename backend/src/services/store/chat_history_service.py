"""Service for storing and retrieving chat history using JSON files.

This module provides functionality to persist chat interactions to JSON files
and retrieve them later. It handles concurrent access using thread locks
and provides basic pagination capabilities.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

from backend.conf.config import Config
from backend.src.data_classes.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChatHistoryRecord:
    """Represents a single chat interaction record.

    Attributes:
        timestamp: When the chat occurred
        question: User's input question
        response: System's generated response
        document_chunks: List of document chunks used to answer the query
        data_sources: Set of data sources used for this chat
        chat_type: Type of chat processing used (standard_rag, adaptive_rag)
    """

    timestamp: str
    question: str
    response: str
    document_chunks: List[DocumentChunk] = field(default_factory=list)
    data_sources: Set[str] = field(default_factory=set)
    chat_type: str = "standard_rag"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for JSON serialization.

        Returns:
            Dictionary containing all record attributes in a JSON-serializable format
        """
        result = asdict(self)

        result["document_chunks"] = [chunk.to_json() for chunk in self.document_chunks]
        result["data_sources"] = list(self.data_sources)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatHistoryRecord":
        """Create a ChatHistoryRecord from a dictionary.

        Args:
            data: Dictionary containing chat history data
        """
        return cls(**data)


class ChatHistoryService:
    """Service for storing and retrieving chat history using JSON files.

    This class provides methods to store chat interactions in a JSON file and
    retrieve them with pagination support. It uses file locking to handle
    concurrent access safely.

    Attributes:
        file_path (str): Path to the JSON file storing chat history
        lock (threading.Lock): Thread lock for safe concurrent file access
    """

    def __init__(self) -> None:
        """Initialize the JSON storage using path from Config."""
        self.file_path = Config.CHAT_HISTORY_PATH
        # Thread lock ensures only one thread can access the file at a time
        self.lock = threading.Lock()
        self.initialize_storage()

    def initialize_storage(self) -> None:
        """Create the chat history file if it doesn't exist.

        Creates an empty JSON array file at the specified file_path if no file
        exists yet. This ensures we have a valid JSON file to work with.
        """
        try:
            if not os.path.exists(self.file_path):
                with open(self.file_path, "w") as f:
                    json.dump([], f)
                logger.info(f"Chat history file initialized at {self.file_path}")
        except Exception as e:
            logger.error(f"Error initializing chat history file: {str(e)}")

    def store_chat_history(self, chat_data: ChatHistoryRecord) -> bool:
        """Store chat history in the JSON file.

        Appends a new chat interaction to the history file. Uses a lock to ensure
        thread-safe file access.

        Args:
            chat_data: Dictionary or ChatHistoryRecord containing chat history data

        Returns:
            bool: True if storage was successful, False otherwise
        """

        with self.lock:
            try:
                # Read existing data if file exists and is not empty
                if (
                    os.path.exists(self.file_path)
                    and os.path.getsize(self.file_path) > 0
                ):
                    with open(self.file_path, "r") as f:
                        try:
                            history_data: List[Dict[str, Any]] = json.load(f)
                            logger.debug(
                                f"Loaded existing history with {len(history_data)} entries"
                            )
                        except json.JSONDecodeError:
                            logger.error(
                                "Invalid JSON in chat history file. Creating new file."
                            )
                            history_data = []
                else:
                    history_data = []
                    logger.debug("Creating new chat history")

                # Add new entry as dictionary (for JSON serialization)
                history_data.append(chat_data.to_dict())
                logger.debug(
                    f"Added new chat history entry. Total entries: {len(history_data)}"
                )

                # Write updated history back to file with pretty formatting
                with open(self.file_path, "w") as f:
                    json.dump(history_data, f, indent=2)

                return True
            except Exception as e:
                logger.error(f"Error storing chat history: {str(e)}")
                return False

    def store_chat_with_chunks(
        self,
        timestamp: str,
        question: str,
        response: str,
        document_chunks: List[DocumentChunk],
        chat_type: str = "standard_rag",
        data_sources: List[str] = [],
    ) -> bool:
        """Store chat history with document chunks in the JSON file.

        This is a convenience method that takes individual parameters and creates
        a ChatHistoryRecord for storage.

        Args:
            timestamp: When the chat occurred
            question: User's input question
            response: System's generated response
            document_chunks: List of document chunks used to answer the query
            chat_type: Type of chat processing used (standard_rag, adaptive_rag)
            data_sources: Optional list of data sources used during retrieval

        Returns:
            bool: True if storage was successful, False otherwise
        """
        # Create a ChatHistoryRecord
        record = ChatHistoryRecord(
            timestamp=timestamp,
            question=question,
            response=response,
            document_chunks=document_chunks,
            data_sources=set(data_sources),
            chat_type=chat_type,
        )

        # Store the record
        return self.store_chat_history(record)

    def get_chat_history(
        self, limit: int = 100, skip: int = 0
    ) -> List[ChatHistoryRecord]:
        """Retrieve chat history from the JSON file.

        Gets chat history with pagination support. Results are sorted by timestamp
        with newest first.

        Args:
            limit: Maximum number of records to retrieve. Defaults to 100.
            skip: Number of records to skip for pagination. Defaults to 0.

        Returns:
            List of ChatHistoryRecord objects representing chat history.
            Returns empty list if no history exists or on error.
        """
        with self.lock:
            try:
                if (
                    os.path.exists(self.file_path)
                    and os.path.getsize(self.file_path) > 0
                ):
                    with open(self.file_path, "r") as f:
                        history_data: List[Dict[str, Any]] = json.load(f)
                        logger.debug(f"Retrieved {len(history_data)} history entries")

                    # Sort by timestamp (newest first)
                    history_data.sort(
                        key=lambda x: x.get("timestamp", ""), reverse=True
                    )

                    # Apply pagination
                    paginated_data = history_data[skip : skip + limit]

                    # Convert dictionaries to ChatHistoryRecord objects
                    result = [
                        ChatHistoryRecord.from_dict(data) for data in paginated_data
                    ]

                    logger.debug(
                        f"Returning {len(result)} entries (skip={skip}, limit={limit})"
                    )
                    return result
                else:
                    logger.debug("No chat history found")
                    return []
            except Exception as e:
                logger.error(f"Error retrieving chat history: {str(e)}")
                return []
