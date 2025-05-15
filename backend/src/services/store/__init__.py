"""Storage services package.

This package provides storage-related services including:
- ChatHistoryService: JSON-based storage for chat interactions

The services handle data persistence and retrieval with thread-safe
concurrent access.
"""

from .chat_history_service import ChatHistoryService

__all__ = ["ChatHistoryService"]
