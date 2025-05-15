"""Retrieval package for document search and retrieval functionality.

The main interface is through the RetrievalService class which coordinates
the various retrieval components. For internet retrieval, use the
InternetRetrievalService.
"""

from .retrieval_service import RetrievalService

# Expose main retrieval service classes at package level
__all__ = [
    "RetrievalService",
]
