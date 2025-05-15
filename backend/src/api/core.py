"""Core API setup and configuration.

This module configures API middleware, error handling, and other global API components.
"""

import logging
from typing import Optional

from flask import Flask

from backend.src.api.endpoints import register_endpoints
from backend.src.api.middleware import register_middleware
from backend.src.services import (
    BaseLLMService,
    ChatHistoryService,
    QueryProcessingService,
    RetrievalService,
    SafetyService,
)

logger = logging.getLogger(__name__)


def setup_api(
    app: Flask,
    retrieval_service: RetrievalService,
    query_processing_service: QueryProcessingService,
    history_service: Optional[ChatHistoryService] = None,
    llm_service: Optional[BaseLLMService] = None,
    safety_service: Optional[SafetyService] = None,
) -> None:
    """Set up API with middleware and endpoints.

    Args:
        app: Flask application
        retrieval_service: Service for retrieval operations
        query_processing_service: Service directing the flow of the query through the system
        history_service: Optional service for storing chat history
        llm_service: Optional local LLM service for generating responses
        safety_service: Optional service for safety checks
    """
    # Register middleware
    register_middleware(app)

    # Register endpoints
    register_endpoints(
        app,
        retrieval_service,
        query_processing_service,
        history_service,
        llm_service,
        safety_service,
    )
