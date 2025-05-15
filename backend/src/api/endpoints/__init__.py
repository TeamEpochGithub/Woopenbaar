"""API endpoints package.

This package contains endpoint definitions for the API.
"""

from typing import Optional

from flask import Flask

from backend.src.api.endpoints.chat import init_chat_routes
from backend.src.api.endpoints.retrieval import init_retrieval_routes
from backend.src.services import (
    BaseLLMService,
    ChatHistoryService,
    QueryProcessingService,
    RetrievalService,
    SafetyService,
)


def register_endpoints(
    app: Flask,
    retrieval_service: RetrievalService,
    query_processing_service: QueryProcessingService,
    history_service: Optional[ChatHistoryService] = None,
    llm_service: Optional[BaseLLMService] = None,
    safety_service: Optional[SafetyService] = None,
) -> None:
    """Register all API endpoints with the application.

    Args:
        app: Flask application
        retrieval_service: Service for retrieval operations
        query_processing_service: Service directing the flow of the query through the system
        history_service: Optional service for storing chat history
        llm_service: Optional local LLM service for generating responses
        safety_service: Optional service for safety checks
    """
    # Import and register chat endpoints
    app.register_blueprint(
        init_chat_routes(
            retrieval_service,
            query_processing_service,
            history_service,
            llm_service,
            safety_service,
        )
    )

    # Import and register retrieval endpoints
    app.register_blueprint(init_retrieval_routes(retrieval_service))
