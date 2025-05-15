"""Chat endpoints module.

This module provides Flask routes for chat functionality including:
1. Standard RAG (Retrieval Augmented Generation)
2. Adaptive RAG that dynamically selects optimal retrieval strategy
"""

import logging
import threading
import time
import traceback
from typing import Any, Dict, Generator, List, Optional, Union

from flask import Blueprint, Response, jsonify
from flask_pydantic import validate  # type: ignore
from pydantic import BaseModel, Field

from backend.conf import Config
from backend.src.api.middleware.exceptions import ServiceError
from backend.src.api.utils.citations import process_response_citations
from backend.src.api.utils.sse import create_sse_event, create_sse_response
from backend.src.data_classes import QueryResult
from backend.src.data_classes.filters import FilterModel, FilterOptions
from backend.src.services import (
    BaseLLMService,
    ChatHistoryService,
    QueryProcessingService,
    RetrievalService,
    SafetyService,
)

logger = logging.getLogger(__name__)

# Blueprint definition
chat_bp = Blueprint("chat", __name__)


# Schema definitions
class ChatRequest(BaseModel):
    """Chat request model for validation."""

    message: str = Field(..., description="User's message")
    filters: Optional[FilterModel] = Field(None, description="Filter options")
    use_internet: bool = Field(
        False, description="Whether to use internet search for retrieval"
    )
    prioritize_earlier: bool = Field(
        False, description="Whether to prioritize earlier documents"
    )

    def get_filter_options(self) -> Optional[FilterOptions]:
        """Get converted filter options.

        Returns:
            Optional[FilterOptions]: Converted filter options
        """
        if self.filters:
            return self.filters.to_filter_options()
        return None


class ChatResponseModel(BaseModel):
    """Chat response model."""

    response: str = Field(..., description="Generated response text")
    chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved document chunks"
    )
    documents: List[Dict[str, Any]] = Field(
        default_factory=list, description="Source documents"
    )
    chunk_ids: List[str] = Field(
        default_factory=list, description="IDs of retrieved document chunks"
    )
    document_ids: List[str] = Field(
        default_factory=list, description="IDs of source documents"
    )
    timestamp: Optional[str] = Field(None, description="Response timestamp")
    reasoning_steps: List[str] = Field(
        default_factory=list, description="Reasoning steps for adaptive chat"
    )
    type: Optional[str] = Field(None, description="Response type (for streaming)")
    suggested_questions: List[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )
    data_sources_used: List[str] = Field(
        default_factory=list, description="List of data sources used during processing"
    )


def init_chat_routes(
    retrieval_service: RetrievalService,
    query_processing_service: QueryProcessingService,
    history_service: Optional[ChatHistoryService] = None,
    llm_service: Optional[BaseLLMService] = None,
    safety_service: Optional[SafetyService] = None,
) -> Blueprint:
    """Initialize chat routes with the provided services.

    Args:
        retrieval_service: Service for retrieval augmented generation.
        query_processing_service: Service for reasoning-based query processing.
        history_service: Optional service for storing chat history.
        llm_service: Optional local LLM service for generating responses.
        safety_service: Optional service for analyzing user input for safety/relevance.

    Returns:
        Blueprint: Flask blueprint with configured chat routes.
    """

    @chat_bp.route("/chat/with-context", methods=["POST"])
    @validate()
    def chat_with_context(body: ChatRequest) -> Union[Dict[str, Any], tuple[Response, int]]:  # type: ignore
        """Retrieve context and generate a response for RAG chat endpoint.

        Args:
            body: Validated request body

        Returns:
            Response with generated chat content
        """
        user_message: str = body.message
        filters: Optional[FilterOptions] = body.get_filter_options()
        logger.info(f"Processing chat request with filters: {filters}")

        # Check if the message is safe and relevant using safety service
        sentence_analysis: Dict[str, Any] = {"flagged": False}
        if safety_service and False:
            sentence_analysis = safety_service.check_text_safety(user_message)
            logger.info(f"Safety analysis result: {sentence_analysis}")

        if sentence_analysis.get("flagged", False):
            # If message is flagged, return a standard rejection message
            logger.warning(f"User message flagged: {user_message}")
            response_text = (
                "Uw verzoek is niet verwerkt omdat het niet relevant is voor de aangeboden "
                "service en/of in strijd is met de regels van de service."
            )
            response = ChatResponseModel(
                response=response_text,
                chunk_ids=[],
                document_ids=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                type="standard_rag",
            )
            return jsonify(response.model_dump()), 200

        try:
            # Process the query using the standard RAG approach
            query_result = query_processing_service.process_standard_query(
                query=user_message,
                filters=filters,
                prioritize_earlier=body.prioritize_earlier,
            )

            # Add timestamp to chat history
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Process citations in the response and get updated chunk order
            chunk_ids = [str(chunk.uuid) for chunk in query_result.chunks]
            processed_response, ordered_chunk_ids = process_response_citations(
                query_result.response, chunk_ids
            )

            # Create a mapping from chunk_id to the corresponding document_id
            chunk_to_doc_map = {
                str(chunk.uuid): str(chunk.parent_document.uuid)
                for chunk in query_result.chunks
            }

            # Get the document IDs in the same order as the reordered chunks
            ordered_doc_ids = [
                chunk_to_doc_map.get(chunk_id, "missing_document")
                for chunk_id in ordered_chunk_ids
            ]

            # Store history with consistent format
            if history_service and query_result.chunks:
                try:
                    history_service.store_chat_with_chunks(
                        timestamp=timestamp,
                        question=user_message,
                        response=processed_response,
                        document_chunks=query_result.chunks,
                        chat_type="standard_rag",
                        data_sources=list(query_result.data_sources_used),
                    )
                except Exception as e:
                    # Log but don't fail the response if history storage fails
                    logger.warning(f"Failed to store chat history: {str(e)}")

            # Format response with the processed text and reordered IDs
            response = ChatResponseModel(
                response=processed_response,
                chunk_ids=ordered_chunk_ids,
                document_ids=ordered_doc_ids,
                timestamp=timestamp,
                suggested_questions=query_result.suggested_questions,
                data_sources_used=list(query_result.data_sources_used),
                type="standard_rag",
            )

            return jsonify(response.model_dump()), 200
        except Exception as e:
            logger.error(f"Failed to process chat: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceError(message="Failed to process chat", details=str(e))

    @chat_bp.route("/chat/adaptive", methods=["POST"])
    @validate()
    def chat_adaptive(body: ChatRequest) -> Response:  # type: ignore
        """Adaptive chat endpoint that optimizes retrieval based on query analysis.

        Args:
            body: Validated request body

        Returns:
            Streaming response with generated chat content
        """
        user_message: str = body.message
        max_iterations: int = Config.MAX_ITERATIONS
        filters: Optional[FilterOptions] = body.get_filter_options()

        # Check if the message is safe and relevant using safety service
        sentence_analysis: Dict[str, Any] = {"flagged": False}
        if safety_service and False:
            sentence_analysis = safety_service.check_text_safety(user_message)
            logger.info(f"Safety analysis result: {sentence_analysis}")

        if sentence_analysis.get("flagged", False):
            # If message is flagged, return a standard rejection message
            logger.warning(f"User message flagged: {user_message}")
            response_text = (
                "Uw verzoek is niet verwerkt omdat het niet relevant is voor de aangeboden "
                "service en/of in strijd is met de regels van de service."
            )

            # For SSE response, we need to use SSE format
            def generate_rejection_events() -> Generator[str, None, None]:
                yield create_sse_event("open", "Connection established")
                yield create_sse_event(
                    "final_response",
                    {
                        "type": "final_response",
                        "response": response_text,
                        "doc_ids": [],
                        "chunks": [],
                        "reasoning_steps": [],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "suggested_questions": [],
                        "internet_used": False,
                    },
                )

            return create_sse_response(generate_rejection_events)

        def generate_sse_events() -> Generator[str, None, None]:
            """Generate SSE events for adaptive chat."""
            try:
                # Add timestamp at start of conversation
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Send initial connection event
                yield create_sse_event("open", "Connection established")

                # Set up a local function for reasoning step updates
                all_reasoning_steps: List[str] = []
                event_queue: List[str] = []

                def reasoning_callback(step: str) -> None:
                    all_reasoning_steps.append(step)
                    event_queue.append(
                        create_sse_event(
                            "reasoning_update",
                            {
                                "type": "reasoning_update",
                                "reasoning_steps": all_reasoning_steps,
                            },
                        )
                    )

                # Process query in a thread to avoid blocking
                query_result: Optional[QueryResult] = None
                processing_error = None

                def process_query() -> None:
                    nonlocal query_result, processing_error
                    try:
                        query_result = query_processing_service.process_adaptive_query(
                            query=user_message,
                            max_iterations=max_iterations,
                            filters=filters,
                            frontend_callback=reasoning_callback,
                            prioritize_earlier=body.prioritize_earlier,
                        )
                    except Exception as e:
                        logger.error(f"Error processing query: {str(e)}")
                        logger.error(traceback.format_exc())
                        processing_error = e

                processing_thread = threading.Thread(target=process_query)
                processing_thread.start()

                # Send reasoning updates while waiting for processing to complete
                no_events = True
                while processing_thread.is_alive():
                    if event_queue:
                        no_events = False
                        for event in event_queue:
                            yield event
                        event_queue.clear()
                    else:
                        time.sleep(0.1)  # Sleep a bit before checking again

                # Wait for processing to complete
                processing_thread.join()

                # Check if there were any reasoning step events that weren't sent yet
                if no_events and all_reasoning_steps:
                    yield create_sse_event(
                        "reasoning_update",
                        {
                            "type": "reasoning_update",
                            "reasoning_steps": all_reasoning_steps,
                        },
                    )

                # Check if there was an error during processing
                if processing_error:
                    error_msg = str(processing_error)
                    error_details = type(processing_error).__name__
                    logger.error(f"Query processing failed: {error_msg}")

                    # Add error as a reasoning step for visibility
                    error_step = f"Er is een fout opgetreden: {error_details}"
                    all_reasoning_steps.append(error_step)

                    error_msg = "Er is een interne systeemfout opgetreden bij het verwerken van uw vraag."
                    logger.error(str(processing_error))

                    yield create_sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": "Query processing failed",
                            "details": error_msg,
                            "reasoning_steps": all_reasoning_steps,
                            "internet_used": body.use_internet,
                        },
                    )
                    return

                # If no results were found
                if not query_result or not query_result.chunks:
                    yield create_sse_event(
                        "final_response",
                        {
                            "type": "final_response",
                            "response": "Ik heb geen relevante informatie kunnen vinden voor uw vraag.",
                            "doc_ids": [],
                            "chunks": [],
                            "reasoning_steps": all_reasoning_steps,
                            "timestamp": timestamp,
                            "suggested_questions": [],
                            "data_sources_used": [],
                        },
                    )
                    return

                # Process citations in the response and get updated chunk order
                # At this point, we know query_result is not None and has chunks
                assert query_result is not None

                # Extract chunk ids for citation processing
                chunk_ids: List[str] = [
                    str(chunk.uuid) for chunk in query_result.chunks
                ]
                processed_response, ordered_chunk_ids = process_response_citations(
                    query_result.response, chunk_ids
                )

                # Create a mapping from chunk_id to the corresponding document_id
                chunk_to_doc_map: Dict[str, str] = {
                    str(chunk.uuid): str(chunk.parent_document.uuid)
                    for chunk in query_result.chunks
                }

                # Get the document IDs in the same order as the reordered chunks
                ordered_doc_ids = [
                    chunk_to_doc_map.get(chunk_id, "missing_document")
                    for chunk_id in ordered_chunk_ids
                ]

                # Store history with consistent format
                if history_service:
                    try:
                        history_service.store_chat_with_chunks(
                            timestamp=timestamp,
                            question=user_message,
                            response=processed_response,
                            document_chunks=query_result.chunks,
                            chat_type="adaptive_rag",
                            data_sources=list(query_result.data_sources_used),
                        )
                    except Exception as e:
                        # Log but don't fail the response if history storage fails
                        logger.warning(f"Failed to store chat history: {str(e)}")

                # Format response with processed text and reordered IDs
                response = ChatResponseModel(
                    response=processed_response,
                    chunk_ids=ordered_chunk_ids,
                    document_ids=ordered_doc_ids,
                    timestamp=timestamp,
                    reasoning_steps=query_result.reasoning_steps,
                    suggested_questions=query_result.suggested_questions,
                    type="final_response",
                    data_sources_used=list(query_result.data_sources_used),
                )

                yield create_sse_event("final_response", response.model_dump())

            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                logger.error(traceback.format_exc())
                yield create_sse_event(
                    "error",
                    {
                        "type": "error",
                        "error": "An error occurred while processing your request",
                        "details": str(e),
                    },
                )

        return create_sse_response(generate_sse_events)

    return chat_bp
