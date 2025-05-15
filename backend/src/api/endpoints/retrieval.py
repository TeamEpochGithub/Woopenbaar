"""Retrieval endpoints module.

This module provides Flask routes for retrieval operations without LLM processing.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from flask import Blueprint, Response, jsonify
from flask_pydantic import validate  # type: ignore
from pydantic import BaseModel, Field, field_validator

from backend.conf.config import Config
from backend.src.api.middleware.exceptions import NotFoundError, ServiceError
from backend.src.data_classes import ChunkedDocument, DocumentChunk
from backend.src.data_classes.filters import FilterModel, FilterOptions
from backend.src.services import RetrievalService

logger = logging.getLogger(__name__)

# Blueprint definition
retrieval_bp = Blueprint("retrieval", __name__)


# Schema definitions
class RetrievalRequest(BaseModel):
    """Retrieval request model for validation."""

    message: str = Field(..., description="User's query for retrieval")
    filters: Optional[FilterModel] = Field(None, description="Filter options")

    def get_filter_options(self) -> Optional[FilterOptions]:
        """Get converted filter options.

        Returns:
            Optional[FilterOptions]: Converted filter options
        """
        if self.filters:
            return self.filters.to_filter_options()
        return None


class RandomRequest(BaseModel):
    """Random content request model for validation."""

    count: int = Field(1, description="Number of items to retrieve")

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: int) -> int:
        """Validate count is greater than zero.

        Args:
            v: The count value to validate

        Returns:
            The validated count value
        """
        if v <= 0:
            return 1
        return v


class ChunkBatchRequest(BaseModel):
    """Request model for retrieving multiple chunks by their IDs."""

    chunk_ids: List[str] = Field(..., description="List of chunk IDs to retrieve")


class DocumentBatchRequest(BaseModel):
    """Request model for retrieving multiple documents by their UUIDs."""

    document_ids: List[str] = Field(
        ..., description="List of document UUIDs to retrieve"
    )


class RetrievalIDsResponseModel(BaseModel):
    """Retrieval response model that only includes IDs."""

    chunk_ids: List[str] = Field(
        default_factory=list, description="IDs of retrieved document chunks"
    )
    document_ids: List[str] = Field(
        default_factory=list, description="IDs of source documents"
    )
    response: str = Field("", description="Generated response text")
    timestamp: str = Field("", description="Response timestamp")
    reasoning_steps: List[str] = Field(
        default_factory=list, description="Reasoning steps for adaptive chat"
    )


class ChunkBatchResponseModel(BaseModel):
    """Response model for batch retrieval of chunks."""

    chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved document chunks"
    )


class DocumentBatchResponseModel(BaseModel):
    """Response model for batch retrieval of documents."""

    documents: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved documents"
    )


def init_retrieval_routes(
    retrieval_service: RetrievalService,
) -> Blueprint:
    """Initialize retrieval routes with the provided services.

    Args:
        retrieval_service: Service for retrieval operations.

    Returns:
        Blueprint: Flask blueprint with configured retrieval routes.
    """

    @retrieval_bp.route("/retrieve-context", methods=["POST"])
    @validate()
    def retrieve_context(body: RetrievalRequest) -> Union[Dict[str, Any], tuple[Response, int]]:  # type: ignore
        """Endpoint that only retrieves relevant chunks without LLM processing.

        Args:
            body: Validated request body

        Returns:
            Response with retrieved content
        """
        user_message: str = body.message
        filters: Optional[FilterOptions] = body.get_filter_options()
        logger.info(f"Processing retrieval request with filters: {filters}")

        try:
            # Retrieve relevant document chunks and their source documents with filters
            relevant_chunks, source_docs = retrieval_service.find(
                query=user_message,
                initial_documents_k=Config.INITIAL_DOCUMENTS_K,
                final_documents_k=Config.FINAL_DOCUMENTS_K,
                initial_chunks_k=Config.INITIAL_CHUNKS_K,
                final_chunks_k=Config.FINAL_CHUNKS_K,
                filters=filters,
                source_name="",  # Empty string to search all sources
            )

            # Generate timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Return only IDs of chunks and documents
            response = RetrievalIDsResponseModel(
                chunk_ids=[str(chunk.uuid) for chunk in relevant_chunks],
                document_ids=[str(doc.uuid) for doc in source_docs],
                response="",  # No LLM response for retrieval-only endpoints
                timestamp=timestamp,
            )

            return jsonify(response.model_dump()), 200
        except Exception as e:
            logger.error(f"Failed to retrieve content: {str(e)}")
            raise ServiceError(message="Failed to retrieve content", details=str(e))

    @retrieval_bp.route("/random-chunks", methods=["GET"])
    @validate()
    def get_random_chunks(query: RandomRequest) -> Union[Dict[str, Any], tuple[Response, int]]:  # type: ignore
        """Endpoint that returns random document chunks for benchmark generation.

        Args:
            query: Validated query parameters

        Returns:
            Response with random chunks
        """
        count: int = query.count

        try:
            chunks: List[DocumentChunk] = retrieval_service.get_random_chunks(count)
            if not chunks:
                raise NotFoundError(
                    message="No chunks available",
                    details="The database does not contain any document chunks",
                )

            response = ChunkBatchResponseModel(
                chunks=[chunk.to_json() for chunk in chunks]
            )

            return jsonify(response.model_dump()), 200
        except NotFoundError:
            # Re-raise specific errors
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve random chunks: {str(e)}")
            raise ServiceError(
                message="Failed to retrieve random chunks", details=str(e)
            )

    @retrieval_bp.route("/random-documents", methods=["GET"])
    @validate()
    def get_random_documents(query: RandomRequest) -> Union[Dict[str, Any], tuple[Response, int]]:  # type: ignore
        """Endpoint that returns random documents for benchmark generation.

        Args:
            query: Validated query parameters

        Returns:
            Response with random documents
        """
        count: int = query.count

        try:
            documents: List[ChunkedDocument] = retrieval_service.get_random_documents(
                count
            )
            if not documents:
                raise NotFoundError(
                    message="No documents available",
                    details="The database does not contain any documents",
                )

            response = DocumentBatchResponseModel(
                documents=[doc.to_json() for doc in documents]
            )

            return jsonify(response.model_dump()), 200
        except NotFoundError:
            # Re-raise specific errors
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve random documents: {str(e)}")
            raise ServiceError(
                message="Failed to retrieve random documents", details=str(e)
            )

    @retrieval_bp.route("/chunks-batch", methods=["POST"])
    @validate()
    def get_chunks_batch(body: ChunkBatchRequest) -> Union[Dict[str, Any], tuple[Response, int]]:  # type: ignore
        """Endpoint that returns multiple chunks by their IDs.

        Args:
            body: Validated request body

        Returns:
            Response with requested chunks
        """
        chunk_ids: List[str] = body.chunk_ids

        try:
            chunks: List[DocumentChunk] = retrieval_service.get_chunks_by_uuids(
                chunk_ids
            )
            response = ChunkBatchResponseModel(
                chunks=[chunk.to_json() for chunk in chunks]
            )

            return jsonify(response.model_dump()), 200
        except KeyError as e:
            logger.error(f"Chunk not found: {str(e)}")
            raise NotFoundError(
                message="One or more chunks not found",
                details=f"Cannot find chunk with ID: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Failed to retrieve chunks batch: {str(e)}")
            raise ServiceError(
                message="Failed to retrieve chunks batch", details=str(e)
            )

    @retrieval_bp.route("/documents-batch", methods=["POST"])
    @validate()
    def get_documents_batch(body: DocumentBatchRequest) -> Union[Dict[str, Any], tuple[Response, int]]:  # type: ignore
        """Endpoint that returns multiple documents by their UUIDs.

        Args:
            body: Validated request body

        Returns:
            Response with requested documents
        """
        document_ids: List[str] = body.document_ids

        try:
            documents: List[ChunkedDocument] = retrieval_service.get_documents_by_uuids(
                document_ids
            )
            response = DocumentBatchResponseModel(
                documents=[doc.to_json() for doc in documents]
            )

            return jsonify(response.model_dump()), 200
        except KeyError as e:
            logger.error(f"Document not found: {str(e)}")
            raise NotFoundError(
                message="One or more documents not found",
                details=f"Cannot find document with UUID: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Failed to retrieve documents batch: {str(e)}")
            raise ServiceError(
                message="Failed to retrieve documents batch", details=str(e)
            )

    return retrieval_bp
