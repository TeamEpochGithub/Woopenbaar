"""Custom exception types for the API.

This module defines custom exception classes for various error scenarios and error response models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Response, jsonify
from pydantic import BaseModel, Field


class ErrorResponseModel(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        None, description="Additional error details"
    )
    status_code: int = Field(400, description="HTTP status code")


class APIError(Exception):
    """Base class for all API errors."""

    status_code = 500
    default_message = "An unexpected error occurred"

    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ):
        """Initialize the API error.

        Args:
            message: Custom error message (uses default_message if None)
            details: Additional error details
        """
        self.message = message or self.default_message
        self.details = details
        super().__init__(self.message)

    def to_response(self) -> Tuple[Response, int]:
        """Convert to Flask response."""
        error_model = ErrorResponseModel(
            error=self.message, details=self.details, status_code=self.status_code
        )
        return jsonify(error_model.model_dump()), self.status_code


class ValidationError(APIError):
    """Error for invalid request data."""

    status_code = 400
    default_message = "Invalid request data"


class NotFoundError(APIError):
    """Error for resource not found."""

    status_code = 404
    default_message = "Resource not found"


class ServiceError(APIError):
    """Error from underlying services."""

    status_code = 500
    default_message = "Service error"
