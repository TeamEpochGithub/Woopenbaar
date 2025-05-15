"""Error handling middleware for API requests.

This module provides error handling for API requests.
"""

import logging
import traceback
from typing import Tuple

from flask import Flask, Response, current_app, jsonify
from pydantic import ValidationError as PydanticValidationError

from backend.src.api.middleware.exceptions import APIError, ErrorResponseModel

logger = logging.getLogger(__name__)


def register_error_handlers(app: Flask) -> None:
    """Register error handlers with the Flask application.

    Args:
        app: Flask application
    """

    @app.errorhandler(PydanticValidationError)
    def handle_pydantic_validation_error(error: PydanticValidationError) -> Tuple[Response, int]:  # type: ignore
        """Handle Pydantic validation errors.

        Args:
            error: Validation error from Pydantic

        Returns:
            JSON response with error details
        """
        logger.warning(f"Validation error: {error}")

        # Convert Pydantic errors to a string representation for consistency
        error_details = "\n".join([str(e) for e in error.errors()])

        response = ErrorResponseModel(
            error="Validation error", details=error_details, status_code=400
        )
        return jsonify(response.model_dump()), 400

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> Tuple[Response, int]:  # type: ignore
        """Handle custom API errors.

        Args:
            error: Custom API error

        Returns:
            JSON response with error details
        """
        logger.error(f"API error ({error.__class__.__name__}): {error.message}")
        if hasattr(error, "details") and error.details:
            logger.error(f"Error details: {error.details}")

        return error.to_response()

    @app.errorhandler(Exception)
    def handle_exception(error: Exception) -> Tuple[Response, int]:  # type: ignore
        """Handle uncaught exceptions.

        Args:
            error: Exception that was raised

        Returns:
            JSON response with error message
        """
        logger.error(f"Unhandled exception: {str(error)}")
        logger.error(traceback.format_exc())

        # Only include detailed error info in debug mode
        details = str(error) if current_app.debug else None

        response = ErrorResponseModel(
            error="Internal server error", details=details, status_code=500
        )
        return jsonify(response.model_dump()), 500
