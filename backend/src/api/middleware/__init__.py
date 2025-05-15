"""Middleware package for API request processing.

This module registers middleware functions for the API.
"""

from flask import Flask


def register_middleware(app: Flask) -> None:
    """Register middleware with the Flask application.

    Args:
        app: Flask application
    """
    # Register error handler middleware
    from backend.src.api.middleware.error_handler import register_error_handlers

    register_error_handlers(app)
