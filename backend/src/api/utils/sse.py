"""Server-Sent Events (SSE) utilities.

This module contains functions for creating and working with SSE responses.
"""

import json
import logging
from typing import Any, Callable, Generator

from flask import Response, stream_with_context

logger = logging.getLogger(__name__)


def create_sse_event(event_type: str, data: Any) -> str:
    """Create a properly formatted SSE event string.

    Args:
        event_type: Type of the event
        data: Data to be JSON serialized

    Returns:
        Formatted SSE event string
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def create_sse_response(
    generator_func: Callable[[], Generator[str, None, None]],
) -> Response:
    """Create a Server-Sent Events (SSE) response.

    Args:
        generator_func: Generator function that yields SSE data

    Returns:
        Flask Response configured for SSE
    """
    return Response(
        stream_with_context(generator_func()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
