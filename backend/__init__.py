"""
Backend package for the RAG-powered chat application.

This package contains the core backend service components including:
- Flask application and API routes
- Document retrieval and chunking services
- LLM integration and chat functionality
- Data models and storage services
- Configuration and utility modules
"""

import logging
import os

# Configure logging with clickable paths before anything else imports logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(pathname)s:%(lineno)d %(message)s"
)


class ClickablePathFilter(logging.Filter):
    """Filter to make file paths clickable in the terminal."""

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "pathname"):
            # Convert absolute path to relative path from workspace root
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            try:
                record.pathname = os.path.relpath(record.pathname, workspace_root)
            except ValueError:
                # If path is not under workspace root, keep it as is
                pass
        return True


# Apply filter to root logger
logging.getLogger().addFilter(ClickablePathFilter())
