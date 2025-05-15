"""Safety analysis package for text content validation.

This package provides tools for assessing text safety, relevance, and sensitivity to
ensure that user inputs and system outputs meet appropriate content guidelines.
"""

from .safety_service import SafetyService

__all__ = ["SafetyService"]
