"""Safety analysis service for text content validation.

This module provides functionality for detecting potentially unsafe, irrelevant, or
sensitive content in text data to ensure appropriate filtering of user inputs.
"""

from typing import Any, Dict, List, Optional

from backend.src.services.safety.sentence_analyzer import SentenceAnalyzer


class SafetyService:
    """Main service for text safety analysis.

    Provides high-level methods for checking if text content is safe and relevant.
    """

    def __init__(self, dangerous_keywords: Optional[List[str]] = None) -> None:
        """Initialize the SafetyService.

        Args:
            dangerous_keywords: Optional list of keywords to check for
        """
        self.analyzer = SentenceAnalyzer(dangerous_keywords=dangerous_keywords)

    def check_text_safety(self, text: str) -> Dict[str, Any]:
        """Analyze a text for safety and relevance concerns.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with analysis results including:
            - text: Original text
            - dangerous_keywords: List of found dangerous keywords
            - has_dangerous_keywords: Boolean indicating if dangerous keywords were found
            - sensitivity_level: Numeric sensitivity level (0-3)
            - sensitivity_description: Text description of sensitivity level
            - Relevant: Boolean indicating if text is relevant
            - flagged: Boolean indicating if text is flagged (dangerous, sensitive, or irrelevant)
        """
        return self.analyzer.analyze_sentence(text)

    def is_text_safe(self, text: str) -> bool:
        """Check if text is safe and relevant.

        Args:
            text: The text to check

        Returns:
            Boolean indicating if text is safe and relevant (not flagged)
        """
        result: Dict[str, Any] = self.analyzer.analyze_sentence(text)
        return not result["flagged"]
