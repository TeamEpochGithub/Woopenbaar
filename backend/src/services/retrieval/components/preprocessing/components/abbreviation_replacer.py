"""Abbreviation handling for text preprocessing.

This module provides functionality for expanding abbreviations and abbreviating full forms
within text data using a dictionary approach. It ensures consistent notation format for
both abbreviations and their full forms.
"""

import abc
import json
import logging
import re
from typing import Match

logger = logging.getLogger(__name__)


class AbbreviationProcessorBase(abc.ABC):
    """Base abstract class for abbreviation processors.

    All implementations should provide methods to expand abbreviations,
    abbreviate full forms, and process text with both operations.
    """

    @abc.abstractmethod
    def expand_abbreviations(self, text: str) -> str:
        """Replace abbreviations with expanded forms.

        Args:
            text: Input text to process

        Returns:
            Text with expanded abbreviations
        """

    @abc.abstractmethod
    def abbreviate_full_forms(self, text: str) -> str:
        """Replace full forms with abbreviated forms.

        Args:
            text: Input text to process

        Returns:
            Text with abbreviated full forms
        """

    @abc.abstractmethod
    def process_text(self, text: str, mode: str = "both") -> str:
        """Process text using specified mode.

        Args:
            text: Input text to process
            mode: Processing mode ("expand", "abbreviate", or "both")

        Returns:
            Processed text
        """


class AbbreviationProcessor(AbbreviationProcessorBase):
    """Handles replacement of abbreviations with their expanded forms in text.

    Loads abbreviations from a JSON file and replaces occurrences in text while
    preserving the original abbreviation in parentheses. Uses regex pattern matching
    for efficient replacement.

    Attributes:
        abbreviation_map: Dictionary mapping abbreviations to their full forms
        pattern: Compiled regex pattern for matching abbreviations
    """

    def __init__(self, abbreviation_file: str):
        """Initialize the abbreviation replacer.

        Args:
            abbreviation_file: Optional path to JSON file containing abbreviation mappings.
                             If provided, abbreviations are loaded during initialization.
        """
        self.abbreviation_map = self.load_abbreviations(abbreviation_file)
        self.reverse_map = self.load_abbreviations(abbreviation_file)
        # Regex pattern to match standalone words consisting of letters and numbers only
        self.abbreviation_pattern = re.compile(r"\b([A-Za-z0-9]+)\b")

    def load_abbreviations(self, abbreviation_file: str) -> dict[str, str]:
        """Load abbreviations from a JSON file and compile regex patterns.

        Args:
            abbreviation_file: Path to JSON file containing abbreviation mappings
        """
        try:
            with open(abbreviation_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading abbreviations from {abbreviation_file}: {e}")
            return {}

    def expand_abbreviations(self, text: str) -> str:
        """Replace abbreviations with their full forms."""
        if not text:
            return text

        try:

            def replacement(match: Match[str]) -> str:
                abbr = match.group(0)
                full_form = self.reverse_map.get(abbr)
                return f"{abbr} ({full_form})" if full_form else abbr

            return self.abbreviation_pattern.sub(replacement, text)
        except Exception as e:
            logger.error(f"Error expanding abbreviations: {e}")
            return text

    def abbreviate_full_forms(self, text: str) -> str:
        """Replace full forms with 'abbr (full form)' format."""
        if not text or not self.abbreviation_pattern:
            return text

        try:

            def replacement(match: Match[str]) -> str:
                full_form = match.group(0)
                abbr = self.reverse_map.get(full_form)
                # Use 'abbr (full form)' format consistently
                return f"{abbr} ({full_form})" if abbr else full_form

            return self.abbreviation_pattern.sub(replacement, text)
        except Exception as e:
            logger.error(f"Error abbreviating full forms: {e}")
            return text

    def full_form_to_abbreviation(self, text: str) -> str:
        """Replaces full forms with abbreviations if the abbreviation isn't already present."""
        if not text:
            return text

        try:
            # Check if the full form is already wrapped with abbreviation
            def replacement(match: Match[str]) -> str:
                full_form = match.group(0)
                abbr = self.reverse_map.get(full_form)
                # Ensure we don't prepend abbreviation if it's already there
                if abbr and f"{abbr} ({full_form})" not in text:
                    return f"{abbr} ({full_form})"
                return full_form

            return self.abbreviation_pattern.sub(replacement, text)

        except Exception as e:
            logger.error(f"Error adding abbreviation before full form: {e}")
            return text

    def process_text(self, text: str, mode: str = "both") -> str:
        """Process text using the specified mode.

        Args:
            text: Input text
            mode: "expand", "abbreviate", or "both"
        """
        if not text:
            return text

        if mode == "expand":
            return self.expand_abbreviations(text)
        elif mode == "abbreviate":
            return self.abbreviate_full_forms(text)
        else:
            expanded = self.expand_abbreviations(text)
            return self.abbreviate_full_forms(expanded)
