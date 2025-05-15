"""Content cleaning and normalization module.

This module provides functionality to clean and normalize document content by removing
unwanted characters, standardizing formatting, and applying document type-specific
cleaning rules. The ContentCleaner class handles various text cleaning operations
while preserving meaningful content.
"""

import abc
import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class EmptyContentError(Exception):
    """Raised when document content is empty or becomes empty after cleaning."""


class ContentCleanerBase(abc.ABC):
    """Base abstract class for content cleaners.

    All content cleaner implementations should provide methods to clean
    and normalize document content based on document type.
    """

    @abc.abstractmethod
    def clean_content(self, content: str, doc_type: str) -> str:
        """Clean and normalize document content.

        Args:
            content: Raw document content to clean
            doc_type: Type of document (e.g. 'Email', 'PDF')

        Returns:
            Cleaned content

        Raises:
            EmptyContentError: If content is empty or becomes empty after cleaning
        """


class ContentCleaner(ContentCleanerBase):
    """Cleans and normalizes document content based on document type in a specific order.

    The following operations are performed in this exact order:
    1.  Ground number removal
    2.  Noise removal
    # Note for self, what should be the order here? Does it really makes a difference as to what order to choose?
    4.  Removal of large upper case content
    5. "Remove of "dubbel dubbel" from content" will replace Removal of long noise (THis doesn't feel like a correct function, perhaps it is better to leave it)
    6.  Removal of newlines
    7. Remove empty brackets
    8.  Punctuation removal and correction
    9.  - or | or ... removals
    10. Double space removal
    11. Small chunks removal
    """

    # all the strings we will directly remove if found per document type
    _anonymization_ids: List[str] = [
        "5.1 .2e",
        "5.12e",
        "5.12i",
        "5.12",
        "(10)(2e)",
        "1.2",
        "1.4",
    ]  # ID's used to replace anonymous data (we can remove it)
    # --- Noise dictionaries ---
    _noise_variants: Set[str] = {
        "\nubbel",
        "UBBEL",
        "Uubbel",
        "EAED",
    }

    # Ground variations in a set, list would make it deterministid
    _ground_variants: List[str] = [
        "5.1 .2e",
        "(10)(26)",
        "(10)(20) (70)",
        "5.1.5",
        "512e",
        "518e",
        "s12e512e",
        "6.1 .2e",
        "6.120",
        "51.2e",
        "10(20)",
        "(10)(20)",
        "0.2 .e",
        "51.ee",
        "s12e",
        "((0)(2))",
        "(0)20",
        "11)(1)",
        "5512e",
        "5.1 .2a1",
        "5.1 .29",
        "5.1 .5",
        "3.1 .2e",
        "6 1 0",
        "5 1 5",
        "10. 2. E",
        "(10)(2e)",
        "10)(2e10)",
        "10.2 .e",
        "5.1 2e",
        "(s. 1.)",
        "5.1 .2c",
        "5.1.2i",
        "5.12e",
        # Smaller variants
        "5. 1.",
        "5.12",
        "5.1.",
        "5.1",
        "(10)",
        "(11)(1)",
        "(20)",
        "2e (20)",
        "(2d)02",
        "(60)",
        "(26)",
        "(26 10)",
        "(10 )",
        "(2e)",
        "(2c)",
        "(29)",
        "2e",
        "10x",
    ]
    # Email headers to watch out for
    _email_headers: Set[str] = {
        "nl",
        "ni",
        "be",
        "net",
        "com",
        "net",
        "gov",
        "uk",
        "us",
        "org",
        "it",
        "uk",
    }

    def __init__(self):
        """Initialize the content cleaner."""

    def clean_content(self, content: str, doc_type: str) -> str:
        """Clean and normalize document content based on document type.

        Applies a series of cleaning operations to standardize content formatting
        and remove unwanted artifacts. The specific cleaning rules vary based on
        the document type.

        Args:
            content: Raw document content to clean
            doc_type: Type of document (e.g. 'spreadsheet', 'word processing')
                     Used to apply type-specific cleaning rules

        Returns:
            str: Cleaned and normalized content

        Raises:
            EmptyContentError: If content is empty or becomes empty after cleaning
            Exception: If a fatal error occurs during cleaning
        """

        # Skip processing if content is empty
        if not content:
            raise EmptyContentError("Input content is empty")

        try:
            content = ContentCleaner._remove_ground_variation(
                content, ground_variants=self._ground_variants
            )
            content = ContentCleaner._remove_noise_variation(
                content, noise_variants=self._noise_variants
            )
            content = ContentCleaner._clean_email_headers(content)
            content = ContentCleaner._removal_of_repeated_words(content)
            content = ContentCleaner._remove_markdown_artefacts(content)
            content = ContentCleaner._remove_excessive_newlines(content)
            content = ContentCleaner._remove_empty_brackets(content)
            content = ContentCleaner._remove_single_brackets(content)
            content = ContentCleaner._clean_document_specific(content, doc_type)
            content = ContentCleaner._fix_punctuation(content, self._email_headers)
            content = ContentCleaner._clean_double_spaces(content)
            content = ContentCleaner._chunk_cleaning(content)

            if len(content.strip()) < 6:
                raise EmptyContentError(
                    f"Content is too small after cleaning (less than 6 characters): {len(content)} characters. Content: '{content[:6]}...'"
                )
            return content

        except EmptyContentError:
            raise
        except Exception as e:
            logger.error(f"Error cleaning content: {e}")
            raise

    @staticmethod
    def _clean_email_headers(content: str) -> str:
        """Improved email header and address cleaning. Note that it doesn't repair email_headers.
        This is done later in ChunkMetadataExtractor.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """

        # Clean up email domains - matches '@something.ni' and replaces with '.nl' (fixing common typo)
        content = re.sub(r"@([a-zA-Z0-9]+)\.ni\b", r"@\1.nl", content)

        # Remove symbols directly adjacent to email addresses
        # Matches any sequence of special characters directly before @ symbol
        content = re.sub(
            r"[\(\)\*\&\^\%\$\#\@\{\}\[\]\+]+(?=@[a-zA-Z0-9]+\.[a-zA-Z]{2,})",
            "",
            content,
        )

        # Remove brackets around email addresses
        # Matches email addresses surrounded by brackets: (user@example.com), [user@example.com], {user@example.com}
        content = re.sub(
            r"[\(\[\{]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s*[\)\]\}]+",
            r"\1",
            content,
        )

        # Remove direct letters before the @ sign
        # Matches any sequence of word characters (letters, numbers, underscores) immediately before @ symbol
        content = re.sub(r"([^\W\d_]+)(?=@)", "", content)

        # Remove space between @ and domain
        # Matches spaces between @ and domain name and removes them
        content = re.sub(r"@(\s+)([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", r"@\2", content)

        return content

    @staticmethod
    def _remove_ground_variation(content: str, ground_variants: List[str]) -> str:
        """Remove ground variations from content.

        Args:
            content: Content to clean
            ground_variants: List of ground variants to remove

        Returns:
            Cleaned content
        """
        for ground in ground_variants:
            escaped_pattern = re.escape(ground)
            content = re.sub(escaped_pattern, "", content)
        return content

    @staticmethod
    def _remove_noise_variation(content: str, noise_variants: Set[str]) -> str:
        """Remove noise variations from content.

        Args:
            content: Content to clean
            noise_variants: Set of noise variants to remove

        Returns:
            Cleaned content
        """
        for noise_term in noise_variants:
            escaped_pattern = re.escape(noise_term)
            content = re.sub(escaped_pattern, "", content)
        return content

    @staticmethod
    def _remove_empty_brackets(content: str) -> str:
        """Remove empty brackets from content.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        # Remove empty brackets with optional whitespace inside - matches (), [], and {}
        content = re.sub(r"\(\s*\)", "", content)
        content = re.sub(r"\[\s*\]", "", content)
        content = re.sub(r"\{\s*\}", "", content)

        return content

    @staticmethod
    def _remove_single_brackets(content: str) -> str:
        """
        Removes brackets if they contain between 4 and 9 elements inside.
        Also removes unmatched opening or closing brackets.
        The content inside remains untouched.
        """
        stack: List[Tuple[str, int, List[str]]] = []
        to_remove: Set[int] = set()
        bracket_pairs: Dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
        opening_brackets: Set[str] = set(bracket_pairs.keys())
        closing_brackets: Set[str] = set(bracket_pairs.values())

        # Track bracket positions and characters
        for i, char in enumerate(content):
            if char in opening_brackets:
                stack.append(
                    (char, i, [])
                )  # Store bracket, position, and inner elements
            elif char in closing_brackets:
                if stack and bracket_pairs[stack[-1][0]] == char:
                    _, start_idx, elements = stack.pop()
                    if 10 > len(elements) > 3:
                        to_remove.add(start_idx)  # Remove opening bracket
                        to_remove.add(i)  # Remove matching closing bracket
                else:
                    to_remove.add(i)  # Unmatched closing bracket

            # If inside a bracket, track non-space characters
            if stack and char.strip():
                stack[-1][2].append(char)

        # Add unmatched opening brackets to removal set
        to_remove.update(pos for _, pos, _ in stack)

        # Reconstruct cleaned content without removed brackets
        return "".join(char for i, char in enumerate(content) if i not in to_remove)

    @staticmethod
    def _remove_excessive_newlines(content: str) -> str:
        """
        Removes more than two newlines, collapsing any spaces in between them, and ensures at most two newlines.
        One newline indicates new information, while two newlines indicate a new paragraph.
        """
        # Remove spaces before any newline (even multiple)
        content = re.sub(r" +(?=\n)", "", content)

        # Preserve terminal newlines
        terminal_newlines = len(content) - len(content.rstrip("\n"))
        terminal_newlines = 2 if terminal_newlines > 1 else terminal_newlines

        # Strip trailing newlines to avoid issues with terminal newlines during replacements
        content = content.rstrip("\n")

        # Collapse excessive newlines and spaces to at most two newlines
        content = re.sub(r"(\n[ \t]*){2,}", "\n\n", content)

        # Reduce any sequence of three or more newlines to exactly two newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Restore terminal newlines
        content += "\n" * terminal_newlines

        return content

    @staticmethod
    def _remove_markdown_artefacts(content: str) -> str:
        """Remove markdown artefacts from content.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        # Remove markdown formatting characters and keep the formatted text
        # Matches text surrounded by ** (bold), * (italic), _ (underscore), and ` (code) and extracts just the text
        content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
        content = re.sub(r"\*(.*?)\*", r"\1", content)
        content = re.sub(r"_(.*?)_", r"\1", content)
        content = re.sub(r"`(.*?)`", r"\1", content)
        return content

    @staticmethod
    def _removal_of_repeated_words(content: str) -> str:
        """
        Only removes the words "dubbel dubbel" from documents. This appears on pages which are fully anonymized.
        """
        # Matches the phrase "dubbel dubbel" (case-insensitive) at word boundaries
        content = re.sub(r"(?i)\bdubbel\s+dubbel\b", "", content)
        #
        return content

    @staticmethod
    def _fix_punctuation(content: str, email_headers: Set[str]) -> str:
        """Fix punctuation in content.

        Args:
            content: Content to clean
            email_headers: Set of email headers to watch out for

        Returns:
            Cleaned content
        """
        # Replace 3 or more consecutive periods with "..."
        content = re.sub(r"\.{3,}", "...", content)
        # Replace 2 or more consecutive periods with a single period
        content = re.sub(r"\.{2,}", ".", content)
        # Replace 2 or more consecutive commas with a single comma
        content = re.sub(r"\,{2,}", ",", content)
        # Remove spaces before punctuation marks
        content = re.sub(r"\s+([.,;:!?])", r"\1", content)
        return content

    @staticmethod
    def _clean_document_specific(content: str, doc_type: str) -> str:
        """Clean document specific content.

        Args:
            content: Content to clean
            doc_type: Type of document

        Returns:
            Cleaned content
        """
        # Document specific cleaning for emails - removes common email headers
        if doc_type.lower() == "email":
            # Matches "From:" followed by any text until a newline
            content = re.sub(r"From:.*?\n", "", content)
            # Matches "To:" followed by any text until a newline
            content = re.sub(r"To:.*?\n", "", content)
            # Matches "Subject:" followed by any text until a newline
            content = re.sub(r"Subject:.*?\n", "", content)
            # Matches "Date:" followed by any text until a newline
            content = re.sub(r"Date:.*?\n", "", content)
        return content

    @staticmethod
    def _clean_double_spaces(content: str) -> str:
        """Remove double spaces from content.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        # Replace 2 or more consecutive whitespace characters with a single space
        content = re.sub(r"\s{2,}", " ", content)
        return content

    @staticmethod
    def _chunk_cleaning(content: str) -> str:
        """Clean small chunks from content.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        # Remove single or two-character words at word boundaries
        content = re.sub(r"\b\w{1,2}\b", "", content)
        return content
