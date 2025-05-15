"""Metadata extraction module.

This module provides functionality to extract metadata like dates, subjects, and other
structured information from document content. It supports both document-level and
chunk-level extraction to enhance document searchability and organization.

The Extractor class uses regex patterns and date parsing to identify and extract
meaningful metadata while handling edge cases and data validation.
"""

import abc
import logging
import re
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from backend.src.data_classes.chunked_document import ChunkedDocument

logger = logging.getLogger(__name__)


class ExtractorBase(abc.ABC):
    """Base abstract class for metadata extractors.

    All extractor implementations should provide methods to extract metadata
    from document content and enrich document objects with that metadata.
    """

    @abc.abstractmethod
    def extract_dates(self, text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract the first and last dates mentioned in text.

        Args:
            text: Document or chunk text to process

        Returns:
            Tuple containing first and last chronological dates found
        """

    @abc.abstractmethod
    def extract_subject(self, text: str) -> Optional[str]:
        """Extract subject/topic information from text.


        Args:
            text: Document or chunk text to process

        Returns:
            Extracted subject string if found, None otherwise
        """

    @abc.abstractmethod
    def extract_email_headers(
        self, text: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract email_headers information from text and removes duplicates.

        Args:
            text: Document or chunk text to process

        Returns:
            Extracted subject string if found, None otherwise
        """

    @abc.abstractmethod
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract all available metadata from text.

        Args:
            text: Document or chunk text to process

        Returns:
            Dictionary containing extracted metadata
        """

    @abc.abstractmethod
    def enrich_chunked_document(self, document: ChunkedDocument) -> ChunkedDocument:
        """Enrich a chunked document with extracted metadata.

        Args:
            document: ChunkedDocument to process

        Returns:
            The same document instance with enriched metadata fields
        """


class Extractor(ExtractorBase):
    """Extract metadata like dates, subjects, and other information from document text.

    This class provides methods to extract various types of metadata from document content,
    supporting the final stage of document preprocessing to enhance document searchability.
    It handles both document-level and chunk-level extraction using configurable patterns.

    Attributes:
        _date_pattern: Compiled regex for matching date strings in various formats
        _date_formats: List of datetime format strings for parsing dates
        _subject_pattern: Compiled regex for matching subject line patterns
    """

    def __init__(self):
        """Initialize regex patterns for metadata extraction.

        Compiles regex patterns once during initialization for efficient reuse across
        multiple documents. Patterns handle common date formats and subject line markers.
        """
        # Regex pattern for matching email addresses - matches the domain part of an email after @
        self._email_pattern = re.compile(r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

        # Match dates in formats like DD-MM-YYYY, YYYY-MM-DD or DD MM YY with various separators
        # Pattern for day-month-year format with various separators, allowing single digits
        self._dd_mm_yyyy_pattern = r"\b(\d{1,2})[-/\. ](\d{1,2})[-/\. ](\d{4})\b"
        # Pattern for year-month-day format
        self._yyyy_mm_dd_pattern = r"\b(\d{4})[-/\. ](\d{1,2})[-/\. ](\d{1,2})\b"
        # Pattern for month-year format without day
        self._mm_yyyy_pattern = r"\b(\d{1,2})[-/\. ](\d{4})\b"
        # Pattern for year-month format without day
        self._yyyy_mm_pattern = r"\b(\d{4})[-/\. ](\d{1,2})\b"

        # Common date format strings for parsing extracted date strings
        self._date_formats = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%d-%m-%y",
            "%d/%m/%y",
            "%d.%m.%y",
        ]
        # Define Dutch months and days (used for date extraction)
        self.dutch_months = {
            "januari": "01",
            "februari": "02",
            "maart": "03",
            "april": "04",
            "mei": "05",
            "juni": "06",
            "juli": "07",
            "augustus": "08",
            "september": "09",
            "oktober": "10",
            "november": "11",
            "december": "12",
        }

        # Match subject lines with various markers in Dutch and English
        # Pattern to match "onderwerp:", "betreft:", or "subject:" followed by text up to 100 chars long
        self._subject_pattern = re.compile(
            r"(?:onderwerp|betreft|subject)[:;\s]+([^\n\.]{3,100})", re.IGNORECASE
        )
        # Pattern to match Dutch date format with written month names: "1 januari 2023"
        self.month_replacement = r"\b(\d{1,2})\s+(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+(\d{4})\b"

    def replace_match(self, match: re.Match[str]) -> str:
        """Replace a match with a formatted date string.

        Args:
            match: The regex match object

        Returns:
            Formatted date string
        """
        day = match.group(1)
        month = self.dutch_months.get(match.group(2).lower(), "01")
        year = match.group(3)
        return f"{day}-{month}-{year}"

    def extract_dates(self, text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract the first and last dates mentioned in text.

        Finds all date strings matching the date pattern and attempts to parse them
        using multiple format strings. Filters out implausible dates and returns the
        chronologically first and last valid dates found.

        Args:
            text: Document or chunk text to process

        Returns:
            Tuple containing:
                - First chronological date found (or None if no valid dates)
                - Last chronological date found (or None if no valid dates)
        """
        if not text:
            return None, None

        dates: List[datetime] = []

        # text being processed for dates
        text = re.sub(self.month_replacement, self.replace_match, text)
        # Step 1: Try to extract DD-MM-YYYY or DD/MM/YYYY first
        date_matches = re.findall(self._dd_mm_yyyy_pattern, text)
        for match in date_matches:
            day, month, year = match
            try:
                date_str = f"{day}-{month}-{year}"
                date = datetime.strptime(date_str, "%d-%m-%Y")
                if 1950 <= date.year <= datetime.now().year + 1:
                    dates.append(date)
            except ValueError:
                continue

        date_matches = re.findall(self._yyyy_mm_dd_pattern, text)
        for match in date_matches:
            year, month, day = match
            try:
                date_str = f"{year}-{month}-{day}"
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if 1950 <= date.year <= datetime.now().year + 1:
                    dates.append(date)
            except ValueError:
                continue

        # Step 2: If no complete date found, look for MM-YYYY or YYYY-MM
        if not dates:
            # Handle MM-YYYY (e.g., 12-2021) or YYYY-MM (e.g., 2021-12)
            date_matches = re.findall(self._mm_yyyy_pattern, text)
            for match in date_matches:
                month, year = match
                try:
                    date_str = f"01-{month}-{year}"
                    date = datetime.strptime(date_str, "%d-%m-%Y")
                    if 1950 <= date.year <= datetime.now().year + 1:
                        dates.append(date)
                except ValueError:
                    continue

            # Handle YYYY-MM format explicitly
            date_matches = re.findall(self._yyyy_mm_pattern, text)
            for match in date_matches:
                year, month = match
                try:
                    date_str = f"01-{month}-{year}"
                    date = datetime.strptime(date_str, "%d-%m-%Y")
                    if 1950 <= date.year <= datetime.now().year + 1:
                        dates.append(date)
                except ValueError:
                    continue

        # Step 4: Find the youngest and oldest dates by sorting
        if dates:
            # Sort the dates to get the youngest (latest) and oldest (earliest)
            dates.sort()
            last_date = dates[0]  # The first date after sorting is the oldest
            first_date = dates[-1]  # The last date after sorting is the youngest
            return last_date, first_date

        return None, None

    def extract_subject(self, text: str) -> Optional[str]:
        """Extract the subject from the text between 'subject' or 'onderwerp' and the first newline."""

        match = re.search(self._subject_pattern, text)
        if match:
            # Extract subject text and clean up any trailing spaces/newlines
            subject = match.group(1).strip()
            return subject.split("\n")[
                0
            ]  # Take only the first line in case of multiple

        return None  # Return None if no subject is found

    def extract_email_headers(
        self, text: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract 'from', 'to', 'cc' email addresses from the given text."""
        email_from, email_to, email_cc = None, None, None

        # Look for 'To' header and extract emails
        if ("to" in text.lower() or "aan" in text.lower()) and "cc" in text.lower():
            try:
                # Matches text between "to/aan" and "cc" to extract recipients
                to_match = re.search(r"(to|aan).*?(cc)", text.lower(), re.DOTALL)
                if to_match:
                    to_section = to_match.group(0)
                    to_emails = re.findall(self._email_pattern, to_section)
                    email_to = list(set(to_emails))  # Remove duplicates
            except Exception as e:
                logger.error(f"Error extracting 'to' emails: {e}")

        # Look for 'From' header and extract emails
        if "from" in text.lower() or "van" in text.lower():
            try:
                # Matches text between "from/van" and the next header or end of text
                from_match = re.search(
                    r"(from|van).*?(to|aan|cc|$)", text.lower(), re.DOTALL
                )
                if from_match:
                    from_section = from_match.group(0)
                    from_emails = re.findall(self._email_pattern, from_section)
                    email_from = list(set(from_emails))  # Remove duplicates
            except Exception as e:
                logger.error(f"Error extracting 'from' emails: {e}")

        # Look for 'CC' header and extract emails
        if "cc" in text.lower():
            try:
                # Matches text between "cc" and the next header or end of text
                cc_match = re.search(
                    r"(cc).*?(to|aan|from|van|$)", text.lower(), re.DOTALL
                )
                if cc_match:
                    cc_section = cc_match.group(0)
                    cc_emails = re.findall(self._email_pattern, cc_section)
                    email_cc = list(set(cc_emails))  # Remove duplicates
            except Exception as e:
                logger.error(f"Error extracting 'cc' emails: {e}")

        # Convert lists to strings or None
        email_from_str = ", ".join(email_from) if email_from else None
        email_to_str = ", ".join(email_to) if email_to else None
        email_cc_str = ", ".join(email_cc) if email_cc else None

        return email_from_str, email_to_str, email_cc_str

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract all available metadata from text.

        Args:
            text: Document or chunk text to process

        Returns:
            Dictionary containing extracted metadata
        """
        metadata: Dict[str, Any] = OrderedDict()

        # Extract dates
        first_date, last_date = self.extract_dates(text)
        if first_date:
            metadata["first_mentioned_date"] = first_date
        if last_date:
            metadata["last_mentioned_date"] = last_date

        # Extract subject
        subject = self.extract_subject(text)
        if subject:
            metadata["subject"] = subject

        # Extract email headers
        email_from, email_to, email_cc = self.extract_email_headers(text)
        if email_from:
            metadata["email_from"] = email_from
        if email_to:
            metadata["email_to"] = email_to
        if email_cc:
            metadata["email_cc"] = email_cc

        return metadata

    def enrich_chunked_document(self, document: ChunkedDocument) -> ChunkedDocument:
        """Enrich a chunked document with extracted metadata.

        Args:
            document: ChunkedDocument to process

        Returns:
            The same document instance with enriched metadata fields
        """
        # Extract metadata from document content
        metadata = self.extract_metadata(document.content)

        # Update document with extracted metadata
        if "first_mentioned_date" in metadata:
            document.first_mentioned_date = metadata["first_mentioned_date"]
        if "last_mentioned_date" in metadata:
            document.last_mentioned_date = metadata["last_mentioned_date"]
        if "subject" in metadata:
            document.subject = metadata["subject"]
        if "email_from" in metadata:
            document.email_from = metadata["email_from"]
        if "email_to" in metadata:
            document.email_to = metadata["email_to"]
        if "email_cc" in metadata:
            document.email_cc = metadata["email_cc"]

        return document
