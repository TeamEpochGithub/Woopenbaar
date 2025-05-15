"""Document parsing module.

This module provides functionality to parse document files into structured RawDocument objects.
It handles reading JSON files, parsing metadata fields like dates and attachments, and
validating document data. The DocumentParser class supports both single file and directory
parsing with error handling and logging.
"""

import abc
import json
import logging
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from dateutil.parser import parse
from tqdm import tqdm

from backend.src.data_classes import RawDocument

logger = logging.getLogger(__name__)


class DocumentParserBase(abc.ABC):
    """Base abstract class for document parsers.

    All parser implementations should provide methods to parse files and directories
    into structured RawDocument objects.
    """

    @abc.abstractmethod
    def parse_file(self, file_path: str) -> Optional[RawDocument]:
        """Parse a single file into a RawDocument.

        Args:
            file_path: Path to the file to parse

        Returns:
            RawDocument object if parsing succeeds, None otherwise
        """

    @abc.abstractmethod
    def parse_directory(self, documents_path: Path) -> List[RawDocument]:
        """Parse all files in a directory into RawDocument objects.

        Args:
            documents_path: Path to directory containing document files

        Returns:
            List of successfully parsed RawDocument objects
        """


class DocumentParser(DocumentParserBase):
    """Parser for loading documents from files and converting them to RawDocument objects.

    This class provides methods to parse both individual JSON files and directories of files
    into RawDocument objects. It handles validation, error checking, and logging while
    parsing document metadata and content.

    The parser expects JSON files with fields matching the RawDocument class structure
    and handles edge cases like date parsing and attachment link formatting.
    """

    def parse_file(self, file_path: str) -> Optional[RawDocument]:
        """Parse a single JSON file into a RawDocument.

        Reads a JSON file and attempts to parse it into a RawDocument object. Handles
        validation and parsing of date strings, attachment links, and other metadata.
        Logs errors if parsing fails.

        Args:
            file_path: Path to the JSON file to parse

        Returns:
            RawDocument object if parsing succeeds, None if any error occurs

        Note:
            Expected JSON format:
            {
                "uuid": str,
                "vws_id": str,
                "create_date": str,
                "content": str,
                "link": str,
                "attachment_links": list[str] or str,
                "type": str
            }
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            date_str = doc_data["create_date"]

            try:
                create_date = (
                    parse(date_str, dayfirst=False, fuzzy=True)
                    if date_str and date_str.lower() != "nan"
                    else None
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse date in file {file_path}. Got: '{date_str}'. Error: {e}. Setting to None."
                )
                create_date = None

            # Handle attachment links that may be string representations of lists
            attachment_links: List[str] = doc_data.get("attachment_links", [])
            if isinstance(attachment_links, str):
                try:
                    attachment_links = eval(attachment_links)
                except Exception:
                    attachment_links = []

            # Create RawDocument with parsed and validated fields
            doc = RawDocument(
                uuid=UUID(doc_data["uuid"]),
                vws_id=doc_data["vws_id"],
                create_date=create_date,
                content=doc_data["content"],
                link=doc_data["link"],
                attachment_links=attachment_links,
                type=doc_data.get("type", None),
            )

            return doc

        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            return None

    def parse_directory(self, documents_path: Path) -> List[RawDocument]:
        """Parse all JSON files in a directory into RawDocument objects.

        Args:
            documents_path: Path to directory containing JSON document files

        Returns:
            Dictionary mapping document UUIDs to their corresponding RawDocument objects.
            Only successfully parsed documents are included.
        """
        documents: List[RawDocument] = []

        doc_paths = list(documents_path.glob("*.json"))

        for file_path in tqdm(doc_paths, desc="Parsing", total=len(doc_paths)):
            doc = self.parse_file(str(file_path))

            if doc is not None:
                documents.append(doc)

        logger.info(f"Parsed {len(documents)} documents from {documents_path}")
        return documents
