from dataclasses import dataclass, field
from typing import Any, Dict, List

from standard_data_format.utils.logger import setup_logger

# Remove lines 5-6 (logging configuration)
logger = setup_logger()


@dataclass
class Document:
    """
    Represents a document in the standard data format.

    This class serves as a standardized container for document data, including
    metadata and content. It provides methods for serialization to JSON and
    creation from metadata.

    Attributes:
        uuid (str): Unique identifier for the document
        vws_id (str): External identifier from the VWS system
        create_date (str): Document creation date in ISO format (YYYY-MM-DD)
        type (str): Document type (e.g., 'Email', 'PDF', etc.)
        link (str): URL or reference to the original document
        attachment_links (List[str]): List of URLs to document attachments
        content (str): The document's content in markdown format
    """

    uuid: str
    vws_id: str
    create_date: str
    type: str
    link: str
    attachment_links: List[str] = field(default_factory=list)
    content: str = ""

    def to_json(self) -> Dict[str, Any]:
        """
        Convert document to JSON-serializable dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the document suitable for JSON serialization.
        """
        logger.info(f"Converting document {self.vws_id} to JSON format")
        return {
            "uuid": self.uuid,
            "vws_id": self.vws_id,
            "create_date": self.create_date,
            "type": self.type,
            "link": self.link,
            "attachment_links": self.attachment_links,
            "content": self.content,
        }

    @classmethod
    def from_metadata(
        cls, metadata_json: Dict[str, Any], uuid: str, content: str
    ) -> "Document":
        """
        Factory method to create a Document from metadata and content.

        Args:
            metadata_json (Dict[str, Any]): Dictionary containing document metadata
            uuid (str): Unique identifier for the document
            content (str): The document's content in markdown format

        Returns:
            Document: A new Document instance populated with the provided data
        """
        logger.debug(f"Creating document from metadata for ID {metadata_json['id']}")
        return cls(
            uuid=uuid,
            vws_id=metadata_json["id"],
            create_date=metadata_json.get("datum", ""),
            type=metadata_json.get("type", ""),
            link=metadata_json.get("link", ""),
            attachment_links=metadata_json.get("attachment_links", []),
            content=content,
        )
