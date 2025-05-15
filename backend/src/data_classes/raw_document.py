"""Data class representing a raw source document before any processing or chunking."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID


class RawDocument:
    """Represents a source document in standard data format.

    Attributes:
        uuid: Internal unique identifier
        vws_id: External identifier for the document
        create_date: Document creation timestamp, can be None if date is invalid or missing
        type: Document type (email, whatsapp, pdf, etc. )
        link: URL or reference to original document
        attachment_links: URLs to all attached documents
        content: Full text content
    """

    uuid: UUID
    vws_id: str
    create_date: Optional[datetime]
    type: str
    link: str
    attachment_links: list[str]
    content: str

    def __init__(
        self,
        uuid: UUID,
        vws_id: str,
        create_date: Optional[datetime],
        content: str,
        link: str,
        attachment_links: list[str],
        type: str,
    ):
        """Initialize a RawDocument.

        Args:
            uuid: Internal unique identifier
            vws_id: External identifier for the document
            create_date: Document creation timestamp, can be None if date is invalid or missing
            content: Full text content
            link: URL or reference to original document
            attachment_links: URLs to all attached documents
            type: Document type (email, whatsapp, pdf, etc.)
        """
        self.uuid = uuid
        self.vws_id = vws_id
        self.create_date = create_date
        self.type = type
        self.link = link
        self.attachment_links = attachment_links
        self.content = content

    def to_json(self) -> Dict[str, Any]:
        """Convert the RawDocument to a JSON-serializable dictionary.

        Returns:
            Dictionary containing all document attributes in a JSON-serializable format
        """
        return {
            "uuid": str(self.uuid),
            "vws_id": self.vws_id,
            "create_date": self.create_date.isoformat() if self.create_date else None,
            "type": self.type,
            "link": self.link,
            "attachment_links": self.attachment_links,
            "content": self.content,
        }

    def __str__(self) -> str:
        """Return a compact string representation of the document.

        Returns:
            String containing all document attributes in a single line
        """
        return f"RawDocument(uuid={self.uuid}, vws_id={self.vws_id}, create_date={self.create_date}, type={self.type}, link={self.link}, attachment_links={self.attachment_links}, content={self.content})"

    def __repr__(self) -> str:
        """Return a concise string representation for debugging.

        Returns:
            String containing the most important document attributes
        """
        return f"RawDocument(uuid={self.uuid}, vws_id={self.vws_id})"

    def __hash__(self) -> int:
        """Get a hash value for the document based on its UUID.

        Returns:
            Hash value derived from the document's UUID
        """
        return hash(self.uuid)

    def __eq__(self, other: object) -> bool:
        """Check if this document equals another object.

        Args:
            other: Object to compare with

        Returns:
            True if other is a RawDocument with the same UUID, False otherwise
        """
        if not isinstance(other, RawDocument):
            return False
        return self.uuid == other.uuid

    def to_str_all(self) -> str:
        """Return a full string representation of the document.

        Returns:
            Multi-line string containing all document attributes in a readable format,
            with content truncated to first 100 characters
        """
        return (
            f"RawDocument(\n"
            f"  uuid={self.uuid},\n"
            f"  vws_id={self.vws_id},\n"
            f"  create_date={self.create_date},\n"
            f"  type={self.type},\n"
            f"  link={self.link},\n"
            f"  attachment_links={self.attachment_links},\n"
            f"  content_preview={self.content[:100]}...\n"
            f")"
        )
