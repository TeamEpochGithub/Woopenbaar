"""Data class representing a chunk/segment of a document with associated metadata."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, NewType, Optional

if TYPE_CHECKING:
    from backend.src.data_classes.chunked_document import ChunkedDocument

# A type alias for the 64-bit ID (required by FAISS)
ChunkID = NewType("ChunkID", int)


class DocumentChunk:
    """Represents a chunk/segment of a document.

    Attributes:
        uuid: unique identifier for the chunk
        type: Document type (email, whatsapp, pdf, etc. )
        link: URL or reference to original document
        content: text content of the chunk
        parent_document: the parent document of the chunk
        content_date: (inferred) date of the content
        subject: (inferred) subject of the chunk
        first_mentioned_date: (inferred) first date mentioned in the chunk
        last_mentioned_date: (inferred) last date mentioned in the chunk
    """

    uuid: ChunkID
    type: Optional[str]
    link: str
    content: str
    parent_document: "ChunkedDocument"

    # inferred metadata
    content_date: Optional[datetime]
    subject: Optional[str]
    first_mentioned_date: Optional[datetime]
    last_mentioned_date: Optional[datetime]
    email_from: Optional[str]
    email_to: Optional[str]
    email_cc: Optional[str]

    def __init__(
        self,
        uuid: "ChunkID",
        type: Optional[str],
        link: str,
        content: str,
        parent_document: "ChunkedDocument",
        content_date: Optional[datetime] = None,
        subject: Optional[str] = None,
        first_mentioned_date: Optional[datetime] = None,
        last_mentioned_date: Optional[datetime] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
        email_cc: Optional[str] = None,
    ):
        """Initialize a DocumentChunk.

        Args:
            uuid: Unique identifier for the chunk
            type: Document type (email, whatsapp, pdf, etc.)
            link: URL or reference to original document
            content: Text content of the chunk
            parent_document: The parent document of the chunk
            content_date: (inferred) Date of the content
            subject: (inferred) Subject of the chunk
            first_mentioned_date: (inferred) First date mentioned in the chunk
            last_mentioned_date: (inferred) Last date mentioned in the chunk
        """
        self.uuid = uuid
        self.type = type
        self.link = link
        self.content = content
        self.parent_document = parent_document
        self.content_date = content_date
        self.subject = subject
        self.first_mentioned_date = first_mentioned_date
        self.last_mentioned_date = last_mentioned_date
        self.email_from = email_from
        self.email_to = email_to
        self.email_cc = email_cc

    def to_json(self) -> Dict[str, Any]:
        """Convert the chunk to a JSON-compatible dictionary.

        Returns:
            Dictionary containing all chunk attributes in a JSON-serializable format
        """
        return {
            "uuid": str(self.uuid),
            "type": self.type,
            "link": self.link,
            "content": self.content,
            "parent_document": (
                str(self.parent_document.uuid) if self.parent_document else None
            ),
            "content_date": (
                self.content_date.isoformat() if self.content_date else None
            ),
            "subject": self.subject if self.subject else None,
            "first_mentioned_date": (
                self.first_mentioned_date.isoformat()
                if self.first_mentioned_date
                else None
            ),
            "last_mentioned_date": (
                self.last_mentioned_date.isoformat()
                if self.last_mentioned_date
                else None
            ),
            "email_from": self.email_from if self.email_from else None,
            "email_to": self.email_to if self.email_to else None,
            "email_cc": self.email_cc if self.email_cc else None,
        }

    def __str__(self) -> str:
        """Return a compact string representation of the chunk.

        Returns:
            String containing basic chunk information including UUID, type, subject and content length
        """
        return f"DocumentChunk(uuid={self.uuid}, type={self.type}, subject={self.subject}, length={len(self.content)} chars)"

    def __repr__(self) -> str:
        """Get a concise string representation for debugging.

        Returns:
            String containing the chunk's UUID and parent document UUID
        """
        parent_uuid = self.parent_document.uuid if self.parent_document else None
        return f"Chunk(uuid={self.uuid}, parent_document={parent_uuid})"

    def __hash__(self) -> int:
        """Get a hash value for the chunk based on its UUID.

        Returns:
            Hash value derived from the chunk's UUID
        """
        return hash(self.uuid)

    def __eq__(self, other: object) -> bool:
        """Check if this chunk equals another object.

        Args:
            other: Object to compare with

        Returns:
            True if other is a DocumentChunk with the same UUID, False otherwise
        """
        if not isinstance(other, DocumentChunk):
            return False
        return self.uuid == other.uuid
