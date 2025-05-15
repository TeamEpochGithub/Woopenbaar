"""Data class for a document with its associated chunks and inferred metadata."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from backend.src.data_classes.document_chunk import ChunkID

if TYPE_CHECKING:
    from backend.src.data_classes.document_chunk import DocumentChunk


class ChunkedDocument:
    """Represents a document with its associated chunks and inferred metadata from the document content.

    Attributes:
        uuid: Internal unique identifier
        vws_id: External identifier for the document
        create_date: Document creation timestamp
        type: Document type (email, whatsapp, pdf, etc. )
        link: URL or reference to original document
        attachment_links: URLs to all attached documents
        content: Full text content
        subject: (inferred) subject of the document
        first_mentioned_date: (inferred) first date mentioned in the document
        last_mentioned_date: (inferred) last date mentioned in the document
        chunks: Dictionary of chunk IDs and their corresponding DocumentChunk objects
    """

    # original document metadata
    uuid: UUID
    vws_id: str
    create_date: Optional[datetime]
    type: Optional[str]
    link: str
    attachment_links: List[str]
    content: str
    weight: float

    # inferred metadata
    subject: Optional[str]
    first_mentioned_date: Optional[datetime]
    last_mentioned_date: Optional[datetime]
    email_from: Optional[str]
    email_to: Optional[str]
    email_cc: Optional[str]
    chunks: Dict[ChunkID, "DocumentChunk"]

    def __init__(
        self,
        uuid: UUID,
        vws_id: str,
        create_date: Optional[datetime],
        type: Optional[str],
        link: str,
        attachment_links: List[str],
        content: str,
        chunks: Dict["ChunkID", "DocumentChunk"],
        subject: Optional[str] = None,
        first_mentioned_date: Optional[datetime] = None,
        last_mentioned_date: Optional[datetime] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
        email_cc: Optional[str] = None,
        weight: float = 1.0,
    ):
        """Initialize a ChunkedDocument.

        Args:
            uuid: Internal unique identifier
            vws_id: External identifier for the document
            create_date: Document creation timestamp, can be None if date is invalid or missing
            type: Document type (email, whatsapp, pdf, etc.)
            link: URL or reference to original document
            attachment_links: URLs to all attached documents
            content: Full text content
            chunks: Dictionary of chunk IDs and their corresponding DocumentChunk objects
            weight: Weight of the document
            subject: (inferred) subject of the document
            first_mentioned_date: (inferred) first date mentioned in the document
            last_mentioned_date: (inferred) last date mentioned in the document
        """
        self.uuid = uuid
        self.vws_id = vws_id
        self.create_date = create_date
        self.type = type
        self.link = link
        self.attachment_links = attachment_links
        self.content = content
        self.chunks = chunks or {}
        self.subject = subject
        self.first_mentioned_date = first_mentioned_date
        self.last_mentioned_date = last_mentioned_date
        self.email_from = email_from
        self.email_to = email_to
        self.email_cc = email_cc

    def to_json(self) -> Dict[str, Any]:
        """Convert the document to a JSON-compatible dictionary.

        Returns:
            Dict containing the document's data in a JSON-serializable format
        """
        return {
            "uuid": str(self.uuid),
            "vws_id": self.vws_id,
            "create_date": (self.create_date.isoformat()) if self.create_date else None,
            "type": self.type,
            "link": self.link,
            "attachment_links": self.attachment_links,
            "content": self.content,
            "chunks": [chunk.to_json() for chunk in self.chunks.values()],
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

    def get_chunk_by_uuid(self, uuid: "ChunkID") -> "DocumentChunk":
        """Retrieve a specific chunk by its UUID.

        Args:
            uuid: The UUID of the chunk to retrieve

        Returns:
            The DocumentChunk with the specified UUID

        Raises:
            KeyError: If no chunk exists with the specified UUID
        """
        return self.chunks[uuid]

    def __str__(self) -> str:
        """Get a compressed string representation of the document.

        Returns:
            String containing basic document info and number of chunks, but not the chunk contents
        """
        return (
            f"ChunkedDocument(uuid={self.uuid}, vws_id={self.vws_id}, create_date={self.create_date}, type={self.type},"
            f"link={self.link}, attachment_links={self.attachment_links}, content={self.content}, chunks={len(self.chunks)}, "
            f"subject={self.subject}, "
            f"first_mentioned_date={self.first_mentioned_date}, last_mentioned_date={self.last_mentioned_date}, "
            f"email_from={self.email_from}, email_to={self.email_to}, email_cc={self.email_cc})"
        )

    def to_str_all(self) -> str:
        """Get a detailed string representation including all chunk contents.

        Returns:
            String containing full document info and the string representation of all chunks
        """
        return (
            self.__str__()
            + "\n"
            + "\n".join([chunk.__str__() for chunk in self.chunks.values()])
        )

    def __repr__(self) -> str:
        """Get a concise string representation for debugging.

        Returns:
            String containing the most important document attributes
        """
        return f"ChunkedDocument(uuid={self.uuid}, vws_id={self.vws_id}, type={self.type}, n_chunks={len(self.chunks)})"

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
            True if other is a ChunkedDocument with the same UUID, False otherwise
        """
        if not isinstance(other, ChunkedDocument):
            return False
        return self.uuid == other.uuid
