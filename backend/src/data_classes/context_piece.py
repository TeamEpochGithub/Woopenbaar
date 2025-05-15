from dataclasses import dataclass
from uuid import UUID

from backend.src.data_classes.document_chunk import ChunkID


@dataclass
class ContextPiece:
    """A piece of context extracted from document chunks.

    Attributes:
        doc_uuid: UUID of the source document
        chunk_id: ID of the source chunk
        content: The extracted information content
        type: type of the document (PDF, Word etc.)
        source: Name of the data source
    """

    doc_uuid: UUID
    chunk_id: ChunkID
    content: str
    type: str
    source: str

    def __hash__(self) -> int:
        """Hash function based on chunk_id.

        Returns:
            Hash value for the context piece
        """
        return hash(self.chunk_id)

    def __eq__(self, other: object) -> bool:
        """Equality check based on chunk_id.

        Args:
            other: Object to compare with

        Returns:
            True if other is a ContextPiece with the same chunk_id, False otherwise
        """
        if not isinstance(other, ContextPiece):
            return False
        return self.chunk_id == other.chunk_id

    def __str__(self) -> str:
        """Return a string representation for display and LLM consumption.

        Returns:
            Formatted string with content and source information
        """
        return (
            f"INHOUD: {self.content}\n"
            f"BRON: {self.source}, Document ID: {self.doc_uuid}, Chunk ID: {self.chunk_id}\n"
            f"----------------------------------------------------------------------"
        )

    def __repr__(self) -> str:
        """Return a technical string representation with essential details.

        Returns:
            Technical representation with all attributes
        """
        return (
            f"ContextPiece(doc_uuid={self.doc_uuid}, "
            f"chunk_id={self.chunk_id}, "
            f"source='{self.source}', "
            f"content='{self.content}"
        )
