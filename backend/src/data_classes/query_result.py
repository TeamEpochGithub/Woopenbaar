"""Query result data class for representing the output of query processing services."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    from backend.src.data_classes.chunked_document import ChunkedDocument
    from backend.src.data_classes.context_piece import ContextPiece
    from backend.src.data_classes.document_chunk import DocumentChunk


@dataclass
class QueryResult:
    """Result of the query processing workflow.

    Attributes:
        query: The original user query
        response: The final answer to the user's query
        chunks: The useful document chunks used to generate the answer
        source_documents: The source documents that contained the useful chunks
        reasoning_steps: List of reasoning steps for frontend display
        known_information: List of known information pieces
        data_sources_used: List of data sources (vws, timelines, etc.) used during processing
        suggested_questions: List of suggested follow-up questions
    """

    query: str
    response: str
    chunks: List["DocumentChunk"]
    source_documents: List["ChunkedDocument"]
    reasoning_steps: List[str]
    context: List["ContextPiece"] = field(default_factory=list)
    data_sources_used: Set[str] = field(default_factory=set)
    suggested_questions: List[str] = field(default_factory=list)
