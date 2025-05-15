"""Query state management for the query processing service."""

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from backend.conf.config import Config
from backend.src.data_classes import ChunkID, ContextPiece, DocumentChunk
from backend.src.services.query_processing.utils import (
    estimate_token_count,
    get_prompt_reserved_tokens,
)

logger = logging.getLogger(__name__)


class QueryState:
    """Manages the state of a query throughout the processing pipeline.

    This class is responsible for:
    1. Tracking query progress
    2. Storing relevant information and chunks
    3. Managing the frontend UI steps
    4. Preserving the reasoning trail

    Attributes:
        query: Original user query
        current_context: Context pieces relevant to the query
        previous_subqueries: Previous subqueries generated during processing
        retrieved_chunks: All chunk IDs seen during processing
        reasoning_trace: Reasoning steps taken during processing
        used_data_sources: Data source names used during processing
        frontend_steps: Reasoning steps formatted for frontend display
        max_available_tokens: Maximum tokens available for context
        _is_complete: Whether the context is full
        _ready_to_answer: Whether enough information has been gathered
        _frontend_callback: Callback function for frontend updates
    """

    def __init__(self, query: str):
        """Initialize the query state.

        Args:
            query: Original user query
        """
        # Core query data
        self.query: str = query
        self.current_context: List[ContextPiece] = []
        self.previous_subqueries: Set[str] = set()
        self.retrieved_chunks: Set[ChunkID] = set()

        # Reasoning and frontend state
        self.reasoning_trace: List[str] = []
        self.used_data_sources: Set[str] = set()
        self.frontend_steps: List[str] = []

        # Internal state
        self._is_complete: bool = False
        self._ready_to_answer: bool = False
        self._frontend_callback: Optional[Callable[[str], None]] = None

        # Token management
        self.max_available_tokens = (
            Config.LLM_MAX_MODEL_LEN
            - Config.LLM_MAX_TOKENS
            - get_prompt_reserved_tokens()
        )

    def add_chunks_to_context(self, chunks: List[DocumentChunk]) -> None:
        """Convert DocumentChunks to ContextPieces and add them to the state.

        Args:
            chunks: List of document chunks to add
        """
        for chunk in chunks:
            # Create a context piece from the chunk
            context_piece = ContextPiece(
                doc_uuid=chunk.parent_document.uuid,
                chunk_id=chunk.uuid,
                content=chunk.content,
                type=chunk.type or "unknown",
                source=getattr(chunk.parent_document, "vws_id", "unknown"),
            )

            # Add the chunk ID to retrieved chunks
            self.add_retrieved_chunk(chunk.uuid)

            # Add the context piece to the state
            self.add_context_piece(context_piece)

    # State management methods
    def update_readiness(self, is_ready: bool) -> None:
        """Update whether enough information has been gathered.

        Args:
            is_ready: Whether we have enough information to answer
        """
        self._ready_to_answer = is_ready
        if is_ready:
            logger.info(f"Query marked as ready to answer: '{self.query}'")

    def is_complete(self) -> bool:
        """Check if query processing is complete."""
        return self._is_complete

    def is_ready_to_answer(self) -> bool:
        """Check if enough information has been gathered."""
        return self._ready_to_answer

    # Context management methods
    def add_context_piece(self, context_piece: ContextPiece) -> bool:
        """Add a context piece if it fits within token limits.

        Args:
            context_piece: Context piece to add

        Returns:
            True if added, False if rejected due to size limits
        """
        if self.is_complete():
            logger.warning("Context already truncated. Not adding more context pieces.")
            return False

        new_context_tokens = self.get_current_token_count() + estimate_token_count(
            str(context_piece)
        )

        if new_context_tokens > self.max_available_tokens:
            logger.warning(
                f"Adding context piece would exceed token limit. Current: {self.get_current_token_count()}, Piece: {estimate_token_count(str(context_piece))}, Max: {self.max_available_tokens}"
            )

            self._is_complete = True
            self.update_readiness(True)

            self.add_reasoning(
                "Contextgrootte bereikt. Stoppen met het ophalen van meer informatie om binnen de LLM contextvenster te passen."
            )
            return False

        self.current_context.append(context_piece)
        if context_piece.source:
            self.used_data_sources.add(context_piece.source)

        return True

    def get_all_context_pieces(self) -> List[ContextPiece]:
        """Get all context pieces."""
        return self.current_context

    def add_retrieved_chunk(self, chunk_id: ChunkID) -> None:
        """Track a retrieved chunk.

        Args:
            chunk_id: Chunk ID to track
        """
        self.retrieved_chunks.add(chunk_id)

    # Reasoning management methods
    def add_reasoning(self, reasoning: str) -> None:
        """Add a reasoning step.

        Args:
            reasoning: Reasoning step to add
        """
        logger.info(f"Adding reasoning: '{reasoning}'")
        self.reasoning_trace.append(reasoning)
        self._ensure_token_limit()

    def add_prev_subquery(self, subquery: str) -> None:
        """Track a processed subquery.

        Args:
            subquery: Subquery to track
        """
        self.previous_subqueries.add(subquery)

    # Frontend management methods
    def set_frontend_callback(self, callback: Callable[[str], None]) -> None:
        """Set the frontend update callback.

        Args:
            callback: Function to call with frontend updates
        """
        self._frontend_callback = callback

    def add_frontend_step(self, step: str) -> None:
        """Add a user-facing step.

        Args:
            step: Step to display on frontend
        """
        logger.info(f"Adding frontend step: '{step}'")
        self.frontend_steps.append(step)

        if self._frontend_callback:
            self._frontend_callback(step)

    # Token management methods
    def get_current_token_count(self) -> int:
        """Get current estimated token count."""
        context_tokens = sum(
            estimate_token_count(str(piece)) for piece in self.current_context
        )
        reasoning_tokens = estimate_token_count("\n".join(self.reasoning_trace))
        query_tokens = estimate_token_count(self.query)
        return context_tokens + reasoning_tokens + query_tokens

    # Properties
    @property
    def full_reasoning_trace(self) -> str:
        """Get complete reasoning trace for LLM processing."""
        return "\n".join(self.reasoning_trace)

    @property
    def is_context_limit_reached(self) -> bool:
        """Check if context limit is reached."""
        buffer_tokens = 500
        current_tokens = self.get_current_token_count()
        return (
            current_tokens >= (self.max_available_tokens - buffer_tokens)
            or self._is_complete
        )

    # Serialization methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "query": self.query,
            "current_context": [str(cp) for cp in self.current_context],
            "previous_subqueries": list(self.previous_subqueries),
            "retrieved_chunks": list(self.retrieved_chunks),
            "reasoning_trace": self.reasoning_trace,
            "frontend_steps": self.frontend_steps,
            "is_complete": self._is_complete,
            "ready_to_answer": self._ready_to_answer,
            "used_data_sources": list(self.used_data_sources),
        }

    # Private methods
    def _ensure_token_limit(self) -> None:
        """Ensure token count stays within limits by truncating if needed."""
        while (
            self.get_current_token_count() > self.max_available_tokens
            and len(self.reasoning_trace) > 1
        ):
            removed = self.reasoning_trace.pop(0)
            logger.info(
                f"Removed oldest reasoning step to fit token limit: '{removed[:50]}...'"
            )

        if self.get_current_token_count() > self.max_available_tokens:
            while (
                self.get_current_token_count() > self.max_available_tokens
                and len(self.current_context) > 0
            ):
                removed = self.current_context.pop(0)
                logger.info(
                    f"Removed oldest context piece to fit token limit: '{str(removed)[:50]}...'"
                )

            if len(self.current_context) < len(self.retrieved_chunks):
                self._is_complete = True
