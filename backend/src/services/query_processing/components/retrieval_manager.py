"""Component for managing document retrieval operations.

This component handles:
1. Intelligent data source selection based on query context
2. Execution of retrieval operations with configurable parameters
3. Tracking of retrieved documents and chunks
4. Frontend progress updates during retrieval
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from backend.conf.config import Config
from backend.conf.prompts import SOURCE_SELECTION_SYSTEM_PROMPT
from backend.src.data_classes import ChunkedDocument, DocumentChunk, FilterOptions
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.query_processing.utils import log_llm_interaction
from backend.src.services.retrieval.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class SourceSelectionResponse(BaseModel):
    """Response model for LLM-based data source selection.

    This model validates and structures the LLM's response when selecting
    the most appropriate data source for a query.

    Attributes:
        source_name: Name of the selected data source
        reasoning: Detailed explanation of why this source was chosen,
                  including relevance to query and expected information quality
    """

    source_name: str = Field(description="Name of the data source to use for retrieval")
    reasoning: str = Field(
        default="No reasoning provided",
        description="Explanation of why this data source was selected",
    )


class RetrievalManager:
    """Manages intelligent document retrieval operations.

    This component is responsible for:
    1. Analyzing queries to select optimal data sources
    2. Executing retrieval with configurable parameters
    3. Managing chunk and document retrieval limits
    4. Tracking retrieved information in query state
    5. Providing frontend progress updates

    The manager uses LLM-based source selection to choose the most appropriate
    data source based on:
    - Query content and context
    - Available data sources and their descriptions
    - Previous retrieval attempts and their success

    Attributes:
        retrieval_service: Service for executing document retrieval operations
        llm_service: Service for LLM-based source selection
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: BaseLLMService,
    ) -> None:
        """Initialize the retrieval manager.

        Args:
            retrieval_service: Service for document retrieval operations
            llm_service: Service for LLM-based source selection

        Raises:
            AssertionError: If either required service is None
        """
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

        assert retrieval_service is not None, "Retrieval service must be provided"
        assert llm_service is not None, "LLM service must be provided"

        logger.info("RetrievalManager initialized")

    def select_data_source(self, query: str, state: QueryState) -> str:
        """Select the optimal data source for a query using LLM-based analysis.

        This method:
        1. Analyzes the query and available data sources
        2. Uses LLM to select the most appropriate source
        3. Validates the selection against available sources
        4. Falls back to default source if selection fails

        The selection considers:
        - Query content and intent
        - Data source descriptions and coverage
        - Previous retrieval attempts in the query state
        - Source availability and status

        Args:
            query: Query to analyze for source selection
            state: Current query state with retrieval history

        Returns:
            str: Name of the selected data source, or empty string if none available

        Note:
            Falls back to the first available source if:
            - Only one source is available
            - LLM selection fails
            - Selected source is invalid
        """
        assert query, "Query must not be empty"
        assert state is not None, "QueryState must be provided"

        # Invariant: Track initial state of reasoning steps
        initial_reasoning_steps_count = len(state.reasoning_trace)

        available_sources = self.retrieval_service.get_sources()

        if not available_sources:
            raise ValueError("No data sources available for selection")

        if len(available_sources) == 1:
            logger.info(
                f"Only one data source available, selecting: {available_sources[0]['name']}"
            )
            source_name = available_sources[0]["name"]
            state.add_reasoning(f"Gekozen gegevensbron: {source_name}")

            # Invariant: Exactly one reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for source selection"

            return source_name

        # Prepare source selection prompt
        system_prompt = SOURCE_SELECTION_SYSTEM_PROMPT
        sources_desc = "\n".join(
            [f"- {s['name']}: {s['description']}" for s in available_sources]
        )

        user_message = f"""
        Oorspronkelijke vraag: {query}
        
        Beschikbare gegevensbronnen:
        {sources_desc}

        Redenering tot nu toe:
        {state.reasoning_trace[:-5]}
        
        Als een bron recent is mislukt, is het misschien verstandig om een andere bron te proberen.

        Kies de BESTE ENKELE gegevensbron voor deze zoekopdracht en leg uit waarom deze specifieke bron de meest relevante informatie zou bevatten.
        """

        json_schema = SourceSelectionResponse.model_json_schema()
        extra_body: Dict[str, Any] = {
            "guided_json": json_schema,
            "guided_backend": "xgrammar:disable-any-whitespace",
        }

        logger.info(f"Selecting data source for query: '{query}'")
        response = ""

        try:
            # Generate and validate source selection
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=system_prompt,
                extra_body=extra_body,
            )

            log_llm_interaction(
                stage="source_selection",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response=response,
                extra_body=extra_body,
            )

            result = SourceSelectionResponse.model_validate_json(response)
            source_name = result.source_name
            reasoning = result.reasoning
            state.add_reasoning(f"Gekozen gegevensbron: {source_name}")

            # Invariant: Exactly one reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for source selection"

            valid_sources = [s["name"] for s in available_sources]

            if source_name in valid_sources:
                logger.info(f"Selected source {source_name} because: {reasoning}")
                return source_name

            logger.warning("Selected source was invalid, using first available source")
            return valid_sources[0]

        except Exception as e:
            logger.warning(
                f"Failed to parse source selection response: {str(e)}\nResponse: {response}"
            )
            state.add_reasoning(
                f"Fout bij het selecteren van een gegevensbron, gebruik fallback: {available_sources[0]['name']}"
            )

            # Invariant: Exactly one reasoning step should be added even in error case
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for source selection error"

            return available_sources[0]["name"]

    def execute_retrieval(
        self,
        state: QueryState,
        subquery: str,
        filters: Optional[FilterOptions] = None,
        prioritize_earlier: bool = False,
    ) -> Tuple[List[DocumentChunk], List[ChunkedDocument]]:
        """Execute retrieval operation with dynamic source selection.

        This method:
        1. Selects the optimal data source for the query
        2. Executes retrieval with configured limits
        3. Filters out previously seen chunks
        4. Updates query state with retrieval results
        5. Provides frontend progress updates

        The retrieval process respects configured limits for:
        - Initial and final document counts
        - Initial and final chunk counts
        - Metadata filtering

        Args:
            state: Query state to update with retrieval results
            subquery: Query or subquery to execute
            filters: Optional metadata filters to apply
            prioritize_earlier: Whether to prioritize earlier documents

        Returns:
            Tuple containing:
            - List of retrieved document chunks (filtered for uniqueness)
            - List of source documents

        Note:
            Returns empty lists if:
            - No data sources are available
            - Retrieval operation fails
            - No matching documents are found
            - All retrieved chunks were already processed
        """
        assert state is not None, "QueryState must be provided"
        assert subquery, "Subquery must not be empty"
        assert (
            subquery not in state.previous_subqueries
        ), "Subquery should not be in previous_subqueries"

        # Invariant: Track initial state of various metrics
        initial_reasoning_steps_count = len(state.reasoning_trace)
        initial_frontend_steps_count = (
            len(state.frontend_steps) if hasattr(state, "frontend_steps") else 0
        )
        initial_retrieved_chunks_count = len(state.retrieved_chunks)
        initial_data_sources_count = len(state.used_data_sources)

        # Add subquery to state history
        if subquery not in state.previous_subqueries:
            state.add_prev_subquery(subquery)
            assert (
                subquery in state.previous_subqueries
            ), "Subquery should be added to previous_subqueries"

        retrieved_chunks: List[DocumentChunk] = []
        retrieved_source_docs: List[ChunkedDocument] = []

        source_name = self.select_data_source(subquery, state)

        # Invariant: We should have exactly one more reasoning step after source selection
        assert (
            len(state.reasoning_trace) == initial_reasoning_steps_count + 1
        ), "Exactly one reasoning step should be added from source selection"

        source_msg = (
            f"Zoeken in bron '{source_name}': {subquery}"
            if source_name
            else "Geen geschikte gegevensbronnen beschikbaar voor deze zoekopdracht"
        )
        state.add_frontend_step(source_msg)

        # Invariant: We should have exactly one more frontend step
        assert (
            len(state.frontend_steps)
            if hasattr(state, "frontend_steps")
            else 0 == initial_frontend_steps_count + 1
        ), "Exactly one frontend step should be added for search notification"

        if not source_name:
            logger.warning("No data sources available for selection")
            return retrieved_chunks, retrieved_source_docs

        state.used_data_sources.add(source_name)
        assert (
            source_name in state.used_data_sources
        ), "Selected source should be added to used_data_sources"

        # Invariant: Data sources should be monotonically increasing
        assert (
            len(state.used_data_sources) >= initial_data_sources_count
        ), "Used data sources should never decrease"

        try:
            logger.info(
                f"Executing find with query: '{subquery}', source: '{source_name}'"
            )

            chunks, source_docs = self.retrieval_service.find(
                query=subquery,
                source_name=source_name,
                initial_documents_k=Config.INITIAL_DOCUMENTS_K,
                final_documents_k=Config.FINAL_DOCUMENTS_K,
                initial_chunks_k=Config.INITIAL_CHUNKS_K,
                final_chunks_k=Config.FINAL_CHUNKS_K,
                filters=filters,
                prioritize_earlier=prioritize_earlier,
            )

            logger.info(
                f"Find returned {len(chunks)} chunks and {len(source_docs)} source documents"
            )

            assert (
                len(chunks) <= Config.FINAL_CHUNKS_K
            ), "Number of chunks should not exceed FINAL_CHUNKS_K"
            if source_docs:
                assert (
                    chunks
                ), "If source documents are found, chunks should not be empty"

            # Filter out previously seen chunks
            new_chunks: List[DocumentChunk] = []
            for chunk in chunks:
                if chunk.uuid not in state.retrieved_chunks:
                    new_chunks.append(chunk)
                    state.add_retrieved_chunk(chunk.uuid)

            for chunk in new_chunks:
                assert (
                    chunk.uuid in state.retrieved_chunks
                ), "New chunk should be registered in retrieved_chunks"

            # Invariant: Retrieved chunks should be monotonically increasing
            assert (
                len(state.retrieved_chunks) >= initial_retrieved_chunks_count
            ), "Retrieved chunks count should never decrease"
            assert len(state.retrieved_chunks) <= initial_retrieved_chunks_count + len(
                chunks
            ), "Cannot add more chunks than retrieved"

            # Invariant: We should have exactly two frontend steps now
            assert (
                len(state.frontend_steps)
                if hasattr(state, "frontend_steps")
                else 0 == initial_frontend_steps_count + 2
            ), "Two frontend steps should be added during retrieval"

            # If all chunks were already seen, return empty
            if not new_chunks and chunks:
                state.add_reasoning(
                    f"All retrieved chunks for subquery '{subquery}' were already processed in previous iterations."
                )
                state.add_frontend_step(
                    "Geen nieuwe informatie gevonden voor deze deelvraag."
                )
                return [], []

            retrieved_chunks.extend(new_chunks)
            retrieved_source_docs.extend(source_docs)

            state.add_reasoning(
                f"Found {len(new_chunks)} new chunks from source '{source_name}' for subquery: '{subquery}'"
            )

            # Invariant: We should have exactly two reasoning steps now
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 2
            ), "Exactly two reasoning steps should be added during retrieval"

            # All retrieved chunks should belong to a source document
            chunk_doc_ids = {
                chunk.parent_document.uuid
                for chunk in new_chunks
                if chunk.parent_document
            }
            source_doc_ids = {doc.uuid for doc in source_docs}
            assert all(
                doc_id in source_doc_ids for doc_id in chunk_doc_ids
            ), "All chunk parent documents should be in source_docs"

            return retrieved_chunks, retrieved_source_docs

        except Exception as e:
            import traceback

            logger.error(f"Failed to execute retrieval: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            state.add_reasoning(f"Error during retrieval: {str(e)}")
            state.add_frontend_step("Er is een fout opgetreden tijdens het zoeken.")

            # Invariant: We should have proper error reporting
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 2
            ), "Two reasoning steps should be present even in error case"
            assert (
                len(state.frontend_steps)
                if hasattr(state, "frontend_steps")
                else 0 >= initial_frontend_steps_count + 2
            ), "At least two frontend steps should be added even in error case"

            return [], []
