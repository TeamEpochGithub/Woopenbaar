"""Query processing service implementing an adaptive RAG (Retrieval-Augmented Generation) pipeline.

This service orchestrates the entire query processing flow by:
1. Breaking down complex queries into subqueries when needed
2. Retrieving relevant documents for each query/subquery
3. Evaluating and filtering retrieved information
4. Synthesizing coherent answers from the gathered context
5. Generating relevant follow-up questions

The service supports two main processing modes:
- Standard RAG: Direct retrieval and answer synthesis
- Adaptive RAG: Iterative retrieval with dynamic subquery generation
"""

import logging
from typing import Any, List, Optional

from backend.conf.config import Config
from backend.src.data_classes import ChunkedDocument, DocumentChunk, FilterOptions
from backend.src.data_classes.query_result import QueryResult
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.components.answer_synthesis import (
    AnswerSynthesizer,
)
from backend.src.services.query_processing.components.chunk_evaluator import (
    ChunkEvaluator,
)
from backend.src.services.query_processing.components.progress_evaluator import (
    ProgressEvaluator,
)
from backend.src.services.query_processing.components.retrieval_manager import (
    RetrievalManager,
)
from backend.src.services.query_processing.components.suggestion import (
    SuggestionGenerator,
)
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.retrieval.retrieval_service import RetrievalService
from backend.src.services.safety.safety_service import SafetyService

logger = logging.getLogger(__name__)


class QueryProcessingService:
    """Orchestrates the query processing pipeline using specialized components.

    This service implements both standard and adaptive RAG approaches, managing:
    - Query analysis and decomposition
    - Document retrieval and filtering
    - Answer synthesis and follow-up suggestion
    - Progress tracking and state management

    Each component handles a specific aspect of the pipeline:
    - ProgressEvaluator: Analyzes query progress and generates subqueries
    - RetrievalManager: Handles document retrieval operations
    - ChunkEvaluator: Evaluates relevance of retrieved chunks
    - AnswerSynthesizer: Generates final and intermediate answers
    - SuggestionGenerator: Creates follow-up questions

    Attributes:
        llm_service: Service for language model interactions
        retrieval_service: Service for document retrieval operations
        safety_service: Service for content safety validation
        progress_evaluator: Component for progress tracking and subquery generation
        retrieval_manager: Component for retrieval operations
        chunk_evaluator: Component for relevance evaluation
        answer_synthesizer: Component for answer generation
        suggestion_generator: Component for follow-up question generation
    """

    def __init__(
        self,
        llm_service: BaseLLMService,
        retrieval_service: RetrievalService,
        safety_service: SafetyService,
    ) -> None:
        """Initialize the service with required components.

        Args:
            llm_service: Service for language model interactions
            retrieval_service: Service for document retrieval
            safety_service: Service for content safety validation

        Raises:
            AssertionError: If any required service is None
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.safety_service = safety_service

        assert llm_service is not None, "LLM service is required"
        assert retrieval_service is not None, "Retrieval service is required"
        assert safety_service is not None, "Safety service is required"

        # Initialize pipeline components
        self.progress_evaluator = ProgressEvaluator(llm_service)
        self.retrieval_manager = RetrievalManager(
            retrieval_service=retrieval_service, llm_service=llm_service
        )
        self.chunk_evaluator = ChunkEvaluator(llm_service)
        self.answer_synthesizer = AnswerSynthesizer(llm_service)
        self.suggestion_generator = SuggestionGenerator(llm_service, safety_service)

        logger.info("QueryProcessingService initialized with all components")

    def process_standard_query(
        self,
        query: str,
        filters: Optional[FilterOptions] = None,
        prioritize_earlier: bool = False,
    ) -> QueryResult:
        """Process a query using the standard RAG approach.

        This method performs direct retrieval and answer synthesis without
        breaking the query into subqueries. It's suitable for simpler queries
        that don't require multiple retrieval steps.

        Args:
            query: The user's query to process
            filters: Optional metadata filters for retrieval
            prioritize_earlier: Whether to prioritize earlier documents

        Returns:
            QueryResult containing the answer, relevant chunks, and metadata

        Note:
            This method will return an empty result with an appropriate message
            if no relevant information is found.
        """
        state = QueryState(query=query)

        # Invariant: Initial state setup
        assert state.query == query, "Query state initialized with incorrect query"
        assert (
            not state.is_ready_to_answer()
        ), "State should not be ready to answer initially"
        assert (
            len(state.reasoning_trace) == 0
        ), "State should have no reasoning steps initially"

        # Execute direct retrieval
        chunks, source_documents = self.retrieval_manager.execute_retrieval(
            state=state,
            subquery=query,
            filters=filters,
            prioritize_earlier=prioritize_earlier,
        )

        assert (
            len(state.reasoning_trace) >= 2
        ), "Retrieval should add at least 2 reasoning steps"

        if not chunks:
            logger.info(f"No chunks found for query: '{query}'")
            return QueryResult(
                query=query,
                response="Ik heb geen relevante informatie kunnen vinden voor uw vraag.",
                chunks=[],
                source_documents=[],
                reasoning_steps=["No relevant information found for the query."],
                context=[],
                data_sources_used=set(),
            )

        # Evaluate chunk relevance
        relevant_chunks = self.chunk_evaluator.evaluate_chunks(
            query=query, subquery=query, chunks=chunks, state=state
        )

        # Invariant: Chunk evaluation should add reasoning steps
        assert (
            len(state.reasoning_trace) >= 4
        ), "Chunk evaluation should add at least 2 reasoning steps"

        state.add_chunks_to_context(relevant_chunks)

        assert len(state.get_all_context_pieces()) == len(
            relevant_chunks
        ), "Number of context pieces should match number of evaluated chunks"

        # Mark state as ready to answer before generating answer
        state.update_readiness(True)
        assert (
            state.is_ready_to_answer()
        ), "State should be ready to answer before generating answer"

        # Generate answer
        answer = self.answer_synthesizer.generate_answer(state)

        # Invariant: Answer generation should add a reasoning step
        assert (
            len(state.reasoning_trace) >= 5
        ), "Answer generation should add at least 1 reasoning step"

        assert (
            answer
        ), "Generated answer should not be empty when relevant chunks are found"

        # Generate follow-up questions if enabled
        suggested_questions: List[str] = []
        if relevant_chunks and Config.ENABLE_SUGGESTED_QUESTIONS:
            try:
                suggestions_result = self.suggestion_generator.generate_suggestions(
                    state=state
                )
                suggested_questions = suggestions_result.suggestions
                logger.info(
                    f"Generated {len(suggested_questions)} suggested follow-up questions"
                )
            except Exception as e:
                logger.error(f"Failed to generate suggested questions: {str(e)}")

        # Create result
        result = QueryResult(
            query=query,
            response=answer,
            chunks=relevant_chunks,
            source_documents=source_documents,
            reasoning_steps=state.frontend_steps,
            context=state.get_all_context_pieces(),
            data_sources_used=state.used_data_sources,
            suggested_questions=suggested_questions,
        )

        # Invariant: Result consistency
        assert (
            result.response == answer
        ), "Response in result should match the generated answer"
        assert result.query == query, "Query in result should match the original query"
        assert len(result.reasoning_steps) == len(
            state.frontend_steps
        ), "All reasoning steps should be included in the result"
        assert len(result.context) == len(
            state.get_all_context_pieces()
        ), "All context pieces should be included in the result"
        assert (
            result.data_sources_used == state.used_data_sources
        ), "Data sources in result should match those in state"

        return result

    def process_adaptive_query(
        self,
        query: str,
        max_iterations: int = 3,
        filters: Optional[FilterOptions] = None,
        frontend_callback: Optional[Any] = None,
        prioritize_earlier: bool = False,
    ) -> QueryResult:
        """Process a query through the adaptive RAG pipeline.

        This method implements an iterative approach that:
        1. Evaluates query progress and generates subqueries
        2. Retrieves and evaluates information for each subquery
        3. Tracks progress and generates intermediate answers
        4. Synthesizes a final answer when sufficient information is gathered

        Args:
            query: The user's query to process
            max_iterations: Maximum number of retrieval iterations (default: 3)
            filters: Optional metadata filters for retrieval
            frontend_callback: Optional callback for frontend progress updates
            prioritize_earlier: Whether to prioritize earlier documents

        Returns:
            QueryResult containing the answer, relevant chunks, and metadata

        Note:
            The method will stop early if:
            - Sufficient information is gathered before max_iterations
            - No useful subqueries can be generated
            - Context size limits are reached
        """
        state = QueryState(query=query)
        if frontend_callback:
            state.set_frontend_callback(frontend_callback)

        # Invariant: Initial state setup
        assert state.query == query, "Query state initialized with incorrect query"
        assert (
            not state.is_ready_to_answer()
        ), "State should not be ready to answer initially"
        assert not state.is_complete(), "State should not be complete initially"
        assert (
            len(state.reasoning_trace) == 0
        ), "State should have no reasoning steps initially"

        state.add_frontend_step(f"Verwerken van vraag: {query}")
        state.add_reasoning(f"Start van de analyse voor vraag: {query}")

        # Invariant: Initial steps added
        assert (
            len(state.reasoning_trace) == 1
        ), "Should have exactly one reasoning step after initialization"
        assert (
            len(state.frontend_steps) == 1
        ), "Should have exactly one frontend step after initialization"

        max_iterations = Config.MAX_ITERATIONS
        iteration = 0
        relevant_chunks: List[DocumentChunk] = []
        source_documents: List[ChunkedDocument] = []
        intermediate_answer: Optional[str] = None

        assert max_iterations > 0, "Max iterations should be a positive integer"

        while not state.is_complete() and iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Starting iteration {iteration} / {max_iterations} ===")

            # Invariant: Reasoning steps should grow with iterations
            assert (
                len(state.reasoning_trace) >= iteration
            ), "Reasoning steps should increase with iterations"

            # Invariant: Frontend steps should grow with iterations
            assert (
                len(state.frontend_steps) >= iteration
            ), "Frontend steps should increase with iterations"

            # Evaluate progress and determine next steps
            evaluation = self.progress_evaluator.evaluate_progress(state)

            # Invariant: Progress evaluation should add reasoning steps
            assert (
                len(state.reasoning_trace) >= iteration + 1
            ), "Progress evaluation should add reasoning steps"

            if evaluation.ready_to_answer:
                state.update_readiness(True)

                # Invariant: State should be ready to answer
                assert (
                    state.is_ready_to_answer()
                ), "State should be ready to answer after positive evaluation"
                break

            elif evaluation.subquery:
                subquery = evaluation.subquery
                logger.info(f"Processing subquery: '{subquery}'")

                assert (
                    subquery not in state.previous_subqueries
                ), "Subquery should not be in previous_subqueries"
                # Execute retrieval for this subquery
                chunks, sources = self.retrieval_manager.execute_retrieval(
                    state=state,
                    subquery=subquery,
                    filters=filters,
                    prioritize_earlier=prioritize_earlier,
                )
                # Invariant: Subquery should be in previous subqueries
                assert (
                    subquery in state.previous_subqueries
                ), "Subquery should be added to previous_subqueries"

                # Invariant: Retrieval should add reasoning steps
                assert (
                    len(state.reasoning_trace) >= iteration + 2
                ), "Retrieval should add reasoning steps"

                # Invariant: Retrieval should add frontend steps
                frontend_steps_after_retrieval = len(state.frontend_steps)
                assert (
                    frontend_steps_after_retrieval >= iteration + 1
                ), "Retrieval should add frontend steps"

                if chunks:
                    # Evaluate chunk relevance
                    evaluated_chunks = self.chunk_evaluator.evaluate_chunks(
                        query=state.query, subquery=subquery, chunks=chunks, state=state
                    )

                    # Invariant: Chunk evaluation should add exactly 2 reasoning steps
                    assert (
                        len(state.reasoning_trace) >= iteration + 4
                    ), "Chunk evaluation should add 2 reasoning steps"

                    # Update state with relevant chunks
                    state.add_chunks_to_context(evaluated_chunks)
                    relevant_chunks.extend(evaluated_chunks)

                    if sources:
                        source_documents.extend(sources)

                    # Generate intermediate answer if appropriate
                    if (
                        relevant_chunks
                        and iteration < max_iterations
                        and not evaluation.ready_to_answer
                    ):
                        try:
                            intermediate_answer = (
                                self.answer_synthesizer.generate_intermediate_answer(
                                    state
                                )
                            )
                            state.add_frontend_step(intermediate_answer)

                            # Invariant: Intermediate answer should add reasoning and frontend steps
                            assert (
                                len(state.reasoning_trace) >= iteration + 5
                            ), "Intermediate answer should add a reasoning step"
                            assert (
                                len(state.frontend_steps)
                                > frontend_steps_after_retrieval
                            ), "Intermediate answer should add a frontend step"

                            assert (
                                intermediate_answer
                            ), "Intermediate answer should not be empty"
                        except Exception as e:
                            logger.error(
                                f"Failed to generate intermediate answer: {str(e)}"
                            )
            else:
                state.update_readiness(True)

                # Invariant: State should be ready to answer
                assert (
                    state.is_ready_to_answer()
                ), "State should be ready to answer when no subquery is generated"
                break

        # Handle max iterations case
        if iteration >= max_iterations and not state.is_ready_to_answer():
            state.update_readiness(True)

        # Invariant: State should be ready to answer after completing all iterations
        assert (
            state.is_ready_to_answer()
        ), "State should be ready to answer after completing iterations"

        # Add final reasoning step before generating answer
        state.add_reasoning(f"Einde van de analyse voor vraag: {query}")

        # Invariant: Final reasoning steps
        total_iterations = min(iteration, max_iterations)
        assert (
            len(state.reasoning_trace) >= total_iterations * 2 + 2
        ), "Should have sufficient reasoning steps after all iterations"

        # Generate final answer
        state.add_frontend_step("Genereren van antwoord...")
        answer = self.answer_synthesizer.generate_answer(state)

        # Invariant: Final frontend steps
        assert (
            len(state.frontend_steps) >= total_iterations + 2
        ), "Should have sufficient frontend steps after all iterations"

        assert answer, "Final answer should not be empty"

        # Generate suggested follow-up questions
        suggested_questions: List[str] = []
        if relevant_chunks and Config.ENABLE_SUGGESTED_QUESTIONS:
            try:
                suggestions_result = self.suggestion_generator.generate_suggestions(
                    state=state
                )
                suggested_questions = suggestions_result.suggestions
                logger.info(
                    f"Generated {len(suggested_questions)} suggested follow-up questions"
                )
            except Exception as e:
                logger.error(f"Failed to generate suggested questions: {str(e)}")

        # Create result
        result = QueryResult(
            query=query,
            response=answer,
            chunks=relevant_chunks,
            source_documents=source_documents,
            reasoning_steps=state.frontend_steps,
            context=state.get_all_context_pieces(),
            data_sources_used=state.used_data_sources,
            suggested_questions=suggested_questions,
        )

        # Invariant: Result consistency
        assert (
            result.response == answer
        ), "Response in result should match the generated answer"
        assert result.query == query, "Query in result should match the original query"
        assert len(result.reasoning_steps) == len(
            state.frontend_steps
        ), "All reasoning steps should be included in the result"
        assert len(result.context) == len(
            state.get_all_context_pieces()
        ), "All context pieces should be included in the result"
        assert (
            result.data_sources_used == state.used_data_sources
        ), "Data sources in result should match those in state"

        return result
