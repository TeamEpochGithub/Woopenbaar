"""Component for evaluating and filtering retrieved document chunks.

This component uses LLM-based analysis to:
1. Evaluate chunk relevance to queries
2. Filter out irrelevant information
3. Provide reasoning about evaluation decisions
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from backend.conf.prompts import BATCH_CHUNK_EVALUATION_SYSTEM_PROMPT
from backend.src.data_classes.document_chunk import DocumentChunk
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.query_processing.utils import log_llm_interaction

logger = logging.getLogger(__name__)


class BatchChunkEvaluationResponse(BaseModel):
    """Response model for LLM-based batch chunk evaluation.

    This model structures and validates the LLM's analysis of multiple chunks,
    capturing both the selection of relevant chunks and the reasoning behind
    the selection.

    Attributes:
        relevant_chunk_indices: Zero-based indices of chunks determined to be
            relevant to the query, considering both information content and
            query context
        summary_reasoning: Detailed explanation of the evaluation results,
            including what relevant information was found and why certain
            chunks were selected or rejected
    """

    relevant_chunk_indices: List[int] = Field(
        default_factory=list,
        description="Indices of chunks that contain relevant information",
    )
    summary_reasoning: str = Field(
        description="Summary explanation of what was found in the relevant chunks"
    )


class ChunkEvaluator:
    """Evaluates and filters document chunks using LLM-based analysis.

    This component is responsible for:
    1. Analyzing chunks for relevance to queries
    2. Filtering out irrelevant or redundant information
    3. Providing detailed reasoning about evaluation decisions

    The evaluation process considers:
    - Relevance to the original query
    - Relevance to specific subqueries
    - Information quality and completeness
    - Context and relationships between chunks

    Attributes:
        llm_service: Service for LLM-based chunk analysis
    """

    def __init__(self, llm_service: BaseLLMService) -> None:
        """Initialize the chunk evaluator.

        Args:
            llm_service: Service for LLM-based chunk analysis

        Raises:
            AssertionError: If LLM service is None
        """
        assert llm_service is not None, "LLM service must be provided"

        self.llm_service = llm_service
        logger.info("ChunkEvaluator initialized")

    def evaluate_chunks(
        self, query: str, subquery: str, chunks: List[DocumentChunk], state: QueryState
    ) -> List[DocumentChunk]:
        """Evaluate and filter chunks using LLM-based analysis.

        This method performs batch evaluation of chunks by:
        1. Analyzing each chunk's relevance to both query and subquery
        2. Identifying chunks containing useful information
        3. Extracting a summary of findings

        The evaluation considers:
        - Direct relevance to the query/subquery
        - Quality and completeness of information
        - Context and relationships between chunks
        - Potential redundancy in information

        Args:
            query: Original user query providing overall context
            subquery: Specific subquery used for retrieval
            chunks: List of document chunks to evaluate
            state: Query state for adding reasoning steps

        Returns:
            List[DocumentChunk]: Filtered list of relevant chunks

        Note:
            Returns an empty list if:
            - No chunks are provided
            - LLM evaluation fails
            - No chunks are found to be relevant
        """
        assert query, "Original query must not be empty"
        assert subquery, "Subquery must not be empty"
        assert state is not None, "QueryState must be provided"

        # Invariant: Subquery should be in the state's previous subqueries
        assert (
            subquery in state.previous_subqueries or subquery == state.query
        ), "Subquery must be tracked in state"

        # Invariant: Initial state of reasoning steps count
        initial_reasoning_steps_count = len(state.reasoning_trace)

        if not chunks:
            logger.warning("No chunks to evaluate")
            return []

        logger.info(f"Evaluating {len(chunks)} chunks for relevance")

        # Prepare chunks for evaluation
        chunk_texts: List[str] = []
        for i, chunk in enumerate(chunks):
            chunk_texts.append(f"FRAGMENT #{i}:\n{chunk.content}\n")

        chunks_content = "\n---\n".join(chunk_texts)

        # Construct evaluation prompt
        user_message = f"""
        Oorspronkelijke vraag: {query}
        
        Deelvraag: {subquery}
        
        Opgehaalde tekstfragmenten om te evalueren:
        {chunks_content}
        
        Evalueer welke fragmenten informatie bevatten die relevant is voor de oorspronkelijke vraag.
        Geef indices van relevante fragmenten en een samenvatting van wat we hebben geleerd van deze ophaalstap.
        """

        json_schema = BatchChunkEvaluationResponse.model_json_schema()
        extra_body: Dict[str, Any] = {
            "guided_json": json_schema,
            "guided_backend": "xgrammar:disable-any-whitespace",
        }
        response: str = ""

        try:
            # Generate and validate evaluation
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=BATCH_CHUNK_EVALUATION_SYSTEM_PROMPT,
                extra_body=extra_body,
            )

            assert response, "LLM response must not be empty"

            log_llm_interaction(
                stage="batch_chunk_evaluation",
                messages=[
                    {"role": "system", "content": BATCH_CHUNK_EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response=response,
                extra_body=extra_body,
            )

            # Process evaluation results
            result = BatchChunkEvaluationResponse.model_validate_json(response)
            relevant_indices: List[int] = result.relevant_chunk_indices
            summary_reasoning: str = result.summary_reasoning

            # Validate results
            assert all(
                0 <= idx < len(chunks) for idx in relevant_indices
            ), "All relevant indices must be within range of chunks list"
            assert summary_reasoning, "Summary reasoning must not be empty"

            # Extract relevant chunks
            relevant_chunks: List[DocumentChunk] = []
            for i, chunk in enumerate(chunks):
                if i in relevant_indices:
                    relevant_chunks.append(chunk)

            # Add reasoning steps
            state.add_reasoning(
                f"Gevonden {len(relevant_chunks)} relevante fragmenten."
            )
            state.add_reasoning(summary_reasoning)
            state.add_frontend_step(
                f"Gevonden {len(relevant_chunks)} relevante informatiefragmenten."
            )

            # Invariant: We should have added exactly 2 reasoning steps
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 2
            ), "ChunkEvaluator must add exactly 2 reasoning steps"

            # Validate consistency
            assert len(relevant_chunks) == len(
                relevant_indices
            ), "Number of relevant chunks should match number of relevant indices"
            assert len(relevant_chunks) <= len(
                chunks
            ), "Number of relevant chunks cannot exceed number of input chunks"

            # Log results
            if relevant_chunks:
                logger.info(
                    f"Found {len(relevant_chunks)} relevant chunks out of {len(chunks)}"
                )
                logger.debug(f"Evaluation summary: {summary_reasoning}")
            else:
                logger.info("No relevant chunks found")

            return relevant_chunks

        except Exception as e:
            logger.warning(
                f"Failed to parse or validate LLM response: {str(e)}\nResponse: {response}"
            )
            return []
