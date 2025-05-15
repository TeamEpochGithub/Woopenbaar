"""Component for evaluating query progress and generating subqueries.

Uses LLM analysis to track progress and adaptively break down complex queries.
"""

import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from backend.conf.prompts import (
    PROGRESS_COMPLETENESS_SYSTEM_PROMPT,
    SUBQUERY_GENERATION_SYSTEM_PROMPT,
)
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.query_processing.utils import log_llm_interaction

logger = logging.getLogger(__name__)


# Models for the two submethods
class CompletenessEvaluation(BaseModel):
    """Result of query completeness evaluation.

    Attributes:
        status: Current progress status (READY/EXHAUSTED/NOT_READY)
        reasoning: Detailed explanation of the assessment
    """

    status: str = Field(
        description="Of we genoeg informatie hebben om de vraag te beantwoorden"
    )
    reasoning: str = Field(
        description="Uitleg waarom we wel of niet klaar zijn om te antwoorden"
    )


class SubqueryGeneration(BaseModel):
    """Result of subquery generation for missing information.

    Attributes:
        subquery: Generated search query to find missing information
        explanation: Reasoning behind the generated subquery
    """

    subquery: str = Field(
        description="Een zoekopdracht om aanvullende informatie te vinden"
    )
    explanation: str = Field(
        description="Uitleg waarom deze zoekopdracht de ontbrekende informatie zal opleveren"
    )


# Original response model (unchanged for compatibility)
class QueryProgressResponse(BaseModel):
    """Response model for LLM-based query progress evaluation.

    This model structures and validates the LLM's assessment of query
    processing progress, including readiness to answer and next steps.

    The status field can be one of:
    - READY: Sufficient information found to answer
    - EXHAUSTED: No more information available, but may be incomplete
    - NOT_READY: More information needed, subquery provided

    Attributes:
        status: Current progress status (READY/EXHAUSTED/NOT_READY)
        reasoning: Detailed explanation of the progress assessment,
                  including analysis of available information and gaps
        subquery: Generated search query to find missing information,
                 only provided when status is NOT_READY
    """

    status: str = Field(
        description="Of we genoeg informatie hebben om de vraag te beantwoorden"
    )
    reasoning: str = Field(
        description="Uitleg waarom we wel of niet klaar zijn om te antwoorden"
    )
    subquery: str = Field(
        default="", description="Een zoekopdracht om aanvullende informatie te vinden"
    )

    @property
    def ready_to_answer(self) -> bool:
        """Check if enough information is available to answer.

        The query is considered ready to answer if:
        - Sufficient relevant information has been found (READY)
        - No more information can be found, even if incomplete (EXHAUSTED)

        Returns:
            bool: True if ready to answer (READY/EXHAUSTED), False if more
                 information needed (NOT_READY)

        Raises:
            AssertionError: If status is not one of the valid values
        """
        assert self.status in [
            "READY",
            "EXHAUSTED",
            "NOT_READY",
        ], "Status should be one of: READY, EXHAUSTED, NOT_READY"

        return self.status == "READY" or self.status == "EXHAUSTED"


class ProgressEvaluator:
    """Evaluates query progress and generates targeted subqueries.

    Attributes:
        llm_service: Service for LLM-based progress analysis
    """

    def __init__(self, llm_service: BaseLLMService) -> None:
        """Initialize the progress evaluator.

        Args:
            llm_service: Service for LLM-based progress analysis

        Raises:
            AssertionError: If LLM service is None
        """
        assert llm_service is not None, "LLM service must be provided"

        self.llm_service = llm_service
        logger.info("ProgressEvaluator initialized")

    def evaluate_progress(self, state: QueryState) -> QueryProgressResponse:
        """Evaluate query progress and determine next steps.

        Args:
            state: Current query state with context and history

        Returns:
            QueryProgressResponse with readiness status and next subquery
        """
        assert state is not None, "QueryState must be provided"
        assert state.query, "QueryState must have a query"

        # Invariant: Track initial state
        initial_reasoning_steps_count = len(state.reasoning_trace)
        initial_readiness = state.is_ready_to_answer()

        try:
            # First evaluate completeness
            completeness = self._evaluate_completeness(state)

            # Validate response
            assert completeness.status in [
                "READY",
                "EXHAUSTED",
                "NOT_READY",
            ], "Status must be one of: READY, EXHAUSTED, NOT_READY"

            # Update state
            is_ready = (
                completeness.status == "READY" or completeness.status == "EXHAUSTED"
            )
            state.update_readiness(is_ready)

            # Invariant: Readiness should be properly updated
            if is_ready:
                assert (
                    state.is_ready_to_answer()
                ), "State should be marked as ready after update"
            else:
                assert (
                    not state.is_ready_to_answer() or initial_readiness
                ), "State readiness should not change from true to false unless initially ready"

            # Record reasoning based on status
            if completeness.status == "READY":
                state.add_reasoning(
                    f"Evaluatie: Voldoende informatie gevonden om te antwoorden. {completeness.reasoning}"
                )
            elif completeness.status == "EXHAUSTED":
                state.add_reasoning(
                    f"Evaluatie: Alle beschikbare informatie verzameld, maar niet compleet. {completeness.reasoning}"
                )
            else:  # NOT_READY
                state.add_reasoning(
                    f"Evaluatie: Meer informatie nodig. {completeness.reasoning}"
                )

            # Invariant: One reasoning step should be added for status
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for status evaluation"

            # Generate subquery if needed
            subquery = ""
            if completeness.status == "NOT_READY":
                # Generate subquery for missing information
                subquery_gen = self._generate_subquery(state)
                subquery = subquery_gen.subquery

                # Invariant: Subquery generation should add one reasoning step
                assert (
                    len(state.reasoning_trace) == initial_reasoning_steps_count + 2
                ), "Subquery generation should add exactly one more reasoning step"

                # Validate new subquery doesn't repeat previous ones
                if subquery in state.previous_subqueries:
                    logger.warning(
                        "Generated subquery repeats a previous one, adjusting..."
                    )
                    subquery = f"{subquery} (meer specifieke details)"

            # Create final response
            response = QueryProgressResponse(
                status=completeness.status,
                reasoning=completeness.reasoning,
                subquery=subquery,
            )

            # Invariant: Response status should match completeness status
            assert (
                response.status == completeness.status
            ), "Response status should match completeness status"

            # Invariant: Subquery should be empty if status is ready
            if is_ready:
                assert (
                    not response.subquery
                ), "Subquery should be empty when status is READY or EXHAUSTED"

            # Invariant: Reasoning steps count should be consistent with actions taken
            if completeness.status == "NOT_READY":
                assert (
                    len(state.reasoning_trace) == initial_reasoning_steps_count + 2
                ), "Two reasoning steps should be added for NOT_READY status"
            else:
                assert (
                    len(state.reasoning_trace) == initial_reasoning_steps_count + 1
                ), "One reasoning step should be added for READY/EXHAUSTED status"

            return response

        except Exception as e:
            logger.error(f"Error evaluating progress: {str(e)}")

            # Generate safe fallback response
            default_subquery = (
                state.query
                if not state.previous_subqueries
                else f"Meer details over {state.query}"
            )

            state.add_reasoning(
                "Er is een fout opgetreden bij het evalueren van de voortgang."
            )

            # Invariant: At least one reasoning step should be added even in error case
            assert (
                len(state.reasoning_trace) >= initial_reasoning_steps_count + 1
            ), "At least one reasoning step should be added in error case"

            return QueryProgressResponse(
                status="NOT_READY",
                reasoning="Er is een fout opgetreden bij het evalueren van de voortgang.",
                subquery=default_subquery,
            )

    def _evaluate_completeness(self, state: QueryState) -> CompletenessEvaluation:
        """Evaluate if the collected information is sufficient to answer the query.

        Args:
            state: Current query state with context and history

        Returns:
            CompletenessEvaluation with status and reasoning
        """
        assert state is not None, "QueryState must be provided"

        # Invariant: Track initial state
        initial_reasoning_steps_count = len(state.reasoning_trace)

        context_pieces = state.get_all_context_pieces()
        previous_subqueries = state.previous_subqueries

        # Format context for analysis
        if context_pieces:
            formatted_context = "\n".join(
                [
                    f"=== CONTEXTDEEL {i+1} === \n" + str(piece)
                    for (i, piece) in enumerate(context_pieces)
                ]
            )
        else:
            formatted_context = "Er is nog geen context verzameld."

        # Format subquery history
        if previous_subqueries:
            formatted_subqueries = "\n".join(
                [f"- {subq}" for subq in previous_subqueries]
            )
        else:
            formatted_subqueries = "Nog geen eerdere deelvragen gesteld."

        # Construct evaluation prompt
        user_message = f"""
            Oorspronkelijke vraag: {state.query}

            Huidige context:
            {formatted_context}

            Eerder gestelde deelvragen:
            {formatted_subqueries}

            Evalueer of er voldoende informatie is gevonden om de vraag volledig te beantwoorden.
            """

        json_schema = CompletenessEvaluation.model_json_schema()
        extra_body: Dict[str, Any] = {
            "guided_json": json_schema,
            "guided_backend": "xgrammar:disable-any-whitespace",
        }

        try:
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=PROGRESS_COMPLETENESS_SYSTEM_PROMPT,
                extra_body=extra_body,
            )

            log_llm_interaction(
                stage="progress_completeness_evaluation",
                messages=[
                    {"role": "system", "content": PROGRESS_COMPLETENESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response=response,
                extra_body=extra_body,
            )

            result = CompletenessEvaluation.model_validate_json(response)

            # Invariant: Result should have valid status
            assert result.status in [
                "READY",
                "EXHAUSTED",
                "NOT_READY",
            ], "Status must be one of the valid values"
            assert result.reasoning, "Reasoning must not be empty"

            # Invariant: Completeness evaluation should not modify reasoning steps
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count
            ), "Completeness evaluation should not modify reasoning steps"

            return result

        except Exception as e:
            logger.error(f"Error evaluating completeness: {str(e)}")
            return CompletenessEvaluation(
                status="NOT_READY",
                reasoning="Er is een fout opgetreden bij het evalueren van de volledigheid.",
            )

    def _generate_subquery(self, state: QueryState) -> SubqueryGeneration:
        """Generate a targeted subquery to find missing information.

        Args:
            state: Current query state with context and history

        Returns:
            SubqueryGeneration with subquery and explanation
        """
        assert state is not None, "QueryState must be provided"

        # Invariant: Track initial state
        initial_reasoning_steps_count = len(state.reasoning_trace)

        context_pieces = state.get_all_context_pieces()
        previous_subqueries = state.previous_subqueries

        # Format context for analysis
        if context_pieces:
            formatted_context = "\n".join(
                [
                    f"=== CONTEXTDEEL {i+1} === \n" + str(piece)
                    for (i, piece) in enumerate(context_pieces)
                ]
            )
        else:
            formatted_context = "Er is nog geen context verzameld."

        # Format subquery history
        if previous_subqueries:
            formatted_subqueries = "\n".join(
                [f"- {subq}" for subq in previous_subqueries]
            )
        else:
            formatted_subqueries = "Nog geen eerdere deelvragen gesteld."

        # Construct subquery prompt
        user_message = f"""
            Oorspronkelijke vraag: {state.query}

            Huidige context:
            {formatted_context}

            Eerder gestelde deelvragen:
            {formatted_subqueries}

            Genereer een gerichte deelvraag om de ontbrekende informatie te vinden.
            """

        json_schema = SubqueryGeneration.model_json_schema()
        extra_body: Dict[str, Any] = {
            "guided_json": json_schema,
            "guided_backend": "xgrammar:disable-any-whitespace",
        }

        try:
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=SUBQUERY_GENERATION_SYSTEM_PROMPT,
                extra_body=extra_body,
            )

            log_llm_interaction(
                stage="subquery_generation",
                messages=[
                    {"role": "system", "content": SUBQUERY_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response=response,
                extra_body=extra_body,
            )

            result = SubqueryGeneration.model_validate_json(response)

            # Invariant: Result should have valid subquery
            assert result.subquery, "Generated subquery must not be empty"
            assert result.explanation, "Subquery explanation must not be empty"

            logger.info(f"Generated next subquery: {result.subquery}")
            state.add_reasoning(f"Volgende deelvraag gegenereerd: {result.subquery}")

            # Invariant: Exactly one reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for subquery generation"

            return result

        except Exception as e:
            logger.error(f"Error generating subquery: {str(e)}")

            # Generate safe fallback subquery
            default_subquery = (
                state.query
                if not state.previous_subqueries
                else f"Meer details over {state.query}"
            )

            state.add_reasoning(
                f"Fout bij het genereren van een deelvraag. Gebruik fallback: {default_subquery}"
            )

            # Invariant: Exactly one reasoning step should be added even in error case
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for error case"

            return SubqueryGeneration(
                subquery=default_subquery,
                explanation="Er is een fout opgetreden bij het genereren van een deelvraag.",
            )
