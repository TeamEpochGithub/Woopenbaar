"""Component for synthesizing coherent answers from retrieved information.

Uses LLM-based synthesis for generating comprehensive answers and progress updates.
Handles source attribution and error cases.
"""

import logging
from typing import Any, Dict, List

from backend.conf.prompts import (
    ANSWER_SYNTHESIS_SYSTEM_PROMPT,
    INTERMEDIATE_ANSWER_SYSTEM_PROMPT,
)
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.query_processing.utils import log_llm_interaction

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """Synthesizes coherent answers from retrieved information using LLM analysis.

    Generates comprehensive answers and progress updates while ensuring proper
    source attribution and error handling.

    Attributes:
        llm_service: Service for LLM-based answer synthesis
    """

    def __init__(self, llm_service: BaseLLMService) -> None:
        """Initialize the answer synthesizer.

        Args:
            llm_service: Service for LLM-based answer synthesis

        Raises:
            AssertionError: If LLM service is None
        """
        assert llm_service is not None, "LLM service must be provided"

        self.llm_service = llm_service
        logger.info("AnswerSynthesizer initialized")

    def generate_answer(self, state: QueryState) -> str:
        """Generate a comprehensive final answer from retrieved information.

        Synthesizes available context into a coherent answer with source citations.
        Presents the answer as if drawn from the complete VWS document database.

        Args:
            state: Query state with original query and retrieved context

        Returns:
            str: Comprehensive answer with source citations, or default message if
                 no context or generation fails
        """
        assert state is not None, "QueryState must be provided"
        assert state.query, "Query must not be empty"

        # Invariant: State should be ready to answer
        assert (
            state.is_ready_to_answer()
        ), "State must be ready to answer before generating final answer"

        # Invariant: Track initial reasoning steps
        initial_reasoning_steps_count = len(state.reasoning_trace)

        context_pieces = state.get_all_context_pieces()
        if not context_pieces:
            logger.warning("No context pieces found to generate answer")
            no_info_response = "Ik heb geen relevante informatie kunnen vinden in de VWS-documentatie om deze vraag te beantwoorden."
            state.add_reasoning(f"Genereerd antwoord: {no_info_response}")

            # Invariant: One reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for empty context case"

            return no_info_response

        logger.info(
            f"Generating answer for query: '{state.query}' with {len(context_pieces)} context pieces"
        )

        formatted_context = "\n".join(
            [
                f"=== CONTEXTDEEL {i+1} === \n" + str(piece)
                for (i, piece) in enumerate(context_pieces)
            ]
        )

        user_message = f"""
            Oorspronkelijke vraag: {state.query}

            Relevante informatie uit VWS-documentatie:
            {formatted_context}

            Genereer een volledig en nauwkeurig antwoord op de vraag gebaseerd op de VWS-documentatie.

            BELANGRIJK: Citeer alle gebruikte informatie met exact deze notatie: [1], [2], [3], etc. direct na elke zin waar je de informatie gebruikt. Begin met [1] en gebruik opeenvolgende nummers.
            """

        try:
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=ANSWER_SYNTHESIS_SYSTEM_PROMPT,
                max_tokens=1000,
            )

            log_llm_interaction(
                stage="answer_synthesis",
                messages=[
                    {"role": "system", "content": ANSWER_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response=response,
            )

            # Invariant: Response should never be empty
            assert response.strip(), "Generated answer must not be empty"

            state.add_reasoning(f"Genereerd antwoord: {response.strip()}")

            # Invariant: Exactly one reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for answer generation"

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            error_msg = "Er is een fout opgetreden bij het genereren van het antwoord."
            state.add_reasoning(f"Fout bij het genereren van antwoord: {error_msg}")

            # Invariant: Exactly one reasoning step should be added even in error case
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for error case"

            return error_msg

    def generate_intermediate_answer(
        self, state: QueryState, max_tokens: int = 150
    ) -> str:
        """Generate a conversational progress update for the user.

        Creates engaging, very short (1-2 sentences) updates during query processing,
        with logical flow based on previous updates.

        Args:
            state: Query state containing current context and progress
            max_tokens: Maximum length of generated answer

        Returns:
            str: JSON-formatted intermediate answer with a conversational progress update
        """
        assert state is not None, "QueryState must be provided"
        assert max_tokens > 0, "Max tokens must be positive"

        # Invariant: Track initial reasoning steps
        initial_reasoning_steps_count = len(state.reasoning_trace)

        # Invariant: We shouldn't generate intermediate answers for completed queries
        assert (
            not state.is_complete()
        ), "Intermediate answers should not be generated for completed queries"

        context_pieces = state.get_all_context_pieces()

        # Create a more engaging progress update when no info is found
        if not context_pieces:
            no_info_response = '{"answer": "Zoeken naar relevante informatie."}'
            state.add_reasoning(f"Genereerd tussentijds antwoord: {no_info_response}")

            # Invariant: Exactly one reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for empty context case"

            return no_info_response

        # Extract previous intermediate answers from reasoning trace
        previous_answers: List[str] = []
        for step in state.reasoning_trace:
            if step.startswith("Genereerd tussentijds antwoord:"):
                try:
                    # Extract the answer part from the JSON
                    import json

                    json_start = step.find('{"answer":')
                    if json_start != -1:
                        json_str = step[json_start:]
                        answer_data = json.loads(json_str)
                        previous_answers.append(answer_data["answer"])
                except Exception:
                    # If parsing fails, just use the whole step
                    previous_answers.append(step)

        previous_answer: str = previous_answers[-1] if previous_answers else ""

        # Sample some context for the progress update (don't overwhelm the LLM)
        sample_size = min(2, len(context_pieces))
        sample_context = context_pieces[
            -sample_size:
        ]  # Use the most recent context pieces
        formatted_context = "\n\n".join([str(piece) for piece in sample_context])

        user_message = f"""
            Oorspronkelijke vraag: {state.query}

            Gevonden informatie (recente voorbeelden):
            {formatted_context}

            Vorige update was:
            {previous_answer}

            Geef een ULTRAKORTE (1-2 zinnen) samenvatting van de INHOUD van wat er tot nu toe is gevonden.
            Focus alleen op de feiten en informatie, niet op het zoekproces.
            """

        extra_body: Dict[str, Any] = {
            "guided_backend": "xgrammar:disable-any-whitespace"
        }

        try:
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=INTERMEDIATE_ANSWER_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )

            log_llm_interaction(
                stage="intermediate_answer",
                messages=[
                    {"role": "system", "content": INTERMEDIATE_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response=response,
                extra_body=extra_body,
            )

            # Invariant: Response should never be empty
            assert response, "Intermediate answer must not be empty"

            state.add_reasoning(f"Genereerd tussentijds antwoord: {response}")

            # Invariant: Exactly one reasoning step should be added
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for intermediate answer generation"

            return response

        except Exception as e:
            logger.error(f"Error generating intermediate answer: {str(e)}")
            error_msg = "Zoeken naar informatie."
            state.add_reasoning(
                f"Fout bij het genereren van tussentijds antwoord: {error_msg}"
            )

            # Invariant: Exactly one reasoning step should be added even in error case
            assert (
                len(state.reasoning_trace) == initial_reasoning_steps_count + 1
            ), "Exactly one reasoning step should be added for error case"

            return error_msg
