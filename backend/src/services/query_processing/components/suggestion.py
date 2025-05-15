"""Component for generating intelligent follow-up suggestions.

Uses LLM-based analysis to generate relevant and safe follow-up questions
with contextual explanations.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from backend.conf.prompts import SUGGESTION_SYSTEM_PROMPT
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.query_processing.utils import log_llm_interaction
from backend.src.services.safety.safety_service import SafetyService

logger = logging.getLogger(__name__)


class SuggestionResponse(BaseModel):
    """Response model for LLM-based follow-up suggestion generation.

    Structures and validates follow-up questions with their explanations.

    Attributes:
        suggestions: List of 2-3 contextually relevant follow-up questions
        explanation: Reasoning about the suggestions' relevance
    """

    suggestions: List[str] = Field(description="Lijst met vervolgvragen")
    explanation: str = Field(
        description="Uitleg waarom deze vervolgvragen relevant zijn"
    )


class SuggestionGenerator:
    """Generates intelligent follow-up suggestions using LLM analysis.

    Creates contextually relevant and safe follow-up questions based on
    query state and retrieved information.

    Attributes:
        llm_service: Service for LLM-based suggestion generation
        safety_service: Service for content safety validation
    """

    def __init__(
        self, llm_service: BaseLLMService, safety_service: SafetyService
    ) -> None:
        """Initialize the suggestion generator.

        Args:
            llm_service: Service for LLM-based suggestion generation
            safety_service: Service for content safety validation

        Raises:
            AssertionError: If either required service is None
        """
        assert llm_service is not None, "LLM service must be provided"
        assert safety_service is not None, "Safety service must be provided"

        self.llm_service = llm_service
        self.safety_service = safety_service
        logger.info("SuggestionGenerator initialized")

    def generate_suggestions(self, state: QueryState) -> SuggestionResponse:
        """Generate safe and relevant follow-up suggestions.

        Creates contextually appropriate follow-up questions based on the current
        query state and retrieved information, applying safety validation.

        Args:
            state: Query state with original query and retrieved context

        Returns:
            SuggestionResponse: Safe follow-up questions with explanation,
                              or empty list if generation fails
        """
        context_pieces = state.get_all_context_pieces()

        if not context_pieces:
            return SuggestionResponse(
                suggestions=[],
                explanation="Geen context beschikbaar om vervolgvragen te genereren.",
            )

        formatted_context = "\n".join(
            [
                f"=== CONTEXTDEEL {i+1} === \n" + str(piece)
                for (i, piece) in enumerate(context_pieces)
            ]
        )

        user_message = f"""
Oorspronkelijke vraag: {state.query}

Huidige context:
{formatted_context}

Genereer 2-3 relevante vervolgvragen die de gebruiker zou kunnen stellen op basis van de huidige context.
"""

        json_schema = SuggestionResponse.model_json_schema()
        extra_body: Dict[str, Any] = {
            "guided_json": json_schema,
            "guided_backend": "xgrammar:disable-any-whitespace",
        }

        try:
            response = self.llm_service.generate_response(
                user_message=user_message,
                system_prompt=SUGGESTION_SYSTEM_PROMPT,
                extra_body=extra_body,
            )

            log_llm_interaction(
                stage="suggestion_generation",
                messages=[
                    {"role": "system", "content": SUGGESTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response=response,
                extra_body=extra_body,
            )

            result = SuggestionResponse.model_validate_json(response)

            safe_suggestions: List[str] = []
            for suggestion in result.suggestions:
                if self.safety_service.is_text_safe(suggestion):
                    safe_suggestions.append(suggestion)
                else:
                    logger.warning(
                        f"Suggestion filtered out by safety service: {suggestion}"
                    )

            explanation = result.explanation
            if len(safe_suggestions) < len(result.suggestions):
                explanation = (
                    "Sommige vervolgvragen zijn gefilterd op basis van veiligheidscontroles. "
                    + explanation
                )

            return SuggestionResponse(
                suggestions=safe_suggestions, explanation=explanation
            )

        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return SuggestionResponse(
                suggestions=[],
                explanation="Er is een fout opgetreden bij het genereren van vervolgvragen.",
            )
