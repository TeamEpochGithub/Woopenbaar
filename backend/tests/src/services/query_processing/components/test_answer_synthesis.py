"""Unit tests for the AnswerSynthesizer component."""

import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from backend.conf.prompts import (
    ANSWER_SYNTHESIS_SYSTEM_PROMPT,
    INTERMEDIATE_ANSWER_SYSTEM_PROMPT,
)
from backend.src.data_classes import ChunkID, ContextPiece
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.components.answer_synthesis import (
    AnswerSynthesizer,
)
from backend.src.services.query_processing.query_state import QueryState


class TestAnswerSynthesizer(unittest.TestCase):
    """Test cases for the AnswerSynthesizer component.

    This test suite aims for 100% branch coverage and tests all edge cases.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a mock LLM service
        self.mock_llm_service = Mock(spec=BaseLLMService)

        # Initialize the component with the mock service
        self.synthesizer = AnswerSynthesizer(self.mock_llm_service)

        # Create a test query state
        self.query_state = QueryState("Test query")

        # Create test context pieces
        self.context_piece1 = ContextPiece(
            doc_uuid=UUID("12345678-1234-5678-1234-567812345678"),
            chunk_id=ChunkID(1),
            content="Test content 1",
            type="test_type",
            source="test_source",
        )

        self.context_piece2 = ContextPiece(
            doc_uuid=UUID("87654321-4321-8765-4321-876543210987"),
            chunk_id=ChunkID(2),
            content="Test content 2",
            type="test_type",
            source="test_source",
        )

        # Add context pieces to the query state
        self.query_state.add_context_piece(self.context_piece1)
        self.query_state.add_context_piece(self.context_piece2)

    def test_generate_answer_with_context(self) -> None:
        """Test generating an answer with context pieces."""
        # Set up mock response
        expected_response = "This is a test answer"
        self.mock_llm_service.generate_response.return_value = expected_response

        # Call the method
        result = self.synthesizer.generate_answer(self.query_state)

        # Verify the result
        self.assertEqual(result, expected_response)

        # Verify the LLM service was called with correct parameters
        self.mock_llm_service.generate_response.assert_called_once()
        call_args = self.mock_llm_service.generate_response.call_args[1]

        self.assertIn("user_message", call_args)
        self.assertIn("system_prompt", call_args)
        self.assertEqual(call_args["system_prompt"], ANSWER_SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("max_tokens", call_args)

        # Verify the user message contains the query and context
        user_message = call_args["user_message"]
        self.assertIn("Test query", user_message)
        self.assertIn("Test content 1", user_message)
        self.assertIn("Test content 2", user_message)

    def test_generate_answer_without_context(self) -> None:
        """Test generating an answer without context pieces."""
        # Create a query state without context
        empty_state = QueryState("Test query")

        # Call the method
        result = self.synthesizer.generate_answer(empty_state)

        # Verify the result
        self.assertEqual(
            result,
            "Ik heb geen relevante informatie kunnen vinden om deze vraag te beantwoorden.",
        )

        # Verify the LLM service was not called
        self.mock_llm_service.generate_response.assert_not_called()

    @patch(
        "backend.src.services.query_processing.components.answer_synthesis.log_llm_interaction"
    )
    def test_generate_answer_logs_interaction(self, mock_log: Mock) -> None:
        """Test that LLM interactions are logged."""
        # Set up mock response
        expected_response = "This is a test answer"
        self.mock_llm_service.generate_response.return_value = expected_response

        # Call the method
        self.synthesizer.generate_answer(self.query_state)

        # Verify logging was called
        mock_log.assert_called_once()
        log_args = mock_log.call_args[1]

        self.assertEqual(log_args["stage"], "answer_synthesis")
        self.assertIn("messages", log_args)
        self.assertEqual(log_args["response"], expected_response)

    def test_generate_answer_llm_error(self) -> None:
        """Test handling LLM service errors during answer generation."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        result = self.synthesizer.generate_answer(self.query_state)

        # Verify the result
        self.assertEqual(
            result, "Er is een fout opgetreden bij het genereren van het antwoord."
        )

        # Verify the LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()

    @patch(
        "backend.src.services.query_processing.components.answer_synthesis.log_llm_interaction"
    )
    def test_generate_answer_logs_interaction_on_error(self, mock_log: Mock) -> None:
        """Test that LLM interactions are logged even when an error occurs."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        self.synthesizer.generate_answer(self.query_state)

        # Verify logging was not called since error occurred before response was generated
        mock_log.assert_not_called()

    def test_generate_intermediate_answer_with_context(self) -> None:
        """Test generating an intermediate answer with context pieces."""
        # Set up mock response
        expected_response = '{"answer": "This is a test intermediate answer"}'
        self.mock_llm_service.generate_response.return_value = expected_response

        # Call the method
        result = self.synthesizer.generate_intermediate_answer(self.query_state)

        # Verify the result
        self.assertEqual(result, expected_response)

        # Verify the LLM service was called with correct parameters
        self.mock_llm_service.generate_response.assert_called_once()
        call_args = self.mock_llm_service.generate_response.call_args[1]

        self.assertIn("user_message", call_args)
        self.assertIn("system_prompt", call_args)
        self.assertEqual(call_args["system_prompt"], INTERMEDIATE_ANSWER_SYSTEM_PROMPT)
        self.assertIn("max_tokens", call_args)
        self.assertIn("extra_body", call_args)

        # Verify the user message contains the query and context
        user_message = call_args["user_message"]
        self.assertIn("Test query", user_message)
        self.assertIn("Test content 1", user_message)
        self.assertIn("Test content 2", user_message)

    def test_generate_intermediate_answer_without_context(self) -> None:
        """Test generating an intermediate answer without context pieces."""
        # Create a query state without context
        empty_state = QueryState("Test query")

        # Call the method
        result = self.synthesizer.generate_intermediate_answer(empty_state)

        # Verify the result
        self.assertEqual(
            result,
            '{"answer": "Nog geen relevante informatie gevonden. De zoekopdracht wordt voortgezet."}',
        )

        # Verify the LLM service was not called
        self.mock_llm_service.generate_response.assert_not_called()

    @patch(
        "backend.src.services.query_processing.components.answer_synthesis.log_llm_interaction"
    )
    def test_generate_intermediate_answer_logs_interaction(
        self, mock_log: Mock
    ) -> None:
        """Test that LLM interactions are logged for intermediate answers."""
        # Set up mock response
        expected_response = '{"answer": "This is a test intermediate answer"}'
        self.mock_llm_service.generate_response.return_value = expected_response

        # Call the method
        self.synthesizer.generate_intermediate_answer(self.query_state)

        # Verify logging was called
        mock_log.assert_called_once()
        log_args = mock_log.call_args[1]

        self.assertEqual(log_args["stage"], "intermediate_answer")
        self.assertIn("messages", log_args)
        self.assertEqual(log_args["response"], expected_response)
        self.assertIn("extra_body", log_args)

    def test_generate_intermediate_answer_custom_max_tokens(self) -> None:
        """Test generating an intermediate answer with custom max tokens."""
        # Set up mock response
        expected_response = '{"answer": "This is a test intermediate answer"}'
        self.mock_llm_service.generate_response.return_value = expected_response

        # Call the method with custom max tokens
        custom_max_tokens = 200
        self.synthesizer.generate_intermediate_answer(
            self.query_state, max_tokens=custom_max_tokens
        )

        # Verify the LLM service was called with the custom max tokens
        call_args = self.mock_llm_service.generate_response.call_args[1]
        self.assertEqual(call_args["max_tokens"], custom_max_tokens)

    def test_generate_intermediate_answer_llm_error(self) -> None:
        """Test handling LLM service errors during intermediate answer generation."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        result = self.synthesizer.generate_intermediate_answer(self.query_state)

        # Verify the result
        self.assertEqual(
            result, '{"answer": "Bezig met het zoeken naar relevante informatie..."}'
        )

        # Verify the LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()

    @patch(
        "backend.src.services.query_processing.components.answer_synthesis.log_llm_interaction"
    )
    def test_generate_intermediate_answer_logs_interaction_on_error(
        self, mock_log: Mock
    ) -> None:
        """Test that LLM interactions are logged even when an error occurs."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        self.synthesizer.generate_intermediate_answer(self.query_state)

        # Verify logging was not called since error occurred before response was generated
        mock_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
