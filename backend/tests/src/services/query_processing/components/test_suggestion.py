"""Unit tests for the SuggestionGenerator component."""

import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from backend.src.data_classes import ChunkID, ContextPiece
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.components.suggestion import (
    SuggestionGenerator,
    SuggestionResponse,
)
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.safety.safety_service import SafetyService


class TestSuggestionGenerator(unittest.TestCase):
    """Test cases for the SuggestionGenerator component.

    This test suite aims for 100% branch coverage and tests all edge cases.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create mock services
        self.mock_llm_service = Mock(spec=BaseLLMService)
        self.mock_safety_service = Mock(spec=SafetyService)

        # Initialize the component with the mock services
        self.suggestion_generator = SuggestionGenerator(
            llm_service=self.mock_llm_service, safety_service=self.mock_safety_service
        )

        # Create a test query state
        self.query_state = QueryState("What is the policy on remote work?")

        # Create test context pieces
        self.context_piece1 = ContextPiece(
            doc_uuid=UUID("12345678-1234-5678-1234-567812345678"),
            chunk_id=ChunkID(1),
            content="Employees can work remotely up to 3 days per week.",
            type="policy",
            source="HR policies",
        )

        self.context_piece2 = ContextPiece(
            doc_uuid=UUID("87654321-4321-8765-4321-876543210987"),
            chunk_id=ChunkID(2),
            content="Remote work requests must be approved by managers.",
            type="policy",
            source="HR policies",
        )

    def test_generate_suggestions_no_context(self) -> None:
        """Test generation of suggestions when there's no context."""
        # Execute
        result = self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert
        self.assertIsInstance(result, SuggestionResponse)
        self.assertEqual(len(result.suggestions), 0)
        self.assertIn("Geen context beschikbaar", result.explanation)

        # Verify services were not called
        self.mock_llm_service.generate_response.assert_not_called()
        self.mock_safety_service.is_text_safe.assert_not_called()

    def test_generate_suggestions_with_context_all_safe(self) -> None:
        """Test successful suggestion generation with all suggestions passing safety check."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)
        self.query_state.add_context_piece(self.context_piece2)

        # Set up LLM response
        expected_suggestions = [
            "What is the approval process for remote work?",
            "Are there any restrictions on which days can be worked remotely?",
        ]

        mock_response = SuggestionResponse(
            suggestions=expected_suggestions,
            explanation="These suggestions explore related aspects of the remote work policy.",
        )
        self.mock_llm_service.generate_response.return_value = (
            mock_response.model_dump_json()
        )

        # Configure safety service to approve all suggestions
        self.mock_safety_service.is_text_safe.return_value = True

        # Execute
        result = self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert
        self.assertIsInstance(result, SuggestionResponse)
        self.assertEqual(result.suggestions, expected_suggestions)
        self.assertEqual(result.explanation, mock_response.explanation)

        # Verify safety service was called for each suggestion
        self.assertEqual(
            self.mock_safety_service.is_text_safe.call_count, len(expected_suggestions)
        )

    def test_generate_suggestions_with_unsafe_suggestions(self) -> None:
        """Test suggestion generation where some suggestions fail safety check."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up LLM response with mix of safe and unsafe suggestions
        suggestions = ["Safe suggestion 1", "Unsafe suggestion", "Safe suggestion 2"]
        mock_response = SuggestionResponse(
            suggestions=suggestions, explanation="Test explanation"
        )
        self.mock_llm_service.generate_response.return_value = (
            mock_response.model_dump_json()
        )

        # Configure safety service to reject the middle suggestion
        def mock_is_text_safe(text: str) -> bool:
            return text != "Unsafe suggestion"

        self.mock_safety_service.is_text_safe.side_effect = mock_is_text_safe

        # Execute
        result = self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert
        self.assertEqual(len(result.suggestions), 2)
        self.assertNotIn("Unsafe suggestion", result.suggestions)
        self.assertIn("Safe suggestion 1", result.suggestions)
        self.assertIn("Safe suggestion 2", result.suggestions)
        self.assertIn("gefilterd op basis van veiligheidscontroles", result.explanation)

        # Verify safety service was called for each suggestion
        self.assertEqual(self.mock_safety_service.is_text_safe.call_count, 3)

    def test_generate_suggestions_all_unsafe(self) -> None:
        """Test suggestion generation where all suggestions fail safety check."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up LLM response
        suggestions = ["Unsafe 1", "Unsafe 2"]
        mock_response = SuggestionResponse(
            suggestions=suggestions, explanation="Test explanation"
        )
        self.mock_llm_service.generate_response.return_value = (
            mock_response.model_dump_json()
        )

        # Configure safety service to reject all suggestions
        self.mock_safety_service.is_text_safe.return_value = False

        # Execute
        result = self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert
        self.assertEqual(len(result.suggestions), 0)
        self.assertIn("gefilterd op basis van veiligheidscontroles", result.explanation)

        # Verify safety service was called for each suggestion
        self.assertEqual(self.mock_safety_service.is_text_safe.call_count, 2)

    def test_generate_suggestions_llm_error(self) -> None:
        """Test handling of LLM service errors during suggestion generation."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up LLM to raise exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Execute
        result = self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert - should return empty suggestions with error explanation
        self.assertIsInstance(result, SuggestionResponse)
        self.assertEqual(len(result.suggestions), 0)
        self.assertIn("fout opgetreden", result.explanation)

        # Verify services were called appropriately
        self.mock_llm_service.generate_response.assert_called_once()
        self.mock_safety_service.is_text_safe.assert_not_called()

    def test_generate_suggestions_invalid_response(self) -> None:
        """Test handling of invalid LLM responses."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up LLM to return invalid JSON
        self.mock_llm_service.generate_response.return_value = "invalid json"

        # Execute
        result = self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert - should return empty suggestions with error explanation
        self.assertIsInstance(result, SuggestionResponse)
        self.assertEqual(len(result.suggestions), 0)
        self.assertIn("fout opgetreden", result.explanation)

        # Verify services were called appropriately
        self.mock_llm_service.generate_response.assert_called_once()
        self.mock_safety_service.is_text_safe.assert_not_called()

    @patch(
        "backend.src.services.query_processing.components.suggestion.log_llm_interaction"
    )
    def test_generate_suggestions_logs_interaction(self, mock_log: Mock) -> None:
        """Test that LLM interactions are logged during suggestion generation."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up LLM response
        mock_response = SuggestionResponse(
            suggestions=["Test suggestion"], explanation="Test explanation"
        )
        self.mock_llm_service.generate_response.return_value = (
            mock_response.model_dump_json()
        )

        # Configure safety service
        self.mock_safety_service.is_text_safe.return_value = True

        # Execute
        self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert - verify logging
        mock_log.assert_called_once()
        log_args = mock_log.call_args[1]

        self.assertEqual(log_args["stage"], "suggestion_generation")
        self.assertIn("messages", log_args)
        self.assertIn("response", log_args)
        self.assertIn("extra_body", log_args)

    @patch(
        "backend.src.services.query_processing.components.suggestion.log_llm_interaction"
    )
    def test_generate_suggestions_logs_interaction_on_error(
        self, mock_log: Mock
    ) -> None:
        """Test that LLM interactions are not logged when an error occurs."""
        # Setup - add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up LLM to raise exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Execute
        self.suggestion_generator.generate_suggestions(self.query_state)

        # Assert - verify logging was not called
        mock_log.assert_not_called()
        self.mock_safety_service.is_text_safe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
