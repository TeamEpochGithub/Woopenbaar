"""Unit tests for the ProgressEvaluator component."""

import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from backend.conf.prompts import PROGRESS_EVALUATION_SYSTEM_PROMPT
from backend.src.data_classes import ChunkID, ContextPiece
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.components.progress_evaluator import (
    ProgressEvaluator,
    QueryProgressResponse,
)
from backend.src.services.query_processing.query_state import QueryState


class TestProgressEvaluator(unittest.TestCase):
    """Test cases for the ProgressEvaluator component."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a mock LLM service
        self.mock_llm_service = Mock(spec=BaseLLMService)

        # Initialize the component with the mock service
        self.evaluator = ProgressEvaluator(self.mock_llm_service)

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

    def test_evaluate_progress_ready(self) -> None:
        """Test evaluation when enough information is found."""
        # Add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)
        self.query_state.add_context_piece(self.context_piece2)

        # Set up mock response indicating ready to answer
        mock_response = {
            "status": "READY",
            "reasoning": "Found sufficient information",
            "subquery": "",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify result
        self.assertEqual(result.status, "READY")
        self.assertTrue(result.ready_to_answer)
        self.assertEqual(result.reasoning, mock_response["reasoning"])

        # Verify state was updated
        self.assertTrue(self.query_state.is_ready_to_answer())
        self.assertIn(
            "Voldoende informatie gevonden", self.query_state.reasoning_trace[-1]
        )

        # Verify LLM service was called with correct parameters
        self.mock_llm_service.generate_response.assert_called_once()
        call_args = self.mock_llm_service.generate_response.call_args[1]

        self.assertIn("user_message", call_args)
        self.assertIn("system_prompt", call_args)
        self.assertEqual(call_args["system_prompt"], PROGRESS_EVALUATION_SYSTEM_PROMPT)
        self.assertIn("extra_body", call_args)

    def test_evaluate_progress_exhausted(self) -> None:
        """Test evaluation when all available information is gathered but incomplete."""
        # Add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up mock response indicating exhausted
        mock_response = {
            "status": "EXHAUSTED",
            "reasoning": "All available information gathered",
            "subquery": "",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify result
        self.assertEqual(result.status, "EXHAUSTED")
        self.assertTrue(result.ready_to_answer)
        self.assertEqual(result.reasoning, mock_response["reasoning"])

        # Verify state was updated
        self.assertTrue(self.query_state.is_ready_to_answer())
        self.assertIn(
            "Alle beschikbare informatie verzameld",
            self.query_state.reasoning_trace[-1],
        )

    def test_evaluate_progress_not_ready(self) -> None:
        """Test evaluation when more information is needed."""
        # Add context pieces to state
        self.query_state.add_context_piece(self.context_piece1)

        # Set up mock response indicating not ready
        mock_response = {
            "status": "NOT_READY",
            "reasoning": "Need more specific information",
            "subquery": "What are the details of X?",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify result
        self.assertEqual(result.status, "NOT_READY")
        self.assertFalse(result.ready_to_answer)
        self.assertEqual(result.reasoning, mock_response["reasoning"])
        self.assertEqual(result.subquery, mock_response["subquery"])

        # Verify state was updated
        self.assertFalse(self.query_state.is_ready_to_answer())
        self.assertIn("Meer informatie nodig", self.query_state.reasoning_trace[-1])

    def test_evaluate_progress_no_context(self) -> None:
        """Test evaluation with no context pieces."""
        # Set up mock response
        mock_response = {
            "status": "NOT_READY",
            "reasoning": "No information found yet",
            "subquery": "Initial search query",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify result
        self.assertEqual(result.status, "NOT_READY")
        self.assertFalse(result.ready_to_answer)

        # Verify LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()

        # Verify user message indicates no context
        call_args = self.mock_llm_service.generate_response.call_args[1]
        user_message = call_args["user_message"]
        self.assertIn("Er is nog geen context verzameld", user_message)

    def test_evaluate_progress_with_previous_subqueries(self) -> None:
        """Test evaluation with previous subqueries."""
        # Add a previous subquery
        self.query_state.add_prev_subquery("Previous test query")

        # Set up mock response
        mock_response = {
            "status": "NOT_READY",
            "reasoning": "Need more information",
            "subquery": "New test query",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify result
        self.assertEqual(result.subquery, mock_response["subquery"])

        # Verify previous subqueries were included in user message
        call_args = self.mock_llm_service.generate_response.call_args[1]
        user_message = call_args["user_message"]
        self.assertIn("Previous test query", user_message)

    @patch(
        "backend.src.services.query_processing.components.progress_evaluator.log_llm_interaction"
    )
    def test_evaluate_progress_logs_interaction(self, mock_log: Mock) -> None:
        """Test that LLM interactions are logged."""
        # Set up mock response
        mock_response = {
            "status": "READY",
            "reasoning": "Found sufficient information",
            "subquery": "",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        self.evaluator.evaluate_progress(self.query_state)

        # Verify logging was called
        mock_log.assert_called_once()
        log_args = mock_log.call_args[1]

        self.assertEqual(log_args["stage"], "progress_evaluation")
        self.assertIn("messages", log_args)
        self.assertIn("response", log_args)
        self.assertIn("extra_body", log_args)

    def test_evaluate_progress_llm_error(self) -> None:
        """Test handling LLM service errors during evaluation."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify default response is returned
        self.assertEqual(result.status, "NOT_READY")
        self.assertFalse(result.ready_to_answer)
        self.assertIn("fout opgetreden", result.reasoning)
        self.assertEqual(result.subquery, "Test query")  # Should use original query

        # Verify state was not marked as ready
        self.assertFalse(self.query_state.is_ready_to_answer())

        # Verify LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()

    @patch(
        "backend.src.services.query_processing.components.progress_evaluator.log_llm_interaction"
    )
    def test_evaluate_progress_logs_interaction_on_error(self, mock_log: Mock) -> None:
        """Test that LLM interactions are not logged when an error occurs."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        self.evaluator.evaluate_progress(self.query_state)

        # Verify logging was not called since error occurred before response was generated
        mock_log.assert_not_called()

    def test_evaluate_progress_invalid_response(self) -> None:
        """Test handling invalid LLM responses."""
        # Set up mock to return invalid JSON
        self.mock_llm_service.generate_response.return_value = "invalid json"

        # Call the method
        result = self.evaluator.evaluate_progress(self.query_state)

        # Verify default response is returned
        self.assertEqual(result.status, "NOT_READY")
        self.assertFalse(result.ready_to_answer)
        self.assertIn("fout opgetreden", result.reasoning)

        # Verify LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()

    def test_evaluate_progress_with_reasoning_trace(self) -> None:
        """Test evaluation with existing reasoning trace."""
        # Add some reasoning steps
        self.query_state.add_reasoning("Previous reasoning step 1")
        self.query_state.add_reasoning("Previous reasoning step 2")

        # Set up mock response
        mock_response = {
            "status": "NOT_READY",
            "reasoning": "Need more information",
            "subquery": "New query",
        }
        self.mock_llm_service.generate_response.return_value = QueryProgressResponse(
            **mock_response
        ).model_dump_json()

        # Call the method
        self.evaluator.evaluate_progress(self.query_state)

        # Verify previous reasoning was included in user message
        call_args = self.mock_llm_service.generate_response.call_args[1]
        user_message = call_args["user_message"]
        self.assertIn("Previous reasoning step 1", user_message)
        self.assertIn("Previous reasoning step 2", user_message)


if __name__ == "__main__":
    unittest.main()
