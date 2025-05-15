"""Unit tests for the ChunkEvaluator component."""

import json
import unittest
from datetime import datetime
from typing import Dict, List
from unittest.mock import Mock, patch
from uuid import UUID

from backend.src.data_classes import ChunkedDocument, ChunkID
from backend.src.data_classes.document_chunk import DocumentChunk
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.components.chunk_evaluator import (
    ChunkEvaluator,
)


class TestChunkEvaluator(unittest.TestCase):
    """Test cases for the ChunkEvaluator component."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a mock LLM service
        self.mock_llm_service = Mock(spec=BaseLLMService)

        # Initialize the component with the mock service
        self.evaluator = ChunkEvaluator(self.mock_llm_service)

        # Create temporary parent document first (will be updated with chunks later)
        self.mock_parent = ChunkedDocument(
            uuid=UUID("99999999-9999-9999-9999-999999999999"),
            content="Full document content",
            type="test_type",
            link="test_link",
            vws_id="test_vws_id",
            create_date=datetime(2024, 1, 1),
            attachment_links=[],
            chunks={},  # Empty initially, will be populated
        )

        # Create test chunks with the parent document
        self.test_chunks: List[DocumentChunk] = [
            DocumentChunk(
                uuid=ChunkID(1),
                content="Test content 1",
                type="test_type",
                link="test_link",
                parent_document=self.mock_parent,
            ),
            DocumentChunk(
                uuid=ChunkID(2),
                content="Test content 2",
                type="test_type",
                link="test_link",
                parent_document=self.mock_parent,
            ),
            DocumentChunk(
                uuid=ChunkID(3),
                content="Test content 3",
                type="test_type",
                link="test_link",
                parent_document=self.mock_parent,
            ),
        ]

        # Create chunks dictionary and update parent
        self.chunks_dict: Dict[ChunkID, DocumentChunk] = {
            chunk.uuid: chunk for chunk in self.test_chunks
        }
        self.mock_parent.chunks = self.chunks_dict

        # Test query and subquery
        self.test_query = "What is the meaning of life?"
        self.test_subquery = "Define meaning of life"

    def test_evaluate_chunks_empty_list(self) -> None:
        """Test evaluating an empty list of chunks."""
        result = self.evaluator.evaluate_chunks(self.test_query, self.test_subquery, [])

        # Verify empty list is returned
        self.assertEqual(result, [])

        # Verify LLM service was not called
        self.mock_llm_service.generate_response.assert_not_called()

    def test_evaluate_chunks_successful(self) -> None:
        """Test successful chunk evaluation."""
        # Setup
        response_data: Dict[str, List[int] | str] = {
            "relevant_chunk_indices": [0, 1],
            "summary_reasoning": "Found relevant information",
        }
        self.mock_llm_service.generate_response.return_value = json.dumps(response_data)

        # Execute
        result = self.evaluator.evaluate_chunks(
            query=self.test_query, subquery=self.test_subquery, chunks=self.test_chunks
        )

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.mock_llm_service.generate_response.called)

    def test_evaluate_chunks_no_relevant_found(self) -> None:
        """Test evaluation when no relevant chunks are found."""
        # Setup
        response_data: Dict[str, List[int] | str] = {
            "relevant_chunk_indices": [],
            "summary_reasoning": "No relevant information found",
        }
        self.mock_llm_service.generate_response.return_value = json.dumps(response_data)

        # Execute
        result = self.evaluator.evaluate_chunks(
            query=self.test_query, subquery=self.test_subquery, chunks=self.test_chunks
        )

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)  # No relevant chunks should be returned
        self.assertTrue(self.mock_llm_service.generate_response.called)

    @patch(
        "backend.src.services.query_processing.components.chunk_evaluator.log_llm_interaction"
    )
    def test_evaluate_chunks_logs_interaction(self, mock_log: Mock) -> None:
        """Test that LLM interactions are logged."""
        # Set up mock response
        response_data = {
            "relevant_chunk_indices": [0],
            "summary_reasoning": "Found relevant information",
        }
        self.mock_llm_service.generate_response.return_value = json.dumps(response_data)

        # Call the method
        self.evaluator.evaluate_chunks(
            self.test_query, self.test_subquery, self.test_chunks
        )

        # Verify logging was called
        mock_log.assert_called_once()
        log_args = mock_log.call_args[1]

        self.assertEqual(log_args["stage"], "batch_chunk_evaluation")
        self.assertIn("messages", log_args)
        self.assertIn("response", log_args)
        self.assertIn("extra_body", log_args)

    def test_evaluate_chunks_llm_error(self) -> None:
        """Test handling LLM service errors during evaluation."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        result = self.evaluator.evaluate_chunks(
            self.test_query, self.test_subquery, self.test_chunks
        )

        # Verify empty list is returned when error occurs
        self.assertEqual(len(result), 0)

        # Verify LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()

    @patch(
        "backend.src.services.query_processing.components.chunk_evaluator.log_llm_interaction"
    )
    def test_evaluate_chunks_logs_interaction_on_error(self, mock_log: Mock) -> None:
        """Test that LLM interactions are not logged when an error occurs."""
        # Set up mock to raise an exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Call the method
        self.evaluator.evaluate_chunks(
            self.test_query, self.test_subquery, self.test_chunks
        )

        # Verify logging was not called since error occurred before response was generated
        mock_log.assert_not_called()

    def test_evaluate_chunks_invalid_response(self) -> None:
        """Test handling invalid LLM responses."""
        # Set up mock to return invalid JSON
        self.mock_llm_service.generate_response.return_value = "invalid json"

        # Call the method
        result = self.evaluator.evaluate_chunks(
            self.test_query, self.test_subquery, self.test_chunks
        )

        # Verify empty list is returned when response is invalid
        self.assertEqual(len(result), 0)

        # Verify LLM service was called
        self.mock_llm_service.generate_response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
