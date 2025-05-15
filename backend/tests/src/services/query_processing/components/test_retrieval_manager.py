"""Unit tests for the RetrievalManager component."""

import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from backend.conf.prompts import SOURCE_SELECTION_SYSTEM_PROMPT
from backend.src.data_classes import ChunkedDocument, ChunkID, DocumentChunk
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.components.retrieval_manager import (
    RetrievalManager,
    SourceSelectionResponse,
)
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.retrieval.retrieval_service import RetrievalService


class TestRetrievalManager(unittest.TestCase):
    """Test cases for the RetrievalManager component.

    This test suite aims for 100% branch coverage and tests all edge cases.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create mock services
        self.mock_retrieval_service = Mock(spec=RetrievalService)
        self.mock_llm_service = Mock(spec=BaseLLMService)

        # Initialize the component with mocked services
        self.retrieval_manager = RetrievalManager(
            retrieval_service=self.mock_retrieval_service,
            llm_service=self.mock_llm_service,
        )

        # Create a test query state
        self.query_state = QueryState("Test query")

        # Sample test data
        self.test_sources = [
            {"name": "source1", "description": "First test source"},
            {"name": "source2", "description": "Second test source"},
        ]

        # Create a test document chunk
        self.test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        self.mock_parent = ChunkedDocument(
            uuid=self.test_uuid,
            content="Full document content",
            type="test_type",
            link="test_link",
            vws_id="test_vws_id",
            create_date=None,
            attachment_links=[],
            chunks={},
        )

        self.test_chunk = DocumentChunk(
            uuid=ChunkID(1),
            content="Test content",
            type="test_type",
            link="test_link",
            parent_document=self.mock_parent,
        )

    def test_select_data_source_no_sources(self) -> None:
        """Test source selection with no available sources."""
        # Setup - empty sources list
        self.mock_retrieval_service.get_sources.return_value = []

        # Execute
        result = self.retrieval_manager.select_data_source(
            "test query", self.query_state
        )

        # Assert
        self.assertEqual(result, "")
        self.mock_retrieval_service.get_sources.assert_called_once()
        self.mock_llm_service.generate_response.assert_not_called()

    def test_select_data_source_single_source(self) -> None:
        """Test source selection with a single available source."""
        # Setup - single source
        single_source = [{"name": "source1", "description": "Test source"}]
        self.mock_retrieval_service.get_sources.return_value = single_source

        # Execute
        result = self.retrieval_manager.select_data_source(
            "test query", self.query_state
        )

        # Assert
        self.assertEqual(result, "source1")
        self.mock_retrieval_service.get_sources.assert_called_once()
        self.mock_llm_service.generate_response.assert_not_called()

    def test_select_data_source_multiple_sources_successful(self) -> None:
        """Test source selection with multiple sources and successful LLM response."""
        # Setup - multiple sources
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Set up LLM response
        selection_response = SourceSelectionResponse(
            source_name="source2", reasoning="This is the most relevant source"
        )
        self.mock_llm_service.generate_response.return_value = (
            selection_response.model_dump_json()
        )

        # Execute
        result = self.retrieval_manager.select_data_source(
            "test query", self.query_state
        )

        # Assert
        self.assertEqual(result, "source2")
        self.mock_retrieval_service.get_sources.assert_called_once()
        self.mock_llm_service.generate_response.assert_called_once()

        # Verify correct parameters were passed to LLM service
        call_args = self.mock_llm_service.generate_response.call_args[1]
        self.assertIn("user_message", call_args)
        self.assertIn("system_prompt", call_args)
        self.assertEqual(call_args["system_prompt"], SOURCE_SELECTION_SYSTEM_PROMPT)
        self.assertIn("extra_body", call_args)

    def test_select_data_source_invalid_source_selected(self) -> None:
        """Test source selection when LLM returns an invalid source name."""
        # Setup - multiple sources
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Set up LLM response with invalid source
        selection_response = SourceSelectionResponse(
            source_name="invalid_source", reasoning="This is the most relevant source"
        )
        self.mock_llm_service.generate_response.return_value = (
            selection_response.model_dump_json()
        )

        # Execute
        result = self.retrieval_manager.select_data_source(
            "test query", self.query_state
        )

        # Assert - should fall back to first source
        self.assertEqual(result, "source1")
        self.mock_retrieval_service.get_sources.assert_called_once()
        self.mock_llm_service.generate_response.assert_called_once()

    def test_select_data_source_llm_error(self) -> None:
        """Test source selection when LLM service throws an error."""
        # Setup - multiple sources
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Set up LLM to raise exception
        self.mock_llm_service.generate_response.side_effect = Exception("LLM error")

        # Execute
        result = self.retrieval_manager.select_data_source(
            "test query", self.query_state
        )

        # Assert - should fall back to first source
        self.assertEqual(result, "source1")
        self.mock_retrieval_service.get_sources.assert_called_once()
        self.mock_llm_service.generate_response.assert_called_once()

    def test_select_data_source_invalid_response(self) -> None:
        """Test source selection when LLM returns invalid JSON."""
        # Setup - multiple sources
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Set up LLM to return invalid JSON
        self.mock_llm_service.generate_response.return_value = "invalid json"

        # Execute
        result = self.retrieval_manager.select_data_source(
            "test query", self.query_state
        )

        # Assert - should fall back to first source
        self.assertEqual(result, "source1")
        self.mock_retrieval_service.get_sources.assert_called_once()
        self.mock_llm_service.generate_response.assert_called_once()

    @patch(
        "backend.src.services.query_processing.components.retrieval_manager.log_llm_interaction"
    )
    def test_select_data_source_logs_interaction(self, mock_log: Mock) -> None:
        """Test that LLM interactions are logged during source selection."""
        # Setup
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        selection_response = SourceSelectionResponse(
            source_name="source2", reasoning="This is the most relevant source"
        )
        self.mock_llm_service.generate_response.return_value = (
            selection_response.model_dump_json()
        )

        # Execute
        self.retrieval_manager.select_data_source("test query", self.query_state)

        # Assert
        mock_log.assert_called_once()
        log_args = mock_log.call_args[1]

        self.assertEqual(log_args["stage"], "source_selection")
        self.assertIn("messages", log_args)
        self.assertIn("response", log_args)
        self.assertIn("extra_body", log_args)

    def test_execute_retrieval_no_sources(self) -> None:
        """Test execute_retrieval when no sources are available."""
        # Setup
        self.mock_retrieval_service.get_sources.return_value = []

        # Execute
        chunks, docs = self.retrieval_manager.execute_retrieval(
            state=self.query_state, subquery="test subquery"
        )

        # Assert
        self.assertEqual(chunks, [])
        self.assertEqual(docs, [])
        self.mock_retrieval_service.find.assert_not_called()

        # Verify state updates
        self.assertIn(
            "Geen geschikte gegevensbronnen", self.query_state.frontend_steps[0]
        )

    def test_execute_retrieval_successful(self) -> None:
        """Test successful retrieval execution."""
        # Setup
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Mock select_data_source to return a known source
        with patch.object(
            self.retrieval_manager, "select_data_source", return_value="source1"
        ) as mock_select:
            # Set up retrieval service to return test data
            self.mock_retrieval_service.find.return_value = (
                [self.test_chunk],
                [self.mock_parent],
            )

            # Execute
            chunks, docs = self.retrieval_manager.execute_retrieval(
                state=self.query_state, subquery="test subquery"
            )

            # Assert
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0], self.test_chunk)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0], self.mock_parent)

            # Verify service calls
            mock_select.assert_called_once_with("test subquery", self.query_state)
            self.mock_retrieval_service.find.assert_called_once()

            # Verify state updates
            self.assertIn("source1", self.query_state.used_data_sources)
            self.assertIn("Found 1 chunks", self.query_state.reasoning_trace[0])

    def test_execute_retrieval_with_filters(self) -> None:
        """Test retrieval execution with filter options."""
        # Setup
        self.mock_retrieval_service.get_sources.return_value = self.test_sources
        # Using a Mock for FilterOptions instead of trying to instantiate it
        test_filters = Mock()

        # Mock select_data_source to return a known source
        with patch.object(
            self.retrieval_manager, "select_data_source", return_value="source1"
        ):
            # Set up retrieval service to return test data
            self.mock_retrieval_service.find.return_value = (
                [self.test_chunk],
                [self.mock_parent],
            )

            # Execute
            chunks, _ = self.retrieval_manager.execute_retrieval(
                state=self.query_state, subquery="test subquery", filters=test_filters
            )

            # Assert
            self.assertEqual(len(chunks), 1)

            # Verify filters were passed correctly
            call_args = self.mock_retrieval_service.find.call_args[1]
            self.assertEqual(call_args["filters"], test_filters)

    def test_execute_retrieval_empty_results(self) -> None:
        """Test retrieval execution with empty results."""
        # Setup
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Mock select_data_source to return a known source
        with patch.object(
            self.retrieval_manager, "select_data_source", return_value="source1"
        ):
            # Set up retrieval service to return empty results
            self.mock_retrieval_service.find.return_value = ([], [])

            # Execute
            chunks, docs = self.retrieval_manager.execute_retrieval(
                state=self.query_state, subquery="test subquery"
            )

            # Assert
            self.assertEqual(chunks, [])
            self.assertEqual(docs, [])

            # Verify state updates - should include "No chunks found" reasoning
            self.assertIn("No chunks found", self.query_state.reasoning_trace[-1])

    def test_execute_retrieval_service_error(self) -> None:
        """Test retrieval execution when retrieval service throws an error."""
        # Setup
        self.mock_retrieval_service.get_sources.return_value = self.test_sources

        # Mock select_data_source to return a known source
        with patch.object(
            self.retrieval_manager, "select_data_source", return_value="source1"
        ):
            # Set up retrieval service to raise exception
            self.mock_retrieval_service.find.side_effect = Exception("Retrieval error")

            # Execute
            chunks, docs = self.retrieval_manager.execute_retrieval(
                state=self.query_state, subquery="test subquery"
            )

            # Assert
            self.assertEqual(chunks, [])
            self.assertEqual(docs, [])

            # Check all reasoning trace entries as implementation adds both error reasoning and "No chunks found" reasoning
            reasoning_text = " ".join(self.query_state.reasoning_trace)
            self.assertIn("Error during retrieval", reasoning_text)
            self.assertIn(
                "Er is een fout opgetreden", self.query_state.frontend_steps[-1]
            )


if __name__ == "__main__":
    unittest.main()
