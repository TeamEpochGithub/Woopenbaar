"""Unit tests for the QueryState class."""

import unittest
from typing import List, Set
from unittest.mock import Mock
from uuid import UUID

from backend.conf.config import Config
from backend.src.data_classes import ChunkID, ContextPiece
from backend.src.services.query_processing.query_state import QueryState
from backend.src.services.query_processing.utils import get_prompt_reserved_tokens


class TestQueryState(unittest.TestCase):
    """Test cases for QueryState class.

    This test suite aims for 100% branch coverage and tests all edge cases.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.test_query: str = "What is the meaning of life?"
        self.query_state = QueryState(query=self.test_query)

        # Create a sample context piece for testing
        self.test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        self.test_chunk_id = ChunkID(123)  # ChunkID is a NewType('ChunkID', int)
        self.context_piece = ContextPiece(
            doc_uuid=self.test_uuid,
            chunk_id=self.test_chunk_id,
            content="Sample context",
            type="test_type",
            source="test_source",
        )

    def test_initialization(self) -> None:
        """Test proper initialization of QueryState."""
        self.assertEqual(self.query_state.query, self.test_query)
        self.assertEqual(len(self.query_state.current_context), 0)
        self.assertEqual(len(self.query_state.previous_subqueries), 0)
        self.assertEqual(len(self.query_state.retrieved_chunks), 0)
        self.assertEqual(len(self.query_state.reasoning_trace), 0)
        self.assertEqual(len(self.query_state.used_data_sources), 0)
        self.assertFalse(self.query_state.is_complete())
        self.assertFalse(self.query_state.is_ready_to_answer())

        # Test max tokens calculation
        expected_max_tokens = (
            Config.LLM_MAX_MODEL_LEN
            - Config.LLM_MAX_TOKENS
            - get_prompt_reserved_tokens()
        )
        self.assertEqual(self.query_state.max_available_tokens, expected_max_tokens)

    def test_frontend_callback(self) -> None:
        """Test frontend callback functionality."""
        mock_callback: Mock = Mock()
        test_step: str = "Test step"

        # Test callback setting and execution
        self.query_state.set_frontend_callback(mock_callback)
        self.query_state.add_frontend_step(test_step)

        mock_callback.assert_called_once_with(test_step)
        self.assertEqual(self.query_state.frontend_steps, [test_step])

        # Test without callback
        self.query_state._frontend_callback = None  # type: ignore
        self.query_state.add_frontend_step("Another step")
        mock_callback.assert_called_once()  # Should not be called again

    def test_reasoning_management(self) -> None:
        """Test reasoning trace management and token limits."""
        # Add multiple reasoning steps
        steps: List[str] = [
            "First reasoning step",
            "Second reasoning step",
            "Third reasoning step",
        ]

        for step in steps:
            self.query_state.add_reasoning(step)

        self.assertEqual(len(self.query_state.reasoning_trace), len(steps))
        self.assertEqual(self.query_state.reasoning_trace, steps)

        # Test full reasoning trace property
        expected_trace = "\n".join(steps)
        self.assertEqual(self.query_state.full_reasoning_trace, expected_trace)

    def test_subquery_management(self) -> None:
        """Test subquery tracking."""
        test_subqueries: Set[str] = {
            "subquery1",
            "subquery2",
            "subquery1",
        }  # Duplicate intentional

        for subquery in test_subqueries:
            self.query_state.add_prev_subquery(subquery)

        self.assertEqual(self.query_state.previous_subqueries, test_subqueries)
        self.assertEqual(
            len(self.query_state.previous_subqueries), len(set(test_subqueries))
        )

    def test_chunk_tracking(self) -> None:
        """Test retrieved chunk tracking."""
        test_chunks: List[ChunkID] = [ChunkID(1), ChunkID(2)]

        for chunk in test_chunks:
            self.query_state.add_retrieved_chunk(chunk)

        self.assertEqual(len(self.query_state.retrieved_chunks), len(test_chunks))
        for chunk in test_chunks:
            self.assertIn(chunk, self.query_state.retrieved_chunks)

    def test_readiness_state(self) -> None:
        """Test readiness state management."""
        self.assertFalse(self.query_state.is_ready_to_answer())

        self.query_state.update_readiness(True)
        self.assertTrue(self.query_state.is_ready_to_answer())

        self.query_state.update_readiness(False)
        self.assertFalse(self.query_state.is_ready_to_answer())

    def test_context_piece_management(self) -> None:
        """Test context piece management including token limits."""
        # Test basic addition
        success = self.query_state.add_context_piece(self.context_piece)
        self.assertTrue(success)
        self.assertEqual(len(self.query_state.current_context), 1)
        self.assertEqual(self.query_state.current_context[0], self.context_piece)
        self.assertIn(self.context_piece.source, self.query_state.used_data_sources)

        # Test getting all context pieces
        self.assertEqual(
            self.query_state.get_all_context_pieces(), [self.context_piece]
        )

    def test_context_token_limits(self) -> None:
        """Test token limit handling for context pieces."""
        # Create a large context piece that would exceed token limits
        large_text = "x" * 100000  # Should exceed token limit
        large_context = ContextPiece(
            doc_uuid=self.test_uuid,
            chunk_id=ChunkID(456),
            content=large_text,
            type="test_type",
            source="test_source",
        )

        # Try adding the large context piece
        success = self.query_state.add_context_piece(large_context)
        self.assertFalse(success)
        self.assertTrue(self.query_state.is_complete())
        self.assertTrue(self.query_state.is_ready_to_answer())

    def test_token_counting(self) -> None:
        """Test token counting functionality."""
        # Test empty state
        initial_tokens = self.query_state.get_current_token_count()
        query_tokens = len(self.test_query.split())  # Rough approximation
        self.assertGreaterEqual(initial_tokens, query_tokens)

        # Add content and verify token count increases
        self.query_state.add_context_piece(self.context_piece)
        self.query_state.add_reasoning("Test reasoning")

        new_tokens = self.query_state.get_current_token_count()
        self.assertGreater(new_tokens, initial_tokens)

    def test_ensure_token_limit(self) -> None:
        """Test token limit enforcement."""
        # Add lots of reasoning steps
        for i in range(100):
            self.query_state.add_reasoning(f"Reasoning step {i} " * 100)

        # Verify that token limit is maintained
        self.assertLessEqual(
            self.query_state.get_current_token_count(),
            self.query_state.max_available_tokens,
        )

    def test_context_limit_reached(self) -> None:
        """Test context limit detection."""
        # Initially should not be reached
        self.assertFalse(self.query_state.is_context_limit_reached)

        # Add large context until limit
        large_text = "x" * 10000
        while not self.query_state.is_context_limit_reached:
            context = ContextPiece(
                doc_uuid=self.test_uuid,
                chunk_id=ChunkID(789),
                content=large_text,
                type="test_type",
                source="test_source",
            )
            self.query_state.add_context_piece(context)

        self.assertTrue(self.query_state.is_context_limit_reached)

    def test_serialization(self) -> None:
        """Test state serialization to dictionary."""
        # Add some test data
        self.query_state.add_context_piece(self.context_piece)
        self.query_state.add_reasoning("Test reasoning")
        self.query_state.add_prev_subquery("Test subquery")
        self.query_state.add_frontend_step("Test frontend step")
        self.query_state.update_readiness(True)

        # Get dictionary representation
        state_dict = self.query_state.to_dict()

        # Verify all expected keys and types
        self.assertEqual(state_dict["query"], self.test_query)
        self.assertIsInstance(state_dict["current_context"], list)
        self.assertIsInstance(state_dict["previous_subqueries"], list)
        self.assertIsInstance(state_dict["retrieved_chunks"], list)
        self.assertIsInstance(state_dict["reasoning_trace"], list)
        self.assertIsInstance(state_dict["frontend_steps"], list)
        self.assertIsInstance(state_dict["is_complete"], bool)
        self.assertIsInstance(state_dict["ready_to_answer"], bool)
        self.assertIsInstance(state_dict["used_data_sources"], list)


if __name__ == "__main__":
    unittest.main()
