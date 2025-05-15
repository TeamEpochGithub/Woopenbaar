"""Unit tests for the QueryProcessingService class.

This test suite aims for 100% branch coverage and tests all edge cases and boundaries.
"""

import unittest
from datetime import datetime
from unittest.mock import ANY, MagicMock, Mock
from uuid import UUID

from backend.src.data_classes import (
    ChunkedDocument,
    ChunkID,
    ContextPiece,
    DocumentChunk,
    FilterOptions,
    PeriodFilter,
)
from backend.src.data_classes.query_result import QueryResult
from backend.src.services.llm import BaseLLMService
from backend.src.services.query_processing.query_processing_service import (
    QueryProcessingService,
)
from backend.src.services.retrieval.retrieval_service import RetrievalService
from backend.src.services.safety.safety_service import SafetyService


class TestQueryProcessingService(unittest.TestCase):
    """Test cases for the QueryProcessingService class.

    This test suite aims for 100% branch coverage and tests all edge cases.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create mock services
        self.mock_llm_service = Mock(spec=BaseLLMService)
        self.mock_retrieval_service = Mock(spec=RetrievalService)
        self.mock_safety_service = Mock(spec=SafetyService)

        # Initialize the service with mock dependencies
        self.service = QueryProcessingService(
            llm_service=self.mock_llm_service,
            retrieval_service=self.mock_retrieval_service,
            safety_service=self.mock_safety_service,
        )

        # Create test data
        self.test_query = "What is the policy on remote work?"
        self.test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        self.test_chunk_id = ChunkID(1)

        # Create test document
        self.test_document = ChunkedDocument(
            uuid=self.test_uuid,
            content="Full document content",
            type="pdf",
            link="https://example.com/policy",
            vws_id="VWS123",
            create_date=datetime(2023, 1, 1),
            attachment_links=[],
            chunks={},
        )

        # Create test chunks
        self.test_chunk = DocumentChunk(
            uuid=self.test_chunk_id,
            content="Employees can work remotely up to 3 days per week.",
            type="pdf",
            link="https://example.com/policy#section1",
            parent_document=self.test_document,
        )

        # Add chunk to document
        self.test_document.chunks[self.test_chunk_id] = self.test_chunk

        # Create test context piece
        self.test_context_piece = ContextPiece(
            doc_uuid=self.test_uuid,
            chunk_id=self.test_chunk_id,
            content=self.test_chunk.content,
            type=self.test_chunk.type or "unknown",  # Ensure type is not None
            source=self.test_document.vws_id,
        )

        # Create test filter options - using correct parameters from FilterOptions class
        period = PeriodFilter(
            start_date=datetime(2023, 1, 1), end_date=datetime(2023, 12, 31)
        )
        self.test_filters = FilterOptions(period=period, doc_types=["pdf"])

        # Create test query result
        self.test_query_result = QueryResult(
            query=self.test_query,
            response="Employees can work remotely up to 3 days per week.",
            chunks=[self.test_chunk],
            source_documents=[self.test_document],
            reasoning_steps=["Found relevant information about remote work policy."],
            context=[self.test_context_piece],
            data_sources_used={"vws"},
            suggested_questions=["What is the approval process for remote work?"],
        )

        # Mock component methods
        self._mock_component_methods()

    def _mock_component_methods(self) -> None:
        """Mock methods of the service's component dependencies."""
        # Mock RetrievalManager
        self.service.retrieval_manager.execute_retrieval = Mock(
            return_value=([self.test_chunk], [self.test_document])
        )

        # Mock ChunkEvaluator
        self.service.chunk_evaluator.evaluate_chunks = Mock(
            return_value=[self.test_chunk]
        )

        # Mock AnswerSynthesizer
        self.service.answer_synthesizer.generate_answer = Mock(
            return_value="Employees can work remotely up to 3 days per week."
        )
        self.service.answer_synthesizer.generate_intermediate_answer = Mock(
            return_value="Based on the information found so far, employees can work remotely."
        )

        # Mock SuggestionGenerator
        self.service.suggestion_generator.generate_suggestions = Mock(
            return_value=MagicMock(
                suggestions=["What is the approval process for remote work?"]
            )
        )

        # Mock ProgressEvaluator
        self.service.progress_evaluator.evaluate_progress = Mock(
            return_value=MagicMock(
                ready_to_answer=True,
                reasoning="Found sufficient information to answer the query.",
                subquery="",
            )
        )

    def test_process_standard_query_success(self) -> None:
        """Test successful processing of a standard query."""
        # Execute
        result = self.service.process_standard_query(
            query=self.test_query, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        # self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.service.retrieval_manager.execute_retrieval.assert_called_once_with(
            state=ANY,  # QueryState is created inside the method
            subquery=self.test_query,
            filters=self.test_filters,
        )
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_standard_query_no_chunks_found(self) -> None:
        """Test processing a standard query when no chunks are found."""
        # Setup - mock retrieval to return no chunks
        self.service.retrieval_manager.execute_retrieval = Mock(return_value=([], []))

        # Execute
        result = self.service.process_standard_query(
            query=self.test_query, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.source_documents, [])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [])
        self.assertEqual(result.data_sources_used, set())
        self.assertEqual(result.suggested_questions, [])
        self.assertIn("geen relevante informatie", result.response.lower())

        # Verify component calls
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_not_called()
        self.service.answer_synthesizer.generate_answer.assert_not_called()
        self.service.suggestion_generator.generate_suggestions.assert_not_called()

    def test_process_standard_query_no_relevant_chunks(self) -> None:
        """Test processing a standard query when chunks are found but none are relevant."""
        # Setup - mock chunk evaluator to return no relevant chunks
        self.service.chunk_evaluator.evaluate_chunks = Mock(return_value=[])

        # Execute
        result = self.service.process_standard_query(
            query=self.test_query, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [])
        self.assertEqual(result.data_sources_used, set())
        self.assertEqual(result.suggested_questions, [])
        self.assertIn("geen relevante informatie", result.response.lower())

        # Verify component calls
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_not_called()
        self.service.suggestion_generator.generate_suggestions.assert_not_called()

    def test_process_standard_query_suggestion_error(self) -> None:
        """Test processing a standard query when suggestion generation fails."""
        # Setup - mock suggestion generator to raise an exception
        self.service.suggestion_generator.generate_suggestions = Mock(
            side_effect=Exception("Suggestion error")
        )

        # Execute
        result = self.service.process_standard_query(
            query=self.test_query, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(result.suggested_questions, [])

        # Verify component calls
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_success(self) -> None:
        """Test successful processing of an adaptive query."""
        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.service.progress_evaluator.evaluate_progress.assert_called_once()
        self.service.retrieval_manager.execute_retrieval.assert_not_called()  # Not called because ready_to_answer=True
        self.service.chunk_evaluator.evaluate_chunks.assert_not_called()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_with_subqueries(self) -> None:
        """Test processing an adaptive query that requires subqueries."""
        # Setup - mock progress evaluator to require subqueries
        self.service.progress_evaluator.evaluate_progress = Mock(
            side_effect=[
                MagicMock(
                    ready_to_answer=False,
                    reasoning="Need more information",
                    subquery="What is the approval process?",
                ),
                MagicMock(
                    ready_to_answer=True,
                    reasoning="Found sufficient information",
                    subquery="",
                ),
            ]
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.assertEqual(
            self.service.progress_evaluator.evaluate_progress.call_count, 2
        )
        self.service.retrieval_manager.execute_retrieval.assert_called_once_with(
            state=ANY,
            subquery="What is the approval process?",
            filters=self.test_filters,
        )
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_max_iterations(self) -> None:
        """Test processing an adaptive query that reaches max iterations."""
        # Setup - mock progress evaluator to always need more information
        self.service.progress_evaluator.evaluate_progress = Mock(
            return_value=MagicMock(
                ready_to_answer=False,
                reasoning="Need more information",
                subquery="What is the approval process?",
            )
        )

        # Execute with low max_iterations
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=1, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.assertEqual(
            self.service.progress_evaluator.evaluate_progress.call_count, 1
        )
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_no_chunks_found(self) -> None:
        """Test processing an adaptive query when no chunks are found."""
        # Setup - mock retrieval to return no chunks
        self.service.retrieval_manager.execute_retrieval = Mock(return_value=([], []))

        # Setup - mock progress evaluator to require a subquery
        self.service.progress_evaluator.evaluate_progress = Mock(
            return_value=MagicMock(
                ready_to_answer=False,
                reasoning="Need more information",
                subquery="What is the approval process?",
            )
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.source_documents, [])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [])
        self.assertEqual(result.data_sources_used, set())
        self.assertEqual(result.suggested_questions, [])
        self.assertIn("geen relevante informatie", result.response.lower())

        # Verify component calls
        self.service.progress_evaluator.evaluate_progress.assert_called_once()
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_not_called()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_not_called()

    def test_process_adaptive_query_no_relevant_chunks(self) -> None:
        """Test processing an adaptive query when chunks are found but none are relevant."""
        # Setup - mock chunk evaluator to return no relevant chunks
        self.service.chunk_evaluator.evaluate_chunks = Mock(return_value=[])

        # Setup - mock progress evaluator to require a subquery
        self.service.progress_evaluator.evaluate_progress = Mock(
            return_value=MagicMock(
                ready_to_answer=False,
                reasoning="Need more information",
                subquery="What is the approval process?",
            )
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [])
        self.assertEqual(result.data_sources_used, set())
        self.assertEqual(result.suggested_questions, [])
        self.assertIn("geen relevante informatie", result.response.lower())

        # Verify component calls
        self.service.progress_evaluator.evaluate_progress.assert_called_once()
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_not_called()

    def test_process_adaptive_query_intermediate_answer(self) -> None:
        """Test processing an adaptive query with intermediate answers."""
        # Setup - mock progress evaluator to require multiple subqueries
        self.service.progress_evaluator.evaluate_progress = Mock(
            side_effect=[
                MagicMock(
                    ready_to_answer=False,
                    reasoning="Need more information",
                    subquery="What is the approval process?",
                ),
                MagicMock(
                    ready_to_answer=False,
                    reasoning="Need more information",
                    subquery="What are the restrictions?",
                ),
                MagicMock(
                    ready_to_answer=True,
                    reasoning="Found sufficient information",
                    subquery="",
                ),
            ]
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.assertEqual(
            self.service.progress_evaluator.evaluate_progress.call_count, 3
        )
        self.assertEqual(self.service.retrieval_manager.execute_retrieval.call_count, 2)
        self.assertEqual(self.service.chunk_evaluator.evaluate_chunks.call_count, 2)
        self.service.answer_synthesizer.generate_intermediate_answer.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_intermediate_answer_error(self) -> None:
        """Test processing an adaptive query when intermediate answer generation fails."""
        # Setup - mock progress evaluator to require multiple subqueries
        self.service.progress_evaluator.evaluate_progress = Mock(
            side_effect=[
                MagicMock(
                    ready_to_answer=False,
                    reasoning="Need more information",
                    subquery="What is the approval process?",
                ),
                MagicMock(
                    ready_to_answer=True,
                    reasoning="Found sufficient information",
                    subquery="",
                ),
            ]
        )

        # Setup - mock intermediate answer generation to raise an exception
        self.service.answer_synthesizer.generate_intermediate_answer = Mock(
            side_effect=Exception("Intermediate answer error")
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.assertEqual(
            self.service.progress_evaluator.evaluate_progress.call_count, 2
        )
        self.service.retrieval_manager.execute_retrieval.assert_called_once()
        self.service.chunk_evaluator.evaluate_chunks.assert_called_once()
        self.service.answer_synthesizer.generate_intermediate_answer.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_suggestion_error(self) -> None:
        """Test processing an adaptive query when suggestion generation fails."""
        # Setup - mock suggestion generator to raise an exception
        self.service.suggestion_generator.generate_suggestions = Mock(
            side_effect=Exception("Suggestion error")
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(result.suggested_questions, [])

        # Verify component calls
        self.service.progress_evaluator.evaluate_progress.assert_called_once()
        self.service.retrieval_manager.execute_retrieval.assert_not_called()
        self.service.chunk_evaluator.evaluate_chunks.assert_not_called()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_with_frontend_callback(self) -> None:
        """Test processing an adaptive query with a frontend callback."""
        # Setup - create a mock callback
        mock_callback = Mock()

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query,
            max_iterations=3,
            filters=self.test_filters,
            frontend_callback=mock_callback,
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)

        # Verify callback was set on the query state
        # Note: We can't directly verify this, but we can check that the service didn't raise an error

        # Verify component calls
        self.service.progress_evaluator.evaluate_progress.assert_called_once()
        self.service.retrieval_manager.execute_retrieval.assert_not_called()
        self.service.chunk_evaluator.evaluate_chunks.assert_not_called()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_with_duplicate_chunks(self) -> None:
        """Test processing an adaptive query with duplicate chunks."""
        # Setup - mock progress evaluator to require multiple subqueries
        self.service.progress_evaluator.evaluate_progress = Mock(
            side_effect=[
                MagicMock(
                    ready_to_answer=False,
                    reasoning="Need more information",
                    subquery="What is the approval process?",
                ),
                MagicMock(
                    ready_to_answer=False,
                    reasoning="Need more information",
                    subquery="What are the restrictions?",
                ),
                MagicMock(
                    ready_to_answer=True,
                    reasoning="Found sufficient information",
                    subquery="",
                ),
            ]
        )

        # Setup - mock retrieval to return the same chunk twice
        self.service.retrieval_manager.execute_retrieval = Mock(
            side_effect=[
                ([self.test_chunk], [self.test_document]),
                ([self.test_chunk], [self.test_document]),
            ]
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [self.test_chunk])
        self.assertEqual(result.source_documents, [self.test_document])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [self.test_context_piece])
        self.assertEqual(result.data_sources_used, {"vws"})
        self.assertEqual(
            result.suggested_questions,
            ["What is the approval process for remote work?"],
        )

        # Verify component calls
        self.assertEqual(
            self.service.progress_evaluator.evaluate_progress.call_count, 3
        )
        self.assertEqual(self.service.retrieval_manager.execute_retrieval.call_count, 2)
        self.assertEqual(
            self.service.chunk_evaluator.evaluate_chunks.call_count, 1
        )  # Only called once for unique chunks
        self.service.answer_synthesizer.generate_intermediate_answer.assert_called_once()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_called_once()

    def test_process_adaptive_query_with_no_subquery(self) -> None:
        """Test processing an adaptive query when no subquery is generated."""
        # Setup - mock progress evaluator to return no subquery
        self.service.progress_evaluator.evaluate_progress = Mock(
            return_value=MagicMock(
                ready_to_answer=False, reasoning="Need more information", subquery=""
            )
        )

        # Execute
        result = self.service.process_adaptive_query(
            query=self.test_query, max_iterations=3, filters=self.test_filters
        )

        # Assert
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, self.test_query)
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.source_documents, [])
        self.assertEqual(len(result.reasoning_steps), 1)
        self.assertEqual(result.context, [])
        self.assertEqual(result.data_sources_used, set())
        self.assertEqual(result.suggested_questions, [])

        # Verify component calls
        self.service.progress_evaluator.evaluate_progress.assert_called_once()
        self.service.retrieval_manager.execute_retrieval.assert_not_called()
        self.service.chunk_evaluator.evaluate_chunks.assert_not_called()
        self.service.answer_synthesizer.generate_answer.assert_called_once()
        self.service.suggestion_generator.generate_suggestions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
