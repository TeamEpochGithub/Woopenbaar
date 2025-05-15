"""Unit tests for query processing utility functions."""

import json
import os
import unittest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from backend.conf.prompts import (
    ANSWER_SYNTHESIS_SYSTEM_PROMPT,
    INTERMEDIATE_ANSWER_SYSTEM_PROMPT,
)
from backend.src.services.query_processing.utils import (
    DEBUG_LLM_DIR,
    estimate_token_count,
    format_document_chunks_for_llm,
    get_prompt_reserved_tokens,
    log_llm_interaction,
    truncate_text_to_max_tokens,
)


class TestQueryProcessingUtils(unittest.TestCase):
    """Test cases for query processing utility functions.

    This test suite aims for 100% branch coverage and tests all edge cases.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Sample document chunks for testing
        self.test_chunks: List[Dict[str, Any]] = [
            {
                "content": "Test content 1",
                "source": "source1",
                "type": "type1",
                "subject": "subject1",
            },
            {
                "content": "Test content 2",
                "source": "source2",
                "type": "",  # Empty type
                "subject": "",  # Empty subject
            },
            {"content": "Test content 3"},  # Missing metadata
        ]

        # Create a temporary debug directory for testing
        self.test_debug_dir = "test_llm_debug"
        self.original_debug_dir = DEBUG_LLM_DIR

        # Clean up any existing test files
        if os.path.exists(self.test_debug_dir):
            for file in os.listdir(self.test_debug_dir):
                os.remove(os.path.join(self.test_debug_dir, file))
            os.rmdir(self.test_debug_dir)

    def tearDown(self) -> None:
        """Clean up after each test method."""
        # Clean up test debug directory
        if os.path.exists(self.test_debug_dir):
            for file in os.listdir(self.test_debug_dir):
                os.remove(os.path.join(self.test_debug_dir, file))
            os.rmdir(self.test_debug_dir)

    def test_format_document_chunks_with_metadata(self) -> None:
        """Test document chunk formatting with metadata."""
        formatted = format_document_chunks_for_llm(
            self.test_chunks, include_metadata=True
        )

        # Check that all chunks are included
        self.assertIn("Fragment 1", formatted)
        self.assertIn("Fragment 2", formatted)
        self.assertIn("Fragment 3", formatted)

        # Check metadata formatting
        self.assertIn("[Bron: source1 (type1) - subject1]", formatted)
        self.assertIn("[Bron: source2]", formatted)  # Empty type and subject
        self.assertIn("[Bron: Onbekende bron]", formatted)  # Missing metadata

        # Check content inclusion
        self.assertIn("Test content 1", formatted)
        self.assertIn("Test content 2", formatted)
        self.assertIn("Test content 3", formatted)

    def test_format_document_chunks_without_metadata(self) -> None:
        """Test document chunk formatting without metadata."""
        formatted = format_document_chunks_for_llm(
            self.test_chunks, include_metadata=False
        )

        # Check simple fragment formatting
        self.assertIn("Fragment 1:\nTest content 1", formatted)
        self.assertIn("Fragment 2:\nTest content 2", formatted)
        self.assertIn("Fragment 3:\nTest content 3", formatted)

        # Verify metadata is not included
        self.assertNotIn("Bron:", formatted)
        self.assertNotIn("type1", formatted)
        self.assertNotIn("subject1", formatted)

    def test_format_document_chunks_empty_list(self) -> None:
        """Test formatting with empty chunk list."""
        formatted = format_document_chunks_for_llm([], include_metadata=True)
        self.assertEqual(formatted, "")

    def test_truncate_text_to_max_tokens(self) -> None:
        """Test text truncation functionality."""
        # Test text within limit
        short_text = "Short text"
        self.assertEqual(truncate_text_to_max_tokens(short_text, 100), short_text)

        # Test text exceeding limit
        long_text = "x" * 1000
        truncated = truncate_text_to_max_tokens(long_text, 50)
        self.assertLess(len(truncated), len(long_text))
        self.assertTrue(truncated.endswith("... [text truncated]"))

        # Test edge case with very small token limit
        tiny_truncated = truncate_text_to_max_tokens(long_text, 5)
        self.assertIn("...", tiny_truncated)
        self.assertTrue(tiny_truncated.endswith("... [text truncated]"))

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open")
    def test_log_llm_interaction(
        self, mock_open: Mock, mock_makedirs: Mock, mock_exists: Mock
    ) -> None:
        """Test LLM interaction logging functionality."""
        # Test data
        test_stage = "test_stage"
        test_messages = [{"role": "user", "content": "test message"}]
        test_response = "test response"
        test_extra_body = {"param": "value"}

        # Test new file creation
        mock_exists.return_value = False
        mock_open.return_value.__enter__.return_value.read.return_value = "[]"

        log_llm_interaction(test_stage, test_messages, test_response, test_extra_body)

        mock_makedirs.assert_called_once()
        mock_open.assert_called()

        # Test appending to existing file
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            [
                {
                    "timestamp": datetime.now().isoformat(),
                    "stage": test_stage,
                    "messages": [],
                    "response": "previous response",
                }
            ]
        )

        log_llm_interaction(test_stage, test_messages, test_response)

        # Test handling of invalid JSON in existing file
        mock_open.return_value.__enter__.return_value.read.return_value = "invalid json"
        log_llm_interaction(test_stage, test_messages, test_response)

        # Test handling of file write errors
        mock_open.side_effect = IOError
        log_llm_interaction(test_stage, test_messages, test_response)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenizer_initialization(self, mock_from_pretrained: Mock) -> None:
        """Test tokenizer initialization and caching."""
        # Test successful initialization
        mock_tokenizer = Mock()
        mock_from_pretrained.return_value = mock_tokenizer

        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer
            count = estimate_token_count("test")
            self.assertIsNotNone(count)

        # Test initialization failure
        mock_from_pretrained.side_effect = Exception("Failed to load")
        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = None
            count = estimate_token_count("test")
            self.assertEqual(count, len("test") // 4)

    def test_estimate_token_count(self) -> None:
        """Test token counting with both tokenizer and fallback."""
        test_text = "This is a test text"

        # Test with working tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer",
            return_value=mock_tokenizer,
        ):
            count = estimate_token_count(test_text)
            self.assertEqual(count, 5)
            mock_tokenizer.encode.assert_called_once_with(test_text)

        # Test with failing tokenizer encode
        mock_tokenizer.encode.side_effect = Exception("Encode failed")
        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer",
            return_value=mock_tokenizer,
        ):
            count = estimate_token_count(test_text)
            self.assertEqual(count, len(test_text) // 4)  # Fallback approximation

        # Test with no tokenizer
        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer",
            return_value=None,
        ):
            count = estimate_token_count(test_text)
            self.assertEqual(count, len(test_text) // 4)  # Fallback approximation

    def test_get_prompt_reserved_tokens(self) -> None:
        """Test calculation of reserved tokens for prompts."""
        # Test with working tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 100  # 100 tokens for any input

        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer",
            return_value=mock_tokenizer,
        ):
            reserved_tokens = get_prompt_reserved_tokens()
            self.assertEqual(reserved_tokens, 350)  # 100 + 250 overhead

            # Verify both prompts were checked
            self.assertEqual(mock_tokenizer.encode.call_count, 2)
            mock_tokenizer.encode.assert_any_call(ANSWER_SYNTHESIS_SYSTEM_PROMPT)
            mock_tokenizer.encode.assert_any_call(INTERMEDIATE_ANSWER_SYSTEM_PROMPT)

        # Test with failing tokenizer
        with patch(
            "backend.src.services.query_processing.utils._get_tokenizer",
            return_value=None,
        ):
            reserved_tokens = get_prompt_reserved_tokens()
            expected = (
                max(
                    len(ANSWER_SYNTHESIS_SYSTEM_PROMPT) // 4,
                    len(INTERMEDIATE_ANSWER_SYSTEM_PROMPT) // 4,
                )
                + 250
            )
            self.assertEqual(reserved_tokens, expected)


if __name__ == "__main__":
    unittest.main()
