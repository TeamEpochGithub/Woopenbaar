from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from standard_data_format.src.metadata import MetadataManager
from standard_data_format.src.pdf_converter import PDFConverterManager


@pytest.fixture
def sample_config():
    """Provide a sample configuration for PDFConverterManager."""
    return {
        "llm_service": {"type": "openai", "model": "gpt-4", "api_key": "test-key"},
        "base_model": "default",
        "output_format": "markdown",
        "batch_size": 10,
        "languages": ["nl", "en"],
        "document_folder": "/path/to/documents",
        "json_output_dir": "/path/to/output",
        "debug_output_dir": "/path/to/debug",
        "metadata_path": "/path/to/metadata.csv",
    }


@pytest.fixture
def mock_pdf_files(tmp_path):
    """Create mock PDF files for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create PDF files
    pdf_files = [
        "doc-100.pdf",
        "doc-101.pdf",
        "besluit-1-doc-200.pdf",
        "besluit-2-doc-300.pdf",
    ]

    for pdf_file in pdf_files:
        (docs_dir / pdf_file).touch()

    return docs_dir


class TestPDFConverterExtended:
    """Extended tests for the PDFConverterManager class."""

    def test_create_pdf_converter_with_min_config(self):
        """Test creating a PDFConverterManager with minimal configuration."""
        # Create minimal config
        min_config = {
            "llm_service": {
                "type": "openai",
                "model": "gpt-4",
            },
            "document_folder": "/path/to/docs",
        }

        # Mock create_model_dict and other dependencies
        with (
            patch(
                "standard_data_format.src.pdf_converter.ConfigParser"
            ) as mock_config_parser,
        ):

            # Mock config parser instance
            mock_config_parser_instance = MagicMock()
            mock_config_parser.return_value = mock_config_parser_instance
            mock_config_parser_instance.generate_config_dict.return_value = {}
            mock_config_parser_instance.get_processors.return_value = []
            mock_config_parser_instance.get_renderer.return_value = MagicMock()

            # Create converter
            converter = PDFConverterManager(min_config)

            # Check that no converters are created until they're accessed
            assert converter._converter_non_forced_ocr is None
            assert converter._converter_forced_ocr is None
            assert converter._converter_llm is None

    @patch("standard_data_format.src.pdf_converter.PdfConverter")
    @patch("standard_data_format.src.pdf_converter.ConfigParser")
    @patch("standard_data_format.src.pdf_converter.create_model_dict")
    def test_converter_lazy_loading(
        self,
        mock_create_model_dict,
        mock_config_parser,
        mock_pdf_converter,
        sample_config,
    ):
        """Test lazy loading of converters."""
        # Mock config parser instance
        mock_config_parser_instance = MagicMock()
        mock_config_parser.return_value = mock_config_parser_instance
        mock_config_parser_instance.generate_config_dict.return_value = {}
        mock_config_parser_instance.get_processors.return_value = []
        mock_config_parser_instance.get_renderer.return_value = MagicMock()

        # Create converter manager
        converter_manager = PDFConverterManager(sample_config)

        # Initially, no converters should be created
        assert converter_manager._converter_non_forced_ocr is None
        assert converter_manager._converter_forced_ocr is None
        assert converter_manager._converter_llm is None

        # Access the converters to trigger lazy loading
        converter_manager.converter_non_forced_ocr
        assert converter_manager._converter_non_forced_ocr is not None
        assert mock_pdf_converter.call_count == 1

        converter_manager.converter_forced_ocr
        assert converter_manager._converter_forced_ocr is not None
        assert mock_pdf_converter.call_count == 2

        converter_manager.converter_llm
        assert converter_manager._converter_llm is not None
        assert mock_pdf_converter.call_count == 3

    @patch("standard_data_format.src.pdf_converter.PdfConverter")
    @patch("standard_data_format.src.pdf_converter.ConfigParser")
    @patch("standard_data_format.src.pdf_converter.create_model_dict")
    def test_create_pdf_converter(
        self,
        mock_create_model_dict,
        mock_config_parser,
        mock_pdf_converter,
        sample_config,
    ):
        """Test the create_pdf_converter method."""
        # Mock config parser instance
        mock_config_parser_instance = MagicMock()
        mock_config_parser.return_value = mock_config_parser_instance
        mock_config_parser_instance.generate_config_dict.return_value = {}
        mock_config_parser_instance.get_processors.return_value = []
        mock_config_parser_instance.get_renderer.return_value = MagicMock()

        # Create converter manager
        converter_manager = PDFConverterManager(sample_config)

        # To test the config update, we need to directly modify the config passed to ConfigParser
        # We'll check the actual calls to ConfigParser
        converter_manager.create_pdf_converter(force_ocr=False, use_llm=False)
        converter_manager.create_pdf_converter(force_ocr=True, use_llm=False)
        converter_manager.create_pdf_converter(use_llm=True)

        # Verify PdfConverter was called with different configs
        assert mock_pdf_converter.call_count == 3

        # Check that each call to ConfigParser received different configs
        config_calls = mock_config_parser.call_args_list

        # Extract the updated configs
        first_call_config = config_calls[0][0][0]
        second_call_config = config_calls[1][0][0]
        third_call_config = config_calls[2][0][0]

        # Verify each config has the expected use_llm setting
        assert first_call_config.get("use_llm") is False
        assert second_call_config.get("use_llm") is False
        assert third_call_config.get("use_llm") is True

    @patch("standard_data_format.src.pdf_converter.PdfConverter")
    @patch("standard_data_format.src.pdf_converter.ConfigParser")
    @patch("standard_data_format.src.pdf_converter.create_model_dict")
    def test_get_markdown_content(
        self,
        mock_create_model_dict,
        mock_config_parser,
        mock_pdf_converter,
        sample_config,
    ):
        """Test the get_markdown_content method."""
        # Create converter manager
        converter_manager = PDFConverterManager(sample_config)

        # Test with object that has markdown attribute
        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Markdown"
        content = converter_manager.get_markdown_content(mock_rendered)
        assert content == "# Test Markdown"

        # Test with dictionary
        mock_dict = {"markdown": "# Dictionary Markdown"}
        content = converter_manager.get_markdown_content(mock_dict)
        assert content == "# Dictionary Markdown"

        # Test with string
        content = converter_manager.get_markdown_content("# String Markdown")
        assert content == "# String Markdown"

        # Test with empty string
        content = converter_manager.get_markdown_content("")
        assert content == ""

    @patch("standard_data_format.src.pdf_converter.PdfConverter")
    @patch("standard_data_format.src.pdf_converter.ConfigParser")
    @patch("standard_data_format.src.pdf_converter.create_model_dict")
    def test_check_markdown_quality(
        self,
        mock_create_model_dict,
        mock_config_parser,
        mock_pdf_converter,
        sample_config,
    ):
        """Test the check_markdown_quality method."""
        # Create converter manager
        converter_manager = PDFConverterManager(sample_config)

        # Test with good content
        good_content = """
        # Sample Document
        
        This is a well-formatted document with proper paragraphs.
        
        It has multiple lines and good structure.
        """
        issues = converter_manager.check_markdown_quality(good_content)
        assert not issues, "No issues should be found in good content"

        # Test with empty content
        empty_content = ""
        issues = converter_manager.check_markdown_quality(empty_content)
        assert "empty_content" in issues

        # Test with very short content
        short_content = "Too short"
        issues = converter_manager.check_markdown_quality(short_content)
        assert "very_short_content" in issues

        # Test with excessive whitespace
        whitespace_content = "Content with\n\n\n\n\n\ntoo many\n\n\n\n\n\nnewlines"
        issues = converter_manager.check_markdown_quality(whitespace_content)
        assert "excessive_newlines" in issues

        # Test with OCR artifacts
        ocr_artifact_content = "Text with ||| too many ||| vertical bars"
        issues = converter_manager.check_markdown_quality(ocr_artifact_content)
        assert "ocr_artifacts" in issues

        # Test with table content (should not flag certain issues)
        table_content = """
        | Header 1 | Header 2 | Header 3 |
        |----------|----------|----------|
        | Cell 1   | Cell 2   | Cell 3   |
        | Cell 4   | Cell 5   | Cell 6   |
        """
        issues = converter_manager.check_markdown_quality(table_content)
        assert (
            "ocr_artifacts" not in issues
        ), "Table content should not trigger OCR artifact detection"

    def test_find_pdf_for_id_patterns(self, mock_pdf_files):
        """Test finding PDF files with different naming patterns."""
        # Create a MetadataManager instance
        metadata_manager = MetadataManager(metadata_df=pd.DataFrame())

        # Test finding a PDF with exact match
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="doc-100", besluit_id=None, search_paths=[mock_pdf_files]
        )
        assert pdf_path is not None
        assert Path(pdf_path).name == "doc-100.pdf"

        # Test ID with besluit_id using hyphen
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="doc-200", besluit_id="besluit-1", search_paths=[mock_pdf_files]
        )
        assert pdf_path is not None
        assert Path(pdf_path).name == "besluit-1-doc-200.pdf"

        # Test ID with besluit_id using hyphen
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="doc-300", besluit_id="besluit-2", search_paths=[mock_pdf_files]
        )
        assert pdf_path is not None
        assert Path(pdf_path).name == "besluit-2-doc-300.pdf"

        # Test non-existent file
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="nonexistent", besluit_id=None, search_paths=[mock_pdf_files]
        )
        assert pdf_path is None

    @patch("standard_data_format.src.pdf_converter.PdfConverter")
    @patch("standard_data_format.src.pdf_converter.ConfigParser")
    @patch("standard_data_format.src.pdf_converter.create_model_dict")
    @patch("standard_data_format.src.pdf_converter.logger")
    def test_process_document(
        self,
        mock_logger,
        mock_create_model_dict,
        mock_config_parser,
        mock_pdf_converter,
        sample_config,
        tmp_path,
    ):
        """Test processing a document with the tiered approach."""
        # Create test metadata
        metadata = {
            "ID": "test-123",
            "Document": "Test Document",
            "Datum": "2023-01-01",
            "File Type": "PDF",
            "Document Link": "http://example.com/test",
        }

        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.touch()

        # Test case 1: Regular conversion succeeds
        # Create a subclass of PDFConverterManager to override problematic methods
        class TestPDFConverter(PDFConverterManager):
            def _try_regular_conversion(self, pdf_path):
                # Override to return a fixed value without raising exceptions
                return "Regular content"

            def _has_quality_issues(self, content):
                # Override to return False to avoid quality checks
                return False

        # Use our test subclass
        converter = TestPDFConverter(sample_config)
        result = converter.process_document(str(test_pdf), metadata)

        # Verify results
        assert result is not None
        assert result["id"] == "test-123"
        assert result["document"] == "Test Document"
        assert result["type"] == "PDF"
        assert result["link"] == "http://example.com/test"
        assert result["markdown_content"] == "Regular content"
        assert result["processed_status"] == "processed"

        # Test case 2: Regular conversion fails, forced OCR succeeds
        class TestPDFConverterFallback(PDFConverterManager):
            def _try_regular_conversion(self, pdf_path):
                # Simulate failure by returning None instead of raising exception
                return None

            def _try_forced_ocr(self, pdf_path):
                # Return successful OCR content
                return "Forced OCR content"

            def _has_quality_issues(self, content):
                # Override to return False to avoid quality checks
                return False

        # Use our test subclass
        converter = TestPDFConverterFallback(sample_config)
        result = converter.process_document(str(test_pdf), metadata)

        # Verify results
        assert result is not None
        assert result["markdown_content"] == "Forced OCR content"
        assert result["processed_status"] == "processed"

        # Test case 3: All conversions fail
        class TestPDFConverterAllFail(PDFConverterManager):
            def _try_regular_conversion(self, pdf_path):
                # Simulate failure by returning None
                return None

            def _try_forced_ocr(self, pdf_path):
                # Simulate failure by returning None
                return None

            def _try_llm_conversion(self, pdf_path):
                # Simulate failure by returning None
                return None

            def _has_quality_issues(self, content):
                # This should not matter since all conversions fail
                return False

        # Use our test subclass
        converter = TestPDFConverterAllFail(sample_config)
        result = converter.process_document(str(test_pdf), metadata)

        # Verify results
        assert result is not None
        assert result["processed_status"] == "failed"
        assert result["markdown_content"] == ""
