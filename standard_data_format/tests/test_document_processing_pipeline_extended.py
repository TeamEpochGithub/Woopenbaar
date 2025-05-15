from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from standard_data_format.src.document import Document
from standard_data_format.src.document_processing_pipeline import (
    DocumentProcessingPipeline,
)


@pytest.fixture
def pipeline_config():
    """Create a sample pipeline configuration."""
    return {
        "batch_size": 2,
        "debug": True,
        "output_format": "markdown",
        "languages": ["nl"],
        "base_dir": "standard_data_format",
        "json_output_dir": "data/output",
        "debug_output_dir": "data/debug",
        "metadata_path": "data/metadata/test_metadata.csv",
        "document_folder": "data/documents",
        "base_path": "/remote/base/path",
    }


@pytest.fixture
def mock_document():
    """Create a mock document for testing."""
    return Document(
        uuid="test-uuid-1",
        vws_id="doc-123",
        create_date="2023-01-01",
        type="PDF",
        link="http://test.com/doc123",
        content="Test document content",
    )


class TestDocumentProcessingPipelineExtended:
    """Extended tests for DocumentProcessingPipeline class."""

    @patch("standard_data_format.src.document_processing_pipeline.subprocess.run")
    @patch("standard_data_format.src.document_processing_pipeline.os.system")
    @patch("standard_data_format.src.document_processing_pipeline.MetadataManager")
    @patch("standard_data_format.src.document_processing_pipeline.PDFConverterManager")
    @patch("standard_data_format.src.document_processing_pipeline.DocumentProcessor")
    @patch("standard_data_format.src.document_processing_pipeline.Path")
    @patch("standard_data_format.src.document_processing_pipeline.pd.read_csv")
    def test_sync_to_base(
        self,
        mock_read_csv,
        mock_path,
        mock_doc_processor,
        mock_pdf_converter,
        mock_metadata_manager,
        mock_os_system,
        mock_subprocess_run,
        pipeline_config,
    ):
        """Test synchronization to a base machine."""
        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.mkdir.return_value = None

        # Mock read_csv to return a DataFrame
        mock_df = pd.DataFrame(
            {"ID": ["doc-1", "doc-2"], "processed": ["true", "true"]}
        )
        mock_read_csv.return_value = mock_df

        # Create pipeline instance with mocked dependencies
        pipeline = DocumentProcessingPipeline(config=pipeline_config)

        # Set required properties for _sync_to_base
        pipeline.base_machine = "user@remote-host"
        pipeline.chunk_id = 1
        pipeline.processed_ids = {"doc-1", "doc-2", "doc-3"}

        # Test sync_to_base
        pipeline._sync_to_base()

        # Verify that sync commands were called
        assert mock_os_system.call_count > 0

        # Check that processed_ids was reset
        assert pipeline.processed_since_sync == 0

    @patch("standard_data_format.src.document_processing_pipeline.Path")
    @patch("standard_data_format.src.document_processing_pipeline.MetadataManager")
    @patch("standard_data_format.src.document_processing_pipeline.PDFConverterManager")
    @patch("standard_data_format.src.document_processing_pipeline.DocumentProcessor")
    @patch("standard_data_format.src.document_processing_pipeline.pd.read_csv")
    def test_save_metadata(
        self,
        mock_read_csv,
        mock_doc_processor,
        mock_pdf_converter,
        mock_metadata_manager,
        mock_path,
        pipeline_config,
    ):
        """Test saving metadata to a CSV file."""
        # Mock path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir.return_value = None
        mock_path_instance.parent = mock_path_instance

        # Mock DataFrame operations
        mock_df = MagicMock()
        mock_df.to_csv = MagicMock()

        # Create pipeline instance
        pipeline = DocumentProcessingPipeline(config=pipeline_config)
        pipeline.metadata_manager = MagicMock()
        pipeline.metadata_manager.metadata_df = mock_df

        # Test save_metadata without base path
        with patch(
            "standard_data_format.src.document_processing_pipeline.subprocess.run"
        ) as mock_run:
            pipeline.config["base_path"] = None
            pipeline._save_metadata(mock_df)

            # Check that to_csv was called
            mock_df.to_csv.assert_called_once()

            # No remote sync should happen
            mock_run.assert_not_called()

        # Test save_metadata with base path
        with patch(
            "standard_data_format.src.document_processing_pipeline.subprocess.run"
        ) as mock_run:
            pipeline.config["base_path"] = "/remote/path"
            pipeline.base_machine = "user@remote-host"
            mock_df.to_csv.reset_mock()
            pipeline._save_metadata(mock_df)

            # Check that to_csv was called again
            assert mock_df.to_csv.call_count > 0

    @patch("standard_data_format.src.document_processing_pipeline.Path")
    @patch("standard_data_format.src.document_processing_pipeline.MetadataManager")
    @patch("standard_data_format.src.document_processing_pipeline.PDFConverterManager")
    @patch("standard_data_format.src.document_processing_pipeline.DocumentProcessor")
    @patch("standard_data_format.src.document_processing_pipeline.pd.read_csv")
    def test_process_documents_distributed(
        self,
        mock_read_csv,
        mock_doc_processor,
        mock_pdf_converter,
        mock_metadata_manager,
        mock_path,
        pipeline_config,
    ):
        """Test processing documents in distributed mode."""
        # Setup mock data
        family_map = {
            "family-1": {
                "base_document": "doc-1",
                "attachment_ids": ["doc-1-1", "doc-1-2"],
                "besluit_id": "besluit-1",
            },
            "family-2": {
                "base_document": "doc-2",
                "attachment_ids": [],
                "besluit_id": "besluit-2",
            },
        }
        mock_df = pd.DataFrame(
            {
                "ID": ["doc-1", "doc-1-1", "doc-1-2", "doc-2"],
                "besluit_id": ["besluit-1", "besluit-1", "besluit-1", "besluit-2"],
                "available": [True, True, True, True],
            }
        )

        # Mock metadata manager
        mock_metadata_instance = MagicMock()
        mock_metadata_manager.return_value = mock_metadata_instance
        mock_metadata_instance.load_metadata_mapping.return_value = (
            family_map,
            mock_df,
        )

        # Mock read_csv to return our test DataFrame
        mock_read_csv.return_value = mock_df

        # Create pipeline instance
        pipeline = DocumentProcessingPipeline(config=pipeline_config)

        # Mock process_documents method to avoid actually processing
        with patch.object(pipeline, "process_documents") as mock_process:
            # Test with default metadata path
            pipeline.process_documents_distributed()

            # Should call process_documents with the family map
            mock_process.assert_called_once()
            args, kwargs = mock_process.call_args

            # Test with custom path by creating a new pipeline with different config
            mock_process.reset_mock()
            custom_path = "custom/metadata.csv"

            # Create modified config with custom metadata path
            custom_config = pipeline_config.copy()
            custom_config["metadata_path"] = custom_path

            # Create new DataFrame for the custom path
            custom_df = pd.DataFrame(
                {
                    "ID": ["custom-1", "custom-2"],
                    "besluit_id": ["besluit-1", "besluit-2"],
                    "available": [True, True],
                }
            )

            # Setup read_csv to return the correct DataFrame based on path
            def mock_read_csv_side_effect(path, **kwargs):
                if path == custom_path:
                    return custom_df
                return mock_df

            mock_read_csv.side_effect = mock_read_csv_side_effect

            # Create new pipeline with custom config
            pipeline_custom = DocumentProcessingPipeline(config=custom_config)

            # Mock its process_documents method
            with patch.object(
                pipeline_custom, "process_documents"
            ) as mock_process_custom:
                pipeline_custom.process_documents_distributed()

                # Verify process_documents was called
                mock_process_custom.assert_called_once()

    @patch("standard_data_format.src.document_processing_pipeline.MetadataManager")
    @patch("standard_data_format.src.document_processing_pipeline.PDFConverterManager")
    @patch("standard_data_format.src.document_processing_pipeline.DocumentProcessor")
    @patch("standard_data_format.src.document_processing_pipeline.pd.read_csv")
    def test_validate_config(
        self,
        mock_read_csv,
        mock_doc_processor,
        mock_pdf_converter,
        mock_metadata_manager,
        pipeline_config,
    ):
        """Test configuration validation."""
        # Create pipeline with valid config
        pipeline = DocumentProcessingPipeline(config=pipeline_config)

        # Test invalid output format
        invalid_config = pipeline_config.copy()
        invalid_config["output_format"] = "invalid_format"

        with pytest.raises(ValueError, match="Unsupported output format"):
            pipeline._validate_config(invalid_config)

        # Test invalid batch size
        invalid_config = pipeline_config.copy()
        invalid_config["batch_size"] = -1

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            pipeline._validate_config(invalid_config)

        # Test missing required fields by setting them to empty values
        # The validation only fails if the value is falsy after defaults are applied
        for required_field in ["base_dir", "json_output_dir", "document_folder"]:
            invalid_config = pipeline_config.copy()
            invalid_config[required_field] = ""  # Empty string which is falsy

            with pytest.raises(
                ValueError, match="Missing required configuration parameter"
            ):
                pipeline._validate_config(invalid_config)

    @patch("standard_data_format.src.document_processing_pipeline.Path")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("standard_data_format.src.document_processing_pipeline.json.dump")
    @patch("standard_data_format.src.document_processing_pipeline.MetadataManager")
    @patch("standard_data_format.src.document_processing_pipeline.PDFConverterManager")
    @patch("standard_data_format.src.document_processing_pipeline.DocumentProcessor")
    @patch("standard_data_format.src.document_processing_pipeline.pd.read_csv")
    def test_save_document(
        self,
        mock_read_csv,
        mock_doc_processor,
        mock_pdf_converter,
        mock_metadata_manager,
        mock_json_dump,
        mock_open,
        mock_path,
        pipeline_config,
        mock_document,
    ):
        """Test saving a document to JSON."""
        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.exists.return_value = True

        # Create pipeline instance
        pipeline = DocumentProcessingPipeline(config=pipeline_config)

        # Test successful save
        result = pipeline._save_document(mock_document)

        # Document should be saved successfully
        assert result is True
        mock_json_dump.assert_called_once()

        # Test error during save
        mock_json_dump.side_effect = Exception("Save error")
        result = pipeline._save_document(mock_document)

        # Should return False on error
        assert result is False
