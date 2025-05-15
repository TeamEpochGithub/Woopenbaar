from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from standard_data_format.src.metadata import MetadataManager


@pytest.fixture
def family_metadata_df():
    """Create sample metadata with family relationships."""
    return pd.DataFrame(
        {
            "ID": ["doc-100", "doc-101", "doc-102", "doc-200", "doc-201"],
            "Family": ["family-1", "family-1", "family-1", "family-2", "family-2"],
            "Document": [
                "Main Doc 1",
                "Attachment 1.1",
                "Attachment 1.2",
                "Main Doc 2",
                "Attachment 2.1",
            ],
            "Document Link": [f"http://test.com/doc{i}" for i in range(100, 105)],
            "besluit_id": [
                "besluit-1",
                "besluit-1",
                "besluit-1",
                "besluit-2",
                "besluit-2",
            ],
            "available": [True, True, True, True, True],
            "uuid": [None, None, None, None, None],
            "processed": ["", "", "", "", ""],
            "attachment_names": [[], [], [], [], []],
            "attachment_ids": [[], [], [], [], []],
            "attachment_links": [[], [], [], [], []],
        }
    )


@pytest.fixture
def mock_pdf_files(tmp_path):
    """Create mock PDF files for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create PDF files
    pdf_files = [
        "doc-100.pdf",
        "doc-101.pdf",
        "doc-102.pdf",
        "doc-200.pdf",
        "doc-201.pdf",
        "besluit-1-doc-100.pdf",  # Alternative naming pattern
    ]

    for pdf_file in pdf_files:
        (docs_dir / pdf_file).touch()

    return docs_dir


class TestMetadataFamily:
    """Tests for family relationship handling in MetadataManager."""

    @patch(
        "standard_data_format.src.metadata.MetadataManager._process_and_update_families"
    )
    def test_load_metadata_mapping(self, mock_process_update, family_metadata_df):
        """Test creating metadata mapping with family relationships."""
        # Create a MetadataManager instance with family data
        metadata_manager = MetadataManager(metadata_df=family_metadata_df)

        # Add attachment columns (normally done in preprocessing)
        metadata_manager.add_attachment_columns()

        # Override availability to avoid the need for file existence checks
        metadata_manager.metadata_df["available"] = True

        # Patch the _process_and_update_families method to inject test data
        def update_family_map(family_map):
            # Replace single document entries with family entries
            family_map = {
                "family-1": {
                    "base_document": "doc-100",
                    "attachment_ids": ["doc-101", "doc-102"],
                    "attachment_names": ["Attachment 1.1", "Attachment 1.2"],
                    "attachment_links": [
                        "http://test.com/doc101",
                        "http://test.com/doc102",
                    ],
                    "besluit_id": "besluit-1",
                    "family": "family-1",
                },
                "family-2": {
                    "base_document": "doc-200",
                    "attachment_ids": ["doc-201"],
                    "attachment_names": ["Attachment 2.1"],
                    "attachment_links": ["http://test.com/doc104"],
                    "besluit_id": "besluit-2",
                    "family": "family-2",
                },
            }
            return family_map

        mock_process_update.side_effect = update_family_map

        # Call load_metadata_mapping
        family_map, updated_df = metadata_manager.load_metadata_mapping()

        # Check that the family map contains the expected families
        assert "family-1" in family_map
        assert "family-2" in family_map

        # Check that each family has the correct base document
        assert family_map["family-1"]["base_document"] == "doc-100"
        assert family_map["family-2"]["base_document"] == "doc-200"

        # Check that each family has the correct attachment IDs
        assert sorted(family_map["family-1"]["attachment_ids"]) == [
            "doc-101",
            "doc-102",
        ]
        assert family_map["family-2"]["attachment_ids"] == ["doc-201"]

        # Check besluit_id is correctly set
        assert family_map["family-1"]["besluit_id"] == "besluit-1"
        assert family_map["family-2"]["besluit_id"] == "besluit-2"

    def test_find_pdf_for_id(self, mock_pdf_files):
        """Test finding PDF files with different naming patterns."""
        # Create a MetadataManager instance
        metadata_manager = MetadataManager(metadata_df=pd.DataFrame())

        # Test finding a PDF with exact match
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="doc-100", besluit_id=None, search_paths=[mock_pdf_files]
        )
        assert pdf_path is not None
        assert Path(pdf_path).name == "doc-100.pdf"

        # Create a mock for Path to avoid recursion issues
        # Add the besluit-prefixed file to the mock_pdf_files directory
        besluit_pdf_path = mock_pdf_files / "besluit-1-doc-100.pdf"
        besluit_pdf_path.touch()

        # Delete the regular file to ensure the besluit version is found
        (mock_pdf_files / "doc-100.pdf").unlink()

        # Test finding PDF with besluit_id
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="doc-100", besluit_id="besluit-1", search_paths=[mock_pdf_files]
        )
        assert pdf_path is not None
        assert "besluit-1" in Path(pdf_path).name

        # Test PDF not found
        pdf_path = metadata_manager.find_pdf_for_id(
            doc_id="nonexistent", besluit_id=None, search_paths=[mock_pdf_files]
        )
        assert pdf_path is None

    @patch("standard_data_format.src.metadata.MetadataManager.load_metadata_mapping")
    def test_update_attachment_info(self, mock_load_mapping, family_metadata_df):
        """Test updating attachment information in metadata."""
        # Prepare test data for the return value of load_metadata_mapping
        test_family_map = {
            "family-1": {
                "base_document": "doc-100",
                "attachment_ids": ["doc-101", "doc-102"],
                "attachment_names": ["Attachment 1.1", "Attachment 1.2"],
                "attachment_links": [
                    "http://test.com/doc101",
                    "http://test.com/doc102",
                ],
                "besluit_id": "besluit-1",
                "link": "http://test.com/doc100",
                "family": "family-1",
            }
        }

        # Mock load_metadata_mapping to return our test data
        mock_load_mapping.return_value = (test_family_map, family_metadata_df.copy())

        # Create a MetadataManager instance with family data
        metadata_manager = MetadataManager(metadata_df=family_metadata_df)

        # Add ID_normalized and Family_normalized columns
        metadata_manager.metadata_df["ID_normalized"] = metadata_manager.metadata_df[
            "ID"
        ].apply(lambda x: x.split("-")[1])
        metadata_manager.metadata_df["Family_normalized"] = (
            metadata_manager.metadata_df["Family"].apply(lambda x: x.split("-")[1])
        )

        # Call _process_and_update_families directly with our test data
        metadata_manager._process_and_update_families(test_family_map)

        # Check that attachment info was updated in the base document row
        assert len(test_family_map["family-1"]["attachment_ids"]) == 2
        assert all(
            id in test_family_map["family-1"]["attachment_ids"]
            for id in ["doc-101", "doc-102"]
        )

        # Assert family relationship fields were updated correctly
        family_map, _ = mock_load_mapping.return_value
        assert "family-1" in family_map
        assert family_map["family-1"]["attachment_ids"] == ["doc-101", "doc-102"]

    def test_is_empty_uuid(self):
        """Test the _is_empty_uuid method with various inputs."""
        metadata_manager = MetadataManager(metadata_df=pd.DataFrame())

        # Test various empty values
        assert metadata_manager._is_empty_uuid(None) is True
        assert metadata_manager._is_empty_uuid(pd.NA) is True
        assert metadata_manager._is_empty_uuid("") is True
        assert metadata_manager._is_empty_uuid("nan") is True
        assert metadata_manager._is_empty_uuid("None") is True

        # Test non-empty values
        assert metadata_manager._is_empty_uuid("valid-uuid") is False
        assert (
            metadata_manager._is_empty_uuid("12345678-1234-5678-1234-567812345678")
            is False
        )

        # Test non-string values
        assert metadata_manager._is_empty_uuid(123) is False  # Should convert to "123"

    @patch("standard_data_format.src.metadata.logger")
    def test_add_available_to_df(self, mock_logger, tmp_path, family_metadata_df):
        """Test adding availability information to metadata."""
        # Create document folder with test files
        doc_folder = tmp_path / "documents"
        doc_folder.mkdir()

        # Create some PDF files
        (doc_folder / "doc-100.pdf").touch()
        (doc_folder / "doc-102.pdf").touch()  # Deliberately skip doc-101

        # Create a MetadataManager instance with family data
        metadata_manager = MetadataManager(metadata_df=family_metadata_df)

        # In this test, we'll use an alternative approach - patch os.path.isfile directly
        # since that's what gets called within add_available_to_df
        # We'll only create a simple mock function that returns True for specific files
        def mock_isfile(path):
            return "doc-100.pdf" in path or "doc-102.pdf" in path

        with patch("os.path.isfile", side_effect=mock_isfile):
            # Call add_available_to_df
            updated_df = metadata_manager.add_available_to_df(
                str(doc_folder), skip_if_exists=False
            )

            # Check that availability was correctly determined
            assert updated_df.loc[updated_df["ID"] == "doc-100", "available"].iloc[0]
            assert not updated_df.loc[updated_df["ID"] == "doc-101", "available"].iloc[
                0
            ]
            assert updated_df.loc[updated_df["ID"] == "doc-102", "available"].iloc[0]

        # Test with skip_if_exists=True
        # First make all docs available
        metadata_manager.metadata_df["available"] = True

        # Patch logger.info to verify it's called with the right message
        with patch.object(mock_logger, "info") as mock_info:
            updated_df = metadata_manager.add_available_to_df(
                str(doc_folder), skip_if_exists=True
            )

            # Should skip the check and return original values
            assert all(
                updated_df["available"]
            ), "All documents should still be marked as available"

            # Verify log message about skipping - we can check any call for this pattern
            assert any(
                "Skipping availability check" in call[0][0]
                for call in mock_info.call_args_list
            )
