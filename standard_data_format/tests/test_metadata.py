from pathlib import Path

import pandas as pd
import pytest

from standard_data_format.src.metadata import MetadataManager


@pytest.fixture
def sample_excel_mapping():
    return [{"base_url": "https://test.com/dossier/123", "csv_file": "test_data.xlsx"}]


@pytest.fixture
def sample_metadata_df():
    return pd.DataFrame(
        {
            "ID": ["test-123", "test-456"],
            "Matter": ["M1", "M2"],
            "Document": ["Test Doc 1", "Test Doc 2"],
            "Document Link": ["http://test1.com", "http://test2.com"],
        }
    )


def test_combine_duplicate_columns(sample_metadata_df):
    manager = MetadataManager(metadata_df=sample_metadata_df)

    # Create DataFrame with duplicate columns
    df_with_dups = sample_metadata_df.copy()
    df_with_dups["Document_2"] = df_with_dups["Document"] + "_alt"
    df_with_dups.columns = ["ID", "Matter", "Document", "Document Link", "Document"]
    print(df_with_dups.head())
    result_df = manager.combine_duplicate_columns(df_with_dups)
    assert len(result_df.columns) == len(set(result_df.columns))
    assert "Document" in result_df.columns
    print(result_df["Document"].iloc[0], "Test Doc 1, Test Doc 1_alt")
    assert result_df["Document"].iloc[0] == "Test Doc 1, Test Doc 1_alt"


def test_find_pdf_for_id(tmp_path):
    """Test finding PDF file by ID."""
    # Create test files in the exact format the function expects to find
    test_files = [
        "test-123.pdf",  # Exact match for 'test-123'
        "test-456.pdf",  # Exact match for 'test-456'
        "test-789.pdf",  # Exact match for 'test-789'
    ]

    for file in test_files:
        (tmp_path / file).touch()

    # Initialize metadata manager
    metadata_df = pd.DataFrame(
        {
            "ID": ["test-123", "test-456", "test-789"],
            "uuid": ["uuid1", "uuid2", "uuid3"],
            "Family": ["family1", "family1", "family2"],
        }
    )
    metadata_manager = MetadataManager(metadata_df, str(tmp_path))

    # Create search_paths structure
    search_paths = [tmp_path]

    # Test exact match
    pdf_path = metadata_manager.find_pdf_for_id(
        "test-123", besluit_id=None, search_paths=search_paths
    )
    assert (
        pdf_path is not None
    ), f"Expected a PDF path but got None for test-123 (files: {list(Path(tmp_path).glob('*.pdf'))})"
    if pdf_path:
        path_obj = Path(pdf_path)
        assert "test-123" in path_obj.name

    # Test exact match for test-456
    pdf_path = metadata_manager.find_pdf_for_id(
        "test-456", besluit_id=None, search_paths=search_paths
    )
    assert pdf_path is not None, "Expected a PDF path but got None for test-456"
    if pdf_path:
        path_obj = Path(pdf_path)
        assert "test-456" in path_obj.name

    # Test exact match for test-789
    pdf_path = metadata_manager.find_pdf_for_id(
        "test-789", besluit_id=None, search_paths=search_paths
    )
    assert pdf_path is not None, "Expected a PDF path but got None for test-789"
    if pdf_path:
        path_obj = Path(pdf_path)
        assert "test-789" in path_obj.name


def test_uuid_validation():
    """Test UUID validation functions"""
    from standard_data_format.src.metadata import MetadataManager

    # Create a dummy instance to access the static method
    metadata_manager = MetadataManager()

    # Test various empty cases
    assert metadata_manager._is_empty_uuid(None)
    assert metadata_manager._is_empty_uuid("nan")
    assert metadata_manager._is_empty_uuid("NA")
    assert metadata_manager._is_empty_uuid("")
    assert metadata_manager._is_empty_uuid(" ")

    # Test valid UUIDs
    assert not metadata_manager._is_empty_uuid("123e4567-e89b-12d3-a456-426614174000")
    assert not metadata_manager._is_empty_uuid("valid-uuid-string")


def test_metadata_division(tmp_path):
    """Test metadata division functionality"""
    # Create test DataFrame
    df = pd.DataFrame(
        {
            "ID": [f"id_{i}" for i in range(10)],
            "uuid": [f"uuid_{i}" if i % 2 == 0 else None for i in range(10)],
        }
    )

    # Save to temporary directory
    input_path = tmp_path / "test_metadata.csv"
    df.to_csv(input_path, index=False)

    # Test division
    chunk_size = 3
    for i in range(0, len(df), chunk_size):
        chunk_path = tmp_path / f"metadata_chunk_{i//chunk_size}.csv"
        chunk = df[i : i + chunk_size]
        chunk.to_csv(chunk_path, index=False)

        # Verify chunk
        loaded_chunk = pd.read_csv(chunk_path)
        assert len(loaded_chunk) <= chunk_size
        assert all(loaded_chunk.columns == df.columns)
