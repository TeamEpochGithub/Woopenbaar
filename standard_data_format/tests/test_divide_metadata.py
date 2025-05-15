import pandas as pd
import pytest

from standard_data_format.src.divide_metadata import divide_metadata


@pytest.fixture
def sample_metadata_df():
    """Create a sample metadata DataFrame for testing."""
    return pd.DataFrame(
        {
            "ID": [f"doc-{i}" for i in range(1, 21)],
            "Family": [
                f"family-{i // 3 + 1}" for i in range(1, 21)
            ],  # 7 families (every 3 docs)
            "Document": [f"Test Document {i}" for i in range(1, 21)],
            "Document Link": [f"http://test.com/doc{i}" for i in range(1, 21)],
        }
    )


@pytest.fixture
def metadata_csv(tmp_path, sample_metadata_df):
    """Create a temporary metadata CSV file."""
    csv_path = tmp_path / "metadata.csv"
    sample_metadata_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output_path = tmp_path / "output"
    output_path.mkdir(exist_ok=True)
    return output_path


class TestDivideMetadata:
    """Tests for the divide_metadata function."""

    def test_divide_by_family(self, metadata_csv, output_dir, sample_metadata_df):
        """Test dividing metadata by family."""
        # Run the function to divide metadata by family
        result = divide_metadata(
            metadata_path=str(metadata_csv),
            output_dir=str(output_dir),
            total_chunks=3,
            by_family=True,
        )

        # Check that the function completed successfully
        assert result is True

        # Check that the correct number of chunk files were created
        chunk_files = list(output_dir.glob("metadata_chunk_*.csv"))
        assert len(chunk_files) == 3

        # Verify each chunk has families grouped together
        for chunk_file in chunk_files:
            chunk_df = pd.read_csv(chunk_file)

            # Get families in this chunk
            families = chunk_df["Family"].unique()

            # Check that each family's documents are all in this chunk
            for family in families:
                original_family_count = len(
                    sample_metadata_df[sample_metadata_df["Family"] == family]
                )
                chunk_family_count = len(chunk_df[chunk_df["Family"] == family])
                assert (
                    original_family_count == chunk_family_count
                ), f"Family {family} was split across chunks"

    def test_divide_by_document(self, metadata_csv, output_dir):
        """Test dividing metadata by individual document."""
        # Run the function to divide metadata by document
        result = divide_metadata(
            metadata_path=str(metadata_csv),
            output_dir=str(output_dir),
            total_chunks=4,
            by_family=False,
        )

        # Check that the function completed successfully
        assert result is True

        # Check that the correct number of chunk files were created
        chunk_files = list(output_dir.glob("metadata_chunk_*.csv"))
        assert len(chunk_files) == 4

        # Read original data
        original_df = pd.read_csv(metadata_csv)

        # Verify that all documents are distributed
        total_docs = 0
        for chunk_file in chunk_files:
            chunk_df = pd.read_csv(chunk_file)
            total_docs += len(chunk_df)

        # The total number of documents should match the original
        assert total_docs == len(original_df)

        # Check that the chunks have some documents in them (except possibly the last one)
        for i, chunk_file in enumerate(sorted(chunk_files)):
            chunk_df = pd.read_csv(chunk_file)
            if i < len(chunk_files) - 1:  # Not the last chunk
                assert len(chunk_df) > 0, f"Non-last chunk {i} should have documents"

    def test_empty_metadata(self, tmp_path):
        """Test with empty metadata."""
        # Create empty metadata file
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)

        output_path = tmp_path / "empty_output"
        output_path.mkdir(exist_ok=True)

        # Run the function
        result = divide_metadata(
            metadata_path=str(empty_csv),
            output_dir=str(output_path),
            total_chunks=2,
            by_family=True,
        )

        # Check that the function completed
        assert result is True

        # Check output - should have created empty chunk files
        chunk_files = list(output_path.glob("metadata_chunk_*.csv"))
        assert len(chunk_files) == 2

        # Verify the files are essentially empty (just headers)
        for chunk_file in chunk_files:
            chunk_df = pd.read_csv(chunk_file)
            assert len(chunk_df) == 0
