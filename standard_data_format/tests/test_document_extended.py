import json
from unittest.mock import patch

from standard_data_format.src.document import Document


class TestDocumentExtended:
    """Extended tests for the Document class."""

    def test_document_equality(self):
        """Test equality comparison of Document instances."""
        # Create identical documents
        doc1 = Document(
            uuid="test-uuid-1",
            vws_id="doc-123",
            create_date="2023-01-01",
            type="PDF",
            link="http://test.com/doc",
        )

        doc2 = Document(
            uuid="test-uuid-1",
            vws_id="doc-123",
            create_date="2023-01-01",
            type="PDF",
            link="http://test.com/doc",
        )

        # Create a different document
        doc3 = Document(
            uuid="test-uuid-2",
            vws_id="doc-456",
            create_date="2023-02-01",
            type="PDF",
            link="http://test.com/doc2",
        )

        # Check equality
        assert doc1.to_json() == doc2.to_json()
        assert doc1.to_json() != doc3.to_json()

    def test_document_serialization_deserialization(self):
        """Test serialization and deserialization of Document."""
        # Create a document
        original_doc = Document(
            uuid="test-uuid-1",
            vws_id="doc-123",
            create_date="2023-01-01",
            type="PDF",
            link="http://test.com/doc",
            content="Test content",
        )

        # Serialize to JSON
        json_data = original_doc.to_json()
        json_str = json.dumps(json_data)

        # Deserialize from JSON
        json_dict = json.loads(json_str)

        # Recreate document from the loaded dict
        recreated_doc = Document(
            uuid=json_dict["uuid"],
            vws_id=json_dict["vws_id"],
            create_date=json_dict["create_date"],
            type=json_dict["type"],
            link=json_dict["link"],
            attachment_links=json_dict["attachment_links"],
            content=json_dict["content"],
        )

        # Verify that the recreated document matches the original
        assert recreated_doc.uuid == original_doc.uuid
        assert recreated_doc.vws_id == original_doc.vws_id
        assert recreated_doc.create_date == original_doc.create_date
        assert recreated_doc.type == original_doc.type
        assert recreated_doc.link == original_doc.link
        assert recreated_doc.content == original_doc.content

    def test_from_metadata_with_missing_fields(self):
        """Test creating a Document from metadata with missing fields."""
        # Create minimal metadata
        metadata = {"id": "doc-123", "Document": "Test Document"}

        # Create document from minimal metadata
        doc = Document.from_metadata(
            metadata_json=metadata, uuid="test-uuid", content="Test content"
        )

        # Check required fields are set from metadata
        assert doc.uuid == "test-uuid"
        assert doc.vws_id == "doc-123"
        assert doc.content == "Test content"

        # Check optional fields have defaults
        assert doc.create_date == ""
        assert doc.type == ""
        assert doc.link == ""
        assert doc.attachment_links == []

    @patch("standard_data_format.src.document.logger")
    def test_document_logging(self, mock_logger):
        """Test that document operations are logged correctly."""
        # Create a document and check that creation is logged
        doc = Document(
            uuid="test-uuid-1",
            vws_id="doc-123",
            create_date="2023-01-01",
            type="PDF",
            link="http://test.com/doc",
        )

        # Call to_json and check that it's logged
        doc.to_json()

        # Verify that to_json was logged at info level
        mock_logger.info.assert_any_call("Converting document doc-123 to JSON format")

        # Create a document from metadata and check that it's logged
        metadata = {"id": "doc-456"}
        Document.from_metadata(
            metadata_json=metadata, uuid="test-uuid-2", content="Test content"
        )

        # Verify that from_metadata was logged at debug level
        mock_logger.debug.assert_any_call(
            "Creating document from metadata for ID doc-456"
        )

    def test_document_with_empty_content(self):
        """Test behavior of Document with empty content."""
        # Create a document with empty content
        doc = Document(
            uuid="test-uuid-1",
            vws_id="doc-123",
            create_date="2023-01-01",
            type="PDF",
            link="http://test.com/doc",
            content="",
        )

        # Check that content is an empty string
        assert doc.content == ""

        # Verify that to_json works with empty content
        json_dict = doc.to_json()
        assert json_dict["content"] == ""

        # Test the default value for content
        doc2 = Document(
            uuid="test-uuid-2",
            vws_id="doc-456",
            create_date="2023-01-01",
            type="PDF",
            link="http://test.com/doc2",
        )

        # Check that default content is an empty string
        assert doc2.content == ""
