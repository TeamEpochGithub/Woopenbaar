from standard_data_format.src.document import Document


def test_document_creation():
    doc = Document(
        uuid="test-uuid",
        vws_id="test-id",
        create_date="2024-01-01",
        type="PDF",
        link="http://test.com",
        attachment_links=["http://test2.com"],
        content="# Test Content",
    )

    assert doc.uuid == "test-uuid"
    assert doc.vws_id == "test-id"
    assert doc.create_date == "2024-01-01"
    assert doc.type == "PDF"
    assert doc.link == "http://test.com"
    assert doc.attachment_links == ["http://test2.com"]
    assert doc.content == "# Test Content"


def test_document_to_json():
    doc = Document(
        uuid="test-uuid",
        vws_id="test-id",
        create_date="2024-01-01",
        type="PDF",
        link="http://test.com",
        attachment_links=["http://test2.com"],
        content="# Test Content",
    )

    json_data = doc.to_json()

    assert isinstance(json_data, dict)
    assert json_data["uuid"] == "test-uuid"
    assert json_data["vws_id"] == "test-id"
    assert json_data["create_date"] == "2024-01-01"
    assert json_data["type"] == "PDF"
    assert json_data["link"] == "http://test.com"
    assert json_data["attachment_links"] == ["http://test2.com"]
    assert json_data["content"] == "# Test Content"


def test_document_from_metadata():
    metadata_json = {
        "id": "test-id",
        "datum": "2024-01-01",
        "type": "PDF",
        "link": "http://test.com",
        "attachment_links": ["http://test2.com"],
    }
    content = "# Test Content"
    uuid = "test-uuid"

    doc = Document.from_metadata(metadata_json, uuid, content)

    assert doc.uuid == uuid
    assert doc.vws_id == metadata_json["id"]
    assert doc.create_date == metadata_json["datum"]
    assert doc.type == metadata_json["type"]
    assert doc.link == metadata_json["link"]
    assert doc.attachment_links == metadata_json["attachment_links"]
    assert doc.content == content


def test_document_uuid_handling():
    """Test document UUID handling"""
    # Test with empty UUID
    doc = Document(
        uuid="",
        vws_id="test-id",
        create_date="2024-01-01",
        type="PDF",
        link="http://test.com",
        content="# Test Content",
    )
    assert doc.uuid == ""

    # Test with None UUID
    doc = Document(
        uuid=None,
        vws_id="test-id",
        create_date="2024-01-01",
        type="PDF",
        link="http://test.com",
        content="# Test Content",
    )
    assert doc.uuid is None


def test_document_metadata_validation():
    """Test document metadata validation"""
    metadata_json = {
        "id": "test-id",
        "datum": "2024-01-01",
        # Missing type and link
    }
    content = "# Test Content"
    uuid = "test-uuid"

    doc = Document.from_metadata(metadata_json, content, uuid)
    assert doc.type == ""
    assert doc.link == ""

    # Test with empty attachment links
    metadata_json["attachment_links"] = []
    doc = Document.from_metadata(metadata_json, content, uuid)
    assert len(doc.attachment_links) == 0
