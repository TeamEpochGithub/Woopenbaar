# Data Classes

The data_classes package provides the core data structures for document processing and retrieval in the Raggle backend. These classes represent documents at various stages of processing and form the foundation for information retrieval operations.

## Core Classes

### RawDocument

Represents an unprocessed document before preprocessing:

- `document_uuid`: Unique identifier for the document
- `type`: Document type (email, pdf, etc.)
- `link`: URL or reference to original document
- `content`: Raw text content
- `content_date`: Date of the document content
- `metadata`: Additional document metadata

### DocumentChunk

Represents a segment of a document after chunking:

- `uuid`: Unique identifier for the chunk
- `document_uuid`: Reference to parent document
- `type`: Document type inherited from parent
- `link`: Reference to original document
- `content`: Text content of the chunk
- `metadata`: Metadata derived from parent document and chunk-specific details
- `embedding`: Optional vector representation of the chunk content

### ChunkedDocument

Represents a document that has been split into chunks:

- `document_uuid`: Unique identifier for the document
- `type`: Document type (email, pdf, etc.)
- `link`: URL or reference to original document
- `metadata`: Additional document metadata
- `chunks`: List of DocumentChunk objects

### ContextPiece

Represents a piece of context for query processing:

- `content`: Text content of the context piece
- `metadata`: Associated metadata
- `evaluation`: Evaluation of relevance to query
- `confidence`: Confidence score of relevance

### QueryResult

Represents the result of a processed query:

- `query`: The original query
- `response`: Generated response
- `chunks`: List of document chunks used in the response
- `source_documents`: List of source documents
- `reasoning_steps`: List of reasoning steps performed
- `context`: Raw context pieces used in the response
- `data_sources_used`: Set of data sources used

## Filter Classes

### FilterOptions

Represents filtering criteria for document retrieval:

- `document_type`: Filter by document type
- `period`: Filter by time period
- `subject`: Filter by document subject
- `min_date`: Filter by minimum date
- `max_date`: Filter by maximum date
- Supports additional custom filters as key-value pairs

### PeriodFilter

Predefined time periods for filtering:

- `ALL_TIME`
- `LAST_WEEK`
- `LAST_MONTH`
- `LAST_3_MONTHS`
- `LAST_6_MONTHS`
- `LAST_YEAR`

## Usage

### Document Processing Flow

The documents follow a processing pipeline that transforms them through these classes:

1. **Raw Document**: Initial unprocessed document
2. **Chunking**: Document is split into chunks
3. **Chunked Document**: Final document with chunks ready for indexing
4. **Retrieval**: DocumentChunks are retrieved and used as context
5. **Context Pieces**: Retrieved chunks are converted to context pieces
6. **Query Result**: Final result including response and relevant chunks

### Serialization

All classes provide serialization methods for easy storage and transmission:

```python
# Convert to JSON-serializable dictionary
chunk_dict = document_chunk.to_json()

# Serialize the entire document with chunks
document_dict = chunked_document.to_json()
```

### Equality and Hashing

The classes implement equality and hashing functions for use in collections:

```python
# Use chunks in sets
chunk_set = {chunk1, chunk2, chunk3}

# Compare chunks and documents
if chunk1 == chunk2:
    print("Same chunk")
```

### Filter Creation

Create filters for document retrieval:

```python
# Create a filter for HR documents from the last month
filters = FilterOptions(
    document_type="HR",
    period=PeriodFilter.LAST_MONTH
)
```

## Implementation Details

- All classes include comprehensive type annotations
- Classes use dataclasses for cleaner code and automatic method generation
- JSON serialization methods handle special types like dates and UUIDs
- Circular imports are avoided with TYPE_CHECKING guards 