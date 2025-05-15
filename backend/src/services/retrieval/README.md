# Retrieval Service

The Retrieval Service provides document indexing, storage, and semantic search capabilities for the Raggle backend. It implements both standard retrieval and adaptive retrieval approaches, forming the core of the RAG system's information retrieval pipeline.

## Features

- **Document Indexing**: Process and index documents for efficient retrieval
- **Semantic Search**: Find relevant documents based on meaning, not just keywords
- **Hybrid Retrieval**: Combine lexical (BM25) and semantic (dense) search methods
- **Reranking**: Improve relevance with cross-encoder models
- **Filtering**: Apply metadata filters to narrow search results
- **Multi-Source Support**: Organize and search documents across multiple data sources
- **Caching**: Save and load indexed documents for performance optimization

## Key Components

### RetrievalService

The standard retrieval service implementing core retrieval functionality:

- `index`: Process and index document collections
- `find`: Retrieve relevant document chunks for a query
- `find_document_by_id`: Fetch document by unique identifier
- `find_chunk_by_id`: Fetch specific document chunk
- `get_sources`: Retrieve available data sources
- `save`/`from_cache`: Cache retrieval pipeline for performance

### Retrieval Components

The service integrates various components:

- **Embedders**: Convert text to vector embeddings (SentenceTransformerEmbedder)
- **Retrievers**: Find potentially relevant documents
  - Document-level retrievers for broad matching
  - Chunk-level retrievers for specific matching
- **Rerankers**: Improve ranking of search results
  - DateDocumentReranker: Prioritize recent documents
  - CrossEncoderReranker: Score relevance contextually

## Retrieval Pipeline

The standard retrieval process follows these steps:

1. **Query Processing**: Tokenize and normalize the user query
2. **Document Retrieval**: Find potentially relevant documents using document retrievers
3. **Document Reranking**: Score and sort documents by relevance (optional)
4. **Chunk Retrieval**: Find relevant chunks within top documents using chunk retrievers
5. **Chunk Reranking**: Score and sort chunks by relevance with cross-encoders
6. **Filtering**: Apply metadata filters to the results
7. **Result Construction**: Return the most relevant chunks and their source documents

## Data Organization

The service organizes data across multiple sources:

- Each data source maintains separate indexes
- Documents and chunks are stored with source attribution
- Metadata allows for filtering within and across sources
- Global lookups facilitate cross-source retrieval

## Training and Evaluation

The service provides tools for:

- Generating training data from indexed documents
- Training/fine-tuning retrieval components
- Evaluating retrieval performance

## Usage

### Standard Retrieval

```python
# Basic retrieval
chunks, docs = retrieval_service.find(
    query="What is the policy on vacation days?",
    source_name="HR_Policies",
    initial_documents_k=10,
    final_documents_k=3,
    initial_chunks_k=20,
    final_chunks_k=5
)

# Retrieval with filters
chunks, docs = retrieval_service.find(
    query="What is the policy on vacation days?",
    source_name="HR_Policies",
    filters={"document_type": "Policy", "department": "HR"},
    prioritize_earlier=True  # Prioritize more recent documents
)
```

## Indexing

To index a new document collection:

```python
# Process documents with a preprocessor
chunked_documents = preprocessor.process_documents("documents/")

# Index documents to a specific source
retrieval_service.index(
    chunked_documents=chunked_documents,
    source_name="HR_Policies",
    source_description="Human Resources policy documents"
)

# Save the indexed data
retrieval_service.save("cache/")
``` 