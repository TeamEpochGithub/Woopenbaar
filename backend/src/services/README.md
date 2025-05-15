# Services Package

The services package provides the core functionality of the Raggle backend through a collection of independent service modules. Each service implements specific business logic and is designed to be independently testable and interchangeable.

## Service Architecture

Services follow a dependency injection pattern, allowing for flexible configuration and testing. The `factory.py` module is responsible for initializing services with their dependencies, centralizing service creation logic.

## Available Services

### Retrieval Service

The retrieval service handles document indexing, storage, and retrieval operations. It provides:

- **Standard Retrieval**: Fixed strategy for document retrieval using a combination of techniques:
  - Document-level retrieval for broad matching
  - Chunk-level retrieval for semantic search
  - Cross-encoder reranking for improved relevance
- **Multi-Source Support**: Organize and search documents across different data sources
- **Filtering**: Apply metadata filters to refine search results
- **Caching**: Save and load indexed documents for performance optimization

### Query Processing Service

The query processing service orchestrates the complete query processing flow, managing how user queries are transformed into answers:

- **Standard Processing**: Direct query processing without subqueries
- **Adaptive Processing**: Multi-step query processing with reasoning
- **Retrieval Management**: Coordinates document retrieval operations
- **Answer Synthesis**: Generates coherent answers from gathered context
- **Follow-up Questions**: Suggests relevant follow-up questions

### LLM Service

The LLM service provides a unified interface for interacting with language models. It supports:

- **Multiple Implementations**: 
  - LocalLLMService: Local models via vLLM
  - GeminiLLMService: Google's Gemini API
  - DeepseekLLMService: DeepSeek's API (via OpenAI SDK)
  - VLLMClientService: External vLLM servers
- **Configurable Generation**: Customizable parameters for text generation
- **Prompt Management**: Consistent handling of system and user prompts

### Safety Service

The safety service validates user inputs and system outputs for safety and relevance:

- **Text Safety Analysis**: Evaluation of text for potentially unsafe content
- **Relevance Classification**: Determination if text is relevant to the system's domain
- **Sensitivity Detection**: Assessment of text sensitivity levels
- **Keyword Detection**: Identification of dangerous or inappropriate keywords

### Store Service

The store service provides persistent storage functionality for:

- **Chat History**: Store and retrieve conversation records with pagination
- **Thread-Safe Operations**: Handle concurrent access with locks
- **Document Chunks**: Track which document chunks were used in responses
- **Data Source Attribution**: Record which data sources provided information

## Service Factory

Services are created through factory methods in `factory.py`, which handle:

1. Component instantiation with appropriate configuration
2. Dependency injection between services
3. Resource initialization and document processing
4. Caching of resource-intensive components

## Usage

Services should be accessed through the factory pattern rather than instantiated directly:

```python
from backend.src.services.factory import (
    create_retrieval_service,
    create_llm_service,
    create_safety_service,
    create_query_processing_service,
    create_store_service
)

# Create services
retrieval_service = create_retrieval_service()
llm_service = create_llm_service()
safety_service = create_safety_service()
store_service = create_store_service()

# Create composite services with dependencies
query_processor = create_query_processing_service(
    llm_service=llm_service,
    retrieval_service=retrieval_service,
    safety_service=safety_service
)

# Use services for a complete RAG pipeline
result = query_processor.process_standard_query(
    query="What is the vacation policy?",
    filters={"document_type": "Policy"}
)

# Store the interaction
store_service.store_chat_with_chunks(
    timestamp="2023-06-01 14:30:45",
    question=result.query,
    response=result.response,
    document_chunks=result.chunks,
    chat_type="standard_rag"
)
```

## Adding New Services

To add a new service:

1. Create a new service class in an appropriate subdirectory
2. Add the service to the `__init__.py` exports
3. Create a factory function in `factory.py` 
4. Add appropriate tests 