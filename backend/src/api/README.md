# API Package

The API package provides the HTTP interface for the Raggle backend, handling request routing, validation, error handling, and response formatting. It's built using Flask and follows a modular architecture with clean separation of concerns.

## Structure

- `core.py`: Central API setup and configuration functions
- `endpoints/`: API route handlers for different features
- `middleware/`: Request processing and error handling middleware
- `utils/`: Utility functions for API operations
- `schemas/`: Data models for request and response validation
- `run_vllm_server.py`: Utility for running a standalone vLLM server

## API Setup

The API is initialized in `core.py` with the `setup_api` function:

```python
def setup_api(
    app: Flask,
    retrieval_service: RetrievalService,
    query_processing_service: QueryProcessingService,
    history_service: Optional[ChatHistoryService] = None,
    llm_service: Optional[BaseLLMService] = None,
    safety_service: Optional[SafetyService] = None,
) -> None:
    # Register middleware and endpoints
```

This function configures the Flask application with necessary middleware and endpoints, injecting all required services.

## Endpoints

The `endpoints/` directory contains API route definitions:

### Chat Endpoints

Conversational interfaces to the RAG system:

- `POST /chat/standard`: Standard RAG chat with context retrieval
- `POST /chat/adaptive`: Adaptive RAG chat with reasoning steps (uses SSE streaming)

### Retrieval Endpoints

Direct access to document retrieval functionality:

- `POST /retrieval/find`: Retrieve relevant document chunks
- `GET /retrieval/document/{id}`: Get a specific document
- `GET /retrieval/chunk/{id}`: Get a specific document chunk
- `GET /retrieval/sources`: Get available data sources
- `GET /retrieval/status`: Check retrieval service status

### LLM Endpoints

Direct access to language model functionality:

- `POST /llm/generate`: Generate text with the LLM
- `GET /llm/status`: Check LLM service status

## Middleware

The `middleware/` directory contains error handling classes:

### Exception Definitions

Custom exception types for API errors and the error response model:

- `ErrorResponseModel`: Pydantic model for consistent error responses
- `APIError`: Base class for all API errors
- `ValidationError`: Invalid request data errors (400)
- `NotFoundError`: Resource not found errors (404)
- `ServiceError`: Service-level errors (500)

### Error Handling

Converts exceptions to appropriate HTTP responses:

- Handles validation errors from Pydantic
- Provides consistent error response format
- Ensures proper HTTP status codes

## Schemas

The `schemas/` directory contains data models for request/response validation:

- Request models with validation rules
- Response models for output formatting
- Conversion methods for domain objects

## Utilities

The `utils/` directory provides helper functions:

### Server-Sent Events

Utilities for streaming responses:

- `create_sse_event`: Creates a formatted SSE event
- `create_sse_response`: Creates a streaming SSE response

### Context Formatting

Helper functions for:

- Formatting document chunks for responses
- Converting domain objects to API schemas

## Request Flow

1. **Request Arrives**: Client sends HTTP request to an endpoint
2. **Validation**: Request data is validated using Pydantic models
3. **Processing**: Endpoint handler processes the request using services
4. **Response**: Data is formatted and returned as JSON or SSE stream
5. **Error Handling**: Exceptions are caught and converted to appropriate responses

## vLLM Server

The `run_vllm_server.py` script provides functionality to:

- Start a standalone vLLM server with OpenAI-compatible API
- Configure model parameters for optimal performance
- Support tensor parallelism for multi-GPU setups

## Adding New Endpoints

To add a new endpoint:

1. Create a new endpoint function in an appropriate module
2. Define a Pydantic schema for request validation
3. Register the endpoint in the `register_endpoints` function
4. Update API documentation if applicable 