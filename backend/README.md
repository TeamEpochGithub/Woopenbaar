# Raggle Backend

The Raggle backend is a modular Python application that provides Retrieval-Augmented Generation (RAG) capabilities through a REST API. It supports document processing, intelligent retrieval, context-aware chat, and adaptive reasoning.

## Directory Structure

```
backend/
├── app.py                  # Application entry point
├── conf/                   # Configuration and prompt definitions
├── data/                   # Data storage directory
├── src/                    # Source code
│   ├── api/                # API definition and endpoints
│   ├── data_classes/       # Core data models
│   ├── services/           # Business logic services
└── tests/                  # Unit and integration tests
```

## Key Components

### API Layer (`src/api/`)

REST API built with Flask, handling:
- Chat endpoints for standard and adaptive processing
- Document and chunk retrieval
- Language model access
- Request validation and error handling

See [API README](src/api/README.md) for detailed documentation.

### Data Classes (`src/data_classes/`)

Core data models representing:
- Documents and document chunks
- Filtering options
- Context pieces for query processing
- Query results

See [Data Classes README](src/data_classes/README.md) for detailed documentation.

### Services (`src/services/`)

Business logic components:
- **Retrieval Service**: Document indexing and semantic search
- **Query Processing Service**: Orchestrates the query pipeline
- **LLM Service**: Language model integrations (vLLM, Gemini, DeepSeek)
- **Store Service**: Persistent storage for chat history
- **Safety Service**: Content filtering and validation

Each service has its own README with detailed documentation.

### Configuration (`conf/`)

Centralized configuration management:
- **Config Module**: System-wide settings and parameters
- **Prompts Module**: System prompts for LLM interactions

See [Configuration README](conf/README.md) for detailed documentation.

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU(s) for efficient inference
- Pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/raggle.git
cd raggle/backend

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Configure the application through:
- Environment variables for sensitive settings
- `conf/config.py` for system-wide parameters

Key configuration options:
- Language model selection: Local, vLLM, Gemini, or DeepSeek
- Model parameters: temperature, max tokens, etc.
- Retrieval parameters: pool sizes, reranking options, etc.
- Path configuration: data directories, model paths, etc.

### Running the Server

```bash
# Start the Flask API server
python app.py

# Optional: Start a separate vLLM server for improved performance
python -m backend.src.api.run_vllm_server
```

## Development Guide

### Adding a New Service

1. Create a new service class in `src/services/`
2. Implement required interfaces
3. Add factory method in `src/services/factory.py`
4. Add configuration in `conf/config.py` if needed
5. Create unit tests in `tests/`

### Adding a New Endpoint

1. Create endpoint function in appropriate file in `src/api/endpoints/`
2. Define request/response schemas in `src/api/schemas/`
3. Register the endpoint in `src/api/endpoints/__init__.py`
4. Add error handling as needed

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_retrieval_service.py

# Run with coverage
pytest --cov=src tests/
```