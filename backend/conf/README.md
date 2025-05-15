# Configuration Package

The configuration package (`conf`) provides centralized configuration management and system-wide settings for the Raggle backend. It includes both application configuration parameters and prompts used throughout the system.

## Components

### Config Module

The `config.py` module implements a singleton configuration class with class-level attributes for all system settings:

```python
class Config(metaclass=ConfigMeta):
    """Singleton configuration class. Access attributes directly via the class."""
    
    # Path Configuration
    BASE_DIR: Path
    DATA_DIR: Path
    # ...
```

Key features:

- **Singleton Pattern**: Prevents direct instantiation, enforcing access via class attributes
- **Environment Awareness**: Uses environment variables for sensitive settings
- **Type Annotations**: All configuration parameters include type annotations
- **Logical Organization**: Settings are grouped by functional area

### Prompts Module

The `prompts.py` module contains system prompts used in various language model interactions:

```python
# Answer synthesis prompts
ANSWER_SYNTHESIS_SYSTEM_PROMPT: str = """
    Je bent een deskundige onderzoeker die informatie beschrijft...
"""
```

Key prompt categories:

- **Answer Synthesis**: Prompts for generating final answers from retrieved context
- **Intermediate Answers**: Prompts for progress updates during adaptive retrieval
- **Chunk Evaluation**: Prompts for evaluating the relevance of retrieved chunks
- **Source Selection**: Prompts for selecting the most appropriate data source
- **Progress Evaluation**: Prompts for determining retrieval completeness
- **Subquery Generation**: Prompts for generating follow-up queries
- **Suggestion Generation**: Prompts for generating related questions

## Configuration Categories

The configuration system covers several functional areas:

### 1. Path Configuration

File and directory paths used throughout the application:

- Base directory, data directory, cache directory
- Model paths, document paths, history paths

### 2. Server Configuration

Server-related settings:

- Flask port
- VLLM server settings
- Hardware configurations (CUDA, etc.)

### 3. LLM Configuration

Language model settings for different providers:

- **Local models**: vLLM-based model parameters
- **API-based models**: Configuration for Gemini, DeepSeek, and OpenAI
- **Shared parameters**: Temperature, max tokens, etc.

### 4. Retrieval Configuration

Settings for the retrieval pipeline:

- Document/chunk pool sizes
- Reranking parameters
- Chunking configurations

### 5. Query Processing Configuration

Parameters for query processing:

- Maximum iterations for adaptive retrieval
- Suggestion generation settings

## Usage

Access configuration values directly from the Config class:

```python
from backend.conf.config import Config

# Use configuration values
data_dir = Config.DATA_DIR
max_tokens = Config.LLM_MAX_TOKENS

# Access prompts
from backend.conf.prompts import ANSWER_SYNTHESIS_SYSTEM_PROMPT

# Use prompt in LLM interaction
response = llm_service.generate_response(
    user_message=query,
    system_prompt=ANSWER_SYNTHESIS_SYSTEM_PROMPT
)
```

## Design Principles

The configuration system follows these principles:

1. **Single Source of Truth**: All configuration in one place
2. **Type Safety**: Type annotations for all parameters
3. **Environment Awareness**: Environment variables for deployment-specific settings
4. **Documentation**: Clear comments explaining each parameter
5. **Logical Organization**: Settings grouped by functional area 