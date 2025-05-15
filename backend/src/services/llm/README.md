# LLM Service

The Language Model (LLM) service provides an abstract interface for interacting with various large language models, supporting text and chat generation in the Raggle backend.

## Features

- **Model Abstraction**: Unified interface for different LLM providers
- **Multiple LLM Implementations**: Supports local models, API-based services, and external providers
- **Chat Formatting**: Handles message formatting for chat interfaces
- **Configurable Parameters**: Flexible generation parameters per implementation

## Components

### BaseLLMService

The abstract base class defining the interface for all LLM implementations:

- `generate_response`: Generates a response based on a user message and system prompt

### LocalLLMService

Implementation for local, vLLM-based models:

- Uses vLLM for efficient GPU inference
- Supports tensor parallelism for multi-GPU setups
- Uses Hugging Face tokenizers for message formatting

### GeminiLLMService

Implementation for Google's Gemini API:

- Generates responses via Google's Generative AI API
- Supports configurable generation parameters
- Transforms system prompts to the Gemini message format

### DeepseekLLMService

Implementation for DeepSeek's API (via OpenAI SDK):

- Generates responses via DeepSeek's API
- Supports standard OpenAI parameters

### VLLMClientService

Client for external vLLM servers with OpenAI-compatible API:

- Connects to an externally running vLLM server
- Supports retry logic for reliable connections
- Implements health checks for server readiness

## Text Generation

Text generation uses vLLM's sampling parameters for local models, or the corresponding parameters for API implementations:

```python
sampling_params = SamplingParams(
    max_tokens=max_tokens,
    temperature=Config.LLM_TEMPERATURE,
    top_p=Config.LLM_TOP_P,
)
```

## Configuration

Key configuration parameters:

- `LLM_MODEL_NAME`: The model to use
- `LOCAL_MAX_MODEL_LEN`: Maximum sequence length
- `LOCAL_MAX_TOKENS`: Maximum tokens to generate
- `LLM_TEMPERATURE`: Randomness parameter (0.0-1.0)
- `LLM_TOP_P`: Nucleus sampling parameter
- `LOCAL_GPU_MEMORY_UTILIZATION`: GPU memory allocation
- `LOCAL_TENSOR_PARALLEL_SIZE`: Parallelism for multi-GPU inference
- `GEMINI_API_KEY`: API key for Gemini
- `GEMINI_MODEL_NAME`: Name of the Gemini model
- `DEEPSEEK_API_KEY`: API key for DeepSeek

## Chat Message Format

The service expects chat messages in the format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
```

## Usage

```python       
from backend.src.services.llm import BaseLLMService, LocalLLMService, GeminiLLMService

# Initialize the service (local or API-based)
llm_service = LocalLLMService()  # Or GeminiLLMService() or DeepseekLLMService()

# Create a message
system_prompt = "You are a helpful assistant."
user_message = "What is the capital of France?"

# Generate a response
response = llm_service.generate_response(
    user_message=user_message, 
    system_prompt=system_prompt, 
    max_tokens=100
)
print(response)  # "The capital of France is Paris..."
``` 