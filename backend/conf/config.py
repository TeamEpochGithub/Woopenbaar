"""Configuration module for the backend."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class ConfigMeta(type):
    """Metaclass to prevent direct instantiation and enforce singleton attributes."""

    def __call__(cls, *args: object, **kwargs: object) -> None:
        """Prevent direct instantiation."""
        raise TypeError("Config cannot be instantiated directly. Use class attributes.")


class Config(metaclass=ConfigMeta):
    """Singleton configuration class. Access attributes directly via the class."""

    # =========================================================================
    # Path Configuration
    # =========================================================================
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "backend" / "cache"
    MODELS_DIR: Path = BASE_DIR / "trained_models"
    DOCUMENTS_PATH: Path = DATA_DIR / "documents200"
    ABBREVIATIONS_PATH: Path = DATA_DIR / "abbreviations.json"
    CHAT_HISTORY_PATH: Path = BASE_DIR / "backend" / "chat_history.json"

    # =========================================================================
    # Server Configuration
    # =========================================================================
    FLASK_PORT: int = 5000
    DEVICE: str = "cuda"
    VLLM_PORT: int = 8001
    VLLM_HOST: str = os.getenv("VLLM_HOST", "localhost")
    VLLM_DISTRIBUTED_INIT_PORT: int = 29500

    # =========================================================================
    # Application Configuration
    # =========================================================================
    FORCE_REPROCESS: bool = False  # Flag to force document reprocessing

    # =========================================================================
    # Chunker Configuration
    # =========================================================================
    USE_SEMANTIC_CHUNKER: bool = False
    IDEAL_CHUNK_SIZE: int = 1000  # (Maximum) ideal chunk size

    # Context sizes for chunking
    SMALL_CONTEXT_SIZE: int = 2
    MEDIUM_CONTEXT_SIZE: int = 3
    LARGE_CONTEXT_SIZE: int = 4

    # Similarity thresholds
    SMALL_SIMILARITY_THRES: float = 0.6
    MEDIUM_SIMILARITY_THRES: float = 0.7
    LARGE_SIMILARITY_THRES: float = 0.85

    # =========================================================================
    # Retrieval Pipeline Configuration
    # =========================================================================
    INITIAL_DOCUMENTS_K: int = 200  # Initial document retrieval pool size
    FINAL_DOCUMENTS_K: int = 50  # Documents to keep after reranking
    INITIAL_CHUNKS_K: int = 10  # Initial chunk retrieval pool size
    FINAL_CHUNKS_K: int = 10  # Chunks to keep after reranking
    DOCUMENT_RERANKER_MODE: str = "ranked"

    # =========================================================================
    # Query Processing Configuration
    # =========================================================================
    MAX_ITERATIONS: int = (
        3  # Maximum number of reasoning (= subquery generation) iterations
    )
    ENABLE_SUGGESTED_QUESTIONS: bool = False

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Tokenizer
    TOKENIZER_MAX_LENGTH: int = 10_000_000

    # Embedding models
    DENSE_EMBEDDER_MODEL: str = str(
        MODELS_DIR / "embedders" / "robbert-new"
    )  # Path for fine-tuned version
    CHUNK_RERANKER_MODEL: str = str(MODELS_DIR / "chunk_rerankers" / "reranker-new")

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    # Service selection
    LLM_SERVICE: str = os.getenv(
        "LLM_SERVICE", "local"
    )  # Options: local, vllm, gemini, deepseek
    VALID_LLM_SERVICES: List[str] = ["local", "vllm", "gemini", "deepseek"]
    LLM_DTYPE: str = "float16"

    # Local model configuration
    LOCAL_MODEL_NAME: str = "neuralmagic/Mistral-Small-24B-Instruct-2501-FP8-Dynamic"
    LOCAL_MAX_MODEL_LEN: int = 24000
    LOCAL_MAX_TOKENS: int = 3000
    LOCAL_TEMPERATURE: float = 0.3
    LOCAL_TENSOR_PARALLEL_SIZE: int = 2
    LOCAL_GPU_MEMORY_UTILIZATION: float = 0.8
    LOCAL_TOP_P: float = 1.0
    ENFORCE_EAGER: bool = False  # No longer needed as it's not using eager execution
    CUDA_MEMORY_CONFIG: Dict[str, Union[int, float, bool]] = {
        "max_split_size_mb": 128,
        "expandable_segments": True,
        "garbage_collection_threshold": 0.6,
    }

    # OpenAI configuration
    OPENAI_MODEL_NAME: str = "gpt-4-turbo-preview"
    # Pricing: $0.01/1K input tokens, $0.03/1K output tokens
    # ~$40 per 1M tokens (mixed input/output)
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MAX_MODEL_LEN: int = 128_000
    OPENAI_MAX_TOKENS: int = 4096

    # DeepSeek configuration
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"
    # Pricing: $0.0008/1K input tokens, $0.0016/1K output tokens
    # ~$2.40 per 1M tokens (mixed input/output)
    DEEPSEEK_TEMPERATURE: float = 0.0  # Completely deterministic
    DEEPSEEK_TOP_P: float = 1.0  # Don't filter tokens
    DEEPSEEK_FREQUENCY_PENALTY: float = 0.0  # No penalties
    DEEPSEEK_PRESENCE_PENALTY: float = 0.0  # No penalties
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MAX_MODEL_LEN: int = 32768
    DEEPSEEK_MAX_TOKENS: int = 2048

    # Gemini configuration
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
    # Pricing: $0.0005/1K input tokens, $0.001/1K output tokens
    # ~$1.50 per 1M tokens (mixed input/output)
    # Much better reasoning than Gemini Pro, with 1M token context
    GEMINI_TEMPERATURE: float = 0.0  # Completely deterministic
    GEMINI_TOP_P: float = 1.0  # Don't filter tokens
    GEMINI_TOP_K: Optional[int] = None  # Don't limit token choices
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MAX_MODEL_LEN: int = 1_000_000  # 1M token context window
    GEMINI_MAX_TOKENS: int = 2048

    # =========================================================================
    # Service Selection and Configuration
    # =========================================================================
    # Validate LLM service selection
    if LLM_SERVICE not in VALID_LLM_SERVICES:
        raise ValueError(
            f"Invalid LLM service: {LLM_SERVICE}. Must be one of {VALID_LLM_SERVICES}"
        )

    # Initialize active LLM settings based on selected service
    # These will be populated based on the selected service below
    LLM_MODEL_NAME: str
    LLM_MAX_MODEL_LEN: int
    LLM_MAX_TOKENS: int
    LLM_TEMPERATURE: float
    LLM_TENSOR_PARALLEL_SIZE: int = 0
    LLM_GPU_MEMORY_UTILIZATION: float = 0.0
    LLM_TOP_P: float = 1.0
    LLM_API_KEY: Optional[str] = None


# Configure active LLM settings based on the selected service
# This avoids redefinition errors by setting attributes after class definition
if Config.LLM_SERVICE == "local":
    Config.LLM_MODEL_NAME = Config.LOCAL_MODEL_NAME
    Config.LLM_MAX_MODEL_LEN = Config.LOCAL_MAX_MODEL_LEN
    Config.LLM_MAX_TOKENS = Config.LOCAL_MAX_TOKENS
    Config.LLM_TEMPERATURE = Config.LOCAL_TEMPERATURE
    Config.LLM_TENSOR_PARALLEL_SIZE = Config.LOCAL_TENSOR_PARALLEL_SIZE
    Config.LLM_GPU_MEMORY_UTILIZATION = Config.LOCAL_GPU_MEMORY_UTILIZATION
    Config.LLM_TOP_P = Config.LOCAL_TOP_P
elif Config.LLM_SERVICE == "gemini":
    Config.LLM_MODEL_NAME = Config.GEMINI_MODEL_NAME
    Config.LLM_MAX_MODEL_LEN = Config.GEMINI_MAX_MODEL_LEN
    Config.LLM_MAX_TOKENS = Config.GEMINI_MAX_TOKENS
    Config.LLM_TEMPERATURE = Config.GEMINI_TEMPERATURE
    Config.LLM_API_KEY = Config.GEMINI_API_KEY
elif Config.LLM_SERVICE == "deepseek":
    Config.LLM_MODEL_NAME = Config.DEEPSEEK_MODEL_NAME
    Config.LLM_MAX_MODEL_LEN = Config.DEEPSEEK_MAX_MODEL_LEN
    Config.LLM_MAX_TOKENS = Config.DEEPSEEK_MAX_TOKENS
    Config.LLM_TEMPERATURE = Config.DEEPSEEK_TEMPERATURE
    Config.LLM_API_KEY = Config.DEEPSEEK_API_KEY
