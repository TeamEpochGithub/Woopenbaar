"""vLLM API server with OpenAI-compatible endpoints."""

import argparse
import logging
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.conf.config import Config
from backend.src.services.llm.llm_service import LocalLLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM OpenAI-Compatible API")

# Global LLM service instance and initialization flag
_llm_service = None
_initialization_failed = False


def initialize_llm_service():
    """Initialize the LLM service once."""
    global _llm_service, _initialization_failed
    if _llm_service is None and not _initialization_failed:
        logger.info("Initializing LLM service...")
        try:
            _llm_service = LocalLLMService()
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            _initialization_failed = True
            raise HTTPException(
                status_code=500, detail="Failed to initialize LLM service"
            )


def get_llm_service():
    """Get the LLM service instance without trying to initialize."""
    if _llm_service is None:
        if _initialization_failed:
            raise HTTPException(
                status_code=503,
                detail="LLM service failed to initialize and is unavailable",
            )
        raise HTTPException(status_code=503, detail="LLM service not yet initialized")
    return _llm_service


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="local-model")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=Config.LOCAL_TEMPERATURE)
    max_tokens: Optional[int] = Field(default=Config.LOCAL_MAX_TOKENS)
    top_p: Optional[float] = Field(default=Config.LOCAL_TOP_P)
    stream: bool = Field(default=False)


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-" + "".join([str(i) for i in range(20)])
    object: str = "chat.completion"
    created: int = 0
    model: str = "local-model"
    choices: List[Dict[str, Union[str, Dict[str, str]]]]
    usage: Dict[str, int]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if _initialization_failed:
        raise HTTPException(
            status_code=503,
            detail="LLM service failed to initialize and is unavailable",
        )
    if _llm_service is None:
        return {
            "status": "starting",
            "ready": False,
            "message": "LLM service is still initializing",
        }
    return {"status": "healthy", "ready": True, "model": Config.LOCAL_MODEL_NAME}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using vLLM."""
    try:
        # Get the LLM service instance
        llm_service = get_llm_service()

        # Convert messages to the format expected by LocalLLMService
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Extract system prompt and user message
        system_prompt = ""
        user_message = ""

        # Find system message if it exists
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break

        # Find the most recent user message
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # If no user message found, use the last message regardless of role
        if not user_message and messages:
            user_message = messages[-1]["content"]

        # Generate response using vLLM with all available parameters
        response = llm_service.generate_response(
            user_message=user_message,
            system_prompt=system_prompt,
            max_tokens=request.max_tokens,
        )

        # Format response in OpenAI style
        return ChatCompletionResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 0,  # We don't track token usage yet
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the OpenAI-compatible API server."""
    parser = argparse.ArgumentParser(description="Run the OpenAI-compatible API server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type for model (default: float16)",
    )

    args = parser.parse_args()

    # Update model config based on arguments
    Config.LLM_DTYPE = args.dtype

    logger.info(f"Starting vLLM server on {args.host}:{args.port}")
    logger.info(f"Configuration: dtype={args.dtype}")

    # Initialize LLM service once at startup
    try:
        initialize_llm_service()
    except Exception as e:
        logger.error(f"Initial LLM service initialization failed: {str(e)}")
        # Continue running the server even if initialization fails
        # The health check will report the failure status

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
