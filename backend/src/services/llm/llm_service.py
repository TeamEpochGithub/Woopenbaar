"""Service module for interacting with large language models.

This module provides a high-level interface for loading and using various LLM providers:
- Local: vLLM-based local models
- Gemini: Google's Gemini API
- DeepSeek: DeepSeek's API (via OpenAI SDK)
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
import requests
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerationConfig
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest,
    LLMGuidedOptions,
)

from backend.conf.config import Config

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """Base class for LLM services.

    This abstract class defines the interface that all LLM services must implement.
    """

    @abstractmethod
    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response from the model based on input.

        Args:
            user_message: The message from the user
            system_prompt: System prompt to use for the generation
            max_tokens: Maximum number of tokens to generate. If None, uses service default
            extra_body: Additional parameters to pass to the model, such as guided_json

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If model generation fails
        """


class LocalLLMService(BaseLLMService):
    """Service for interacting with a local large language model.

    This class provides an interface for generating responses using a vLLM-based
    language model with configurable parameters.

    Attributes:
        model_name (str): Name of the loaded model
        model (LLM): vLLM model instance
        tokenizer (Any): Tokenizer for the model
    """

    def __init__(self) -> None:
        """Initialize the LLM service.

        Loads the configured language model and tokenizer, setting up GPU memory and
        parallel processing parameters according to the configuration.
        """
        os.environ["NO_PROXY"] = "*"  # TODO: Remove this
        # Fix type annotations for the lambda
        requests.utils.should_bypass_proxies = (
            lambda *args, **kwargs: True  # type: ignore
        )
        self.model_name: str = Config.LLM_MODEL_NAME
        logger.info("Loading LLM...")
        self.model: LLM = LLM(
            model=self.model_name,
            max_model_len=Config.LOCAL_MAX_MODEL_LEN,  # Changed from LLM_MAX_MODEL_LEN
            kv_cache_dtype="fp8",
            gpu_memory_utilization=Config.LOCAL_GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=Config.LOCAL_TENSOR_PARALLEL_SIZE,
            dtype=Config.LLM_DTYPE,
        )
        # Use direct assignment with type ignores for partially unknown types
        self.tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore
        logger.info("LLM loaded!")

    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response from the model based on input.

        Takes a user message and generates an appropriate response using the
        loaded language model. The response generation can be configured with parameters
        like maximum token length and sampling settings.

        Args:
            user_message: The message from the user
            system_prompt: System prompt to use for the generation
            max_tokens: Maximum number of tokens to generate. If None, uses Config default
            extra_body: Additional parameters to pass to the model, such as guided_json

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If model generation fails
        """
        if max_tokens is None:
            max_tokens = Config.LOCAL_MAX_TOKENS  # Changed from LLM_MAX_TOKENS

        # Create messages with system prompt and user message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        prompt: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=Config.LLM_TEMPERATURE,
            top_p=Config.LLM_TOP_P,
        )

        # Create guided options for the request
        guided_options: Optional[Union[LLMGuidedOptions, GuidedDecodingRequest]] = None

        # Handle guided_json for structured output if provided
        if extra_body and "guided_json" in extra_body:
            guided_json = extra_body["guided_json"]
            logger.debug("Using guided_json schema for structured output")

            # Create the guided options based on the provided schema
            guided_options = LLMGuidedOptions(guided_json=guided_json)

        outputs: List[RequestOutput] = self.model.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
            guided_options_request=guided_options,
        )

        if not outputs or not outputs[0].outputs or len(outputs[0].outputs) == 0:
            raise RuntimeError("Model failed to generate any output")

        return outputs[0].outputs[0].text


class GeminiLLMService(BaseLLMService):
    """Service for interacting with Google's Gemini API."""

    def __init__(self):
        """Initialize the Gemini LLM service."""

        if not Config.GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable."
            )
        # Configure the API key first
        genai.configure(api_key=Config.GEMINI_API_KEY)  # type: ignore

        # Create a proper GenerationConfig object
        generation_config = GenerationConfig(
            temperature=Config.GEMINI_TEMPERATURE,
            top_p=Config.GEMINI_TOP_P,
            top_k=Config.GEMINI_TOP_K,
            max_output_tokens=Config.GEMINI_MAX_TOKENS,
        )

        # Then create the client
        self.client = GenerativeModel(
            model_name=Config.GEMINI_MODEL_NAME,
            generation_config=generation_config,
        )
        logger.info(
            f"Initialized Gemini LLM service with model: {Config.GEMINI_MODEL_NAME}"
        )

    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response using Gemini's API.

        Args:
            user_message: The message from the user
            system_prompt: System prompt to use for the generation
            max_tokens: Maximum number of tokens to generate. If None, uses Config default
            extra_body: Additional parameters to pass to the model, such as guided_json

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Create Gemini-formatted message with system prompt and user message
            gemini_messages = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"{system_prompt}\n\n{user_message}"
                                if system_prompt
                                else user_message
                            )
                        }
                    ],
                }
            ]

            # Set max tokens if provided
            from google.generativeai.types import GenerationConfig

            generation_config: Optional[GenerationConfig] = None
            if max_tokens is not None:
                generation_config = GenerationConfig(max_output_tokens=max_tokens)

            # Generate response
            response = self.client.generate_content(  # type: ignore
                gemini_messages,
                generation_config=generation_config,
            )

            if not response or not response.text:
                raise RuntimeError("Empty response from Gemini API")

            return response.text

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")


class DeepseekLLMService(BaseLLMService):
    """Service for interacting with DeepSeek's API."""

    def __init__(self):
        """Initialize the DeepSeek LLM service."""
        if not Config.DEEPSEEK_API_KEY:
            raise ValueError(
                "DeepSeek API key not found. Please set the DEEPSEEK_API_KEY environment variable."
            )
        self.client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1"
        )
        logger.info(
            f"Initialized DeepSeek LLM service with model: {Config.DEEPSEEK_MODEL_NAME}"
        )

    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response using DeepSeek's API.

        Args:
            user_message: The message from the user
            system_prompt: System prompt to use for the generation
            max_tokens: Maximum number of tokens to generate. If None, uses Config default
            extra_body: Additional parameters to pass to the model, such as guided_json

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Create messages with system prompt and user message
            from openai.types.chat import ChatCompletionMessageParam

            messages: List[ChatCompletionMessageParam] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

            # Set max tokens if provided
            max_tokens = max_tokens or Config.DEEPSEEK_MAX_TOKENS

            response = self.client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=Config.DEEPSEEK_TEMPERATURE,
                top_p=Config.DEEPSEEK_TOP_P,
                frequency_penalty=Config.DEEPSEEK_FREQUENCY_PENALTY,
                presence_penalty=Config.DEEPSEEK_PRESENCE_PENALTY,
            )

            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Empty response from DeepSeek API")

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")


# For backward compatibility
LLMService = LocalLLMService
