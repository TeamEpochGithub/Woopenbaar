"""LLM interaction utilities.

This module contains functions for interacting with various LLM providers:
- Local (default): Uses locally hosted LLM service
- Gemini: Google's Gemini 1.5 Pro model
- DeepSeek: DeepSeek's 67B chat model

The provider can be configured using the LLM_SERVICE configuration option.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

import requests

from backend.conf.config import Config
from backend.src.services.llm import BaseLLMService

logger = logging.getLogger(__name__)


class OpenAIRequest(TypedDict):
    """Type for OpenAI API request data."""

    model: str
    messages: List[Dict[str, str]]
    temperature: float


def call_openai_llm(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI's API with the provided messages.

    Args:
        messages: List of message dictionaries with role and content.

    Returns:
        The content of the generated response from OpenAI.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
    """
    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data: OpenAIRequest = {
        "model": Config.OPENAI_MODEL_NAME,
        "messages": messages,
        "temperature": Config.OPENAI_TEMPERATURE,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", json=data, headers=headers
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_deepseek_llm(messages: List[Dict[str, str]]) -> str:
    """Call DeepSeek's API with the provided messages.

    Args:
        messages: List of message dictionaries with role and content.

    Returns:
        The content of the generated response from DeepSeek.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
    """
    headers = {
        "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": Config.DEEPSEEK_MODEL_NAME,
        "messages": messages,
        "temperature": Config.DEEPSEEK_TEMPERATURE,
    }
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",  # Update with actual endpoint
        json=data,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_gemini_llm(messages: List[Dict[str, str]]) -> str:
    """Call Google's Gemini API with the provided messages.

    Args:
        messages: List of message dictionaries with role and content.

    Returns:
        The content of the generated response from Gemini.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
    """
    headers = {
        "Authorization": f"Bearer {Config.GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": Config.GEMINI_MODEL_NAME,
        "messages": messages,
        "temperature": Config.GEMINI_TEMPERATURE,
    }
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
        json=data,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def log_llm_interaction(
    stage: str,
    messages: List[Dict[str, str]],
    response: str,
    extra_body: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an LLM interaction for debugging.

    Args:
        stage: Name of the processing stage
        messages: List of message dicts sent to LLM
        response: Response received from LLM
        extra_body: Optional extra parameters sent to LLM
    """
    try:
        # Log basic interaction info
        logger.debug(f"LLM Interaction - Stage: {stage}")
        logger.debug("Messages sent:")
        for msg in messages:
            logger.debug(f"  {msg['role']}: {msg['content'][:100]}...")

        # Log response (truncated for readability)
        logger.debug(f"Response received: {response[:100]}...")

        # Log any extra parameters
        if extra_body:
            logger.debug(f"Extra parameters: {str(extra_body)}")

    except Exception as e:
        logger.error(f"Error logging LLM interaction: {e}")


def generate_answer(
    llm_service: BaseLLMService,
    query: str,
    context: str,
    system_prompt: str,
    thinking_summary: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate an answer using the LLM service.

    Args:
        llm_service: Service for LLM interactions
        query: User's query
        context: Retrieved context
        system_prompt: System prompt to use for generation
        thinking_summary: Optional summary of reasoning process
        max_tokens: Optional maximum tokens for response

    Returns:
        str: Generated answer
    """
    try:
        # Add thinking summary if provided
        enhanced_system_prompt = system_prompt
        if thinking_summary:
            enhanced_system_prompt += thinking_summary

        # Add context
        enhanced_system_prompt += "\n\nGevonden Context:\n" + context

        # Generate response using the new interface
        response = llm_service.generate_response(
            user_message=query,
            system_prompt=enhanced_system_prompt,
            max_tokens=max_tokens,
        )

        # Log the interaction (for logging purposes, we still need to create messages)
        log_llm_interaction(
            stage="answer_generation",
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": query},
            ],
            response=response,
        )

        return response

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Er is een fout opgetreden bij het genereren van het antwoord."


def generate_llm_response(
    user_message: str,
    context: str,
    system_prompt: str,
    llm_service: Optional[BaseLLMService],
    reasoning_steps: Optional[List[str]] = None,
) -> str:
    """Generate LLM response using configured LLM service.

    Args:
        user_message: User's input message
        context: Retrieved context to include in prompt
        system_prompt: System prompt to use for generation
        llm_service: Optional local LLM service
        reasoning_steps: Optional list of reasoning steps to include

    Returns:
        Generated response text
    """
    logger.debug(f"Using LLM service: {Config.LLM_SERVICE}")

    if reasoning_steps:
        thinking_summary = "\n\nZoekstappen:\n"
        for step in reasoning_steps:
            thinking_summary += f"- {step}\n"
        system_prompt += thinking_summary

    system_prompt += "\n\nGevonden Context:\n" + context

    # Use the configured LLM service
    if Config.LLM_SERVICE in Config.VALID_LLM_SERVICES:
        if not llm_service:
            raise ValueError(
                "Local LLM service not initialized but local service is configured"
            )
        logger.debug("Using local LLM service")
        return llm_service.generate_response(
            user_message=user_message, system_prompt=system_prompt
        )

    # Map service names to their functions
    provider_map = {
        "gemini": call_gemini_llm,
        "deepseek": call_deepseek_llm,
    }

    if Config.LLM_SERVICE not in provider_map:
        raise ValueError(f"Unsupported LLM provider: {Config.LLM_SERVICE}")

    logger.debug(f"Using {Config.LLM_SERVICE} LLM service")

    # For external services, we need to create messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return provider_map[Config.LLM_SERVICE](messages)
