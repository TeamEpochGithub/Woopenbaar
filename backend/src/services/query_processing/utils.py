"""Utility functions for the query processing service."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.conf.config import Config
from backend.conf.prompts import (
    ANSWER_SYNTHESIS_SYSTEM_PROMPT,
    INTERMEDIATE_ANSWER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Directory for debug files - in root project directory
DEBUG_LLM_DIR = "llm_debug"

# Global tokenizer reference to avoid repeated initialization
_llm_tokenizer: Any = None


def log_llm_interaction(
    stage: str,
    messages: List[Dict[str, str]],
    response: str,
    extra_body: Optional[Dict[str, Any]] = None,
) -> None:
    """Log LLM interaction to a debug file specific to the processing stage.

    This function logs the full prompt sent to the LLM and the response
    received, along with metadata about the interaction stage.

    Each processing stage (e.g., chunk_evaluation, answer_synthesis) gets
    its own separate debug file to make analysis easier.

    Args:
        stage: The processing stage (e.g., "progress_evaluation", "answer_synthesis")
        messages: The messages sent to the LLM
        response: The response received from the LLM
        extra_body: Any additional parameters passed to the LLM (e.g., guided_json)
    """
    try:
        # Create debug directory if it doesn't exist
        if not os.path.exists(DEBUG_LLM_DIR):
            os.makedirs(DEBUG_LLM_DIR)

        # Define the stage-specific debug file path
        stage_file = os.path.join(DEBUG_LLM_DIR, f"llm_{stage}_debug.json")

        # Create a record of this interaction
        interaction: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "messages": messages,
            "extra_body": extra_body,
            "response": response,
        }

        # Read existing interactions or start with an empty list
        interactions: List[Dict[str, Any]] = []

        try:
            if os.path.exists(stage_file):
                with open(stage_file, "r", encoding="utf-8") as f:
                    interactions = json.load(f)
            else:
                interactions = []
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is empty or invalid, start fresh
            interactions = []

        # Add the new interaction
        interactions.append(interaction)

        # Write back to file
        with open(stage_file, "w", encoding="utf-8") as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)

        logger.debug(f"Logged LLM interaction for stage '{stage}' to {stage_file}")
    except Exception as e:
        logger.error(f"Failed to log LLM interaction to debug file: {e}")


def _get_tokenizer() -> Any:
    """Get or initialize a tokenizer instance.

    Returns:
        The tokenizer instance from the loaded LLM or None if initialization fails
    """
    global _llm_tokenizer

    if _llm_tokenizer is None:
        try:
            # Import here to avoid circular imports
            from transformers import AutoTokenizer  # type: ignore

            logger.info(f"Initializing tokenizer with model {Config.LLM_MODEL_NAME}")
            # Using type: ignore for the next line since from_pretrained type is unknown
            _llm_tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME)  # type: ignore
            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            # Fall back to simple approximation
            _llm_tokenizer = None

    return _llm_tokenizer  # type: ignore


def estimate_token_count(text: str) -> int:
    """Count the number of tokens in a text string using the actual LLM tokenizer.

    This uses the actual tokenizer from the LLM for accurate token counting.
    Falls back to a simple approximation if tokenizer is unavailable.

    Args:
        text: The text to count tokens for

    Returns:
        int: Token count
    """
    tokenizer = _get_tokenizer()

    if tokenizer is not None:
        try:
            # Use the actual tokenizer to count tokens
            tokens: List[int] = tokenizer.encode(text)  # type: ignore
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error using tokenizer, falling back to approximation: {e}")
    # Fallback to simple approximation if tokenizer fails
    return len(text) // 4


def get_prompt_reserved_tokens() -> int:
    """Get the number of tokens reserved for system prompt and overhead.

    This calculates the token count for the largest system prompt used in answer synthesis,
    plus some overhead for processing.

    Returns:
        int: Token count for system prompt and processing overhead
    """
    # Get token counts for both prompts
    synthesis_tokens = estimate_token_count(ANSWER_SYNTHESIS_SYSTEM_PROMPT)
    intermediate_tokens = estimate_token_count(INTERMEDIATE_ANSWER_SYSTEM_PROMPT)

    # Use the larger of the two prompts
    system_prompt_tokens = max(synthesis_tokens, intermediate_tokens)

    # Add overhead for processing
    overhead_tokens = 250

    return system_prompt_tokens + overhead_tokens
