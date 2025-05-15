"""Client for connecting to a vLLM server running the OpenAI-compatible API."""

import logging
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.conf.config import Config
from backend.src.services.llm.llm_service import BaseLLMService

logger = logging.getLogger(__name__)


class VLLMClientService(BaseLLMService):
    """LLM service that connects to a running vLLM server with OpenAI-compatible API."""

    def __init__(
        self, api_base_url: str, max_retries: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize the vLLM client service.

        Args:
            api_base_url: Base URL of the vLLM server (e.g., "http://localhost:8001")
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_base_url = api_base_url
        self.chat_completions_url = f"{api_base_url}/v1/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        """Wait for the vLLM server to be ready."""
        logger.info(f"Waiting for vLLM server at {self.api_base_url}...")

        start_time = time.time()
        max_wait_time = 300  # 5 minutes maximum wait time

        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(
                    f"{self.api_base_url}/health", timeout=10  # Increased timeout
                )

                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("ready", False):
                        logger.info("Successfully connected to vLLM server")
                        logger.info(
                            f"Model loaded: {health_data.get('model', 'unknown')}"
                        )
                        return
                    else:
                        logger.warning("Server responding but not ready yet...")
                else:
                    logger.warning(f"Server not ready (status {response.status_code})")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Connection attempt failed: {str(e)}")

            # Progressive backoff
            wait_time = min(self.retry_delay * 2, 30)  # Max 30 seconds between retries
            logger.info(f"Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)

        raise ConnectionError(
            f"Could not connect to vLLM server at {self.api_base_url} "
            f"after waiting {max_wait_time} seconds"
        )

    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response based on user message and system prompt.

        Args:
            user_message: The message from the user
            system_prompt: System prompt to use for the generation
            max_tokens: Maximum number of tokens to generate. If None, uses Config default
            extra_body: Additional parameters to pass to the model

        Returns:
            Generated response text

        Raises:
            RuntimeError: If the vLLM server returns an error
        """
        try:
            # Create messages with system prompt and user message
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Prepare request payload with config defaults
            payload = {
                "model": Config.LOCAL_MODEL_NAME,
                "messages": messages,
                "max_tokens": max_tokens or Config.LOCAL_MAX_TOKENS,
                # Add other potential parameters from kwargs if needed
            }

            # Remove None values
            payload = {k: v for k, v in payload.items()}

            # Prepare headers (only Content-Type needed based on server code)
            headers = {
                "Content-Type": "application/json",
                # Removed the "Authorization": "Bearer dummy-key" line
            }
            logger.debug(
                f"Sending request to {self.chat_completions_url} with payload: {payload} and headers: {headers}"
            )

            response = self.session.post(
                self.chat_completions_url,
                json=payload,
                headers=headers,  # Pass the simplified headers
                timeout=600,
            )

            if response.status_code != 200:
                error_msg = f"vLLM server error: {response.status_code}"
                error_detail = response.json().get("detail", "No detail provided")
                error_msg += f" - {error_detail}"
                logger.error(error_msg)

            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text

        except requests.exceptions.Timeout:
            error_msg = "Request to vLLM server timed out"
            logger.error(error_msg)

        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to vLLM server at {self.api_base_url}"
            logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error generating text with vLLM: {str(e)}"
            logger.error(error_msg)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Convert standard generate call to chat format for vLLM server."""
        return self.generate_response(
            user_message=prompt,
            system_prompt=system_prompt or "",
            max_tokens=max_tokens,
            extra_body=(
                {"temperature": temperature} if temperature is not None else None
            ),
        )
