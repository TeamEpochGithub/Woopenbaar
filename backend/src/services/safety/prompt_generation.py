"""Prompt generation utility for creating training data.

This module uses the Google Generative AI API to generate batches of questions
that can be used for training or testing the safety and relevance models. It handles
API interactions, rate limiting, and saves the generated content to files.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
import tqdm
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerateContentResponse


class PromptGenerator:
    """A class for generating and saving batches of prompts using Google's Generative AI.

    This class handles the configuration of the Google Generative AI API,
    generation of content based on provided prompts, and saving the generated
    content to output files with proper error handling and progress tracking.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-pro",
        output_folder: str = "generated_text",
    ) -> None:
        """Initialize the PromptGenerator with API credentials and settings.

        Args:
            api_key: The Google Generative AI API key.
            model_name: The name of the generative model to use.
            output_folder: Directory path where the generated content will be saved.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.output_folder = output_folder
        self.model: Optional[GenerativeModel] = None

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Configure the API
        self._configure_api()

    def _configure_api(self) -> None:
        """Configure the Google Generative AI API with the provided key."""
        genai.configure(api_key=self.api_key)  # type: ignore
        self.model = GenerativeModel(self.model_name)

    def generate_batch(
        self, prompt: str, num_iterations: int, start_index: int = 1, delay: float = 1.0
    ) -> List[Path]:
        """Generate multiple batches of content based on the provided prompt.

        Args:
            prompt: The text prompt to send to the generative model.
            num_iterations: Number of batches to generate.
            start_index: Starting index for file naming.
            delay: Delay between API calls in seconds to avoid rate limiting.

        Returns:
            List of Path objects pointing to the saved output files.
        """
        generated_files: List[Path] = []

        print(f"Generating {num_iterations} batches of questions...")

        with tqdm.tqdm(
            total=num_iterations,
            desc="Processing",
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}",
        ) as pbar:
            for i in range(start_index, start_index + num_iterations):
                try:
                    # Generate the content
                    response = self._generate_content(prompt)

                    # Save the response to file
                    file_path = self._save_response(response, i)
                    generated_files.append(file_path)

                    # Add delay to avoid rate limits
                    time.sleep(delay)

                except Exception as e:
                    print(f"Error encountered in batch {i}: {e}")

                pbar.update(1)

        print(
            f"\nAll {num_iterations} batches have been generated and saved in '{self.output_folder}'."
        )
        return generated_files

    def _generate_content(self, prompt: str) -> GenerateContentResponse:
        """Generate content using the configured model.

        Args:
            prompt: The text prompt to send to the generative model.

        Returns:
            The response from the generative model.
        """
        if self.model is None:
            raise ValueError("API not configured. Call _configure_api() first.")

        return self.model.generate_content(prompt)  # type: ignore

    def _save_response(self, response: GenerateContentResponse, index: int) -> Path:
        """Save the generated response to a file.

        Args:
            response: The response from the generative model.
            index: The batch index used for file naming.

        Returns:
            Path to the saved file.
        """
        file_path = Path(self.output_folder) / f"questions_{index}.txt"

        with open(file_path, "w") as file:
            file.write(response.text)

        return file_path


def main() -> None:
    """Run the prompt generation process with default parameters."""
    # Set your API key - replace with actual key or environment variable
    api_key = os.environ.get("GOOGLE_API_KEY", "")

    # Initialize the generator
    generator = PromptGenerator(api_key=api_key)

    # Create the prompt
    prompt = "Generate 10 questions"

    # Generate batches
    generator.generate_batch(prompt=prompt, num_iterations=19, start_index=108)


if __name__ == "__main__":
    main()
