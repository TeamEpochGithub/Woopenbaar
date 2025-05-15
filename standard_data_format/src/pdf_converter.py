import os
import re
import threading
from typing import Any, Dict

from dotenv import load_dotenv
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from standard_data_format.utils.logger import setup_logger

logger = setup_logger()

load_dotenv("standard_data_format/.env")


class PDFConverterManager:
    """
    Manages the conversion of PDF documents to markdown using a tiered approach.

    This class implements a fallback strategy for PDF conversion:
    1. First attempts regular conversion
    2. If that fails, tries forced OCR
    3. As a last resort, uses LLM-based conversion (if configured)

    Each tier is progressively more resource-intensive but potentially more effective
    for challenging documents.
    """

    def __init__(
        self,
        config,
        use_local_model=False,
        allow_regular_conversion=True,
        allow_ocr=True,
        allow_llm=True,
    ):
        """
        Initialize the PDF converter manager with lazy loading.

        Args:
            config (dict): Configuration dictionary containing parameters for the converters.
                           Should include settings for OCR, layout, output format, etc.
            use_local_model (bool): If True, uses Ollama for local LLM processing. If False, uses Google's Gemini.

        Raises:
            ValueError: If using cloud model and Google API key is not found in environment variables.
        """
        self.config = config
        self.use_local_model = use_local_model
        # Initialize converter attributes as None
        self._converter_non_forced_ocr = None
        self._converter_forced_ocr = None
        self._converter_llm = None
        self.allow_regular_conversion = allow_regular_conversion
        self.allow_ocr = allow_ocr
        self.allow_llm = allow_llm

        # Only validate API key if using cloud model
        if not use_local_model and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "API key not found. Please set the GOOGLE_API_KEY environment variable in the standard_data_format/.env file."
            )
        if config.get("llm_service") == "marker.services.gemini.GoogleGeminiService":
            logger.info("Using Gemini API key from environment variable")
            config["gemini_api_key"] = os.getenv("GOOGLE_API_KEY")

        # Add thread-safe buffer management
        self._buffer_lock = threading.Lock()
        self._converter_buffers = {}  # Store per-thread buffers

    @property
    def converter_non_forced_ocr(self):
        """Lazy load the regular converter."""
        if self._converter_non_forced_ocr is None:
            self._converter_non_forced_ocr = self.create_pdf_converter(
                force_ocr=False, use_llm=False
            )
        return self._converter_non_forced_ocr

    @converter_non_forced_ocr.setter
    def converter_non_forced_ocr(self, value):
        self._converter_non_forced_ocr = value

    @property
    def converter_forced_ocr(self):
        """Lazy load the forced OCR converter."""
        if self._converter_forced_ocr is None:
            self._converter_forced_ocr = self.create_pdf_converter(
                force_ocr=True, use_llm=False
            )
        return self._converter_forced_ocr

    @converter_forced_ocr.setter
    def converter_forced_ocr(self, value):
        self._converter_forced_ocr = value

    @property
    def converter_llm(self):
        """Lazy load the LLM converter."""
        if self._converter_llm is None:
            self._converter_llm = self.create_pdf_converter(use_llm=True)
        return self._converter_llm

    def create_pdf_converter(self, force_ocr=False, use_llm=False):
        """
        Create a PdfConverter instance with the specified settings.
        """
        logger.debug("Creating PDF converter with settings:")
        logger.debug(f"force_ocr: {force_ocr}")
        logger.debug(f"use_llm: {use_llm}")
        logger.debug(f"use_local_model: {self.use_local_model}")

        config = self.config.copy()  # Create a copy to avoid modifying original
        config["use_llm"] = use_llm
        config["use_local_model"] = self.use_local_model
        logger.debug(f"Final config after updates: {config}")
        config_parser = ConfigParser(config)

        # Create converter with explicit llm_service from config
        pdf_converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=(
                config["llm_service"] if use_llm else None
            ),  # Use the service directly from config
        )
        logger.debug(f"Created PDF converter: {pdf_converter}")
        print(f"Created PDF converter: {pdf_converter}")
        return pdf_converter

    def _check_ollama_available(self) -> bool:
        """Check if Ollama service is available and the required model is loaded."""
        import time

        import requests
        from requests.exceptions import RequestException

        base_url = self.config.get("llm_service", {}).get(
            "base_url", "http://localhost:11434"
        )
        model_name = self.config.get("llm_service", {}).get("model", "gemma3:12b")
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Check if service is up
                response = requests.get(f"{base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.warning(
                        f"Ollama service returned status {response.status_code}"
                    )
                    continue

                # Check if our model is loaded
                models = response.json().get("models", [])
                if not models:
                    logger.warning("No models found in Ollama")
                    continue

                model_loaded = any(m.get("name") == model_name for m in models)
                if not model_loaded:
                    logger.warning(
                        f"Model {model_name} not found. Available models: {[m.get('name') for m in models]}"
                    )
                    # Optionally try to pull the model
                    try:
                        pull_response = requests.post(
                            f"{base_url}/api/pull",
                            json={"name": model_name},
                            timeout=10,
                        )
                        if pull_response.status_code == 200:
                            logger.info(f"Successfully pulled model {model_name}")
                            return True
                    except RequestException as e:
                        logger.warning(f"Failed to pull model: {str(e)}")
                    continue

                return True

            except RequestException as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries}: Failed to connect to Ollama service: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

        return False

    def _get_thread_buffer(self):
        """Get a thread-specific buffer"""
        thread_id = threading.get_ident()
        with self._buffer_lock:
            if thread_id not in self._converter_buffers:
                self._converter_buffers[thread_id] = {
                    "content": None,
                    "converter": None,
                }
            return self._converter_buffers[thread_id]

    def process_document(self, pdf_path: str, metadata_row) -> Dict[str, Any]:
        """
        Process document with proper return value handling.

        Returns:
            Dict containing metadata and processed content, or None if processing failed
        """
        # try:
        content = None
        processed_status = "failed"

        # Try regular conversion first
        if self.allow_regular_conversion:
            content = self._try_regular_conversion(pdf_path)
        if not content or self._has_quality_issues(content) and self.allow_ocr:
            content = self._try_forced_ocr(pdf_path)
        if not content or self._has_quality_issues(content) and self.allow_llm:
            content = self._try_llm_conversion(pdf_path)

        if content:
            processed_status = "processed"

        metadata_json = self._prepare_metadata_json(metadata_row, content or "")
        metadata_json["processed_status"] = processed_status

        return metadata_json

        # except Exception as e:
        #     logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        #     return None

    def get_markdown_content(self, rendered):
        """
        Extract markdown content from the rendered output of the converter.

        Handles different output formats from the Marker library.

        Args:
            rendered: The output from the PDF converter, which may be an object with a 'markdown'
                     attribute, a dictionary with a 'markdown' key, or a string.

        Returns:
            str: The extracted markdown content as a string.
        """
        if hasattr(rendered, "markdown"):
            return str(rendered.markdown).strip()
        elif isinstance(rendered, dict):
            return str(rendered.get("markdown", "")).strip()
        return str(rendered).strip()

    def check_markdown_quality(self, content):
        """
        Check markdown content for quality issues that might indicate conversion problems.

        This method analyzes the converted content for common issues like:
        - OCR errors (common character confusions)
        - High ratio of special characters
        - Empty or very short content
        - Image-only content without proper descriptions
        - Single line content
        - OCR artifacts like excessive brackets or vertical bars
        - Excessive whitespace, newlines, or tab characters

        Args:
            content (str): The markdown content to check.

        Returns:
            list: A list of identified quality issues (empty if no issues found).
        """
        logger.debug("Checking markdown content quality")
        issues = []
        clean_content_normalized = " ".join(content.split())
        clean_content_chars = "".join(content.split())
        content_lines = [line.strip() for line in content.split("\n") if line.strip()]

        # Check if content is a properly formatted table
        is_table_content = (
            "|" in content
            and content.count("|") > content.count("\n") * 0.5
            and "-|-" in content
        )

        # Check for excessive whitespace issues
        if content:
            # Count consecutive newlines
            max_consecutive_newlines = 0
            current_consecutive_newlines = 0
            for char in content:
                if char == "\n":
                    current_consecutive_newlines += 1
                    max_consecutive_newlines = max(
                        max_consecutive_newlines, current_consecutive_newlines
                    )
                else:
                    current_consecutive_newlines = 0

            # Count consecutive spaces
            max_consecutive_spaces = 0
            current_consecutive_spaces = 0
            for char in content:
                if char == " ":
                    current_consecutive_spaces += 1
                    max_consecutive_spaces = max(
                        max_consecutive_spaces, current_consecutive_spaces
                    )
                else:
                    current_consecutive_spaces = 0

            # Count tab characters
            tab_count = content.count("\t")

            # Calculate whitespace ratio (excluding normal single spaces between words)
            total_length = len(content)
            whitespace_chars = content.count("\n") + content.count("\t")
            # Count spaces that are part of multiple consecutive spaces (2 or more)
            whitespace_chars += sum(
                max(0, len(s) - 1)
                for s in content.split("\n")
                for s in " ".join(s.split()).split(" ")
                if s.strip() == ""
            )

            whitespace_ratio = (
                whitespace_chars / total_length if total_length > 0 else 0
            )

            # Flag excessive whitespace issues
            if max_consecutive_newlines > 5:
                issues.append("excessive_newlines")

            if max_consecutive_spaces > 10:
                issues.append("excessive_spaces")

            if tab_count > 10:
                issues.append("excessive_tabs")

            if (
                whitespace_ratio > 0.3 and not is_table_content
            ):  # Allow more whitespace in tables
                issues.append("high_whitespace_ratio")

        # Adjusted OCR error detection with more lenient thresholds
        # Only check for the most problematic patterns with higher thresholds
        common_ocr_errors = {
            "l l": 4,  # Increased from 2 to 4
            "rn": 5,  # Increased from 3 to 5
            "ii": 5,  # Increased from 3 to 5
            "0O": 4,  # Increased from 2 to 4
        }

        # Check for OCR errors with adjusted thresholds
        content_length = len(clean_content_normalized.split())
        for error_pattern, threshold in common_ocr_errors.items():
            # Only flag if the error appears frequently relative to content length
            error_count = clean_content_normalized.count(error_pattern)
            if error_count >= threshold and (error_count / content_length) > 0.1:
                issues.append("potential_ocr_errors")
                break

        # Rest of the quality checks
        total_chars = len(clean_content_chars)
        if total_chars > 0:
            if not is_table_content:
                special_chars = sum(1 for c in clean_content_chars if not c.isalnum())
                special_ratio = special_chars / total_chars
                if special_ratio > 0.3:  # More than 30% special characters
                    issues.append("high_special_char_ratio")

        if not clean_content_chars:
            issues.append("empty_content")

        # Improved image-only detection
        content_stripped = content.strip()
        if (
            content_stripped.startswith("![")
            and "/page" in content_stripped.lower()
            and len(content_lines) <= 2
            and "description:" not in content_stripped.lower()
        ):
            issues.append("image_only")

        # Improved line count check
        if len(content_lines) < 2 and not any(
            line.startswith(("#", "---", "![", "|")) for line in content_lines
        ):
            issues.append("single_line")

        # Improved length check with exceptions
        if len(clean_content_chars) < 50 and not any(
            [
                "description:" in content.lower(),
                "Image" in content and "/page/" in content,
                is_table_content,
                content.strip().startswith("!["),
            ]
        ):
            issues.append("very_short_content")

        # Improved OCR artifacts detection - skip for proper tables
        if not is_table_content:
            ocr_artifacts = {
                "|": 4,  # Increased from 3 to 4
                "[]": 3,  # Increased from 2 to 3
                "{}": 3,  # Increased from 2 to 3
                "()": 3,  # Increased from 2 to 3
            }

            for artifact, threshold in ocr_artifacts.items():
                if clean_content_normalized.count(artifact) >= threshold:
                    issues.append("ocr_artifacts")
                    break
        if issues:
            logger.warning(f"Quality issues found in content: {', '.join(issues)}")
        return issues

    def _try_regular_conversion(self, pdf_path):
        """
        Attempt to convert a PDF using the regular (non-forced OCR) converter.

        This is the fastest method and works well for PDFs with good text layers.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The extracted markdown content if successful.

        Raises:
            ValueError: If quality issues are detected in the converted content.
        """
        logger.info(f"Attempting regular conversion for {pdf_path}")
        rendered = self.converter_non_forced_ocr(str(pdf_path))
        content = self.get_markdown_content(rendered)
        content = self._clean_content(content)
        quality_issues = self.check_markdown_quality(content)
        if quality_issues:
            logger.info(f"Quality issues found: {', '.join(quality_issues)}")
            raise ValueError(f"Quality issues detected: {', '.join(quality_issues)}")

        else:
            logger.info("Document processed successfully with regular converter")
            return content

    def _try_forced_ocr(self, pdf_path):
        """
        Attempt to convert a PDF using forced OCR.

        This method ignores any existing text layers and performs OCR on the entire document.
        It's slower but can handle PDFs with problematic text layers.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The extracted markdown content if successful.

        Raises:
            ValueError: If quality issues are detected in the converted content.
        """
        logger.info(f"Attempting forced OCR conversion for {pdf_path}")
        rendered = self.converter_forced_ocr(str(pdf_path))
        content = self.get_markdown_content(rendered)
        content = self._clean_content(content)
        quality_issues = self.check_markdown_quality(content)
        if quality_issues:
            logger.info(f"Quality issues found: {', '.join(quality_issues)}")
            raise ValueError(f"Quality issues detected: {', '.join(quality_issues)}")
        else:
            logger.info("Document processed successfully with forced OCR")
            return content

    def _try_llm_conversion(self, pdf_path):
        """
        Attempt to convert a PDF using LLM-assisted processing.
        """
        logger.info(f"Attempting LLM-assisted conversion for {pdf_path}")
        converter = self.converter_llm  # Get the converter instance
        logger.info(f"Converter: {converter}")

        # Add more debug logging
        logger.debug(f"PDF path type: {type(pdf_path)}")
        logger.debug(f"Converter type: {type(converter)}")

        # Call the converter with the path
        rendered = converter(pdf_path)
        content = self.get_markdown_content(rendered)
        content = self._clean_content(content)
        return content

    def _prepare_metadata_json(self, metadata_row, content):
        """
        Prepare a metadata JSON object with the extracted content.

        Args:
            metadata_row (dict or pd.Series): Metadata for the document.
            content (str): The extracted markdown content.

        Returns:
            dict: A dictionary containing the document metadata and content.
        """
        metadata_json = {
            "id": str(metadata_row["ID"]),
            "document": str(metadata_row.get("Document", "")),
            "datum": str(metadata_row.get("Datum", "")),
            "type": metadata_row.get("File Type", ""),
            "markdown_content": content,
            "link": metadata_row.get("Document Link", ""),  # Add document link
            "attachment_links": metadata_row.get(
                "attachment_links", []
            ),  # Add attachment links
        }
        return metadata_json

    def _clean_content(self, content):
        """
        Clean the content by removing excessive whitespace while preserving document structure.

        This method performs several text cleaning operations:
        1. Preserves paragraph breaks by marking them temporarily
        2. Normalizes whitespace (multiple spaces to single space)
        3. Converts tabs to spaces
        4. Preserves reference numbers (like 1.2a)
        5. Restores paragraph breaks

        Args:
            content (str): The raw content to clean

        Returns:
            str: The cleaned content with preserved structure
        """
        logger.debug("Cleaning document content")

        # Skip cleaning if content is empty
        if not content:
            return content

        # Preserve paragraph breaks by temporarily marking them
        content = re.sub(r"\n\s*\n", "\n<PARAGRAPH_BREAK>\n", content)

        # Replace multiple spaces with a single space
        content = re.sub(r" +", " ", content)

        # Replace tabs with spaces
        content = re.sub(r"\t", " ", content)

        # Preserve reference numbers (e.g., 1.2a)
        content = re.sub(r"(\d+\.\d+\w?)", r" \1 ", content)

        # Restore paragraph breaks
        content = re.sub(r"<PARAGRAPH_BREAK>", "\n\n", content)

        return content

    def _has_quality_issues(self, content):
        """Check if the content has quality issues"""
        return bool(self.check_markdown_quality(content))
