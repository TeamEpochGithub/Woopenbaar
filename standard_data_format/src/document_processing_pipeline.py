import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from filelock import FileLock
from tqdm import tqdm

from standard_data_format.src.document import Document
from standard_data_format.src.document_processor import DocumentProcessor
from standard_data_format.src.metadata import MetadataManager
from standard_data_format.src.pdf_converter import PDFConverterManager
from standard_data_format.utils.logger import setup_logger

# Remove all the logging setup code and replace with:
logger = setup_logger()


class DocumentProcessingPipeline:
    """
    Manages the entire document processing pipeline.

    This class orchestrates the end-to-end process of converting PDF documents to
    the standard data format, including:
    - Setting up directories
    - Initializing necessary managers
    - Processing document families in parallel
    - Saving processed documents and metadata
    - Tracking and reporting progress

    Thread safety is ensured through the use of locks when accessing shared resources.
    """

    def __init__(
        self,
        config: dict,
        test: bool = False,
        save_interval: int = 1,
        base_machine: str = None,
        sync_interval: int = 10,
        metadata_path: str = None,
        chunk_id: Optional[int] = None,
    ):
        """
        Initialize the document processing pipeline.

        Args:
            config (dict): Configuration dictionary containing processing parameters and paths
            test (bool): Whether to run in test mode
            save_interval (int): How often to save metadata
            base_machine (str): SSH connection string for base machine
            sync_interval (int): How many documents to process before syncing
            chunk_id (int): ID of the current chunk being processed
            custom_metadata_path (str): Path to a pre-divided metadata file
        """
        self.config = config
        self.test = test
        self.base_machine = base_machine
        self.sync_interval = sync_interval
        self.chunk_id = chunk_id

        # Debug log the configuration
        logger.debug(f"Pipeline initialized with config: {config}")
        logger.debug(
            f"allow_regular_conversion: {config.get('allow_regular_conversion')}"
        )
        logger.debug(f"allow_ocr: {config.get('allow_ocr')}")
        logger.debug(f"allow_llm: {config.get('allow_llm')}")

        # Override metadata path if provided
        if metadata_path:
            self.config["metadata_path"] = metadata_path

        # Initialize PDF converter with explicit settings
        self.pdf_converter = PDFConverterManager(
            config,
            use_local_model=config.get("use_local_model", True),
            allow_regular_conversion=config.get("allow_regular_conversion", True),
            allow_ocr=config.get("allow_ocr", True),
            allow_llm=config.get("allow_llm", True),
        )

        # Debug log the PDF converter settings
        logger.debug("PDF converter initialized with:")
        logger.debug(f"use_local_model: {self.pdf_converter.use_local_model}")
        logger.debug(
            f"allow_regular_conversion: {self.pdf_converter.allow_regular_conversion}"
        )
        logger.debug(f"allow_ocr: {self.pdf_converter.allow_ocr}")
        logger.debug(f"allow_llm: {self.pdf_converter.allow_llm}")

        self.setup_directories()
        self.initialize_managers()
        logger.info(
            f"Initial metadata size: {len(self.metadata_manager.metadata_df)} rows"
        )
        logger.info(
            f"Unique families: {self.metadata_manager.metadata_df['Family'].nunique()}"
        )
        logger.info(
            f"Sample of first few families: {sorted(self.metadata_manager.metadata_df['Family'].unique())[:5]}"
        )

        if self.test:
            test_families = [810811]
            original_size = len(self.metadata_manager.metadata_df)
            self.metadata_manager.metadata_df = self.metadata_manager.metadata_df[
                self.metadata_manager.metadata_df["Family"].isin(test_families)
            ]
            logger.info(
                f"Test mode: Filtered from {original_size} to {len(self.metadata_manager.metadata_df)} rows for families {test_families}"
            )
        self.save_interval = save_interval
        self.processed_since_sync = 0
        self.file_locks = {}  # Add file lock dictionary

    def _validate_config(self, config: dict) -> dict:
        """
        Validate and set default configuration values.

        This method ensures that all required configuration parameters are present
        and have valid values. It applies default values for missing parameters.

        Args:
            config (dict): The configuration dictionary to validate.

        Returns:
            dict: A validated configuration dictionary with defaults applied.

        Raises:
            ValueError: If required parameters are missing or have invalid values.
        """
        defaults = {
            "batch_size": 1,
            "debug": True,
            "output_format": "markdown",
            "languages": "nl",
            # Default directory paths
            "base_dir": "standard_data_format",
            "json_output_dir": "output/json_files",
            "debug_output_dir": "output/debug_output",
            "metadata_path": "data/metadata/combined_documents_full_data_merged.csv",
            "document_folder": "data/woo_scraped/documents",
        }

        # Start with defaults
        validated = defaults.copy()

        # Update with provided config
        validated.update(config)

        # Validate required parameters
        required_params = [
            "base_dir",
            "json_output_dir",
            "metadata_path",
            "document_folder",
        ]
        for param in required_params:
            if not validated.get(param):
                raise ValueError(f"Missing required configuration parameter: {param}")

        # Validate batch_size is a positive integer
        if not isinstance(validated["batch_size"], int) or validated["batch_size"] < 1:
            raise ValueError(
                f"batch_size must be a positive integer, got {validated['batch_size']}"
            )

        # Validate output_format is supported
        supported_formats = ["markdown", "text", "html", "json"]
        if validated["output_format"] not in supported_formats:
            raise ValueError(
                f"Unsupported output format: {validated['output_format']}. "
                f"Supported formats are: {', '.join(supported_formats)}"
            )

        # Validate debug is boolean
        if not isinstance(validated["debug"], bool):
            validated["debug"] = bool(validated["debug"])

        return validated

    def setup_directories(self):
        """
        Initialize necessary directories for the processing pipeline.

        This method creates the output directories if they don't exist:
        - JSON output directory for processed documents
        - Debug output directory for debugging information

        It also sets up paths for metadata and document folders.
        """
        self.base_dir = Path(self.config["base_dir"])
        self.output_dir = self.base_dir / self.config["json_output_dir"]
        self.debug_output_dir = self.base_dir / self.config["debug_output_dir"]
        self.metadata_path = Path(self.config["metadata_path"])
        self.document_folder = Path(self.config["document_folder"])

        for directory in [self.output_dir, self.debug_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def initialize_managers(self):
        """
        Initialize the necessary managers for document processing.

        This method creates instances of:
        - PDFConverterManager: For converting PDFs to markdown
        - MetadataManager: For handling document metadata
        - DocumentProcessor: For processing documents and their attachments

        If reinitialize_metadata is True, metadata is loaded from Excel files.
        Otherwise, it's loaded from the existing CSV file if available.
        The scraped_metadata_path parameter takes precedence when provided.
        """
        try:
            # Initialize metadata manager
            self.metadata_manager = MetadataManager(
                metadata_df=pd.read_csv(self.metadata_path, low_memory=False),
            )

            # Initialize document processor
            self.document_processor = DocumentProcessor(
                self.pdf_converter,
                self.metadata_manager,
                max_workers=self.config.get("batch_size", 4),
            )
        except Exception as e:
            logger.error(f"Failed to initialize managers: {str(e)}")
            raise

    def standardize_family_id(self, family_id):
        """
        Standardize family ID format to ensure consistent matching.

        Examples:
            10-264352 -> 10264352
            202227 -> 202227
            10264352 -> 10264352
        """
        if family_id is None or pd.isna(family_id):
            return None

        # Convert to string if not already
        family_id = str(family_id)

        # Remove any hyphens
        family_id = family_id.replace("-", "")

        # If it starts with a leading zero, remove it
        if family_id.startswith("0"):
            family_id = family_id.lstrip("0")

        return family_id

    def get_file_lock(self, file_path: str) -> FileLock:
        """Get or create a FileLock for a given path"""
        if file_path not in self.file_locks:
            self.file_locks[file_path] = FileLock(f"{file_path}.lock")
        return self.file_locks[file_path]

    def _sync_to_base(self):
        """Sync processed files and chunk-specific metadata to base machine"""
        if not self.base_machine:
            return

        try:
            # First, ensure remote directories exist
            remote_base = f"{self.base_machine}:{self.config['base_path']}"
            remote_metadata_dir = f"{remote_base}/data/metadata"
            remote_json_dir = (
                f"{remote_base}/standard_data_format/output_scraped/json_files"
            )

            # Create remote directories
            for remote_dir in [remote_metadata_dir, remote_json_dir]:
                remote_path = remote_dir.split(":")[1]
                mkdir_cmd = f"ssh {self.base_machine} 'mkdir -p {remote_path}'"
                logger.info(f"Creating remote directory: {mkdir_cmd}")
                os.system(mkdir_cmd)

            # Sync JSON files
            json_dir = str(self.output_dir)
            sync_cmd = (
                f"cd {json_dir} && "
                f"find . -name '*.json' -print0 | "
                f"rsync -av --files-from=- --from0 . {remote_json_dir}/"
            )
            logger.info(f"Syncing JSON files to {remote_json_dir}")
            os.system(sync_cmd)

            # Sync chunk-specific metadata file
            chunk_metadata_name = f"metadata_chunk_{self.chunk_id}.csv"
            if self.test:
                chunk_metadata_name = f"metadata_chunk_{self.chunk_id}_test.csv"

            # Use the actual local path of the metadata file
            local_metadata_path = Path(self.metadata_path).parent / chunk_metadata_name

            if local_metadata_path.exists():
                sync_cmd = f"rsync -avz {local_metadata_path} {remote_base}/{local_metadata_path}"
                logger.info(
                    f"Syncing metadata file {local_metadata_path} to {remote_base}/{local_metadata_path}"
                )
                os.system(sync_cmd)
            else:
                logger.warning(f"Metadata file not found at {local_metadata_path}")

            self.processed_since_sync = 0
            logger.info("Sync completed successfully")

        except Exception as e:
            logger.error(f"Error during sync: {str(e)}")

    def process_documents(self, family_map, metadata_df):
        """Main processing loop with sync support"""
        try:
            remaining_families = len(family_map)
            logger.info(f"Starting to process {remaining_families} families")

            metadata_lock = threading.RLock()
            processed_count = 0

            with tqdm(
                desc="Processing documents",
                total=remaining_families,
                ncols=80,  # Fixed width
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",  # Simplified format
            ) as pbar:
                for family_id, family_data in family_map.items():
                    try:
                        logger.info(f"Starting to process family {family_id}")
                        document = self._process_family(
                            family_id, family_data, metadata_df
                        )
                        processed_count += 1
                        self.processed_since_sync += 1

                        if document:
                            self._save_document(document)
                            logger.info(
                                f"Processed family {family_id} ({processed_count}/{remaining_families})"
                            )

                        # Sync after processing sync_interval documents
                        if self.processed_since_sync >= self.sync_interval:
                            with metadata_lock:
                                self._save_metadata(metadata_df)
                                self._sync_to_base()
                            logger.info(
                                f"Synced after processing {processed_count} documents"
                            )

                    except Exception as e:
                        logger.error(f"Error processing family {family_id}: {str(e)}")
                        # Save and sync on error
                        with metadata_lock:
                            self._save_metadata(metadata_df)
                            self._sync_to_base()

                    pbar.update(1)

        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
        finally:
            # Final save and sync
            self._save_metadata(metadata_df)
            self._sync_to_base()
            logger.info("Completed document processing")

    def _process_family(
        self, family_id: str, family_data: dict, metadata_df: pd.DataFrame
    ) -> Optional[Document]:
        """
        Process a single family of documents.

        This method delegates the processing of a document family to the DocumentProcessor.

        Args:
            family_id (str): The ID of the family to process
            family_data (dict): Dictionary containing family information
            metadata_df (pd.DataFrame): DataFrame containing metadata for all documents

        Returns:
            Optional[Document]: A Document object if processing was successful, None otherwise
        """
        try:
            # Log original and standardized family IDs
            std_family_id = self.standardize_family_id(family_id)
            if std_family_id != family_id:
                logger.info(
                    f"Standardized family ID from {family_id} to {std_family_id}"
                )

            return self.document_processor.process_family(
                family_data, metadata_df, self.document_folder
            )
        except Exception as e:
            logger.error(f"Error processing family {family_id}: {str(e)}")
            return None

    def _save_document(self, document: Document):
        """
        Save processed document to JSON file using atomic write operations.

        Args:
            document (Document): The document to save
        """
        import shutil
        import tempfile
        from pathlib import Path

        output_path = self.output_dir / f"{document.uuid}.json"

        try:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, encoding="utf-8"
            ) as temp_file:
                json.dump(document.to_json(), temp_file, ensure_ascii=False, indent=2)

            # Atomic rename of temp file to final destination
            shutil.move(temp_file.name, output_path)

            logger.info(
                f"Saved document VWS-{document.vws_id} to standard data format (UUID: {document.uuid})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error saving document {document.uuid}: {str(e)}", exc_info=True
            )

            # Clean up temporary file if it exists
            if Path(temp_file.name).exists():
                Path(temp_file.name).unlink()

            return False

    def _save_metadata(self, metadata_df: pd.DataFrame) -> None:
        """
        Save metadata locally and sync with base machine.

        Args:
            metadata_df: DataFrame containing metadata to save
        """
        try:
            # Determine local save path
            local_path, parent_dir = self._get_local_metadata_path_and_parent_path()

            # Save metadata locally
            self._save_local_metadata(metadata_df, parent_dir, local_path)

            # Sync with base machine if configured
            if self.base_machine:
                self._sync_metadata_to_base(parent_dir, local_path)

        except Exception as e:
            logger.error(f"Error in metadata saving process: {str(e)}")
            raise

    def _get_local_metadata_path_and_parent_path(self):
        """Get the local metadata path and its parent directory."""
        # Convert PosixPath to string if needed
        if hasattr(self.metadata_path, "parent"):
            # It's a Path object
            local_path = str(self.metadata_path)
            parent_dir = str(self.metadata_path.parent)
        else:
            # It's a string
            local_path = self.metadata_path
            parent_dir = "/".join(self.metadata_path.split("/")[:-1])

        return local_path, parent_dir

    def _save_local_metadata(self, metadata_df, parent_dir, local_path):
        """Save metadata to a local CSV file."""
        try:
            # Convert parent_dir to Path if it's a string
            if isinstance(parent_dir, str):
                parent_dir = Path(parent_dir)

            # Create parent directory if it doesn't exist
            parent_dir.mkdir(parents=True, exist_ok=True)

            # Save metadata to CSV
            metadata_df.to_csv(local_path, index=False)
            logger.info(f"Saved metadata to {local_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {local_path}: {str(e)}")
            raise

    def _sync_metadata_to_base(self, parent_dir: Path, csv_path: Path) -> None:
        """Sync metadata file to base machine."""
        # try:
        # Construct remote path with base machine and base path
        remote_path = f"{self.base_machine}:{self.config['base_path']}"
        parent_path = os.path.join(remote_path, parent_dir)
        local_path = os.path.join(csv_path)
        remote_csv_path = os.path.join(parent_path)
        # Create remote directory
        remote_dir_cmd = f"mkdir -p {parent_path}"
        subprocess.run(
            ["ssh", self.base_machine, remote_dir_cmd], check=True, capture_output=True
        )

        # Sync file
        subprocess.run(
            ["rsync", "-avz", str(local_path), remote_csv_path],
            check=True,
            capture_output=True,
        )

        logger.info(f"Successfully synced metadata to {remote_csv_path}")

        # except subprocess.CalledProcessError as e:
        #     logger.error(f"Failed to sync metadata: {e.stderr.decode()}")
        #     raise
        # except Exception as e:
        #     logger.error(f"Error during metadata sync: {str(e)}")
        #     raise

    def process_documents_distributed(self, chunk_id=None, total_chunks=None):
        """
        Process a subset of documents in a distributed manner.

        Args:
            chunk_id (int, optional): ID of the current chunk. Defaults to None.
            total_chunks (int, optional): Total number of chunks. Defaults to None.
        """
        start_time = time.time()

        logger.info(f"Using pre-divided metadata file: {self.metadata_path}")

        # Now the mapping will be created using the chunk-specific metadata
        family_map, metadata_df = self.metadata_manager.load_metadata_mapping()
        # Process all documents in this chunk
        self.process_documents(family_map, metadata_df)

        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
