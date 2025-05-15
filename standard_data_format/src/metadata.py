import logging  # Add logging import
import os
import threading  # Add this import
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from filelock import FileLock  # Add this import
from tqdm import tqdm

from standard_data_format.utils.logger import setup_logger

# Remove any existing logging configuration

# Configure logging
logger = setup_logger()

# Ensure the logger level is set to INFO
logger.setLevel(logging.INFO)

# Add a stream handler if you want to see output in the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MetadataManager:
    """Manages document metadata and relationships.

    Required Metadata Columns:
        - ID: Unique document identifier
        - Family: Groups related documents (optional)
        - Document: Document name/title
        - Datum: Document date
        - File Type: Type of document
        - Document Link: URL to original document
        - besluit_id: Decision ID reference

    Auto-generated Columns:
        - uuid: Generated unique identifier
        - processed: Processing status
        - available: PDF file existence
        - attachment_names: List of attachment names
        - attachment_ids: List of attachment IDs
        - attachment_links: List of attachment URLs

    Assumptions:
        1. PDF files are named as either "{ID}.pdf" or "{ID}"
        2. Document IDs may have prefixes (e.g., "VWS-12345")
        3. Family relationships are based on matching IDs
        4. Document processing follows: regular -> forced OCR -> LLM
        5. Thread-safe operations are required for concurrent processing
    """

    def __init__(self, metadata_df: pd.DataFrame = None, metadata_path: str = None):
        self.metadata_path = metadata_path
        self.metadata_df = metadata_df
        if metadata_path is not None and metadata_df is None:
            self.load_metadata()

        self.lock = threading.RLock()
        self.file_locks = {}
        self._processing_locks = {}  # Add document-specific locks

    def get_file_lock(self, file_path: str) -> FileLock:
        """Get or create a FileLock for a given path"""
        if file_path not in self.file_locks:
            self.file_locks[file_path] = FileLock(f"{file_path}.lock")
        return self.file_locks[file_path]

    def get_document_lock(self, doc_id: str) -> threading.Lock:
        """Get a document-specific lock"""
        with self.lock:
            if doc_id not in self._processing_locks:
                self._processing_locks[doc_id] = threading.Lock()
            return self._processing_locks[doc_id]

    def save_metadata(self, path: Union[str, Path]) -> None:
        """
        Save metadata to CSV file with proper path handling.

        Args:
            path: Path where to save the metadata (string or Path object)
        """
        try:
            # Convert string path to Path object
            save_path = Path(path)

            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with self.lock:
                # Save metadata with proper index handling
                self.metadata_df.to_csv(str(save_path), index=False)
                logger.info(f"Successfully saved metadata to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata to {path}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise

    def load_metadata(self):
        """Load metadata with file locking"""
        if self.metadata_path:
            with self.lock:  # Thread lock
                with self.get_file_lock(self.metadata_path):  # File lock
                    try:
                        self.metadata_df = pd.read_csv(
                            self.metadata_path, low_memory=False
                        )
                        logger.info(f"Loaded metadata from {self.metadata_path}")
                    except Exception as e:
                        logger.error(f"Error loading metadata: {str(e)}")
                        self.metadata_df = pd.DataFrame()

    def combine_duplicate_columns(self, df):
        """Combine duplicate columns in the DataFrame."""
        logger.info("Combining duplicate columns")
        # Get initial list of duplicate columns
        duplicates = df.columns[df.columns.duplicated()].unique()

        # Process each duplicate column only once
        for col in duplicates:
            # Get exact column matches only
            duplicate_cols = [c for c in df.columns if c == col]
            logger.debug(f"Combining {len(duplicate_cols)} instances of column '{col}'")
            # Get values from all duplicate columns as a list
            values = [df[col_name] for col_name in duplicate_cols]
            # Combine values only once
            combined_values = pd.concat(values, axis=1).apply(
                lambda x: ", ".join(x.dropna().astype(str).unique()), axis=1
            )
            # Assign combined values to first column and drop others
            df[duplicate_cols[0]] = combined_values
            # Drop all but the first occurrence
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        return df

    def add_available_to_df(self, document_folder, skip_if_exists=True):
        """
        Check if the document has been downloaded, and add available boolean.

        Args:
            document_folder: Folder containing document files
            skip_if_exists: If True and 'available' column already exists with values,
                            skip the check to save time

        Returns:
            Updated metadata DataFrame
        """
        with self.lock:
            # Skip if column already exists and has values
            if skip_if_exists and "available" in self.metadata_df.columns:
                existing_values = self.metadata_df["available"].sum()
                if existing_values > 0:
                    logger.info(
                        f"Skipping availability check - already found {existing_values} available documents"
                    )
                    return self.metadata_df

            logger.info(f"Checking document availability in {document_folder}")

        # Create a set of all existing files in the document folder for O(1) lookups
        existing_files_set = set()

        # Use os.scandir for better performance - only scan once
        with os.scandir(document_folder) as entries:
            for entry in entries:
                if entry.is_file():
                    filename = entry.name
                    # Strip extension for matching
                    base_name = os.path.splitext(filename)[0]
                    existing_files_set.add(base_name)
                    existing_files_set.add(filename)  # Also keep full filename

        logger.info(f"Found {len(existing_files_set)} files in document folder")

        with self.lock:
            # Process in batches to improve progress bar accuracy
            batch_size = 5000
            total_rows = len(self.metadata_df)
            availability = []

            for start_idx in tqdm(
                range(0, total_rows, batch_size),
                desc="Checking document availability",
                total=(total_rows + batch_size - 1) // batch_size,
            ):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = self.metadata_df.iloc[start_idx:end_idx]

                batch_availability = []
                for id_val in batch["ID"]:
                    # Convert ID to string and check if it exists in our set
                    id_str = str(id_val).strip() if pd.notna(id_val) else ""
                    # Fast O(1) lookup in set
                    found = bool(id_str) and (
                        id_str in existing_files_set
                        or f"{id_str}.pdf" in existing_files_set
                    )
                    batch_availability.append(found)

                availability.extend(batch_availability)

            # Add the availability status to the DataFrame
            with self.get_file_lock(self.metadata_path):
                self.metadata_df["available"] = availability

            available_count = sum(availability)
            logger.info(
                f"Found {available_count} available documents out of {len(availability)}"
            )
            return self.metadata_df

    def add_processed_to_df(self) -> pd.DataFrame:
        """Add 'processed' column to the DataFrame."""
        with self.lock:  # Thread lock
            with self.get_file_lock(self.metadata_path):  # File lock
                logger.info("Adding 'processed' column to metadata")
                self.metadata_df["processed"] = self.metadata_df.get(
                    "processed", pd.Series(dtype="object")
                ).apply(lambda x: x if isinstance(x, str) else "")
                return self.metadata_df

    def add_uuid_column(self) -> pd.DataFrame:
        """Add 'uuid' column to the DataFrame."""
        with self.lock:  # Thread lock
            with self.get_file_lock(self.metadata_path):  # File lock
                logger.info("Adding 'uuid' column to metadata")
                self.metadata_df["uuid"] = self.metadata_df.get(
                    "uuid", pd.Series(dtype="object")
                ).apply(lambda x: x if isinstance(x, str) else "")
                return self.metadata_df

    def add_attachment_columns(self) -> pd.DataFrame:
        """Add 'attachment_names', 'attachment_ids' and 'attachment_links' columns to the DataFrame."""
        with self.lock:  # Thread lock
            with self.get_file_lock(self.metadata_path):  # File lock
                logger.info("Adding attachment columns to metadata")
                self.metadata_df["attachment_names"] = self.metadata_df.get(
                    "attachment_names", pd.Series(dtype="object")
                ).apply(lambda x: x if isinstance(x, list) else [])
                self.metadata_df["attachment_ids"] = self.metadata_df.get(
                    "attachment_ids", pd.Series(dtype="object")
                ).apply(lambda x: x if isinstance(x, list) else [])
                self.metadata_df["attachment_links"] = self.metadata_df.get(
                    "attachment_links", pd.Series(dtype="object")
                ).apply(lambda x: x if isinstance(x, list) else [])
                return self.metadata_df

    def load_metadata_mapping(self) -> Tuple[Dict[str, Dict], pd.DataFrame]:
        """
        Load metadata and create a mapping of family IDs to their attachments.

        Returns:
            Tuple[Dict, DataFrame]: Family mapping and updated metadata DataFrame

        Note:
            - Uses Matter as fallback for empty besluit_id
            - Normalizes IDs by removing prefixes
            - Handles duplicate IDs by keeping first occurrence
            - Excludes already processed families (those with UUIDs)
        """
        with self.lock:
            # Step 1: Initial metadata preparation
            self.metadata_df = self._prepare_initial_metadata()

            # Step 2: Process and filter available documents
            available_docs, valid_ids = self._get_available_documents()

            # Step 3: Create family mappings
            family_map = self._create_family_map(available_docs, valid_ids)

            # Step 4: Filter out processed families and update attachments
            family_map = self._process_and_update_families(family_map)

            return family_map, self.metadata_df

    def _prepare_initial_metadata(self) -> pd.DataFrame:
        """Prepare and clean the initial metadata."""
        df = self.add_attachment_columns()
        logger.info(f"Initial metadata rows: {len(df)}")

        # Handle duplicates
        df = df.drop_duplicates(subset=["ID"], keep="first")
        logger.info(f"After removing duplicates: {len(df)} rows")

        # Handle besluit_id
        df["besluit_id"] = df.apply(
            lambda row: (
                row["Matter"]
                if pd.isna(row["besluit_id"]) or not row["besluit_id"]
                else row["besluit_id"]
            ),
            axis=1,
        )

        # Normalize IDs
        df["ID_normalized"] = df["ID"].apply(self._normalize_id)
        df["Family_normalized"] = df["Family"].apply(self._normalize_id)

        return df

    def _get_available_documents(self) -> Tuple[pd.DataFrame, set]:
        """Filter for available documents and get valid IDs."""
        valid_ids = set(id for id in self.metadata_df["ID_normalized"] if id)
        logger.info(f"Found {len(valid_ids)} valid IDs")

        available_docs = self.metadata_df[
            self.metadata_df["Family_normalized"].notna()
            & (self.metadata_df["available"])
        ]

        logger.info(
            f"Filtered from {len(self.metadata_df)} to {len(available_docs)} available documents"
        )
        return available_docs, valid_ids

    def _create_family_map(self, available_docs: pd.DataFrame, valid_ids: set) -> Dict:
        """Create initial family mapping from available documents."""
        family_map = {}

        for _, row in available_docs.iterrows():
            family_id = row["Family_normalized"]
            doc_data = {
                "id": row["ID"],
                "document": row.get("Document", ""),
                "link": row.get("Document Link", ""),
                "besluit_id": row.get("besluit_id", ""),
            }

            if family_id in valid_ids:
                self._add_to_family(
                    family_map, family_id, doc_data, row["ID_normalized"]
                )
            else:
                self._add_single_document(family_map, doc_data)

        logger.info(f"Created initial family map with {len(family_map)} entries")
        return family_map

    def _process_and_update_families(self, family_map: Dict) -> Dict:
        """Process families and update attachment information."""
        # Remove processed families (those with UUIDs)
        unprocessed_map = {}
        for family_id, family_data in family_map.items():
            base_doc = self.metadata_df[
                self.metadata_df["ID"] == family_data["base_document"]
            ]
            if not base_doc.empty and self._is_empty_uuid(base_doc.iloc[0].get("uuid")):
                unprocessed_map[family_id] = family_data

        logger.info(f"Filtered to {len(unprocessed_map)} unprocessed families")

        # Update attachment information
        self._update_attachment_info(unprocessed_map)

        # Clean up temporary columns
        self.metadata_df = self.metadata_df.drop(
            ["ID_normalized", "Family_normalized"], axis=1
        )

        return unprocessed_map

    @staticmethod
    def _normalize_id(id_str: str) -> str:
        """Normalize ID by removing prefixes."""
        if pd.isna(id_str):
            return ""
        id_str = str(id_str).strip()
        return id_str.split("-")[-1] if "-" in id_str else id_str

    @staticmethod
    def _is_empty_uuid(uuid_value) -> bool:
        """Check if a UUID value is empty or invalid."""
        if pd.isna(uuid_value):
            return True
        if not isinstance(uuid_value, str):
            uuid_value = str(uuid_value)
        return uuid_value.lower().strip() in ["", "nan", "na", "none"]

    def _add_to_family(
        self, family_map: Dict, family_id: str, doc_data: Dict, id_normalized: str
    ):
        """Add document to family map."""
        if family_id not in family_map:
            family_map[family_id] = {
                "base_document": doc_data["id"],
                "link": doc_data["link"],
                "attachment_ids": [],
                "attachment_names": [],
                "attachment_links": [],
                "besluit_id": doc_data["besluit_id"],
            }

        if id_normalized != family_id:
            family_map[family_id]["attachment_ids"].append(doc_data["id"])
            family_map[family_id]["attachment_names"].append(doc_data["document"])
            family_map[family_id]["attachment_links"].append(doc_data["link"])
            family_map[family_id]["besluit_id"] = doc_data["besluit_id"]
        else:
            family_map[family_id]["link"] = doc_data["link"]

    def _add_single_document(self, family_map: Dict, doc_data: Dict):
        """Add single document to family map."""
        doc_map_id = f"single_{doc_data['id']}"
        family_map[doc_map_id] = {
            "base_document": doc_data["id"],
            "link": doc_data["link"],
            "attachment_ids": [],
            "attachment_names": [],
            "attachment_links": [],
            "besluit_id": doc_data["besluit_id"],
        }

    def _update_attachment_info(self, family_map: Dict):
        """Update metadata DataFrame with attachment information."""
        update_count = 0
        for family_id, family_data in family_map.items():
            if not family_data["attachment_ids"]:
                continue

            mask = self.metadata_df["ID"] == family_id
            if not any(mask):
                continue

            idx = self.metadata_df[mask].index[0]
            self._update_attachment_fields(idx, family_data)
            update_count += 1

        logger.info(f"Updated {update_count} documents with attachment information")

    def _update_attachment_fields(self, idx: int, family_data: Dict):
        """Update attachment fields for a specific row."""
        fields = ["names", "ids", "links"]
        for field in fields:
            current = self.metadata_df.at[idx, f"attachment_{field}"]
            current = [] if pd.isna(current) else current
            updated = family_data[f"attachment_{field}"] + current
            self.metadata_df.at[idx, f"attachment_{field}"] = updated

    def find_pdf_for_id(
        self, doc_id: str, besluit_id: str, search_paths: List[Path]
    ) -> Optional[str]:
        """
        Find PDF file for a given document ID with enhanced logging and validation.

        File Naming Patterns Checked:
            - {doc_id}.pdf
            - {doc_id}
            - {normalized_id}.pdf
            - {normalized_id}
            - {besluit_id}_{doc_id}.pdf This one happens most of the time
            - {besluit_id}_{doc_id}
            Where normalized_id removes any prefix (e.g., 'VWS-123' -> '123')

        Args:
            doc_id: Document ID to search for
            besluit_id: Decision ID that might be part of the filename
            search_paths: List of directories to search

        Returns:
            Optional[str]: Path to PDF file if found, None otherwise
        """
        logger.info(f"Searching for PDF with ID: {doc_id}, besluit_id: {besluit_id}")

        # Validate search paths
        for path in search_paths:
            if not path.exists():
                logger.error(f"Search path does not exist: {path}")
                continue

            # Log directory contents for debugging
            logger.debug(f"Contents of {path}:")
            try:
                files = list(path.glob("*"))
                logger.debug(f"Found {len(files)} files")
                logger.debug(f"Sample files: {files[:5]}")
            except Exception as e:
                logger.error(f"Error reading directory {path}: {e}")
                continue

        # Generate possible filenames
        normalized_id = (
            str(doc_id).split("-")[-1] if "-" in str(doc_id) else str(doc_id)
        )
        patterns = [
            f"{doc_id}.pdf",
            f"{doc_id}",
            f"{normalized_id}.pdf",
            f"{normalized_id}",
        ]

        # Add besluit_id patterns if provided
        if besluit_id:
            print(f"Adding besluit_id patterns for {besluit_id}")
            patterns.extend(
                [
                    f"{besluit_id}-{doc_id}.pdf",
                    f"{besluit_id}-{doc_id}",
                    f"{besluit_id}-{normalized_id}.pdf",
                    f"{besluit_id}-{normalized_id}",
                ]
            )

        logger.debug(f"Checking patterns: {patterns}")

        # Search all paths with all patterns
        for path in search_paths:
            if not path.exists():
                continue

            for pattern in patterns:
                pdf_path = path / pattern
                logger.debug(f"Checking path: {pdf_path}")

                if pdf_path.exists():
                    logger.info(f"Found PDF at {pdf_path}")
                    return str(pdf_path)

        logger.warning(f"No PDF found for ID {doc_id} in paths {search_paths}")
        logger.debug("Attempted patterns: " + ", ".join(patterns))
        return None
