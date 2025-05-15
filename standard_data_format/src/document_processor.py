import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

import pandas as pd
from filelock import FileLock

from standard_data_format.src.document import Document
from standard_data_format.src.metadata import MetadataManager
from standard_data_format.src.pdf_converter import PDFConverterManager
from standard_data_format.utils.logger import setup_logger

logger = setup_logger()


class DocumentProcessor:
    """
    Handles the processing of documents and their attachments.

    Required Metadata Structure:
        Base Document:
            - ID: Unique identifier
            - Document: Title/name
            - Document Link: URL to original
            - Datum: Creation date
            - File Type: Document type
            - Family: Group identifier (optional)
            - besluit_id: Decision reference

        Attachments:
            - Must share same Family ID as base document
            - Must have unique ID within family
            - Must have Document name and Link

    Processing Flow:
        1. Base document is processed first
        2. Attachments are processed in order
        3. Attachment content is inserted:
           - At mention of attachment name in base content
           - Or appended at end if name not found
        4. All content is combined into single Document object

    Thread Safety:
        - Metadata access is protected by RLock
        - PDF processing is thread-safe
        - File operations use atomic writes
    """

    def __init__(
        self,
        pdf_converter: PDFConverterManager,
        metadata_manager: MetadataManager,
        max_workers: int = 4,
    ):
        """
        Initialize the DocumentProcessor with necessary components.

        Args:
            pdf_converter (PDFConverterManager): Manager for PDF conversion operations
            metadata_manager (MetadataManager): Manager for document metadata
            max_workers (int, optional): Maximum number of concurrent worker threads. Defaults to 4.
        """
        self.pdf_converter = pdf_converter
        self.metadata_manager = metadata_manager
        self.max_workers = max_workers
        self.metadata_lock = (
            threading.RLock()
        )  # Reentrant lock for thread-safe metadata access
        self.file_locks = {}  # Dictionary to store file locks
        self.content_cache = {}  # Cache to store document content during processing
        self.cache_lock = threading.Lock()

    def get_file_lock(self, file_path: str) -> FileLock:
        """Get or create a FileLock for a given path"""
        with self.metadata_lock:  # Protect the file_locks dictionary
            if file_path not in self.file_locks:
                self.file_locks[file_path] = FileLock(f"{file_path}.lock")
            return self.file_locks[file_path]

    def _save_document(self, document: Document) -> None:
        """Thread-safe document saving with content verification"""
        file_path = Path(self.config["json_output_dir"]) / f"{document.uuid}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate content hash before saving
        content_hash = hashlib.md5(document.content.encode()).hexdigest()

        # Cache the content with its hash
        with self.cache_lock:
            self.content_cache[document.uuid] = {
                "content": document.content,
                "hash": content_hash,
            }

        try:
            # Use file lock for atomic writing
            with self.get_file_lock(str(file_path)):
                # Verify no other process has written to this file
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        if existing_data.get("content"):
                            existing_hash = hashlib.md5(
                                existing_data["content"].encode()
                            ).hexdigest()
                            if existing_hash != content_hash:
                                logger.warning(
                                    f"Content collision detected for {document.uuid}"
                                )
                                self._handle_content_collision(
                                    document, existing_data, file_path
                                )
                                return

                # Write the document
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)

                # Verify written content
                with open(file_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    saved_hash = hashlib.md5(saved_data["content"].encode()).hexdigest()
                    if saved_hash != content_hash:
                        logger.error(f"Content verification failed for {document.uuid}")
                        raise ValueError("Content verification failed")

                logger.info(
                    f"Successfully saved document {document.uuid} with hash {content_hash}"
                )

        except Exception as e:
            logger.error(f"Error saving document {document.uuid}: {str(e)}")
            raise
        finally:
            # Clean up cache
            with self.cache_lock:
                self.content_cache.pop(document.uuid, None)

    def _handle_content_collision(
        self, new_doc: Document, existing_data: dict, file_path: Path
    ) -> None:
        """Handle cases where different content exists for the same UUID"""
        # Compare document metadata to determine which version to keep
        existing_date = existing_data.get("create_date")
        new_date = new_doc.create_date

        # Create backup of existing file
        backup_path = file_path.with_suffix(".json.backup")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        # Log the collision for analysis
        logger.warning(
            f"""
            Content collision:
            UUID: {new_doc.uuid}
            Existing date: {existing_date}
            New date: {new_date}
            Backup created: {backup_path}
        """
        )

    def process_document(
        self, doc_id: str, pdf_path: Path, metadata_row
    ) -> Tuple[Optional[Document], str]:
        """Process a single document with error handling."""
        try:
            metadata_json = self.pdf_converter.process_document(
                str(pdf_path), metadata_row
            )

            if not metadata_json:
                logger.warning(f"Failed to process document {doc_id}")
                return None, ""

            # Generate UUID and create document
            uuid = self._generate_uuid(metadata_json)
            # Pass the content from metadata_json directly
            document = Document.from_metadata(
                metadata_json=metadata_json,
                uuid=uuid,
                content=metadata_json["markdown_content"],  # Pass content explicitly
            )

            return document, metadata_json["processed_status"]

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            return None, ""

    def process_family(
        self, family_data: dict, metadata_df: pd.DataFrame, input_folder: Path
    ) -> Optional[Document]:
        """
        Process a document family (main document + attachments).

        Args:
            family_data: Dictionary with base document and attachment information
            metadata_df: DataFrame containing metadata for all documents
            input_folder: Path to the folder containing PDF files

        Returns:
            Optional[Document]: Combined family document or None if processing failed
        """
        # Get family information
        base_id = family_data["base_document"]
        besluit_id = self._get_besluit_id(family_data)

        # Process base document
        base_document = self._process_base_document(
            base_id, besluit_id, input_folder, metadata_df
        )
        if not base_document:
            return None

        # Process attachments
        attachment_contents = self._process_attachments(
            family_data["attachment_ids"],
            besluit_id,
            input_folder,
            metadata_df,
            base_id,
        )

        # Combine all content
        combined_content = self._combine_content(
            base_document.content, attachment_contents, metadata_df
        )

        # Create final document
        return Document(
            uuid=base_document.uuid,
            vws_id=base_id,
            create_date=base_document.create_date,
            type=base_document.type,
            link=family_data["link"],
            attachment_links=family_data.get("attachment_links", []),
            content=combined_content,
        )

    def _get_besluit_id(self, family_data: dict) -> str:
        """Get besluit_id with Matter fallback."""
        besluit_id = family_data["besluit_id"]
        if pd.isna(besluit_id) or not besluit_id:
            besluit_id = family_data.get("Matter")
            logger.info(f"Using Matter '{besluit_id}' as fallback for empty besluit_id")
        return besluit_id

    def _process_base_document(
        self,
        base_id: str,
        besluit_id: str,
        input_folder: Path,
        metadata_df: pd.DataFrame,
    ) -> Optional[Document]:
        """Process the base document of a family."""
        base_pdf = self.metadata_manager.find_pdf_for_id(
            base_id, besluit_id, [input_folder]
        )
        if not base_pdf:
            logger.warning(f"Base document PDF not found for family {base_id}")
            return None

        base_metadata = metadata_df[metadata_df["ID"] == base_id].iloc[0]
        base_document, _ = self.process_document(base_id, Path(base_pdf), base_metadata)

        if not base_document:
            logger.warning(f"Failed to process base document for family {base_id}")
            return None

        return base_document

    def _process_attachments(
        self,
        attachment_ids: List[str],
        besluit_id: str,
        input_folder: Path,
        metadata_df: pd.DataFrame,
        base_id: str,
    ) -> List[Tuple[str, str]]:
        """Process all attachments for a family."""
        processed_attachments = []

        for att_id in attachment_ids:
            att_pdf = self.metadata_manager.find_pdf_for_id(
                att_id, besluit_id, [input_folder]
            )
            if not att_pdf:
                logger.warning(
                    f"Attachment PDF not found for ID {att_id} in family {base_id}"
                )
                continue

            att_metadata = metadata_df[metadata_df["ID"] == att_id].iloc[0]
            att_document, _ = self.process_document(att_id, Path(att_pdf), att_metadata)

            if att_document:
                processed_attachments.append((att_id, att_document.content))
            else:
                logger.warning(
                    f"Failed to process attachment {att_id} for family {base_id}"
                )

        return processed_attachments

    def _combine_content(
        self,
        base_content: str,
        attachment_contents: List[Tuple[str, str]],
        metadata_df: pd.DataFrame,
    ) -> str:
        """Combine base content with attachment contents."""
        combined_content = base_content

        for attachment_id, content in attachment_contents:
            att_metadata = metadata_df[metadata_df["ID"] == attachment_id].iloc[0]
            combined_content = self._insert_attachment_content(
                combined_content,
                content,
                attachment_id,
                att_metadata["Document"],
                att_metadata["Document Link"],
            )

        return combined_content

    def _generate_uuid(self, metadata_json: dict) -> str:
        """
        Generate a deterministic UUID from core metadata.

        Uses only the most stable fields for UUID generation to ensure
        the same document always gets the same UUID regardless of
        date parsing or other volatile fields.

        Args:
            metadata_json (dict): The metadata from which to generate the UUID.

        Returns:
            str: A string representation of the generated UUID.
        """
        # Create a simplified metadata dict with only stable fields
        stable_metadata = {
            "vws_id": metadata_json.get("vws_id", ""),
            "link": metadata_json.get("link", ""),
            "title": metadata_json.get("title", ""),
            # First 100 chars of content provides good uniqueness without being too sensitive
            "content_hash": hashlib.md5(
                metadata_json.get("markdown_content", "")[:100].encode()
            ).hexdigest(),
        }

        metadata_str = json.dumps(stable_metadata, sort_keys=True)
        hash_object = hashlib.sha256(metadata_str.encode())
        hash_bytes = hash_object.digest()[:16]  # Use first 16 bytes for UUID
        return str(UUID(bytes=hash_bytes))

    def _insert_attachment_content(
        self,
        base_content: str,
        attachment_content: str,
        attachment_id: str,
        attachment_name: str,
        attachment_link: str,
    ) -> str:
        """
        Insert attachment content into base content with proper formatting.

        Format Structure:
            [Base Content]

            Begin Attachment: {id}, {name}, {link}
            [Attachment Content]
            End Attachment: {id}, {name}, {link}

        Insertion Logic:
            1. If attachment name found in base content:
               - Insert after the line containing the name
            2. If name not found:
               - Append to end of document
        """
        if not attachment_content:
            return base_content

        separator = "\n\n"
        begin_marker = f"\nBegin Attachment: {attachment_id}, {attachment_name}, {attachment_link}\n\n"
        end_marker = f"\n\nEnd Attachment: {attachment_id}, {attachment_name}, {attachment_link}\n"
        attachment_block = f"{separator}{begin_marker}{attachment_content}{end_marker}"

        # Convert attachment_name to string and handle NaN
        attachment_name = str(attachment_name) if pd.notna(attachment_name) else ""

        # Try to find the attachment name in the base content
        if attachment_name and attachment_name in base_content:
            # Find the end of the line containing the attachment name
            name_pos = base_content.find(attachment_name)
            line_end = base_content.find("\n", name_pos)
            if line_end == -1:  # If no newline found, use the end of the content
                line_end = len(base_content)

            # Insert the attachment block after the line containing the name
            return base_content[:line_end] + attachment_block + base_content[line_end:]
        else:
            # If name not found, append to the end
            return base_content + attachment_block

    def _is_already_processed(
        self, id_to_check: str, metadata_df: pd.DataFrame, json_output_dir: Path
    ) -> bool:
        """
        Check if a document or family has already been processed.

        This method checks both the metadata DataFrame for UUIDs and the output directory
        for the corresponding JSON files to determine if a document/family has been processed.

        Args:
            id_to_check (str): The unique identifier (can be either document ID or family ID)
            metadata_df (pd.DataFrame): The DataFrame containing metadata for documents
            json_output_dir (Path): The directory where JSON output files are stored

        Returns:
            bool: True if the document/family has already been processed, False otherwise
        """
        # Use lock when accessing the DataFrame
        with self.metadata_lock:
            # First try as document ID
            doc_rows = metadata_df[metadata_df["ID"] == id_to_check]

            logger.info(
                f"Checking ID as document ID: {id_to_check}, Found rows: {len(doc_rows)}"
            )

            if doc_rows.empty:
                # If not found as document ID, try as family ID
                doc_rows = metadata_df[metadata_df["Family"] == id_to_check]

                logger.info(
                    f"Checking ID as family ID: {id_to_check}, Found rows: {len(doc_rows)}"
                )
                if doc_rows.empty:
                    return False

            # For each document in the family/single document
            for _, doc_row in doc_rows.iterrows():
                uuid = doc_row.get("uuid")
                processed_flag = doc_row.get("processed", "")

                logger.info(
                    f"Checking document: ID={doc_row['ID']}, UUID={uuid}, Processed={processed_flag}"
                )
                # If any document lacks UUID or processed flag, not fully processed
                if not (uuid and processed_flag):

                    logger.info(
                        f"Document {doc_row['ID']} is not fully processed (UUID or processed flag missing)."
                    )
                    return False

                # Check if JSON file exists locally
                json_path = json_output_dir / f"{uuid}.json"

                logger.info(f"Checking for JSON file at: {json_path}")
                if not json_path.exists():
                    # If we have a base machine, check if it exists there via ssh
                    if hasattr(self, "base_machine") and self.base_machine:
                        try:
                            # Use ssh to check if file exists on base machine
                            result = subprocess.run(
                                [
                                    "ssh",
                                    self.base_machine,
                                    f"test -f {json_output_dir / f'{uuid}.json'} && echo 'exists'",
                                ],
                                capture_output=True,
                                text=True,
                            )
                            if "exists" not in result.stdout:

                                logger.info(
                                    f"JSON file for {uuid} does not exist on base machine."
                                )
                                return False
                        except Exception as e:
                            logger.warning(
                                f"Error checking base machine for document {uuid}: {e}"
                            )
                            return False
                    else:
                        logger.info(f"JSON file for {uuid} does not exist locally.")
                        return False

            # If we get here, all documents are processed
            logger.info(f"All documents for ID {id_to_check} are processed.")
            return True
