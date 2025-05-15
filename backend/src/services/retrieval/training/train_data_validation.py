"""Training data validation module.

This module provides functionality to validate and clean training data for retrieval models.
It handles validation of query-chunk pairs, embedder triplets, and reranker pairs using
LLM-based quality checks. The module supports both structured training data formats and
simple query-chunk pair lists.

The TrainingDataValidator class coordinates validation using an LLM service to assess
the quality and relevance of training examples. The validate_training_files function
provides a command-line interface for batch processing training data files.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from backend.src.services.llm import BaseLLMService, LocalLLMService

logger = logging.getLogger(__name__)


class TrainingDataValidator:
    """Validates training data quality using LLM-based assessment.

    This class handles validation of training examples by using an LLM to assess
    the quality of query-chunk pairs. It supports validation of embedder triplets,
    chunk reranker pairs, and document reranker pairs.

    The validator ensures that queries are well-formed, chunks contain meaningful
    information, and that queries can be answered using the chunk content.

    Attributes:
        llm_service: LLM service used for validation assessments
    """

    def __init__(self, llm_service: BaseLLMService):
        """Initialize the training data validator.

        Args:
            llm_service: LLM service for validation assessments
        """
        self.llm_service = llm_service

    def validate_query_chunk_pair(self, query: str, chunk: str) -> Tuple[bool, str]:
        """Validate if a query-chunk pair is suitable for training.

        Uses the LLM service to validate whether a query-chunk pair meets
        the requirements for training data quality.

        Args:
            query: The query to validate
            chunk: The text chunk to validate against the query

        Returns:
            Tuple containing:
            - bool: Whether the pair is valid
            - str: Reason for the validation result
        """
        message_content = f"""Beoordeel of deze query en tekstfragment geschikt zijn voor het trainen van een retrieval-model:

Query: {query}
Tekstfragment: {chunk}

Vereisten:
1. Is de query een natuurlijke vraag die eindigt met een vraagteken?
2. Bevat het tekstfragment betekenisvolle en coherente informatie?
3. Is de vraag beantwoordbaar op basis van de informatie in het tekstfragment?
4. Is de informatie in het tekstfragment van voldoende kwaliteit voor training?

Geef antwoord in het volgende formaat:
GELDIG: [JA/NEE]
REDEN: [Korte uitleg waarom]"""

        response = self.llm_service.generate_response(
            user_message=message_content
        ).strip()

        # Parse the LLM response into validity and reason
        is_valid = False
        reason = "Unknown"

        try:
            lines = response.split("\n")
            if len(lines) >= 2:
                validity_line = lines[0].strip()
                reason_line = lines[1].strip()

                is_valid = "JA" in validity_line.upper()

                if reason_line.startswith("REDEN:"):
                    reason = reason_line[6:].strip()
        except Exception as e:
            reason = f"Error parsing response: {str(e)}"

        return is_valid, reason

    def validate_training_data(
        self, data: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """Validate and filter structured training data.

        Processes embedder triplets, chunk reranker pairs, and document reranker pairs.
        For each type, validates the positive examples and retains corresponding negatives
        when valid.

        Args:
            data: Dictionary containing training data of different types:
                - embedder_triplets: List of query/positive/negative triplets
                - chunk_reranker_pairs: List of query/text pairs with labels
                - document_reranker_pairs: List of query/document pairs with labels

        Returns:
            Dictionary containing filtered valid training examples of each type
        """
        validated_data: Dict[str, List[Any]] = {
            "embedder_triplets": [],
            "chunk_reranker_pairs": [],
            "document_reranker_pairs": [],
        }

        # Process embedder triplets
        logger.info("Validating embedder triplets...")
        triplets = data.get("embedder_triplets", [])

        valid_triplets: List[Dict[str, Any]] = []
        for triplet in tqdm(triplets, desc="Validating triplets"):
            query = triplet.get("query", "")
            positive = triplet.get("positive", "")

            is_valid, reason = self.validate_query_chunk_pair(query, positive)
            if is_valid:
                valid_triplets.append(triplet)
            else:
                logger.debug(f"Rejected triplet: {reason}")

        validated_data["embedder_triplets"] = valid_triplets
        logger.info(f"Kept {len(valid_triplets)}/{len(triplets)} triplets")

        # Process chunk reranker pairs
        logger.info("Validating chunk reranker pairs...")
        pairs = data.get("chunk_reranker_pairs", [])

        valid_pairs: List[Dict[str, Any]] = []
        for pair in tqdm(pairs, desc="Validating chunk pairs"):
            query = pair.get("query", "")
            text = pair.get("text", "")
            label = pair.get("label", 0)

            # Only validate positive pairs (label=1)
            if label == 1:
                is_valid, reason = self.validate_query_chunk_pair(query, text)
                if is_valid:
                    # Find the corresponding negative pair
                    neg_pair = next(
                        (
                            p
                            for p in pairs
                            if p.get("query") == query and p.get("label") == 0
                        ),
                        None,
                    )
                    if neg_pair:
                        valid_pairs.append(pair)
                        valid_pairs.append(neg_pair)
                else:
                    logger.debug(f"Rejected positive pair: {reason}")

        validated_data["chunk_reranker_pairs"] = valid_pairs
        logger.info(f"Kept {len(valid_pairs)}/{len(pairs)} chunk pairs")

        # Process document reranker pairs (similar to chunk reranker pairs)
        logger.info("Validating document reranker pairs...")
        doc_pairs = data.get("document_reranker_pairs", [])

        valid_doc_pairs: List[Dict[str, Any]] = []
        for pair in tqdm(doc_pairs, desc="Validating document pairs"):
            query = pair.get("query", "")
            text = pair.get("text", "")
            label = pair.get("label", 0)

            # Only validate positive pairs (label=1)
            if label == 1:
                is_valid, reason = self.validate_query_chunk_pair(query, text)
                if is_valid:
                    # Find the corresponding negative pair
                    neg_pair = next(
                        (
                            p
                            for p in doc_pairs
                            if p.get("query") == query and p.get("label") == 0
                        ),
                        None,
                    )
                    if neg_pair:
                        valid_doc_pairs.append(pair)
                        valid_doc_pairs.append(neg_pair)
                else:
                    logger.debug(f"Rejected document pair: {reason}")

        validated_data["document_reranker_pairs"] = valid_doc_pairs
        logger.info(f"Kept {len(valid_doc_pairs)}/{len(doc_pairs)} document pairs")

        return validated_data

    def save_validated_data(self, data: Dict[str, List[Any]], output_path: str):
        """Save validated training data to a JSON file.

        Creates the output directory if it doesn't exist and saves the validated
        training data with proper JSON formatting.

        Args:
            data: Validated training data to save
            output_path: Path where the JSON file should be saved
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Validated training data saved to {output_path}")


def validate_training_files(input_folder_path: str):
    """Process training data files from a directory and create cleaned versions.

    Handles both structured training data (dictionary with typed examples) and
    simple query-chunk pair lists. For simple lists, converts them to the structured
    format during validation.

    The function:
    1. Processes all JSON files in the input directory
    2. Skips already cleaned files (containing "_cleaned")
    3. Validates examples using LLM assessment
    4. Saves cleaned versions with "_cleaned" suffix
    5. Logs validation statistics

    Args:
        input_folder_path: Path to folder containing training data files
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize LLM service
    llm_service = LocalLLMService()
    validator = TrainingDataValidator(llm_service)

    # Find training data files
    for file_name in os.listdir(input_folder_path):
        if not file_name.endswith(".json"):
            continue

        if "_cleaned" in file_name:
            # Skip already cleaned files
            continue

        input_path = os.path.join(input_folder_path, file_name)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(input_folder_path, f"{base_name}_cleaned{ext}")

        logger.info(f"Processing file: {input_path}")

        # Load training data
        with open(input_path, "r") as f:
            training_data: List[Dict[str, Any]] = json.load(f)

        logger.info(
            f"Detected list format in {file_name}, converting to structured format"
        )

        # Convert to expected dictionary format
        structured_data: Dict[str, List[Any]] = {
            "embedder_triplets": [],
            "chunk_reranker_pairs": [],
            "document_reranker_pairs": [],
        }

        # Process as query-chunk pairs
        valid_items: List[Dict[str, Any]] = []
        for item in tqdm(training_data, desc="Validating pairs"):
            query = item.get("query", "")
            chunk = item.get("chunk", "")

            # Skip empty chunks or queries
            if not query.strip() or not chunk.strip() or chunk.strip() == ".":
                logger.debug(f"Skipping empty pair: {query[:30]} - {chunk[:30]}")
                continue

            is_valid, reason = validator.validate_query_chunk_pair(query, chunk)
            if is_valid:
                valid_items.append(item)
            else:
                logger.debug(f"Rejected pair: {reason}")

        # Convert valid query-chunk pairs to both triplets and reranker pairs
        for item in valid_items:
            query = item.get("query", "")
            chunk = item.get("chunk", "")

            # Add as triplet
            structured_data["embedder_triplets"].append(
                {
                    "query": query,
                    "positive": chunk,
                    "negative": "",  # Optional: could add negative examples
                }
            )

            # Add as chunk reranker pair
            structured_data["chunk_reranker_pairs"].append(
                {"query": query, "text": chunk, "label": 1}
            )

        logger.info(
            f"Converted {len(valid_items)}/{len(training_data)} valid pairs to structured format"
        )
        validated_data = structured_data

        # Save validated data
        validator.save_validated_data(validated_data, output_path)

        # Print summary
        logger.info(f"Validation complete for {file_name}")
        logger.info(f"Original pairs: {len(training_data)}")
        logger.info(
            f"Validated embedder triplets: {len(validated_data.get('embedder_triplets', []))}"
        )
        logger.info(
            f"Validated chunk reranker pairs: {len(validated_data.get('chunk_reranker_pairs', []))}"
        )
        logger.info(
            f"Validated embedder triplets: {len(validated_data.get('embedder_triplets', []))}"
        )

        logger.info(
            f"Validated chunk reranker pairs: {len(validated_data.get('chunk_reranker_pairs', []))}"
        )

        logger.info(
            f"Validated document reranker pairs: {len(validated_data.get('document_reranker_pairs', []))}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        validate_training_files(sys.argv[1])
    else:
        print("Please provide a path to the training data folder")
        print(
            "Usage: python -m backend.src.retrieval.training.train_data_validation /path/to/training/data"
        )
