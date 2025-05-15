"""Utility module for handling citations in responses.

This module provides functions for processing and reordering citations in text responses.
"""

import logging
import re
from typing import Dict, List, Match, Set, Tuple

logger = logging.getLogger(__name__)


def reorder_citations(text: str) -> Tuple[str, Dict[int, int]]:
    """Reorder citations in a text to be sequential while maintaining references.

    This function finds all citation numbers in the format [N] or [N,M,...] and reorders them
    to be sequential (starting from 1). It maintains a mapping between the original
    citation numbers and the new sequential ones.

    Args:
        text: The text containing citations in the format [N] or [N,M,...]

    Returns:
        Tuple containing:
        - The text with reordered citations
        - A mapping from new citation numbers to original citation numbers
    """
    # Find all citations with pattern [N] where N is a number (single citation)
    single_citation_pattern = r"\[(\d+)\]"
    # Find all citations with pattern [N,M,...] where N,M are numbers (multi-citation)
    multi_citation_pattern = r"\[(\d+(?:,\d+)+)\]"

    # First extract all citation numbers from both single and multi-citations
    single_citations: List[str] = re.findall(single_citation_pattern, text)
    multi_citations_groups: List[str] = re.findall(multi_citation_pattern, text)

    # Extract individual citation numbers from multi-citation groups
    all_citation_numbers: Set[str] = set(single_citations)
    for group in multi_citations_groups:
        for citation in group.split(","):
            all_citation_numbers.add(citation)

    if not all_citation_numbers:
        logger.debug("No citations found in the text")
        return text, {}

    # Create ordered list of unique citations
    unique_citations: List[str] = sorted(list(all_citation_numbers), key=int)

    # Create mapping from original citation number to new sequential number
    original_to_new: Dict[str, int] = {}
    new_to_original: Dict[int, int] = {}

    for i, orig_citation in enumerate(unique_citations, start=1):
        original_to_new[orig_citation] = i
        new_to_original[i] = int(orig_citation)

    # Replace single citations in the text
    def replace_single_citation(match: Match[str]) -> str:
        orig_citation = match.group(1)
        new_citation = original_to_new[orig_citation]
        return f"[{new_citation}]"

    # Replace multi-citations in the text
    def replace_multi_citation(match: Match[str]) -> str:
        orig_group = match.group(1)
        orig_citations = orig_group.split(",")
        new_citations = [str(original_to_new[citation]) for citation in orig_citations]
        return f"[{','.join(new_citations)}]"

    # First replace multi-citations to avoid conflicts with single-citation replacements
    processed_text = re.sub(multi_citation_pattern, replace_multi_citation, text)
    # Then replace single citations
    processed_text = re.sub(
        single_citation_pattern, replace_single_citation, processed_text
    )

    logger.debug(f"Reordered {len(unique_citations)} unique citations")
    return processed_text, new_to_original


def process_response_citations(
    response_text: str, chunk_ids: List[str]
) -> Tuple[str, List[str]]:
    """Process citations in a response and keep track of which chunks are actually cited.

    This function reorders the citations to be sequential and updates the chunk_ids
    list to contain only the chunks that are actually cited, in the correct order.
    It also handles multi-citations in the format [1,2,4].

    Args:
        response_text: The text response with citations
        chunk_ids: The list of chunk IDs corresponding to the original citations

    Returns:
        Tuple containing:
        - The text with reordered citations
        - A list of chunk IDs corresponding to the new citation order
    """
    # Handle the case where there are no chunks
    if not chunk_ids:
        return response_text, []

    # Reorder citations in the text
    processed_text, citation_mapping = reorder_citations(response_text)

    # If no citations were found or processed, return original
    if not citation_mapping:
        return response_text, chunk_ids

    # Create a new list of chunk IDs based on citation usage and order
    ordered_chunk_ids: List[str] = []

    for _, orig_citation in citation_mapping.items():
        # Handle the case where citations exceed available chunks
        if orig_citation <= len(chunk_ids):
            # Citations in the text are 1-based, but list indices are 0-based
            chunk_index = orig_citation - 1
            chunk_id = chunk_ids[chunk_index]
            # Only add if it's not already in the list
            if chunk_id not in ordered_chunk_ids:
                ordered_chunk_ids.append(chunk_id)
        else:
            logger.warning(
                f"Citation [{orig_citation}] exceeds available chunks (total: {len(chunk_ids)})"
            )
            # Add a placeholder for missing chunk IDs
            placeholder = f"missing_chunk_{orig_citation}"
            if placeholder not in ordered_chunk_ids:
                ordered_chunk_ids.append(placeholder)

    return processed_text, ordered_chunk_ids
