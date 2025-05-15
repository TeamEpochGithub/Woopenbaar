import argparse
import os
import random

import pandas as pd

from standard_data_format.utils.logger import setup_logger

logger = setup_logger()


def divide_metadata(
    metadata_path: str, output_dir: str, total_chunks: int, by_family: bool = True
):
    """
    Divide metadata into non-overlapping chunks to prevent duplicate processing.

    Args:
        metadata_path: Path to the metadata CSV file
        output_dir: Directory to save the chunk-specific metadata files
        total_chunks: Number of chunks to divide the metadata into
        by_family: Whether to divide by family (recommended) or by individual documents
    """
    logger.info(f"Loading metadata from {metadata_path}")

    try:
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        logger.info(f"Loaded metadata with {len(metadata_df)} documents")
    except pd.errors.EmptyDataError:
        # Handle empty CSV file by creating an empty DataFrame with basic columns
        logger.warning(f"Empty CSV file detected at {metadata_path}")
        metadata_df = pd.DataFrame(
            columns=["ID", "Family", "Document", "Document Link"]
        )
        logger.info("Created empty DataFrame with standard columns")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if by_family and not metadata_df.empty and "Family" in metadata_df.columns:
        # Get unique families and distribute them randomly
        families = list(metadata_df["Family"].unique())
        logger.info(f"Found {len(families)} unique families to distribute")

        # Randomly shuffle families
        random.shuffle(families)

        # Allocate families to chunks
        family_chunks = {}
        for i, family in enumerate(families):
            chunk_id = i % total_chunks
            if chunk_id not in family_chunks:
                family_chunks[chunk_id] = []
            family_chunks[chunk_id].append(family)

        # Create chunk-specific metadata files
        for chunk_id, chunk_families in family_chunks.items():
            chunk_df = metadata_df[metadata_df["Family"].isin(chunk_families)]
            output_path = os.path.join(output_dir, f"metadata_chunk_{chunk_id}.csv")
            chunk_df.to_csv(output_path, index=False)
            logger.info(
                f"Chunk {chunk_id}: {len(chunk_df)} documents from {len(chunk_families)} families"
            )
    else:
        # If empty DataFrame or no Family column, divide evenly by document rows
        # Or if by_family is False
        if metadata_df.empty:
            # For empty DataFrame, create empty chunk files
            for chunk_id in range(total_chunks):
                output_path = os.path.join(output_dir, f"metadata_chunk_{chunk_id}.csv")
                metadata_df.to_csv(output_path, index=False)
                logger.info(f"Chunk {chunk_id}: 0 documents (empty file)")
        else:
            # Randomly shuffle the dataframe with a fixed seed for consistent results
            metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(
                drop=True
            )

            # Divide documents evenly using ceiling division for consistent chunk sizes
            total_docs = len(metadata_df)
            base_chunk_size = total_docs // total_chunks
            remainder = total_docs % total_chunks

            start_idx = 0
            for chunk_id in range(total_chunks):
                # Give one extra document to the first 'remainder' chunks
                chunk_size = base_chunk_size + (1 if chunk_id < remainder else 0)
                end_idx = start_idx + chunk_size

                # Extract chunk and save to file
                chunk_df = metadata_df.iloc[start_idx:end_idx]
                output_path = os.path.join(output_dir, f"metadata_chunk_{chunk_id}.csv")
                chunk_df.to_csv(output_path, index=False)
                logger.info(f"Chunk {chunk_id}: {len(chunk_df)} documents")

                # Update start index for next chunk
                start_idx = end_idx

    logger.info(f"Divided metadata into {total_chunks} chunks successfully")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide metadata into chunks for distributed processing"
    )
    parser.add_argument(
        "--metadata-path", type=str, required=True, help="Path to the metadata CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the chunk-specific metadata files",
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        required=True,
        help="Number of chunks to divide the metadata into",
    )
    parser.add_argument(
        "--by-family", action="store_true", help="Divide by family (recommended)"
    )

    args = parser.parse_args()
    divide_metadata(
        args.metadata_path, args.output_dir, args.total_chunks, args.by_family
    )

    # 855412 to standard data format (UUID: 56c42f89-b06b-e405-a0c4-afbec762c07a)
