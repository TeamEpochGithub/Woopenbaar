import json
from pathlib import Path
from typing import Union


def compare(
    self, documents_path: Union[str, Path], clean_document_path: Union[str, Path]
) -> None:
    """
    Compare the number of elements before and after cleaning across all JSON files in the directories.

    Args:
        documents_path: Path to the directory containing the original JSON files.
        clean_document_path: Path to the directory containing the cleaned JSON files.
    """
    documents_path = Path(documents_path)
    clean_document_path = Path(clean_document_path)

    total_before = 0
    total_after = 0
    removed_counts = []

    # Iterate over all JSON files in the original documents directory
    for file_path in documents_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)

            # Locate the corresponding cleaned file
            cleaned_file_path = clean_document_path / file_path.name
            if not cleaned_file_path.exists():
                print(f"Warning: No cleaned file found for {file_path.name}, skipping.")
                continue

            with open(cleaned_file_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)

            # Count elements before and after cleaning
            before_count = len(original_data.get("content", "").split())
            after_count = len(cleaned_data.get("content", "").split())

            total_before += before_count
            total_after += after_count
            removed_counts.append(before_count - after_count)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Compute overall statistics
    if removed_counts:
        min(removed_counts)
        max_removed = max(removed_counts)
        total_removed = total_before - total_after
        percentage_removed = (total_removed / total_before) * 100 if total_before else 0

        # Print results
        print(f"Total elements before cleaning: {total_before}")
        print(f"Total elements after cleaning: {total_after}")
        print(f"Total elements removed: {total_removed}")
        print(f"Percentage of elements removed: {percentage_removed:.2f}%")
        print(f"Maximum elements removed from a document: {max_removed}")
    else:
        print("No valid documents found for comparison.")
