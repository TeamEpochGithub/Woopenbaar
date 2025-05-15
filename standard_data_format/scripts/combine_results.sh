#!/bin/bash
# Run this after all workers complete to combine results

# Activate virtual environment
source ./.venv/bin/activate

# Combine metadata
python -c "
from pathlib import Path
from standard_data_format.src.multiprocess import combine_metadata_chunks

metadata_path = Path('data/metadata/all_synced_metadata.csv')
output_path = Path('data/metadata/all_synced_metadata_final.csv')
combine_metadata_chunks(metadata_path, output_path)
"

# Run duplicate cleanup as a precaution
python trials/check_json_remove_duplicates.py

echo "Processing complete. Final metadata saved to data/metadata/all_synced_metadata_final.csv" 