#!/bin/bash
# Divide metadata into chunks for distributed processing

# Set variables
METADATA_PATH="data/metadata/vws_metadata_standardized.csv"
OUTPUT_DIR="data/metadata/divided"
TOTAL_CHUNKS=8 # Adjust based on your total worker count

# Get the full path to the script directory

# Change to project root directory
cd "$(dirname "$0")/../.."

# Activate virtual environment with full path
if [ -d "./standard_data_format/.venv" ]; then
    source ./standard_data_format/.venv/bin/activate
elif [ -d "./.venv" ]; then
    source ./.venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Verify Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Check your virtual environment."
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the division script with python3 explicitly
python3 -m standard_data_format.src.divide_metadata \
  --metadata-path $METADATA_PATH \
  --output-dir $OUTPUT_DIR \
  --total-chunks $TOTAL_CHUNKS \
  --by-family || {
    echo "Error running metadata division script"
    exit 1
  }

echo "Metadata division complete. Files saved to $OUTPUT_DIR" 