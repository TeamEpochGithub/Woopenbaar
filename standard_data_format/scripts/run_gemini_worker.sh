#!/bin/bash

# Configuration Parameters
BASE_MACHINE="epochvpc8@145.94.40.35"  # SSH connection string for base machine
TOTAL_CHUNKS=8                         # Total number of processing chunks
START_CHUNK=0                          # Start chunk number
END_CHUNK=7                            # End chunk number
GPU_COUNT=2                           # Number of GPUs to use
LOCK_DIR="/tmp/chunk_locks"           # Directory for process locks
SYNC_INTERVAL=10                      # How often to sync with base machine (in documents)
LAUNCH_DELAY=30                       # Seconds to wait between launching processes
YAML_CONFIG="standard_data_format/config/gemini_config.yaml" # Path to the YAML configuration file
CUSTOM_METADATA_PATH="data/metadata/divided"
# Worker Process Management Script
#
# This script manages the distributed processing of document chunks on a worker machine.
#
# Required Environment:
#   - Python virtual environment at ./standard_data_format/.venv/
#   - CUDA-capable GPUs (minimum 2)
#   - /tmp/chunk_locks/ directory for process management
#
# Process Flow:
#   1. Activates virtual environment
#   2. Creates process lock directory
#   3. Distributes chunks across GPUs
#   4. Monitors and manages chunk processing
#   5. Cleans up on completion

# Change to the project root directory
cd "$(dirname "$0")/../.."

# Activate virtual environment
source ./standard_data_format/.venv/bin/activate

# Create a temporary directory to track running processes
mkdir -p $LOCK_DIR

# Function to check if chunk is already running
is_chunk_running() {
    local chunk_id=$1
    local lock_file="$LOCK_DIR/chunk_${chunk_id}.lock"
    
    if [ -f "$lock_file" ]; then
        # Check if process is still running
        local pid=$(cat "$lock_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # Chunk is running
        else
            rm "$lock_file"  # Clean up stale lock
        fi
    fi
    return 1  # Chunk is not running
}

# Start processes 0-7 (8 processes total)
for i in $(seq $START_CHUNK $((END_CHUNK))); do
    # Check if chunk is already running
    if is_chunk_running $i; then
        echo "Chunk $i is already being processed, skipping..."
        continue
    fi
    
    # Create lock file with current process ID
    echo $$ > "$LOCK_DIR/chunk_${i}.lock"
    
    # Distribute across available GPUs
    gpu_id=$((i % GPU_COUNT))
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Use custom metadata path
    custom_metadata="${CUSTOM_METADATA_PATH}/metadata_chunk_$i.csv"
    
    # Verify metadata file exists
    if [ ! -f "$custom_metadata" ]; then
        echo "Error: Metadata file not found: $custom_metadata"
        rm "$LOCK_DIR/chunk_${i}.lock"
        continue
    fi
    
    echo "Starting chunk $i on GPU $gpu_id"
    python -m standard_data_format.src.multiprocess \
        --chunk $i \
        --total-chunks $TOTAL_CHUNKS \
        --base-machine $BASE_MACHINE \
        --sync-interval $SYNC_INTERVAL \
        --custom-metadata $custom_metadata \
        --config $YAML_CONFIG &
        
    # Store the background process PID in the lock file
    echo $! > "$LOCK_DIR/chunk_${i}.lock"
    
    # Add a small delay between launches
    sleep $LAUNCH_DELAY
done

# Wait for all background processes to complete
wait

# Clean up lock files
rm -f $LOCK_DIR/chunk_*

echo "Processing complete"