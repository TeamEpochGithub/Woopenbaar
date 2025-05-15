#!/bin/bash

# Configuration Parameters
BASE_MACHINE="epochvpc8@145.94.40.35"  # SSH connection string for base machine
TOTAL_CHUNKS=1                        # Total number of processing chunks
START_CHUNK=0                          # Start chunk number
END_CHUNK=0                            # End chunk number
OLLAMA_GPU=0                          # GPU ID for Ollama service
PROCESSING_GPU=1                       # GPU ID for document processing
OLLAMA_STARTUP_WAIT=10                # Seconds to wait for Ollama to start
LOCK_DIR="/tmp/chunk_locks"           # Directory for process locks
OLLAMA_LOG="/tmp/ollama.log"          # Ollama service log file
SYNC_INTERVAL=10                      # How often to sync with base machine (in documents)
YAML_CONFIG="standard_data_format/config/local_config.yaml" # Path to the YAML configuration file
CUSTOM_METADATA_PATH="data/metadata/divided"
# Worker Process Management Script
#
# This script manages the distributed processing of document chunks on a worker machine.
# GPU 0 is dedicated to the LLM service (Ollama)
# GPU 1 is used for document processing
#
# Required Environment:
#   - Python virtual environment at ./standard_data_format/.venv/
#   - CUDA-capable GPUs (minimum 2)
#   - /tmp/chunk_locks/ directory for process management
#
# Process Flow:
#   1. Starts Ollama on GPU 0
#   2. Processes documents using GPU 1
#   3. Monitors and manages chunk processing
#   4. Cleans up on completion

# Change to the project root directory
cd "$(dirname "$0")/../../"

# Activate virtual environment
source ./standard_data_format/.venv/bin/activate

# Create a temporary directory to track running processes
mkdir -p $LOCK_DIR

# Start Ollama on GPU 0
export CUDA_VISIBLE_DEVICES=$OLLAMA_GPU
echo "Starting Ollama service on GPU $OLLAMA_GPU..."
pkill ollama  # Stop any existing Ollama processes
ollama serve > $OLLAMA_LOG 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to start
sleep $OLLAMA_STARTUP_WAIT

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

# Start processes 0-3 (4 processes total)
# All processing will use GPU 1
for i in $(seq $START_CHUNK $((END_CHUNK))); do
    # Check if chunk is already running
    if is_chunk_running $i; then
        echo "Chunk $i is already being processed, skipping..."
        continue
    fi
    
    # Create lock file with current process ID
    echo $$ > "$LOCK_DIR/chunk_${i}.lock"
    
    # Use GPU 1 for all processing
    export CUDA_VISIBLE_DEVICES=$PROCESSING_GPU
    
    # Use custom metadata path
    custom_metadata="${CUSTOM_METADATA_PATH}/metadata_chunk_$i.csv"
    
    # Verify metadata file exists
    if [ ! -f "$custom_metadata" ]; then
        echo "Error: Metadata file not found: $custom_metadata"
        rm "$LOCK_DIR/chunk_${i}.lock"
        continue
    fi
    
    echo "Starting chunk $i on GPU $PROCESSING_GPU"
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
    sleep 30
done

# Wait for all background processes to complete
wait

# Clean up
echo "Stopping Ollama service..."
kill $OLLAMA_PID
rm -f $LOCK_DIR/chunk_*
rm -f $OLLAMA_LOG

echo "Processing complete"