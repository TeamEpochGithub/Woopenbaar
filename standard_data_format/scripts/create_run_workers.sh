#!/bin/bash

# Configuration
TOTAL_CHUNKS=24
NUM_WORKERS=3              # Total number of workers
CHUNKS_PER_WORKER=$((TOTAL_CHUNKS / NUM_WORKERS))
SOURCE_SCRIPT="./standard_data_format/scripts/run_gemini_worker.sh"    # Source script to copy from

# Loop through workers and create their scripts
for ((WORKER_NUM=1; WORKER_NUM<=NUM_WORKERS; WORKER_NUM++)); do
    # Calculate chunk range for this worker
    START_CHUNK=$(( (WORKER_NUM-1) * CHUNKS_PER_WORKER ))
    END_CHUNK=$(( (WORKER_NUM * CHUNKS_PER_WORKER) - 1 ))

    # Create worker-specific script
    WORKER_SCRIPT="./standard_data_format/scripts/run_worker_${WORKER_NUM}.sh"
    cp "$SOURCE_SCRIPT" "$WORKER_SCRIPT"

    # Update variables in the new script
    sed -i "s/TOTAL_CHUNKS=.*/TOTAL_CHUNKS=${TOTAL_CHUNKS}/" "$WORKER_SCRIPT"
    sed -i "s/START_CHUNK=.*/START_CHUNK=${START_CHUNK}/" "$WORKER_SCRIPT"
    sed -i "s/END_CHUNK=.*/END_CHUNK=${END_CHUNK}/" "$WORKER_SCRIPT"

    echo "Created $WORKER_SCRIPT with chunks $START_CHUNK to $END_CHUNK out of $TOTAL_CHUNKS total chunks"
done


