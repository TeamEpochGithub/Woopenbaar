#!/bin/bash
# Base Machine Setup Script
#
# This script prepares the distributed processing environment across multiple machines.
#
# Required Environment:
#   - SSH access to all worker machines
#   - Python 3 and venv module on all machines
#   - Sufficient disk space for document storage
#   - Network connectivity between all machines
#
# Directory Structure Created:
#   workers/
#   ├── standard_data_format/
#   │   ├── output_scraped/
#   │   │   └── json_files/
#   │   └── src/
#   ├── data/
#   │   ├── metadata/
#   │   │   └── divided/
#   │   └── woo_scraped/
#   │       └── documents/
#   └── venv/
#
# Security Assumptions:
#   - API keys are stored in .env files (we made use of 3 keys for 3 workers)
#   - SSH keys are properly configured
#   - Worker machines are on secure network
#
# Process Flow:
#   1. Verify machine accessibility
#   2. Create directory structure
#   3. Distribute code and configuration
#   4. Set up Python environments
#   5. Distribute data chunks

# Configuration Variables
WORKER_MACHINES=(
    "epochvpc2@145.94.40.47"
    "epochvpc1@145.94.40.176"
    # "epochvpc5@145.94.40.189"
    # "epochvpc9@145.94.40.198"
)

# Directory Structure
BASE_DIR="workers"
SDF_DIR="${BASE_DIR}/standard_data_format"
SCRIPTS_DIR="${SDF_DIR}/scripts"
DATA_DIR="${BASE_DIR}/data"
METADATA_DIR="${DATA_DIR}/metadata"
OUTPUT_DIR="${SDF_DIR}/output_scraped/json_files"
DOCUMENTS_DIR="${DATA_DIR}/woo_scraped/documents"
CHUNK_LOCKS_DIR="/tmp/chunk_locks"

# Extract API keys from original .env
KEY1=$(grep "GOOGLE_API_KEY=" standard_data_format/.env | cut -d'=' -f2)    # felipe
KEY2=$(grep "GOOGLE_API_KEY_2=" standard_data_format/.env | cut -d'=' -f2)  # felipe
KEY3=$(grep "GOOGLE_API_KEY_3=" standard_data_format/.env | cut -d'=' -f2)  # marcin

# Machine-specific chunk ranges
declare -A CHUNK_RANGES=(
    ["epochvpc2"]="8 15"    # Worker 2: Chunks 8-15
    ["epochvpc1"]="16 23"   # Worker 1: Chunks 16-23
    # ["epochvpc5"]="16 20"   # Worker 5: Chunks 16-20
    # ["epochvpc9"]="21 24"   # Worker 9: Chunks 21-24
)

# Run this on the base machine (epochvpc8)
# Function to check if a machine is reachable
check_machine() {
    if ! ping -c 1 -W 2 "$1" >/dev/null 2>&1; then
        echo "Error: Cannot reach $1"
        return 1
    fi
    return 0
}

# Function to check if Python is installed on remote machine
check_python() {
    if ! ssh "$1" "command -v python3" >/dev/null 2>&1; then
        echo "Error: Python3 not found on $1. Please install Python3 first."
        echo "Run: ssh $1 'sudo apt update && sudo apt install -y python3 python3-venv'"
        return 1
    fi
    return 0
}

# Create local directories
mkdir -p standard_data_format/output_scraped/json_files
mkdir -p data/metadata
mkdir -p data/woo_scraped/documents
mkdir -p /tmp/chunk_locks

# Generate .env files for each machine
for machine in "${WORKER_MACHINES[@]}"; do
    machine_name=${machine%@*}
    api_key_var="KEY"
    case $machine_name in
        "epochvpc1") api_key_var="KEY2";;
        "epochvpc2") api_key_var="KEY3";;
        "epochvpc5"|"epochvpc9") api_key_var="KEY1";;
    esac
    
    cat > ".env.${machine_name}" << EOL
$(grep -v "GOOGLE_API_KEY" standard_data_format/.env | grep -v "GOOGLE_API_KEY_2")
GOOGLE_API_KEY=${!api_key_var}
EOL
done

# Run metadata division if not already done
if [ ! -d "data/metadata/divided" ]; then
  echo "Dividing metadata for distributed processing..."
  ./standard_data_format/divide_metadata.sh
fi

# Copy code and data to worker machines
for machine in "${WORKER_MACHINES[@]}"; do
    echo "Setting up $machine..."
    machine_name=${machine%@*}
    machine_ip=${machine#*@}

    # Check connectivity
    if ! check_machine "$machine_ip" || ! check_python "$machine"; then
        echo "Skipping setup for $machine"
        continue
    }

    # Create remote directories
    ssh $machine "mkdir -p ${OUTPUT_DIR} ${METADATA_DIR} ${DOCUMENTS_DIR} ${CHUNK_LOCKS_DIR}" || continue

    # Copy code files
    echo "Copying code files..."
    rsync -avz standard_data_format/src/ $machine:${SDF_DIR}/src/ || continue
    for worker_script in {1..3}; do
        rsync -avz standard_data_format/scripts/run_worker_${worker_script}.sh $machine:${SCRIPTS_DIR}/ || continue
    done
    rsync -avz standard_data_format/requirements.txt $machine:${SDF_DIR}/ || continue
    rsync -avz .env.${machine_name} $machine:${SDF_DIR}/.env || continue

    # Copy data files
    echo "Copying data files..."
    rsync -avz data/metadata/all.csv $machine:${METADATA_DIR}/ || continue
    rsync -avz data/woo_scraped/documents/ $machine:${DOCUMENTS_DIR}/ || continue

    # Copy chunk files based on machine-specific range
    if [[ -n "${CHUNK_RANGES[$machine_name]}" ]]; then
        read start_chunk end_chunk <<< "${CHUNK_RANGES[$machine_name]}"
        for i in $(seq $start_chunk $end_chunk); do
            rsync -avz data/metadata/divided/metadata_chunk_$i.csv $machine:${METADATA_DIR}/divided/ || continue
        done
    fi

    # Setup Python environment
    echo "Setting up Python environment..."
    ssh $machine "python3 -m venv ${BASE_DIR}/venv" || continue
    ssh $machine "source ${BASE_DIR}/venv/bin/activate && pip install -r ${SDF_DIR}/requirements.txt" || continue

    echo "Setup completed successfully for $machine"
done

# Cleanup
rm .env.epochvpc1 .env.epochvpc2
echo "Script completed."