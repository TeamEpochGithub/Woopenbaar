#!/bin/bash

# Store the root directory path
ROOT_DIR="$(realpath "$(dirname "$0")")"

echo "Running tests from: $ROOT_DIR"

# Run backend tests
echo -e "\nPreparing Python environment..."
VENV_PATH="$ROOT_DIR/backend/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    pip install -r "$ROOT_DIR/backend/requirements.txt"
    
    echo -e "\nRunning Python tests..."
    cd "$ROOT_DIR"
    pytest backend/tests/
    PYTEST_EXIT_CODE=$?
    
    deactivate
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Run frontend tests
echo -e "\nPreparing JavaScript environment..."
cd "$ROOT_DIR/frontend"
echo "Installing npm dependencies..."
npm install

echo -e "\nRunning JavaScript tests..."
npm test
NPM_EXIT_CODE=$?

# Exit with failure if either test suite failed
if [ $PYTEST_EXIT_CODE -ne 0 ] || [ $NPM_EXIT_CODE -ne 0 ]; then
    echo -e "\nTests failed!"
    exit 1
else
    echo -e "\nAll tests passed!"
    exit 0
fi