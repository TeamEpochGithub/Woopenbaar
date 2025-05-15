import os
import sys

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force CPU usage before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as an integration test")

    # Force CPU usage by setting environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DEVICE"] = "cpu"


@pytest.fixture(autouse=True)
def force_cpu_device():
    """Force tests to use CPU instead of GPU."""
    import torch

    if torch.cuda.is_available():
        # Move all existing tensors to CPU
        torch.cuda.empty_cache()

    # Make sure any new tensors are created on CPU
    old_device = None
    if hasattr(torch, "default_device"):
        old_device = torch.default_device()
        torch.set_default_device("cpu")

    yield

    # Restore the previous device if needed
    if old_device is not None:
        torch.set_default_device(old_device)
