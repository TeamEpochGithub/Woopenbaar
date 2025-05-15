"""Configure pytest for the project."""

import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
backend_src = os.path.join(project_root, "backend", "src")
sys.path.insert(0, project_root)
sys.path.insert(0, backend_src)
