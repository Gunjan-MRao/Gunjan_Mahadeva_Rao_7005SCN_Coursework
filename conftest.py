"""Pytest path configuration for repository-local package imports."""

import sys, pathlib

# Add the repository root so tests resolve the ``src`` package consistently.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
