"""Pytest config for Wan Studio.

Adds the project root to sys.path so tests can `from pipelines import ...`
without an installed package.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
