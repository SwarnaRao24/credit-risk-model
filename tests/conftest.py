# This file is intentionally empty.
# Its presence tells pytest to add the project root to sys.path
# so that `from app import app` works from within the tests/ folder.
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))