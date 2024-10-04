"""This module contains utility functions for testing the crawfish package."""

from crawfish.core import ROOT
from pathlib import Path

TEST_FILES_DIR = Path(ROOT) / ".." / "tests" / "files"
EXAMPLE_FILES_DIR = TEST_FILES_DIR / "io" / "example_files"
EXAMPLE_CALC_DIRS_DIR = TEST_FILES_DIR / "io" / "example_calc_dirs"
TMP_FILES_DIR = TEST_FILES_DIR / "io" / "tmp_files"
