"""This module contains utility functions for testing the crawfish package."""

from crawfish.core import ROOT
from pathlib import Path
import numpy as np
from crawfish.utils.typing import REAL_DTYPE

TEST_FILES_DIR = Path(ROOT) / ".." / "tests" / "files"
EXAMPLE_FILES_DIR = TEST_FILES_DIR / "io" / "example_files"
EXAMPLE_CALC_DIRS_DIR = TEST_FILES_DIR / "io" / "example_calc_dirs"
TMP_FILES_DIR = TEST_FILES_DIR / "io" / "tmp_files"


def approx_idx(arr: np.ndarray[REAL_DTYPE], val: REAL_DTYPE) -> int:
    """Return index of value in array closest to input value.

    Return index of value in array closest to input value.

    Parameters
    ----------
    arr : np.ndarray[REAL_DTYPE]
        The (1D) array of interest.
    val : REAL_DTYPE
        The value of interest.
    """
    difs = np.abs(arr - val)
    return np.argsort(difs)[0]


def get_pocket_idx(e_sabcj: np.ndarray[REAL_DTYPE], erange: np.ndarray[REAL_DTYPE]) -> int:
    """Return the index of greatest gap in eigenvalues within energy range.

    Return the index of greatest gap in eigenvalues within energy range.

    Parameters
    ----------
    e_sabcj : np.ndarray[REAL_DTYPE]
        The eigenvalues of interest.
    erange : np.ndarray[REAL_DTYPE]
        The energy range of interest.
    """
    spaces = np.diff(e_sabcj.flatten())
    pocket_mean = np.mean(e_sabcj.flatten()[np.argmax(spaces) : np.argmax(spaces) + 2])
    return approx_idx(erange, pocket_mean)
