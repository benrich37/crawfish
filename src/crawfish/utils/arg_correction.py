"""Module for methods to correct input arguments if None.

Module for methods to correct input arguments if None.
"""

from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
import numpy as np
from pathlib import Path


def edata_input_to_edata(edata_input: ElecData | str | Path) -> ElecData:
    """Return ElecData object from input.

    Return ElecData object from input.

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    """
    if isinstance(edata_input, ElecData):
        return edata_input
    elif isinstance(edata_input, (str, Path)):
        return ElecData(edata_input)
    else:
        raise ValueError("edata_input must be ElecData or path to ElecData directory.")


def get_erange(
    edata: ElecData, erange: np.ndarray[REAL_DTYPE] | None, res: REAL_DTYPE = REAL_DTYPE(0.01)
) -> np.ndarray[REAL_DTYPE]:
    """Return energy range from input.

    Return energy range from input.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.
    erange : np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    """
    if erange is None:
        return np.arange(np.min(edata.e_sabcj) - 10 * res, np.max(edata.e_sabcj) + 10 * res, res, dtype=REAL_DTYPE)
    elif isinstance(erange, np.ndarray):
        return np.array(erange, dtype=REAL_DTYPE)
    else:
        raise ValueError("erange must be None or np.ndarray.")


def check_repeat(orbs_u: list[int], orbs_v: list[int]):
    """Check if there are repeated orbitals.

    Check if there are repeated orbitals.

    Parameters
    ----------
    orbs_u : list[str]
        The orbitals of interest for the first set.
    orbs_v : list[str]
        The orbitals of interest for the second set.
    """
    if any([orb in orbs_v for orb in orbs_u]):
        raise ValueError("Orbitals cannot be repeated between sets.")
    return None
