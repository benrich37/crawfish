"""Module for type checking functions.

Methods for checking the types of arrays and other objects.
"""

import numpy as np

REAL_DTYPE = np.float32
COMPLEX_DTYPE = np.complex64


def check_arr_typing_all(arrs: list[np.ndarray]) -> None:
    """Raise ValueError if all arrays are not of the same type (REAL_DTYPE or COMPLEX_DTYPE).

    Raise ValueError if all arrays are not of the same type (REAL_DTYPE or COMPLEX_DTYPE).

    Parameters
    ----------
    arrs : list[np.ndarray]
        The list of arrays to check
    """
    num_dtype = arrs[0].dtype
    if num_dtype not in [REAL_DTYPE, COMPLEX_DTYPE]:
        raise ValueError("All arrays must be of type REAL_DTYPE or COMPLEX_DTYPE")
    for arr in arrs:
        if not arr.dtype == num_dtype:
            raise ValueError("All arrays must have the same dtype (REAL_DTYPE or COMPLEX_DTYPE)")


def check_arr_typing(arrs: list[np.ndarray]) -> None:
    """Raise ValueError if any arrays are not of type REAL_DTYPE or COMPLEX_DTYPE.

    Raise ValueError if any arrays are not of type REAL_DTYPE or COMPLEX_DTYPE.

    Parameters
    ----------
    arrs : list[np.ndarray]
        The list of arrays to check
    """
    for arr in arrs:
        if arr.dtype not in [REAL_DTYPE, COMPLEX_DTYPE]:
            raise ValueError("All arrays must be of type REAL_DTYPE or COMPLEX_DTYPE")


def cs_formatter(cs: list[np.ndarray], spin_pol: bool) -> np.ndarray:
    """Return the cs array in the appropriate format.

    Return the cs array in the appropriate format.

    Parameters
    ----------
    cs : list[np.ndarray]
        List of dos/pdos/pcohp arrays of length N (parallel to Erange of length N)
    spin_pol : bool
        Return as (up, down) or (tot)
    """
    if len(cs) > 2:
        raise ValueError(f"Unexpected numbers of spin ({len(cs)} found, only 1-2 supported)")
    if spin_pol:
        if len(cs) == 1:
            raise ValueError("Spin-polarized output not supported for spin-paired output")
        else:
            return np.array([cs[0], cs[1]])
    else:
        if len(cs) == 1:
            return cs[0]
        else:
            return cs[0] + cs[1]
