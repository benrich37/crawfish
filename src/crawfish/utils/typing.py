"""Module for type checking functions.

Methods for checking the types of arrays and other objects.
"""

import numpy as np


def check_arr_typing_all(arrs: list[np.ndarray]) -> None:
    """Raise ValueError if all arrays are not of the same type (np.float32 or np.complex64).

    Raise ValueError if all arrays are not of the same type (np.float32 or np.complex64).

    Parameters
    ----------
    arrs : list[np.ndarray]
        The list of arrays to check
    """
    num_dtype = arrs[0].dtype
    if num_dtype not in [np.float32, np.complex64]:
        raise ValueError("All arrays must be of type np.float32 or np.complex64")
    for arr in arrs:
        if not arr.dtype == num_dtype:
            raise ValueError("All arrays must have the same dtype (np.float32 or np.complex64)")


def check_arr_typing(arrs: list[np.ndarray]) -> None:
    """Raise ValueError if any arrays are not of type np.float32 or np.complex64.

    Raise ValueError if any arrays are not of type np.float32 or np.complex64.

    Parameters
    ----------
    arrs : list[np.ndarray]
        The list of arrays to check
    """
    for arr in arrs:
        if arr.dtype not in [np.float32, np.complex64]:
            raise ValueError("All arrays must be of type np.float32 or np.complex64")
