"""Module for methods to correct input arguments if None.

Module for methods to correct input arguments if None.
"""

from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
import numpy as np
from pathlib import Path
from functools import wraps
from typing import TypeVar, Callable, Any, cast, ParamSpec


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


def get_use_cache(use_cache_arg: bool | None, use_cache_default: bool | None) -> bool:
    """Return use_cache from input.

    Return use_cache from input.

    Parameters
    ----------
    use_cache_arg : bool | None
        Whether to use caching.
    """
    if ((use_cache_default is None) and (use_cache_arg is None)):
        return True
    elif not use_cache_arg is None:
        return use_cache_arg
    else:
        return use_cache_default
    



T = TypeVar('T')  # Type of the first argument
P = ParamSpec('P')  # Parameter specification for remaining arguments

def create_wrapper(transformer: Callable[[T], T]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Creates a wrapper for functions that transforms their first argument.
    
    Args:
        transformer: Function that transforms the first argument
        
    Returns:
        A decorator that can be applied to any function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not args:
                return func(*args, **kwargs)
            
            # Transform the first argument and preserve all others
            new_args = (transformer(cast(T, args[0])),) + args[1:]
            return func(*new_args, **kwargs)
        
        return wrapper
    
    return decorator

def _assert_complex_bandprojections(edata_input: ElecData | str | Path) -> None:
    """
    Assert that the band projections are complex. Wrapper for functions that require complex band projections.
    
    Args:
        edata: The ElecData object to check
        
    Raises:
        ValueError: If the band projections are not complex
    """
    edata = edata_input_to_edata(edata_input)
    if edata.jdftx and (not edata.bandprojfile_is_complex):
        raise ValueError(
            "Band projections are not complex - remember to set 'band-projection-params yes no' "
            "in your jdftx in file. Aborting bonding analysis. (Data still good for pDOS analysis)"
            )
    
assert_complex_bandprojections = create_wrapper(_assert_complex_bandprojections)