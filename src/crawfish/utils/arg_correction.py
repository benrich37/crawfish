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
    edata: ElecData, erange: np.ndarray[REAL_DTYPE] | None, res: REAL_DTYPE | None = None,
    sig: REAL_DTYPE | None = None,
) -> np.ndarray[REAL_DTYPE]:
    """Return energy range from input.

    An automatic energy range (erange None) runs from the lowest to the highest eigenvalue in steps of `res`,
    padded on both sides by 10 res, or, for a Gaussian of width parameter `sig` (standard deviation sqrt(sig/2)),
    by 5 standard deviations if that is more, so the outermost peaks aren't cut. Pass sig=None for LTI spectra.
    With res = 0.01 Ha and the default sig, the pad is 10 res, the grid crawfish used by default before finding F17.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.
    erange : np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    res : REAL_DTYPE | None
        Grid step (Ha) of an automatic range; None means crawfish.funcs.general.RES_DEFAULT.
    sig : REAL_DTYPE | None
        Width parameter of the Gaussian the range is for (None: no Gaussian, e.g. LTI).
    """
    if erange is None:
        if res is None:
            from crawfish.funcs.general import RES_DEFAULT  # funcs.general imports this module

            res = RES_DEFAULT
        pad = 10 * res
        if sig is not None:
            pad = max(pad, 5 * np.sqrt(sig / 2))
        return np.arange(np.min(edata.e_sabcj) - pad, np.max(edata.e_sabcj) + pad, res, dtype=REAL_DTYPE)
    elif isinstance(erange, np.ndarray):
        return np.array(erange, dtype=REAL_DTYPE)
    else:
        raise ValueError("erange must be None or np.ndarray.")


def check_repeat(orbs_u: list[int], orbs_v: list[int], image: str | list[int] | None = None):
    """Check that the two orbital sets of a pair spectrum share no orbital, unless `image` allows it.

    An orbital paired with itself in the same cell is its on-site term, not a pair interaction, so sets that share
    orbitals are refused for image="nearest" (an atom's nearest image of itself is itself), image="all" (which
    would include that on-site term) and image=(0, 0, 0). For an explicit lattice vector R != (0, 0, 0), every
    pair is between an orbital in cell 0 and one in cell R, so sets may overlap: this is how an atom's bond with
    its own periodic image is selected (finding F19).

    Parameters
    ----------
    orbs_u : list[int]
        The orbital indices of the first set.
    orbs_v : list[int]
        The orbital indices of the second set.
    image : str | list[int] | None
        The (normalized) lattice-image selection of the pair spectrum.
    """
    if not any(orb in orbs_v for orb in orbs_u):
        return None
    if isinstance(image, (list, tuple)) and any(int(x) != 0 for x in image):
        return None
    raise ValueError(
        "Orbitals cannot be repeated between sets: within one cell, an orbital paired with itself is its on-site "
        "energy, not a pair interaction, and image='nearest' pairs an atom with itself in the same cell. For an "
        "atom's bond with its own periodic image, pass image=<lattice vector> (e.g. image=(1, 0, 0)); "
        "crawfish.funcs.get_nearest_images lists the nearest ones (finding F19)."
    )


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

def _assert_complex_bandprojections(edata_input: ElecData | str | Path) -> ElecData:
    """
    Assert that the band projections are complex, and return the ElecData. Pair spectra (pCOHP, pCOOP, pCOBI,
    pCOMO) call it from their `_get_<mode>_tj`; it is also the transformer of the `assert_complex_bandprojections`
    decorator, which was commented out because this returned None (replacing the decorated function's first
    argument). See finding F18.

    Args:
        edata: The ElecData object to check

    Raises:
        ValueError: If the band projections are not complex
    """
    edata = edata_input_to_edata(edata_input)
    if edata.projections_are_norms:
        raise ValueError(
            "This bandProjections file holds |projection|^2, without phases, so pair spectra (pCOHP, pCOOP, pCOBI, "
            "pCOMO) can't be computed from it; DOS and pDOS still work. Rerun JDFTx with the second argument of "
            "band-projection-params set to 'no' (complex projections), e.g. 'band-projection-params yes no' "
            "(finding F18)."
            )
    return edata

assert_complex_bandprojections = create_wrapper(_assert_complex_bandprojections)