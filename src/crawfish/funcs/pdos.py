"""Module for user methods to get PDOS spectrum.

Module for user methods to get PDOS spectrum.
"""

from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata
from crawfish.core.operations.matrix import get_pdos_sabcj, _add_kweights
from crawfish.funcs.general import get_generic_gsmear_spectrum
from pathlib import Path
import numpy as np


def get_pdos(
    edata_input: ElecData | str | Path,
    idcs: list[int] | int | None = None,
    elements: list[str] | str | None = None,
    orbs: list[str] | str | None = None,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = REAL_DTYPE(0.00001),
    res: REAL_DTYPE = REAL_DTYPE(0.01),
    spin_pol: bool = False,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the PDOS spectrum for the system of interest.

    Get the energy array / PDOS spectrum pair for the system of interest.

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    idcs : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements : list[str] | str | None
        The elements of interest.
    orbs : list[str] | str | None
        The orbitals of interest.
    """
    edata = edata_input_to_edata(edata_input)
    orb_idcs = get_orb_idcs(edata, idcs, elements, orbs)
    pdos_sabcj = get_pdos_sabcj(edata.proj_sabcju, orb_idcs)
    pdos_sabcj = _add_kweights(pdos_sabcj, edata.wk_sabc)
    return get_generic_gsmear_spectrum(edata, pdos_sabcj, erange, spin_pol, sig, res=res)
