"""Module for user methods to get PCOHP spectrum.

Module for user methods to get PCOHP spectrum.
"""

from crawfish.utils.arg_correction import check_repeat
from crawfish.core.operations.matrix import get_pcohp_sabcj, get_p_uvjsabc, get_h_uvsabc, _add_kweights
from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata
from crawfish.funcs.general import get_generic_gsmear_spectrum
from pathlib import Path
import numpy as np


def get_pcohp(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = REAL_DTYPE(0.00001),
    res: REAL_DTYPE = REAL_DTYPE(0.01),
    spin_pol: bool = False,
    lite: bool = False,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the PDOS spectrum for the system of interest.

    Get the energy array / PDOS spectrum pair for the system of interest.

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    idcs(1/2) : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements(1/2) : list[str] | str | None
        The elements of interest.
    orbs(1/2) : list[str] | str | None
        The orbitals of interest.
    lite : bool
        Use the lite version of the method
        (less expenive if performing multiple pcoop/pcohps with the same ElecData object).
    """
    edata = edata_input_to_edata(edata_input)
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    if lite:
        p_uvjsabc = get_p_uvjsabc(edata.proj_sabcju, orbs_u, orbs_v)
        h_uvsabc = get_h_uvsabc(p_uvjsabc, edata.e_sabcj, orbs_u, orbs_v)
    else:
        p_uvjsabc = edata.p_uvjsabc
        h_uvsabc = edata.h_uvsabc
    pcohp_sabcj = get_pcohp_sabcj(p_uvjsabc, h_uvsabc, orbs_u, orbs_v)
    pcohp_sabcj = _add_kweights(pcohp_sabcj, edata.wk_sabc)
    return get_generic_gsmear_spectrum(edata, pcohp_sabcj, erange, spin_pol, sig, res=res)
