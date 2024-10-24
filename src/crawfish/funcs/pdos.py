"""Module for user methods to get PDOS spectrum.

Module for user methods to get PDOS spectrum.
"""

from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata
from crawfish.core.operations.matrix import get_pdos_sabcj
from crawfish.funcs.general import get_generic_spectrum, SIGMA_DEFAULT, RES_DEFAULT
from pathlib import Path
import numpy as np


def get_pdos(
    edata_input: ElecData | str | Path,
    idcs: list[int] | int | None = None,
    elements: list[str] | str | None = None,
    orbs: list[str] | str | None = None,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
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
    erange: np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    sig : REAL_DTYPE
        The sigma value for the Gaussian smearing (if lti is False).
    res : REAL_DTYPE
        The resolution of the energy range (if erange is None)
    spin_pol : bool
        If the spectrum should be returned with up/down intensities separated.
    lti : bool
        Use the linear tetrahedron integration method.
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    """
    edata = edata_input_to_edata(edata_input)
    orb_idcs = get_orb_idcs(edata, idcs, elements, orbs)
    pdos_tj = get_pdos_tj(edata.proj_tju, orb_idcs)
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
    }
    return get_generic_spectrum(edata, pdos_tj, **kwargs)
