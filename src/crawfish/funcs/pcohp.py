"""Module for user methods to get PCOHP spectrum.

Module for user methods to get PCOHP spectrum.
"""

from crawfish.utils.arg_correction import check_repeat
from crawfish.core.operations.matrix import _get_gen_tj
from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata
from crawfish.funcs.general import get_generic_spectrum, get_generic_integrate
from pathlib import Path
import numpy as np
from crawfish.funcs.general import SIGMA_DEFAULT, RES_DEFAULT


def get_pcohp(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
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
    idcs(1/2) : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements(1/2) : list[str] | str | None
        The elements of interest.
    orbs(1/2) : list[str] | str | None
        The orbitals of interest.
    lite : bool
        Use the lite version of the method
        (less expenive if performing multiple pcoop/pcohps with the same ElecData object).
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
    pcohp_tj = _get_pcohp_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2)
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
    return get_generic_spectrum(edata, pcohp_tj, **kwargs)


def get_ipcohp(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    spin_pol: bool = False,
):
    edata = edata_input_to_edata(edata_input)
    pcohp_tj = _get_pcohp_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2)
    kwargs = {
        "spin_pol": spin_pol,
    }
    es, cs = get_generic_integrate(edata, pcohp_tj, **kwargs)
    return es, cs


def _get_pcohp_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    h_uu = edata.h_uu
    proj_tju = edata.proj_tju
    wk_t = edata.wk_t
    pcohp_tj = _get_gen_tj(proj_tju, h_uu, wk_t, orbs_u, orbs_v)
    return pcohp_tj
