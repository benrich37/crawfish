"""Module for user methods to get PCOHP spectrum.

Module for user methods to get PCOHP spectrum.
"""

from crawfish.utils.arg_correction import check_repeat
from crawfish.core.operations.matrix import _get_gen_tj
from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata, get_use_cache
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
    use_cache: bool | None = None,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the pCOHP spectrum for the system of interest.

    Get the energy array / pCOHP spectrum pair for the system of interest.

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
    use_cache : bool
        If True, cache the results of the calculation and retrieve them if they are already computed.
        This will be more time efficient for subsequent redundant pcohp evaluations
        (enumerating over the same orbitals) for calculations with more states and bands.
        Note: for each unique pair of orbitals, an array of size (nStates, nBands) is stored
        by a crawfish.utils.caching.CachedFunction object at ElecData._pcohp_tj_cache,
        where each value occupies 4 bytes. To free memory, the cache can be cleared
        via ElecData._pcohp_tj_cache.clear().
    """
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcohp_tj = _get_pcohp_tj(
        edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2,
        use_cache=use_cache
        )
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
    use_fillings: bool = True,
    use_cache: bool | None = None,
    ):
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcohp_tj = _get_pcohp_tj(
        edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2,
        use_cache=use_cache)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcohp_tj, **kwargs)
    return es, cs


def _get_pcohp_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        use_cache: bool = False,
        ):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    compute_func = lambda u, v: _compute_pcohp_tj(edata, u, v)
    if use_cache:
        pcohp_tj = edata.pcohp_tj_cache.compute_or_retrieve(
            orbs_u, orbs_v, compute_func
        )
    else:
        pcohp_tj = compute_func(orbs_u, orbs_v)
    #     tj_func =

    #     edata._pcohp_tj_cache.compute_or_retrieve(
    #         orbs_u, orbs_v, _compute_pcohp_tj
    #     )
    #     cache = edata._pcohp_tj_cache
    # h_uu = edata.h_uu
    # proj_tju = edata.proj_tju
    # wk_t = edata.wk_t
    # pcohp_tj = _get_gen_tj(proj_tju, h_uu, wk_t, orbs_u, orbs_v)
    return pcohp_tj

def _compute_pcohp_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        ):
    h_uu = edata.h_uu
    proj_tju = edata.proj_tju
    wk_t = edata.wk_t
    pcohp_tj = _get_gen_tj(proj_tju, h_uu, wk_t, orbs_u, orbs_v)
    return pcohp_tj

def get_pcobi(
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
    lite: bool = False,
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
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    """
    edata = edata_input_to_edata(edata_input)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2)
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
    return get_generic_spectrum(edata, pcoop_tj, **kwargs)


def get_ipcobi(
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
    _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2)
    kwargs = {
        "spin_pol": spin_pol,
    }
    es, cs = get_generic_integrate(edata, pcohp_tj, **kwargs)
    return es, cs


def _get_pcobi_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    p_uu = edata.p_uu
    proj_tju = edata.proj_tju
    wk_t = edata.wk_t
    pcoop_tj = _get_gen_tj(proj_tju, p_uu, wk_t, orbs_u, orbs_v)
    return pcoop_tj


def get_pcoop(
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
    lite: bool = False,
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
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    """
    edata = edata_input_to_edata(edata_input)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2)
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
    return get_generic_spectrum(edata, pcoop_tj, **kwargs)


def get_ipcoop(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    spin_pol: bool = False,
    use_fillings: bool = True,
):
    edata = edata_input_to_edata(edata_input)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcoop_tj, **kwargs)
    return es, cs


def _get_pcoop_tj(
        edata: ElecData,
        idcs1,
        elements1,
        orbs1,
        idcs2,
        elements2,
        orbs2
        ):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    s_uu = edata.s_uu
    proj_tju = edata.proj_tju
    wk_t = edata.wk_t
    pcoop_tj = _get_gen_tj(proj_tju, s_uu, wk_t, orbs_u, orbs_v)
    return pcoop_tj
