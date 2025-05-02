"""Module for user methods to get PCOHP spectrum.

Module for user methods to get PCOHP spectrum.
"""

from crawfish.utils.arg_correction import check_repeat
from crawfish.core.operations.matrix import _get_gen_tj_coef_uu, _get_gen_tj_coef_tuu
from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata, get_use_cache, assert_complex_bandprojections
from crawfish.funcs.general import evaluate_or_retrieve_generic_spectrum, get_generic_integrate
from pathlib import Path
import numpy as np
from crawfish.funcs.general import SIGMA_DEFAULT, RES_DEFAULT

@assert_complex_bandprojections
def get_pcohp(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    state_sep: bool = False,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
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
    state_sep : bool
        If the Hamiltonian coefficients should be resolved by state.
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
        use_cache=use_cache, state_sep=state_sep
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
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "state_sep": state_sep,
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcohp_tj, "pcohp", **kwargs)



@assert_complex_bandprojections
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
        state_sep: bool = False,
        ):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    compute_func = lambda u, v: _compute_pcohp_tj(edata, u, v, state_sep)
    if use_cache:
        if state_sep:
            pcohp_tj = edata.tsep_pcohp_tj_cache.compute_or_retrieve(
                orbs_u, orbs_v, compute_func
            )
        else:
            pcohp_tj = edata.pcohp_tj_cache.compute_or_retrieve(
                orbs_u, orbs_v, compute_func
            )
    else:
        pcohp_tj = compute_func(orbs_u, orbs_v)
    return pcohp_tj

def _compute_pcohp_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        state_sep: bool,
        ):
    weighted_overlaps_tjuv = edata.weighted_overlap_tjuv
    if state_sep:
        h_t_uu = edata.h_t_uu * 1/edata.wk_t[:, np.newaxis, np.newaxis]
        pcohp_tj = _get_gen_tj_coef_tuu(weighted_overlaps_tjuv, h_t_uu, orbs_u, orbs_v)
    else:
        h_uu = edata.h_uu
        pcohp_tj = _get_gen_tj_coef_uu(weighted_overlaps_tjuv, h_uu, orbs_u, orbs_v)
    return pcohp_tj



# Cite (C. A. Coulson, Proc. Roy. Soc. (London) 169A, 419 (1939).)
@assert_complex_bandprojections
def get_pcomo(
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
    sep_channels: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the pCOMO spectrum for the system of interest.

    Get the energy array / pCOMO spectrum pair for the system of interest.

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
    pcomo_tj = _get_pcomo_tj(
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
        "sep_channels": sep_channels,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcomo_tj, "pcomo", **kwargs)

@assert_complex_bandprojections
def get_ipcomo(
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
    pcobi_tj = _get_pcomo_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, use_cache=use_cache)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcobi_tj, **kwargs)
    return es, cs


def _get_pcomo_tj(
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
    compute_func = lambda u, v: _compute_pcomo_tj(edata, u, v)
    if use_cache:
        pcohp_tj = edata.pcomo_tj_cache.compute_or_retrieve(
            orbs_u, orbs_v, compute_func
        )
    else:
        pcohp_tj = compute_func(orbs_u, orbs_v)
    return pcohp_tj

def _compute_pcomo_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        ):
    h_uu = np.ones_like(edata.h_uu)
    weighted_overlaps_tjuv = edata.weighted_overlap_tjuv
    pcohp_tj = _get_gen_tj_coef_uu(weighted_overlaps_tjuv, h_uu, orbs_u, orbs_v)
    return pcohp_tj


@assert_complex_bandprojections
def get_pcoop(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    state_sep: bool = False,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
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
    state_sep : bool
        If the Hamiltonian coefficients should be resolved by state.
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
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, use_cache=use_cache)
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
        "sep_channels": sep_channels,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "state_sep": state_sep,
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcoop_tj, "pcoop", **kwargs)

@assert_complex_bandprojections
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
    use_cache: bool | None = None,
    state_sep: bool = False,
):
    edata = edata_input_to_edata(edata_input)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, use_cache=use_cache)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcoop_tj, **kwargs)
    return es, cs


def _get_pcoop_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        state_sep: bool = False,
        use_cache: bool = False,
        ):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    compute_func = lambda u, v: _compute_pcoop_tj(edata, u, v, state_sep)
    if use_cache:
        if state_sep:
            pcoop_tj = edata.tsep_pcoop_tj_cache.compute_or_retrieve(
                orbs_u, orbs_v, compute_func
            )
        else:
            pcoop_tj = edata.pcoop_tj_cache.compute_or_retrieve(
                orbs_u, orbs_v, compute_func
            )
    else:
        pcoop_tj = compute_func(orbs_u, orbs_v)
    return pcoop_tj

@assert_complex_bandprojections
def _compute_pcoop_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        state_sep: bool,
        ):
    check_repeat(orbs_u, orbs_v)
    weighted_overlaps_tjuv = edata.weighted_overlap_tjuv
    if state_sep:
        s_t_uu = edata.s_t_uu * 1/edata.wk_t[:, np.newaxis, np.newaxis]
        pcoop_tj = _get_gen_tj_coef_tuu(weighted_overlaps_tjuv, s_t_uu, orbs_u, orbs_v)
    else:
        s_uu = edata.s_uu
        pcoop_tj = _get_gen_tj_coef_uu(weighted_overlaps_tjuv, s_uu, orbs_u, orbs_v)
    return pcoop_tj

##
@assert_complex_bandprojections
def get_pcobi(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    state_sep: bool = False,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
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
    state_sep : bool
        If the Hamiltonian coefficients should be resolved by state.
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
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcobi_tj = _get_pcobi_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, use_cache=use_cache)
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
        "sep_channels": sep_channels,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "state_sep": state_sep,
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcobi_tj, "pcobi", **kwargs)

@assert_complex_bandprojections
def get_ipcobi(
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
    state_sep: bool = False,
):
    edata = edata_input_to_edata(edata_input)
    pcobi_tj = _get_pcobi_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, use_cache=use_cache)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcobi_tj, **kwargs)
    return es, cs



def _get_pcobi_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        state_sep: bool = False,
        use_cache: bool = False,
        ):
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    check_repeat(orbs_u, orbs_v)
    compute_func = lambda u, v: _compute_pcobi_tj(edata, u, v, state_sep)
    if use_cache:
        if state_sep:
            pcobi_tj = edata.tsep_pcobi_tj_cache.compute_or_retrieve(
                orbs_u, orbs_v, compute_func
            )
        else:
            pcobi_tj = edata.pcobi_tj_cache.compute_or_retrieve(
                orbs_u, orbs_v, compute_func
            )
    else:
        pcobi_tj = compute_func(orbs_u, orbs_v)
    return pcobi_tj

def _compute_pcobi_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        state_sep: bool,
        ):
    check_repeat(orbs_u, orbs_v)
    weighted_overlaps_tjuv = edata.weighted_overlap_tjuv
    if state_sep:
        p_t_uu = edata.p_t_uu * 1/edata.wk_t[:, np.newaxis, np.newaxis]
        pcobi_tj = _get_gen_tj_coef_tuu(weighted_overlaps_tjuv, p_t_uu, orbs_u, orbs_v)
    else:
        p_uu = edata.p_uu
        pcobi_tj = _get_gen_tj_coef_uu(weighted_overlaps_tjuv, p_uu, orbs_u, orbs_v)
    return pcobi_tj