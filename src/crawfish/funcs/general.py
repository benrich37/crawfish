"""Module for general methods to work alongside spectrum-getting methods.

Module for general methods to work alongside spectrum-getting methods.
"""

from crawfish.core.elecdata import ElecData
from crawfish.core.operations.vector import get_gauss_smear_spectrum, get_lti_spectrum, get_uneven_integrated_array
from crawfish.utils.arg_correction import get_erange
from crawfish.utils.typing import REAL_DTYPE, cs_formatter, COMPLEX_DTYPE
from crawfish.utils.caching import get_generic_cache_dir, get_erange_cache_dir
from copy import deepcopy
import numpy as np
from scipy.integrate import trapezoid
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:
    from typing import Any

# Gaussian width parameter sig (kernel exp(-(E - e)^2 / sig)): standard deviation sqrt(sig / 2) = 2.2e-3 Ha
SIGMA_DEFAULT = REAL_DTYPE(0.00001)
# Grid step (Ha) of an automatic energy range. 1e-3 Ha is under half the default standard deviation, so the area of
# a peak on the grid is its weight to well below 1e-6. It was 0.01 Ha (4.5 standard deviations) before finding F17,
# when a peak's area on the grid ranged from 0.29 to 1.78 of its weight; pass res=0.01 for that grid.
RES_DEFAULT = REAL_DTYPE(0.001)
# A Gaussian peak's area on a grid of step h is off by up to 2 exp(-2 pi^2 (std / h)^2): 1.4% at h = 2 std
GRID_STEP_WARN_STD = 2.0


def _resolve_erange(edata: ElecData, erange, res, sig, lti: bool) -> np.ndarray[REAL_DTYPE]:
    """Return the energy grid of a spectrum: `erange` as given, or the automatic one (finding F17).

    Every function that needs the grid of one spectrum call (the spectrum itself, its cache path, the returned
    erange) gets it from here, so they always agree. The automatic range is padded by max(10 res, 5 standard
    deviations) for Gaussian spectra and by 10 res for LTI (see utils/arg_correction.get_erange).
    """
    return get_erange(edata, erange, res=res, sig=None if lti else sig)


def _warn_if_grid_too_coarse(erange: np.ndarray, sig) -> None:
    """Warn when the grid step exceeds GRID_STEP_WARN_STD standard deviations of the Gaussian (finding F17)."""
    if len(erange) < 2:
        return
    step = float(np.max(np.diff(np.asarray(erange, dtype=np.float64))))
    std = float(np.sqrt(float(sig) / 2))
    if step > GRID_STEP_WARN_STD * std:
        warnings.warn(
            f"The energy grid step ({step:.2g} Ha) is more than {GRID_STEP_WARN_STD:g} standard deviations of the "
            f"Gaussian ({std:.2g} Ha, from sig={float(sig):g} Ha^2), so the area of each peak on the grid depends on "
            "where its eigenvalue falls between grid points, by more than 1% (up to several times its weight for "
            f"steps of 4 or more standard deviations). Pass a finer `res` or `erange` (step <= "
            f"{GRID_STEP_WARN_STD * std:.2g} Ha) or a larger `sig`. Integrated spectra (get_i*) are exact sums and "
            "aren't affected (finding F17).",
            stacklevel=4,
        )


def get_generic_integrate(
    edata: ElecData, weights_tj: np.ndarray[REAL_DTYPE | COMPLEX_DTYPE],
    spin_pol: bool = False,
    use_fillings: bool = False,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE | COMPLEX_DTYPE]]:
    """Return a generic integrated spectrum.

    Return a generic integrated spectrum.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.
    weights_tj : np.ndarray[REAL_DTYPE]
        The weights of the spectrum of interest.
    spin_pol : bool
        Return separate spectra for up/down intensities if True.
    """
    _weights_tj = weights_tj.copy()
    weights_sabcj = _weights_tj.reshape([edata.nspin] + list(edata.kfolding) + [edata.nbands])
    if use_fillings:
        occ_sabcj = edata.occ_tj.reshape([edata.nspin] + list(edata.kfolding) + [edata.nbands])
        if edata.legacy_fillings:
            # Pre-F4 behaviour: rescale by the largest occupation (hid the x2 of no-spin fillings files)
            occ_sabcj = occ_sabcj / np.max(edata.occ_tj)
        weights_sabcj *= occ_sabcj
    es = None
    if spin_pol:
        cs = []
        for s in range(edata.nspin):
            w_sabcj = weights_sabcj.copy()
            for s2 in range(edata.nspin):
                if s2 != s:
                    w_sabcj[s2, :, :, :, :] *= 0
            es, _cs = get_uneven_integrated_array(edata.e_sabcj, w_sabcj)
            cs.append(_cs)
        cs = np.array(cs)
    else:
        es, cs = get_uneven_integrated_array(edata.e_sabcj, weights_sabcj)
    return es, cs



def evaluate_or_retrieve_generic_spectrum(
    edata: ElecData,
    weights_tj: np.ndarray[REAL_DTYPE],
    func_name: str,
    func_args_dict: dict,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    spin_pol: bool = False,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
):
    # Resolved once, so the cache path, the spectrum and the returned erange share one grid (finding F17)
    erange = _resolve_erange(edata, erange, res, sig, lti)
    spectrum_file_path = get_spectrum_file_path(
        edata,
        func_name,
        func_args_dict,
        erange=erange,
        spin_pol=spin_pol,
        sig=sig,
        res=res,
        lti=lti,
        rattle_eigenvals=rattle_eigenvals,
        norm_max=norm_max,
        norm_intg=norm_intg,
        sep_channels=sep_channels,
    )
    if spectrum_file_path.is_file() and use_cached_spectrum:
        # print(f"Using cached spectrum at {spectrum_file_path}")
        spectrum = np.load(spectrum_file_path)
        if not lti:
            _warn_if_grid_too_coarse(erange, sig)
    else:
        erange, spectrum = get_generic_spectrum(
            edata,
            weights_tj,
            erange=erange,
            spin_pol=spin_pol,
            sig=sig,
            res=res,
            lti=lti,
            rattle_eigenvals=rattle_eigenvals,
            norm_max=norm_max,
            norm_intg=norm_intg,
            sep_channels=sep_channels,
        )
        if save_spectrum:
            np.save(spectrum_file_path, spectrum)
    return erange, spectrum


def get_generic_spectrum(
    edata: ElecData,
    weights_tj: np.ndarray[REAL_DTYPE],
    erange: np.ndarray[REAL_DTYPE] | None = None,
    spin_pol: bool = False,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Return a generic spectrum.

    Return a generic spectrum.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.
    weights_tj : np.ndarray[REAL_DTYPE]
        The weights of the spectrum of interest.
    erange : np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    spin_pol : bool
        Return separate spectra for up/down intensities if True.
    sig : REAL_DTYPE
        Width parameter of the Gaussian kernel exp(-(E - e)^2 / sig), i.e. 2 * (standard deviation)^2. Not used
        with lti. The default is a standard deviation of 2.2e-3 Ha. A grid step above 2 standard deviations warns,
        since peak areas on the grid then depend on where each eigenvalue falls (finding F17).
    res : REAL_DTYPE
        The resolution of the energy range if erange is None (default 1e-3 Ha; it was 1e-2 Ha before finding F17,
        and res=0.01 still gives that grid). The automatic range is padded by max(10 res, 5 standard deviations)
        beyond the extreme eigenvalues, or 10 res with lti.
    lti : bool
        Use the linear tetrahedron integration method.
    rattle_eigenvals : bool
        Randomly perturb the eigenvalues up to magnitude of twice the energy array resolution.
        (Needed for fully localized bands to appear in LTI spectra.)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    sep_channels : bool
        Return separate spectra for positive/negative intensities if True.
    """
    erange = _resolve_erange(edata, erange, res, sig, lti)
    if not lti:
        _warn_if_grid_too_coarse(erange, sig)
    weights_sabcj = weights_tj.reshape([edata.nspin] + list(edata.kfolding) + [edata.nbands])
    if not sep_channels:
        spectrum = _get_generic_spectrum_helper(edata, weights_sabcj, erange, spin_pol, sig, res, lti, rattle_eigenvals, norm_max, norm_intg)
    else:
        # only positive weights (else 0)
        weights_sabcj_pos = np.maximum(weights_sabcj, np.zeros_like(weights_sabcj))
        weights_sabcj_neg = -np.maximum(-weights_sabcj, np.zeros_like(weights_sabcj))
        spectrum1 = _get_generic_spectrum_helper(edata, weights_sabcj_pos, erange, spin_pol, sig, res, lti, rattle_eigenvals, norm_max, norm_intg)
        spectrum2 = _get_generic_spectrum_helper(edata, weights_sabcj_neg, erange, spin_pol, sig, res, lti, rattle_eigenvals, norm_max, norm_intg)
        spectrum = np.array([spectrum1, spectrum2])
    return erange, spectrum

def _get_generic_spectrum_helper(edata, weights_sabcj, erange, spin_pol, sig, res, lti, rattle_eigenvals, norm_max, norm_intg):
    if not lti:
        erange, spectrum = get_generic_gsmear_spectrum(edata, weights_sabcj, erange, spin_pol, sig, res=res)
    elif not edata.lti_allowed:
        raise ValueError("LTI is not allowed for this ElecData object due to incomplete k-point data.")
    else:
        erange, spectrum = get_generic_lti_spectrum(
            edata, weights_sabcj, erange, spin_pol, res=res, rattle_eigenvals=rattle_eigenvals
        )
    if norm_max:
        spectrum = spectrum / np.max(np.abs(spectrum))
    elif norm_intg:
        spectrum = spectrum / trapezoid(np.abs(spectrum), erange)
    return spectrum


def get_generic_lti_spectrum(
    edata: ElecData,
    weights_sabcj: np.ndarray[REAL_DTYPE],
    erange: np.ndarray[REAL_DTYPE] | None,
    spin_pol: bool,
    res: REAL_DTYPE = RES_DEFAULT,
    rattle_eigenvals: bool = False,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Return a generic linear-tetrahedron integrated spectrum.

    Return a generic linear-tetrahedron integrated spectrum.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.
    weights_sabcj : np.ndarray[REAL_DTYPE]
        The weights of the spectrum of interest.
    erange : np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    spin_pol : bool
        Return separate spectra for up/down intensities if True.
    res : REAL_DTYPE
        The resolution of the energy range if erange is None (padded by 10 res). The default was 1e-5 Ha before
        finding F17; get_generic_spectrum always passed its own res and erange, so that only affected direct calls.
    rattle_eigenvals : bool
        Randomly perturb the eigenvalues up to magnitude of twice the energy array resolution.
        (Needed for fully localized bands to appear in LTI spectra.)
    """
    erange = _resolve_erange(edata, erange, res, None, lti=True)
    res = erange[1] - erange[0]
    bvec = edata.structure.lattice.reciprocal_lattice.matrix
    # vbz = abs(numpy.linalg.det(bvec))
    if rattle_eigenvals:
        e_sabcj = np.array(
            deepcopy(edata.e_sabcj) + (np.random.rand(*edata.e_sabcj.shape) - 0.5) * res * 2,
            dtype=REAL_DTYPE,
        )
    else:
        e_sabcj = edata.e_sabcj
    if not edata.legacy_normalization:
        # libtetrabz weights already carry the 1/N_k of the BZ average, and weights_sabcj carry w_t (= spin
        # weight / N_k on the full mesh LTI needs), so N_k is counted twice without this (F6)
        weights_sabcj = weights_sabcj * weights_sabcj.dtype.type(np.prod(edata.kfolding))
    cs = get_lti_spectrum(e_sabcj, erange, weights_sabcj, bvec)
    spectrum = cs_formatter(cs, spin_pol)
    return erange, spectrum


def get_generic_gsmear_spectrum(
    edata: ElecData,
    weights_sabcj: np.ndarray[REAL_DTYPE],
    erange: np.ndarray[REAL_DTYPE] | None,
    spin_pol: bool,
    sig: REAL_DTYPE,
    res: REAL_DTYPE = RES_DEFAULT,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Return a generic Gaussian-smeared spectrum.

    Return a generic Gaussian-smeared spectrum.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.
    weights_sabcj : np.ndarray[REAL_DTYPE]
        The weights of the spectrum of interest.
    erange : np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    spin_pol : bool
        Return separate spectra for up/down intensities if True.
    sig : REAL_DTYPE
        Width parameter of the Gaussian kernel exp(-(E - e)^2 / sig), i.e. 2 * (standard deviation)^2.
    res : REAL_DTYPE
        The resolution of the energy range if erange is None (padded by max(10 res, 5 standard deviations)).
    """
    erange = _resolve_erange(edata, erange, res, sig, lti=False)
    cs = get_gauss_smear_spectrum(erange, edata.e_sabcj, weights_sabcj, sig, normalize=not edata.legacy_normalization)
    spectrum = cs_formatter(cs, spin_pol)
    return erange, spectrum

def get_spectrum_file_path(
    edata: ElecData,
    func_name: str,
    func_args_dict: dict,
    erange: np.ndarray[REAL_DTYPE] | None = None,
    spin_pol: bool = False,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
):
    # The cache is keyed by the grid (erange_* directory) and these settings (metadata)
    erange = _resolve_erange(edata, erange, res, sig, lti)
    erange_cache_dir = get_erange_cache_dir(edata.cache_sub_dir, erange)
    metadata_dict = func_args_dict.copy()
    metadata_dict["spin_pol"] = spin_pol
    metadata_dict["sig"] = sig
    metadata_dict["lti"] = lti
    metadata_dict["rattle_eigenvals"] = rattle_eigenvals
    metadata_dict["norm_max"] = norm_max
    metadata_dict["norm_intg"] = norm_intg
    metadata_dict["sep_channels"] = sep_channels
    spec_cache_dir = get_generic_cache_dir(
        erange_cache_dir,
        func_name,
        metadata_dict,
    )
    file_path = spec_cache_dir / "spec.npy"
    return file_path