"""Module for general methods to work alongside spectrum-getting methods.

Module for general methods to work alongside spectrum-getting methods.
"""

from crawfish.core.elecdata import ElecData
from crawfish.core.operations.vector import get_gauss_smear_spectrum, get_lti_spectrum, get_uneven_integrated_array
from crawfish.utils.arg_correction import get_erange
from crawfish.utils.typing import REAL_DTYPE, cs_formatter, COMPLEX_DTYPE
from copy import deepcopy
import numpy as np
from scipy.integrate import trapezoid

SIGMA_DEFAULT = REAL_DTYPE(0.00001)
RES_DEFAULT = REAL_DTYPE(0.01)


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
        weights_sabcj *= edata.occ_tj.reshape([edata.nspin] + list(edata.kfolding) + [edata.nbands]) / np.max(edata.occ_tj)
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






def get_erange_cache_dir(edata: ElecData, erange: np.ndarray):
    """Return the energy range cache directory.

    Return the energy range cache directory.
    """
    emin = erange[0]
    emax = erange[-1]
    estep = erange[1] - erange[0]
    gen_cache_dir = edata.cache_sub_dir
    erange_cache_dir = gen_cache_dir / f"erange_{emin}_{emax}_{estep}"
    erange_cache_dir.mkdir(parents=True, exist_ok=True)
    return erange_cache_dir


def get_spectrum_file_name_func_args_prefix(func_args_dict: dict) -> str:
    """Return the spectrum file name function arguments prefix.

    Return the spectrum file name function arguments prefix. Gives portion
    of file name that encodes the function arguments.
    """
    # Sort the keys by alphabetical order
    sorted_keys = sorted(func_args_dict.keys())
    sorted_vals = [str(func_args_dict[key]) for key in sorted_keys]
    # Join the sorted keys and values into a string
    func_args_str = "_".join([f"{key}_{val}" for key, val in zip(sorted_keys, sorted_vals)])
    # Remove any whitespace
    func_args_str = func_args_str.replace(" ", "")
    return func_args_str

def get_spectrum_file_name_spectrum_args_suffix(
    spin_pol: bool = False,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
):
    """Return the spectrum file name spectrum arguments suffix.

    Return the spectrum file name spectrum arguments suffix. Gives portion
    of file name that encodes the spectrum evaluation arguments.
    """
    p1 = f"lti_{lti}"
    if lti:
        p1 += f"_rattle_{rattle_eigenvals}"
    else:
        p1 += f"_sig_{sig}"
    p2s = [
        f"norm_max_{norm_max}",
        f"norm_intg_{norm_intg}",
        f"sep_channels_{sep_channels}",
        f"spin_pol_{spin_pol}",
    ]
    p2 = "_".join([p for p in p2s])
    return f"{p1}_{p2}"

        
    

def get_spectrum_file_name(
    func_args_dict: dict,
    spin_pol: bool = False,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    ) -> str:
    """Return the spectrum file name.

    Return the spectrum file name.
    """
    prefix = get_spectrum_file_name_func_args_prefix(func_args_dict)
    suffix = get_spectrum_file_name_spectrum_args_suffix(
        spin_pol=spin_pol,
        sig=sig,
        lti=lti,
        rattle_eigenvals=rattle_eigenvals,
        norm_max=norm_max,
        norm_intg=norm_intg,
        sep_channels=sep_channels,
    )
    return f"{prefix}_{suffix}.npy"

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
    if erange is None:
        erange = get_erange(edata, erange, res=res)
    erange_cache_dir = get_erange_cache_dir(edata, erange)
    func_cache_dir = erange_cache_dir / func_name
    func_cache_dir.mkdir(parents=True, exist_ok=True)
    file_name = get_spectrum_file_name(
        func_args_dict,
        spin_pol=spin_pol,
        sig=sig,
        lti=lti,
        rattle_eigenvals=rattle_eigenvals,
        norm_max=norm_max,
        norm_intg=norm_intg,
        sep_channels=sep_channels,
    )
    file_path = func_cache_dir / file_name
    return file_path

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
        spectrum = np.load(spectrum_file_path)
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
    if erange is None:
        erange = get_erange(edata, erange, res=res)
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
        The standard deviation of the Gaussian/Lorentzian-Tetrahedron smearing.
    res : REAL_DTYPE
        The resolution of the energy range if erange is None.
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
    erange = get_erange(edata, erange, res=res)
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
        spectrum = spectrum / np.max(spectrum)
    elif norm_intg:
        spectrum = spectrum / trapezoid(spectrum, erange)
    return spectrum


def get_generic_lti_spectrum(
    edata: ElecData,
    weights_sabcj: np.ndarray[REAL_DTYPE],
    erange: np.ndarray[REAL_DTYPE] | None,
    spin_pol: bool,
    res: REAL_DTYPE = REAL_DTYPE(0.00001),
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
    sig : REAL_DTYPE
        The standard deviation of the Lorentzian-Tetrahedron smearing.
    res : REAL_DTYPE
        The resolution of the energy range if erange is None.
    rattle_eigenvals : bool
        Randomly perturb the eigenvalues up to magnitude of twice the energy array resolution.
        (Needed for fully localized bands to appear in LTI spectra.)
    """
    erange = get_erange(edata, erange, res=res)
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
        The standard deviation of the Gaussian smearing.
    res : REAL_DTYPE
        The resolution of the energy range if erange is None.
    """
    erange = get_erange(edata, erange, res=res)
    cs = get_gauss_smear_spectrum(erange, edata.e_sabcj, weights_sabcj, sig)
    spectrum = cs_formatter(cs, spin_pol)
    return erange, spectrum
