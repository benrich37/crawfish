"""Module for general methods to work alongside spectrum-getting methods.

Module for general methods to work alongside spectrum-getting methods.
"""

from crawfish.core.elecdata import ElecData
from crawfish.core.operations.vector import get_gauss_smear_spectrum, get_lti_spectrum
from crawfish.core.operations.matrix import _add_kweights
from crawfish.utils.arg_correction import get_erange
from crawfish.utils.typing import REAL_DTYPE, cs_formatter
from copy import deepcopy
import numpy as np
from scipy.integrate import trapezoid

SIGMA_DEFAULT = REAL_DTYPE(0.00001)
RES_DEFAULT = REAL_DTYPE(0.01)


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
    """
    weights_sabcj = weights_tj.reshape([edata.nspin] + list(edata.kfolding) + [edata.nbands])
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
    return erange, spectrum


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
    weights_tj: np.ndarray[REAL_DTYPE],
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
    weights_tj : np.ndarray[REAL_DTYPE]
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
