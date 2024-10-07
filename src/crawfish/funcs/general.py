"""Module for general methods to work alongside spectrum-getting methods.

Module for general methods to work alongside spectrum-getting methods.
"""

from crawfish.core.elecdata import ElecData
from crawfish.core.operations.vector import get_gauss_smear_spectrum
from crawfish.utils.arg_correction import get_erange
from crawfish.utils.typing import REAL_DTYPE, cs_formatter
import numpy as np


def get_generic_gsmear_spectrum(
    edata: ElecData,
    weights_sabcj: np.ndarray[REAL_DTYPE],
    erange: np.ndarray[REAL_DTYPE] | None,
    spin_pol: bool,
    sig: REAL_DTYPE,
    res: REAL_DTYPE = REAL_DTYPE(0.01),
) -> np.ndarray[REAL_DTYPE]:
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
