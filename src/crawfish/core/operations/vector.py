"""Module for common vector operations.

Module for common methods yielding vectors.
"""

from __future__ import annotations
from ase.dft.dos import linear_tetrahedron_integration as lti
import numpy as np
from crawfish.core.operations.scalar import gauss
from crawfish.utils.typing import check_arr_typing
from numba import jit


def get_lti_spectrum(
    e_sabcj: np.ndarray[np.float32],
    erange: np.ndarray[np.float32],
    weights_sabcj: np.ndarray[np.float32],
    lattice: np.ndarray[np.float32],
) -> list[np.ndarray[np.float32]]:
    """Return the linear tetrahedron integration spectrum for a given set of energies, weights, and lattice.

    Return the linear tetrahedron integration spectrum for a given set of energies, weights, and lattice.

    Parameters
    ----------
    e_sabcj : np.ndarray
        The set of energies for the system of interest
    erange : np.ndarray
        The energy range to evaluate the spectrum
    weights_sabcj : np.ndarray
        The set of weights for the energies of interest
    lattice : np.ndarray
        The lattice of the system of interest
    """
    check_arr_typing([e_sabcj, erange, weights_sabcj, lattice])
    cs = []
    nspin = np.shape(e_sabcj)[0]
    for s in range(nspin):
        cs.append(lti(lattice, e_sabcj[s], erange, weights=weights_sabcj[s]))
    return cs


def get_gauss_smear_spectrum(
    erange: np.ndarray[np.float32],
    e_sabcj: np.ndarray[np.float32],
    weights_sabcj: np.ndarray[np.float32] | np.ndarray[[np.complex64]],
    sig: np.float32,
) -> list[np.ndarray]:
    """Return the Gaussian smeared spectrum for a given set of energies, weights, and sigma smearing parameter.

    Return the Gaussian smeared spectrum for a given set of energies, weights, and sigma smearing parameter.
    (larger sigma means more smearing)

    Parameters
    ----------
    erange : np.ndarray
        The energy range to evaluate the spectrum
    e_sabcj : np.ndarray
        The set of energies for the system of interest
    weights_sabcj : np.ndarray
        The set of weights for the energies of interest
    sig : float
        The sigma smearing parameter
    """
    check_arr_typing([erange, e_sabcj, weights_sabcj])
    num_dtype = weights_sabcj.dtype
    nspin = np.shape(e_sabcj)[0]
    ws = []
    es = []
    cs = []
    for s in range(nspin):
        ws.append(weights_sabcj[s].flatten())
        es.append(e_sabcj[s].flatten())
        cs.append(np.zeros(np.shape(erange), dtype=num_dtype))
        cs[s] = _get_gauss_smear_spectrum_jit(erange, es[s], ws[s], cs[s], sig)
    return cs


@jit(nopython=True)
def _get_gauss_smear_spectrum_jit(
    erange: np.ndarray[np.float32],
    eflat: np.ndarray[np.float32],
    wflat: np.ndarray[np.float32] | np.ndarray[[np.complex64]],
    cflat: np.ndarray[np.float32] | np.ndarray[[np.complex64]],
    sig: np.float32,
) -> np.ndarray:
    for i in range(len(eflat)):
        cflat += gauss(erange, eflat[i], sig) * wflat[i]
    return cflat


def get_uneven_integrated_array(
    e_sabcj: np.ndarray[np.float32],
    weights_sabcj: np.ndarray[np.float32] | np.ndarray[[np.complex64]],
    wk_sabcj: np.ndarray[np.float32],
    occ_sabcj: np.ndarray[np.float32],
) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32] | np.ndarray[np.complex64]]:
    """Integrate the unevenly spaced array of energies and weights.

    Integrate the unevenly spaced array of energies and weights. Used for efficient representation integrated
    dos-like quantities without interpolation.

    Parameters
    ----------
    e_sabcj : np.ndarray
        The set of eigenvalues for the system of interest
    weights_sabcj : np.ndarray
        The weights at the eigenvalues
    wk_sabcj : np.ndarray
        The set of k-point weights at the eigenvalues
    occ_sabcj : np.ndarray
        The fillings at the eigenvalues

    Returns
    -------
    e : np.ndarray
        The eigenvalues sorted in ascending order
    integrated : np.ndarray
        The integrated weights at the eigenvalues
    """
    check_arr_typing([e_sabcj, weights_sabcj, wk_sabcj, occ_sabcj])
    e_flat = e_sabcj.flatten()
    idcs = np.argsort(e_flat)
    e = e_flat[idcs]
    pw = weights_sabcj.flatten()[idcs]
    kw = wk_sabcj.flatten()[idcs]
    occ = occ_sabcj.flatten()[idcs]
    rawvals = pw * kw * occ
    _integrated = np.zeros(len(rawvals) + 1)
    _integrated[1:] = np.cumsum(rawvals)
    integrated = _integrated[1:]
    return e, integrated
