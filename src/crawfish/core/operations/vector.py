"""Module for common vector operations.

Module for common methods yielding vectors.
"""

from __future__ import annotations
import numpy as np
from crawfish.core.operations.scalar import gauss
from crawfish.utils.typing import check_arr_typing
from numba import jit
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
import libtetrabz


def get_lti_spectrum(
    e_sabcj: np.ndarray[REAL_DTYPE],
    erange: np.ndarray[REAL_DTYPE],
    weights_sabcj: np.ndarray[REAL_DTYPE],
    bvec: np.ndarray[REAL_DTYPE],
) -> list[np.ndarray[REAL_DTYPE]]:
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
    bvec : np.ndarray
        Reciprocal lattice of the system of interest
    """
    check_arr_typing([e_sabcj, erange, weights_sabcj])
    cs = []
    nspin = np.shape(e_sabcj)[0]
    for s in range(nspin):
        wght = libtetrabz.dos(bvec, e_sabcj[s], erange)
        c = np.zeros(np.shape(erange), dtype=weights_sabcj.dtype)
        cs.append(get_lti_spectrum_jit(erange, weights_sabcj[s], wght, c))
    return cs


@jit(nopython=True)
def get_lti_spectrum_jit(
    erange: np.ndarray[REAL_DTYPE],
    weights_abcj: np.ndarray[REAL_DTYPE],
    wght: np.ndarray[REAL_DTYPE],
    c: np.ndarray[REAL_DTYPE],
) -> list[np.ndarray[REAL_DTYPE]]:
    """Return the linear tetrahedron integration spectrum for a given set of energies, weights, and lattice.

    Return the linear tetrahedron integration spectrum for a given set of energies, weights, and lattice.

    Parameters
    ----------
    e_sabcj : np.ndarray
        The set of energies for the system of interest
    erange : np.ndarray
        The energy range to evaluate the spectrum
    weights_abcj : np.ndarray
        The set of weights for the energies of interest
    bvec : np.ndarray
        Reciprocal lattice of the system of interest
    """
    for i, e in enumerate(erange):
        c[i] = (wght[:, :, :, :, i] * weights_abcj[:, :, :, :]).sum()
    return c


def get_gauss_smear_spectrum(
    erange: np.ndarray[REAL_DTYPE],
    e_sabcj: np.ndarray[REAL_DTYPE],
    weights_sabcj: np.ndarray[REAL_DTYPE] | np.ndarray[[COMPLEX_DTYPE]],
    sig: REAL_DTYPE,
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
    erange: np.ndarray[REAL_DTYPE],
    eflat: np.ndarray[REAL_DTYPE],
    wflat: np.ndarray[REAL_DTYPE] | np.ndarray[[COMPLEX_DTYPE]],
    cflat: np.ndarray[REAL_DTYPE] | np.ndarray[[COMPLEX_DTYPE]],
    sig: REAL_DTYPE,
) -> np.ndarray:
    for i in range(len(eflat)):
        cflat += gauss(erange, eflat[i], sig) * wflat[i]
    return cflat


def get_uneven_integrated_array(
    e_sabcj: np.ndarray[REAL_DTYPE], weights_sabcj: np.ndarray[REAL_DTYPE] | np.ndarray[[COMPLEX_DTYPE]]
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE] | np.ndarray[COMPLEX_DTYPE]]:
    """Integrate the unevenly spaced array of energies and weights.

    Integrate the unevenly spaced array of energies and weights. Used for efficient representation integrated
    dos-like quantities without interpolation.

    Parameters
    ----------
    e_sabcj : np.ndarray
        The set of eigenvalues for the system of interest
    weights_sabcj : np.ndarray
        The weights at the eigenvalues

    Returns
    -------
    e : np.ndarray
        The eigenvalues sorted in ascending order
    integrated : np.ndarray
        The integrated weights at the eigenvalues
    """
    check_arr_typing([e_sabcj, weights_sabcj])
    e_flat = e_sabcj.flatten()
    idcs = np.argsort(e_flat)
    e = e_flat[idcs]
    pw = weights_sabcj.flatten()[idcs]
    _integrated = np.zeros(len(pw) + 1)
    _integrated[1:] = np.cumsum(pw)
    integrated = _integrated[1:]
    return e, integrated
