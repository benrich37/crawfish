"""Module for scalar operations for the crawfish package.

Module for common methods yielding scalars.

"""

from __future__ import annotations
import numpy as np
from crawfish.utils.typing import check_arr_typing
from numba import jit


@jit(nopython=True)
def gauss(x: np.float32, mu: np.float32, sig: np.float32) -> np.float32:
    """Return the Gaussian function evaluated at a given point.

    Return the Gaussian function evaluated at a given point.

    Parameters
    ----------
    x : np.float32
        The point at which to evaluate the Gaussian
    mu : np.float32
        The mean of the Gaussian
    sig : np.float32
        The standard deviation of the Gaussian
    """
    return np.exp(-((x - mu) ** 2) / sig)


def integrate(occ_sabcj: np.ndarray, weights_sabcj: np.ndarray, wk_sabc: np.ndarray) -> np.float32 | np.complex64:
    """Integrate the product of the weights and the k-point weights and occupations.

    Integrate the product of the weights and the k-point weights and occupations.

    Parameters
    ----------
    occ_sabcj : np.ndarray
        The set of occupations for the system of interest
    weights_sabcj : np.ndarray
        The set of weights for the occupations of interest
    wk_sabc : np.ndarray
        The set of k-point weights for the system of interest
    """
    check_arr_typing([occ_sabcj, weights_sabcj, wk_sabc])
    num_dtype = weights_sabcj.dtype
    shape = np.shape(occ_sabcj)
    nspin = shape[0]
    nka = shape[1]
    nkb = shape[2]
    nkc = shape[3]
    nbands = shape[4]
    integrand = num_dtype(0.0)
    wweights_sabcj = occ_sabcj * weights_sabcj
    integrand = _integrate_jit(wweights_sabcj, wk_sabc, nspin, nka, nkb, nkc, nbands, integrand)
    return integrand


@jit(nopython=True)
def _integrate_jit(
    weights_sabcj: np.ndarray[np.float32] | np.ndarray[np.complex64],
    wk_sabc: np.ndarray[np.float32],
    nspin: int,
    nka: int,
    nkb: int,
    nkc: int,
    nbands: int,
    integrand: np.float32 | np.complex64,
) -> np.float32 | np.complex64:
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        integrand += weights_sabcj[s, a, b, c, j] * wk_sabc[s, a, b, c]
    return integrand
