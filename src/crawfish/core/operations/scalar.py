"""Module for scalar operations for the crawfish package.

Module for common methods yielding scalars.

"""

from __future__ import annotations
import numpy as np
from crawfish.utils.typing import check_arr_typing
from numba import jit
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE


@jit(nopython=True)
def gauss(x: REAL_DTYPE, mu: REAL_DTYPE, sig: REAL_DTYPE) -> REAL_DTYPE:
    """Return the Gaussian function evaluated at a given point.

    Return the Gaussian function evaluated at a given point.

    Parameters
    ----------
    x : REAL_DTYPE
        The point at which to evaluate the Gaussian
    mu : REAL_DTYPE
        The mean of the Gaussian
    sig : REAL_DTYPE
        The standard deviation of the Gaussian
    """
    return np.exp(-((x - mu) ** 2) / sig)


def integrate(occ_sabcj: np.ndarray, weights_sabcj: np.ndarray, wk_sabc: np.ndarray) -> REAL_DTYPE | COMPLEX_DTYPE:
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
    weights_sabcj: np.ndarray[REAL_DTYPE] | np.ndarray[COMPLEX_DTYPE],
    wk_sabc: np.ndarray[REAL_DTYPE],
    nspin: int,
    nka: int,
    nkb: int,
    nkc: int,
    nbands: int,
    integrand: REAL_DTYPE | COMPLEX_DTYPE,
) -> REAL_DTYPE | COMPLEX_DTYPE:
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        integrand += weights_sabcj[s, a, b, c, j] * wk_sabc[s, a, b, c]
    return integrand
