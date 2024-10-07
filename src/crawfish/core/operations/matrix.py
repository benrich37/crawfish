"""Module for common matrix operations.

Module for common methods yielding matrices/tensors
"""

from __future__ import annotations
import numpy as np
from numba import jit
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE


def get_p_uvjsabc(
    proj_sabcju: np.ndarray[REAL_DTYPE], orbs_u: list[int] | None = None, orbs_v: list[int] | None = None
) -> np.ndarray[REAL_DTYPE]:
    r"""Return the projection matrix P_{uv}^{j,s,a,b,c} = <\phi_{j,s,a,b,c}^j | u><v | \phi_{j,s,a,b,c}>.

    Return the projection matrix P_{uv}^{j,s,a,b,c} = <\phi_{j,s,a,b,c}^j | u><v | \phi_{j,s,a,b,c}>.
    Evaluated at P_uv as T_u^* T_v, where T is the bandprojections vector.
    Parameters
    ----------
    proj_sabcju : np.ndarray
        The projection matrix T_{a,b,c,j,u} = <\phi_{s,a,b,c,j}^j | u>
    orbs_u : list[int] | None
        The list of orbitals to evaluate for species 1
    orbs_v : list[int] | None
        The list of orbitals to evaluate for species 2
    """
    shape = np.shape(proj_sabcju)
    nspin = shape[0]
    nka = shape[1]
    nkb = shape[2]
    nkc = shape[3]
    nbands = shape[4]
    nproj = shape[5]
    if orbs_u is None:
        orbs_u = list(range(nproj))
    if orbs_v is None:
        orbs_v = list(range(nproj))
    p_uvjsabc = np.zeros([nproj, nproj, nbands, nspin, nka, nkb, nkc], dtype=REAL_DTYPE)
    _orbs_u = np.asarray(orbs_u)
    _orbs_v = np.asarray(orbs_v)
    p_uvjsabc = _get_p_uvjsabc_jit(proj_sabcju, p_uvjsabc, nproj, nbands, nka, nkb, nkc, nspin, _orbs_u, _orbs_v)
    return np.real(p_uvjsabc)


@jit(nopython=True)
def _get_p_uvjsabc_jit(
    proj_sabcju: np.ndarray[COMPLEX_DTYPE],
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    nproj: int,
    nbands: int,
    nka: int,
    nkb: int,
    nkc: int,
    nspin: int,
    orbs_u: np.ndarray[int],
    orbs_v: np.ndarray[int],
):
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nbands):
                for a in range(nka):
                    for b in range(nkb):
                        for c in range(nkc):
                            for s in range(nspin):
                                t1 = proj_sabcju[s, a, b, c, j, u]
                                t2 = proj_sabcju[s, a, b, c, j, v]
                                p_uvjsabc[u, v, j, s, a, b, c] += np.real(np.conj(t1) * t2)
    return p_uvjsabc


@jit(nopython=True)
def _get_h_uvsabc_jit(
    h_uvsabc: np.ndarray[REAL_DTYPE],
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    e_sabcj: np.ndarray[REAL_DTYPE],
    nproj: int,
    nbands: int,
    nka: int,
    nkb: int,
    nkc: int,
    nspin: int,
    orbs_u: np.ndarray[int],
    orbs_v: np.ndarray[int],
) -> np.ndarray:
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nbands):
                for s in range(nspin):
                    for a in range(nka):
                        for b in range(nkb):
                            for c in range(nkc):
                                h_uvsabc[u, v, s, a, b, c] += p_uvjsabc[u, v, j, s, a, b, c] * e_sabcj[s, a, b, c, j]
    return h_uvsabc


def get_h_uvsabc(
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    e_sabcj: np.ndarray[REAL_DTYPE],
    orbs_u: list[int] | None = None,
    orbs_v: list[int] | None = None,
) -> np.ndarray[REAL_DTYPE]:
    r"""Get the Hamiltonian matrix H_{u,v}^{s,a,b,c} = Sum_j <\phi_{s,a,b,c,j} | u><u | \hat{H} | v><v | \phi_{s,a,b,c,j}>.

    Get the Hamiltonian matrix  H_{u,v}^{s,a,b,c} = Sum_j <\phi_{s,a,b,c,j} | u><u | \hat{H} | v><v | \phi_{s,a,b,c,j}>.
    Evaluated at H_uv as T_u^* E T_v, where T is the bandprojections vector, and E is the band energies.

    Parameters
    ----------
    p_uvjsabc : np.ndarray
        The projection matrix P_{u,v}^{j,s,a,b,c} = <\phi_{j,s,a,b,c} | u><v | \phi_{j,s,a,b,c}>
    e_sabcj : np.ndarray
        The band energies E_{s,a,b,c,j} = <\phi_{s,a,b,c,j} | \hat{H} | \phi_{s,a,b,c,j}>
    orbs_u : list[int] | None
        The list of orbitals to evaluate for species 1
    orbs_v : list[int] | None
        The list of orbitals to evaluate for species 2
    """
    shape = np.shape(p_uvjsabc)
    nproj = shape[0]
    nbands = shape[2]
    nspin = shape[3]
    nka = shape[4]
    nkb = shape[5]
    nkc = shape[6]
    h_uvsabc = np.zeros([nproj, nproj, nspin, nka, nkb, nkc], dtype=REAL_DTYPE)
    if orbs_u is None:
        orbs_u = list(range(nproj))
    if orbs_v is None:
        orbs_v = list(range(nproj))
    _orbs_u = np.asarray(orbs_u)
    _orbs_v = np.asarray(orbs_v)
    return _get_h_uvsabc_jit(h_uvsabc, p_uvjsabc, e_sabcj, nproj, nbands, nka, nkb, nkc, nspin, _orbs_u, _orbs_v)


@jit(nopython=True)
def _get_pcohp_sabcj_jit(
    nspin: int,
    nka: int,
    nkb: int,
    nkc: int,
    nbands: int,
    orbs_u: np.ndarray[int],
    orbs_v: np.ndarray[int],
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    h_uvsabc: np.ndarray[REAL_DTYPE],
    pcohp_sabcj: np.ndarray[REAL_DTYPE],
) -> np.ndarray:
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                p1 = p_uvjsabc[u, v, j, s, a, b, c]
                                p2 = h_uvsabc[u, v, s, a, b, c]
                                uv_sum += p1 * p2
                        pcohp_sabcj[s, a, b, c, j] += uv_sum
    return pcohp_sabcj


def get_pcohp_sabcj(
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    h_uvsabc: np.ndarray[REAL_DTYPE],
    orbs_u: list[int],
    orbs_v: list[int],
) -> np.ndarray:
    r"""Return the pCOHP tensor pCOHP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} H_{u,v}^{s,a,b,c} w_{s,a,b,c}.

    Return the pCOHP tensor pCOHP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} H_{u,v}^{s,a,b,c} w_{s,a,b,c}.
    where u encompasses indices for orbitals of interest for species 1, and v for species 2.

    Parameters
    ----------
    p_uvjsabc : np.ndarray
        The projection matrix P_{u,v}^{j,s,a,b,c} = <\phi_{j,s,a,b,c} | u><v | \phi_{j,s,a,b,c}>
    h_uvsabc : np.ndarray
        The Hamiltonian matrix H_{u,v}^{s,a,b,c} = Sum_j <\phi_{s,a,b,c,j} | u><u | \hat{H} | v><v | \phi_{s,a,b,c,j}>
    orbs_u : list[int]
        The list of orbitals to evaluate for species 1
    orbs_v : list[int]
        The list of orbitals to evaluate for species 2
    """
    shape = np.shape(p_uvjsabc)
    nbands = shape[2]
    nspin = shape[3]
    nka = shape[4]
    nkb = shape[5]
    nkc = shape[6]
    pcohp_sabcj = np.zeros([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE)
    _orbs_u = np.asarray(orbs_u)
    _orbs_v = np.asarray(orbs_v)
    return _get_pcohp_sabcj_jit(nspin, nka, nkb, nkc, nbands, _orbs_u, _orbs_v, p_uvjsabc, h_uvsabc, pcohp_sabcj)


@jit(nopython=True)
def _get_pcoop_sabcj_jit(
    nspin: int,
    nka: int,
    nkb: int,
    nkc: int,
    nbands: int,
    orbs_u: np.ndarray[int],
    orbs_v: np.ndarray[int],
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    pcoop_sabcj: np.ndarray[REAL_DTYPE],
) -> np.ndarray:
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                uv_sum += p_uvjsabc[u, v, j, s, a, b, c]
                        pcoop_sabcj[s, a, b, c, j] += uv_sum
    return pcoop_sabcj


def get_pcoop_sabcj(
    p_uvjsabc: np.ndarray[REAL_DTYPE],
    orbs_u: list[int],
    orbs_v: list[int],
) -> np.ndarray[REAL_DTYPE]:
    r"""Return the pCOOP tensor pCOOP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} w_{s,a,b,c}.

    Return the pCOOP tensor pCOOP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} w_{s,a,b,c}.
    where u encompasses indices for orbitals of interest for species 1, and v for species 2.

    Parameters
    ----------
    p_uvjsabc : np.ndarray
        The projection matrix P_{u,v}^{j,s,a,b,c} = <\phi_{j,s,a,b,c} | u><v | \phi_{j,s,a,b,c}>
    orbs_u : list[int]
        The list of orbitals to evaluate for species 1
    orbs_v : list[int]
        The list of orbitals to evaluate for species 2
    """
    shape = np.shape(p_uvjsabc)
    nbands = shape[2]
    nspin = shape[3]
    nka = shape[4]
    nkb = shape[5]
    nkc = shape[6]
    pcoop_sabcj = np.zeros([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE)
    _orbs_u = np.asarray(orbs_u)
    _orbs_v = np.asarray(orbs_v)
    return _get_pcoop_sabcj_jit(nspin, nka, nkb, nkc, nbands, _orbs_u, _orbs_v, p_uvjsabc, pcoop_sabcj)


def get_pdos_sabcj(proj_sabcju: np.ndarray[COMPLEX_DTYPE], orbs: list[int]) -> np.ndarray[REAL_DTYPE]:
    r"""Return the projected density of states tensor PDOS_{s,a,b,c,j} = Sum_{u} |P_{u}^{j,s,a,b,c}|^2 w_{s,a,b,c}.

    Return the projected density of states tensor PDOS_{s,a,b,c,j} = Sum_{u} |P_{u}^{j,s,a,b,c}|^2 w_{s,a,b,c}.
    where u encompasses indices for orbitals of interest.

    Parameters
    ----------
    proj_sabcju : np.ndarray
        The projection vector T_{u}^{j,s,a,b,c} = <u | \phi_{j,s,a,b,c}>
    orbs : list[int]
        The list of orbitals to evaluate
    """
    pdos_sabcj = np.zeros(np.shape(proj_sabcju)[:5], dtype=REAL_DTYPE)
    for orb in orbs:
        pdos_sabcj += np.abs(proj_sabcju[:, :, :, :, :, orb]) ** 2
    return pdos_sabcj


def mod_weights_for_ebounds(
    weights_sabcj: np.ndarray[REAL_DTYPE], e_sabcj: np.ndarray[REAL_DTYPE], ebounds: list[REAL_DTYPE]
) -> np.ndarray:
    """Modify the weights array for the energy bounds of interest.

    Modify the weights array for the energy bounds of interest. Entries within the weights array
    are set to zero if the corresponding energy is not within the bounds.

    All arrays must be real (REAL_DTYPE).

    Parameters
    ----------
    weights_sabcj : np.ndarray
        The set of weights for the energies of interest
    e_sabcj : np.ndarray
        The set of energies for the system of interest
    ebounds : list[float]
        The energy bounds of interest
    """
    shape = np.shape(weights_sabcj)
    bool_arr = get_ebound_arr(ebounds, e_sabcj)
    weights_sabcj = _mod_weights_for_ebounds_jit(weights_sabcj, shape, bool_arr)
    return weights_sabcj


@jit(nopython=True)
def _mod_weights_for_ebounds_jit(
    _weights_sabcj: np.ndarray[REAL_DTYPE],
    sabcj_shape: list[int] | tuple[int, int, int, int, int],
    bool_arr: np.ndarray[bool],
) -> np.ndarray:
    for s in range(sabcj_shape[0]):
        for a in range(sabcj_shape[1]):
            for b in range(sabcj_shape[2]):
                for c in range(sabcj_shape[3]):
                    for j in range(sabcj_shape[4]):
                        if not bool_arr[s, a, b, c, j]:
                            _weights_sabcj[s, a, b, c, j] *= 0
    return _weights_sabcj


def get_ebound_arr(ebounds: list[REAL_DTYPE], arr: np.ndarray[REAL_DTYPE]) -> np.ndarray[bool]:
    """Return a boolean array for the energy bounds of interest.

    Return a boolean array for the energy bounds of interest. Entries within the boolean
    array are True if the corresponding argument entry is within the bounds, and False otherwise.

    Ebounds must even length (such that the boundaries are paired).
    Ebounds and arr must be real (REAL_DTYPE)

    Parameters
    ----------
    ebounds : list[float]
        The energy bounds of interest
    arr : np.ndarray
        The array to evaluate
    """
    if not len(ebounds) % 2 == 0:
        raise ValueError("The ebounds list must have an even number of elements.")
    nlows = ebounds[::2]
    nhighs = ebounds[1::2]

    vec = arr.flatten()
    min_low = np.min(nlows)
    max_high = np.max(nhighs)

    bools = np.zeros(len(vec), dtype=int)
    valid_mask = (vec >= min_low) & (vec <= max_high)

    valid_nums = vec[valid_mask]

    if valid_nums.size > 0:
        lb_indices = _get_lb_idx_vec(valid_nums, nlows)
        ub_indices = _get_ub_idx_vec(valid_nums, nhighs)
        within_ebound = lb_indices == ub_indices
        bools[valid_mask] = within_ebound.astype(int)

    bool_arr = bools.reshape(arr.shape)
    return bool_arr


def _get_lb_idx(num, lb_list):
    idcs = np.argsort(lb_list)
    for i, idx in enumerate(idcs):
        if lb_list[idx] > num:
            return idcs[i - 1]
    return None


def _get_ub_idx(num, ub_list):
    idcs = np.argsort(ub_list)[::-1]
    for i, idx in enumerate(idcs):
        if ub_list[idx] < num:
            return idcs[i - 1]
    return None


def _get_lb_idx_vec(nums, lb_list):
    lb_array = np.array(lb_list)
    sorted_indices = np.argsort(lb_array)
    sorted_lb = lb_array[sorted_indices]
    idxs = np.searchsorted(sorted_lb, nums, side="right") - 1
    return sorted_indices[idxs]


def _get_ub_idx_vec(nums, ub_list):
    ub_array = np.array(ub_list)
    sorted_indices = np.argsort(ub_array)
    sorted_ub = ub_array[sorted_indices]
    idxs = np.searchsorted(sorted_ub, nums, side="right")
    return sorted_indices[idxs]


def _add_kweights(weights_sabcj: np.ndarray[REAL_DTYPE], wk_sabc: np.ndarray[REAL_DTYPE]) -> np.ndarray[REAL_DTYPE]:
    """Add k-point weights to the weights array.

    Add k-point weights to the weights array. The weights array is multiplied by the k-point weights.

    Parameters
    ----------
    weights_sabcj : np.ndarray
        The set of weights for the energies of interest
    wk_sabc : np.ndarray
        The set of k-point weights for the system of interest
    """
    wk_sabcj = np.moveaxis(np.array([wk_sabc] * np.shape(weights_sabcj)[-1]), 0, -1)
    return weights_sabcj * wk_sabcj
