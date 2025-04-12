r"""Module for common matrix operations.

Module for common methods yielding matrices/tensors
r"""

from __future__ import annotations
import numpy as np
from numba import jit
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE


def get_s_tj_uu(proj_tju: np.ndarray[COMPLEX_DTYPE]) -> np.ndarray[REAL_DTYPE | COMPLEX_DTYPE]:
    r"""Get the real part of the overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*).

    Get the real part of the overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*).

    Parameters
    ----------
    proj_tju : np.ndarray
        The projection tensor T_{t,j,u} = <\phi_{t,j} | u>
    r"""
    t, j, u = np.shape(proj_tju)
    s_tj_uu = np.zeros([t, j, u, u], dtype=COMPLEX_DTYPE)
    s_tj_uu = get_overlap_tjuv(proj_tju)
    s_tj_uu = np.abs(s_tj_uu) ** 2
    s_tj_uu = np.array(np.real(s_tj_uu), dtype=REAL_DTYPE)
    return s_tj_uu


def get_p_tj_uu(
    s_tj_uu: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    sys_consistent: bool = True,
) -> np.ndarray[REAL_DTYPE]:
    r"""Get the projected density of states tensor P_{t,j,u,u} = Sum_{u} |T_{t,j,u}|^2 w_{t}.

    Get the projected density of states tensor P_{t,j,u,u} = Sum_{u} |T_{t,j,u}|^2 w_{t}.
    Creates with restriction of electron conservation (sum_{u,v}P_{t,j,u,v} = occ_{t,j}*wk_{t}).

    Parameters
    ----------
    s_tj_uu : np.ndarray
        The overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*)
    occ_tj : np.ndarray
        The occupation tensor occ_{t,j} = <\phi_{t,j} | \phi_{t,j}>
    wk_t : np.ndarray
        The k-point weights w_{t}
    r"""
    t, j, u, _ = np.shape(s_tj_uu)
    p_tj_uu = np.zeros([t, j, u, u], dtype=REAL_DTYPE)
    if sys_consistent:
        p_tj_uu = get_p_tj_uu_sys_consistent(p_tj_uu, occ_tj, wk_t, s_tj_uu)
    else:
        p_tj_uu = get_p_tj_uu_direct(p_tj_uu, occ_tj, wk_t, s_tj_uu)
    return p_tj_uu

def get_p_tj_uu_sys_consistent(
    p_tj_uu: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    s_tj_uu: np.ndarray[REAL_DTYPE],
    ):
    _t, _j, _u, _ = np.shape(p_tj_uu)
    for t in range(_t):
        for j in range(_j):
            p_tj_uu[t, j, :, :] = np.abs(s_tj_uu[t, j, :, :]) * (
                occ_tj[t, j] * wk_t[t] / np.sum(np.abs(s_tj_uu[t, j, :, :]), axis=(0, 1))
            )
    return p_tj_uu


def get_p_tj_uu_direct(
    p_tj_uu: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    s_tj_uu: np.ndarray[REAL_DTYPE],
    ):
    p_tj_uu += (occ_tj[:, :, np.newaxis, np.newaxis] * wk_t[:, np.newaxis, np.newaxis, np.newaxis]) * np.abs(
            s_tj_uu
        )
    return p_tj_uu



def get_h_tj_uu(p_tj_uu: np.ndarray[REAL_DTYPE], e_tj: np.ndarray[REAL_DTYPE]) -> np.ndarray[REAL_DTYPE]:
    r"""Get the Hamiltonian tensor H_{t,j,u,u} = P_{t,j,u,u} * E_{t,j}.

    Get the Hamiltonian tensor H_{t,j,u,u} = P_{t,j,u,u} * E_{t,j}.
    Assumes P_{t,j,u,u} is created with electron count preserved for each state/band (j/t).

    Parameters
    ----------
    p_tj_uu : np.ndarray
        The projected density of states tensor P_{t,j,u,u} = Sum_{u} |T_{t,j,u}|^2 w_{t}
    e_tj : np.ndarray
        The planewave eigenvalue tensor E_{t,j} = <\phi_{t,j} | \hat{H} | \phi_{t,j}>
    r"""
    t, j, u, _ = np.shape(p_tj_uu)
    h_tj_uu = np.zeros([t, j, u, u], dtype=REAL_DTYPE)
    h_tj_uu = get_h_tj_uu_helper(h_tj_uu, p_tj_uu, e_tj)
    return h_tj_uu

@jit(nopython=True)
def get_h_tj_uu_helper(h_tj_uu, p_tj_uu, e_tj):
    h_tj_uu += p_tj_uu
    _t, _j = np.shape(e_tj)
    for t in range(_t):
        for j in range(_j):
            h_tj_uu[t, j, :, :] *= e_tj[t, j]
    return h_tj_uu


def get_h_tj_uu_p_tj_uu_s_tj_uu(
    proj_tju: np.ndarray[COMPLEX_DTYPE],
    e_tj: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    p_sc: bool = True,
    norm_eigs_for_h: bool = True,
    mu: float | None = None
):
    r"""Get the Hamiltonian, Population, and Overlap band/state tensors.

    Parameters
    ----------
    proj_tju : np.ndarray
        The projection tensor T_{t,j,u} = <\phi_{t,j} | u>
    e_tj : np.ndarray
        The planewave eigenvalue tensor E_{t,j} = <\phi_{t,j} | \hat{H} | \phi_{t,j}>
    occ_tj : np.ndarray
        The occupation tensor occ_{t,j} = <\phi_{t,j} | \phi_{t,j}>
    wk_t : np.ndarray
        The k-point weights w_{t}
    r"""
    s_tj_uu = get_s_tj_uu(proj_tju)
    p_tj_uu = get_p_tj_uu(s_tj_uu, occ_tj, wk_t, sys_consistent=p_sc)
    if norm_eigs_for_h:
        h_tj_uu = get_h_tj_uu(p_tj_uu, e_tj - mu)
    else:
        h_tj_uu = get_h_tj_uu(p_tj_uu, e_tj)
    return h_tj_uu, p_tj_uu, s_tj_uu

def get_h_t_uu_p_t_uu_s_t_uu(
    proj_tju: np.ndarray[COMPLEX_DTYPE],
    e_tj: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    p_sc: bool = True,
    norm_eigs_for_h: bool = True,
    mu: float | None = None
):
    r"""Get the Hamiltonian, Population, and Overlap state tensors.

    Parameters
    ----------
    proj_tju : np.ndarray
        The projection tensor T_{t,j,u} = <\phi_{t,j} | u>
    e_tj : np.ndarray
        The planewave eigenvalue tensor E_{t,j} = <\phi_{t,j} | \hat{H} | \phi_{t,j}>
    occ_tj : np.ndarray
        The occupation tensor occ_{t,j} = <\phi_{t,j} | \phi_{t,j}>
    wk_t : np.ndarray
        The k-point weights w_{t}
    r"""
    h_tj_uu, p_tj_uu, s_tj_uu = get_h_tj_uu_p_tj_uu_s_tj_uu(
        proj_tju, e_tj, occ_tj, wk_t, p_sc=p_sc, mu=mu, norm_eigs_for_h=norm_eigs_for_h
    )
    h_t_uu = np.sum(h_tj_uu, axis=1)
    p_t_uu = np.sum(p_tj_uu, axis=1)
    s_t_uu = np.sum(s_tj_uu, axis=1)
    return h_t_uu, p_t_uu, s_t_uu

def get_h_uu_p_uu_s_uu(
    proj_tju: np.ndarray[COMPLEX_DTYPE],
    e_tj: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    p_sc: bool = True,
    norm_eigs_for_h: bool = True,
    mu: float | None = None
):
    r"""Get the Hamiltonian, Population, and Overlap matrices.

    Get the Hamiltonian tensor H_{u,u} = Sum_{t,j} P_{t,j,u,u} * E_{t,j}.
    Assumes P_{t,j,u,u} is created with electron count preserved for each state/band (j/t).

    Parameters
    ----------
    proj_tju : np.ndarray
        The projection tensor T_{t,j,u} = <\phi_{t,j} | u>
    e_tj : np.ndarray
        The planewave eigenvalue tensor E_{t,j} = <\phi_{t,j} | \hat{H} | \phi_{t,j}>
    occ_tj : np.ndarray
        The occupation tensor occ_{t,j} = <\phi_{t,j} | \phi_{t,j}>
    wk_t : np.ndarray
        The k-point weights w_{t}
    r"""
    h_tj_uu, p_tj_uu, s_tj_uu = get_h_tj_uu_p_tj_uu_s_tj_uu(
        proj_tju, e_tj, occ_tj, wk_t, p_sc=p_sc, norm_eigs_for_h=norm_eigs_for_h, mu=mu
    )
    h_uu = np.sum(h_tj_uu, axis=(0, 1))
    p_uu = np.sum(p_tj_uu, axis=(0, 1))
    s_uu = np.sum(s_tj_uu, axis=(0, 1))
    return h_uu, p_uu, s_uu


def _get_gen_tj_coef_uu(
    weighted_overlaps_tj_uv: np.ndarray[COMPLEX_DTYPE],
    coef_uu: np.ndarray[REAL_DTYPE],
    orbs_u: list[int],
    orbs_v: list[int],
) -> np.ndarray:
    t, j, u, _ = np.shape(weighted_overlaps_tj_uv)
    gen_tj = np.zeros([t, j], dtype=REAL_DTYPE)
    return np.array(np.real(_get_gen_tj_coef_uu_jit(gen_tj, coef_uu, weighted_overlaps_tj_uv, orbs_u, orbs_v)), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_gen_tj_coef_uu_jit(gen_tj, coef_uu, weighted_overlaps_tj_uv, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            if not u == v:
                gen_tj += coef_uu[u, v] * weighted_overlaps_tj_uv[:, :, u, v]
    return gen_tj


def get_weighted_overlap_tjuv(proj_tju, wk_t):
    t, j, u = np.shape(proj_tju)
    wk_cw_tj_uu = np.zeros([t, j, u, u], dtype=COMPLEX_DTYPE)
    return np.array(np.real(_get_weighted_overlap_tjuv_jit(wk_cw_tj_uu, proj_tju, wk_t)), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_weighted_overlap_tjuv_jit(wk_cw_tj_uv, proj_tju, wk_t):
    _t, _j, _u = np.shape(proj_tju)
    for t in range(_t):
        for j in range(_j):
            for u in range(_u):
                for v in range(u, _u):
                    val = np.conj(proj_tju[t, j, u]) * proj_tju[t, j, v] * wk_t[t]
                    wk_cw_tj_uv[t, j, u, v] = val
                    wk_cw_tj_uv[t, j, v, u] = np.conj(val)
    return wk_cw_tj_uv

def get_overlap_tjuv(proj_tju):
    t, j, u = np.shape(proj_tju)
    wk_cw_tj_uu = np.zeros([t, j, u, u], dtype=COMPLEX_DTYPE)
    return np.array(np.real(_get_overlap_tjuv_jit(wk_cw_tj_uu, proj_tju)), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_overlap_tjuv_jit(wk_cw_tj_uv, proj_tju):
    _t, _j, _u = np.shape(proj_tju)
    for t in range(_t):
        for j in range(_j):
            for u in range(_u):
                for v in range(u, _u):
                    val = np.conj(proj_tju[t, j, u]) * proj_tju[t, j, v]
                    wk_cw_tj_uv[t, j, u, v] = val
                    wk_cw_tj_uv[t, j, v, u] = np.conj(val)
    return wk_cw_tj_uv



def _get_gen_tj_coef_tuu(
    weighted_overlaps_tj_uv: np.ndarray[COMPLEX_DTYPE],
    coef_tuu: np.ndarray[REAL_DTYPE],
    orbs_u: list[int],
    orbs_v: list[int],
) -> np.ndarray:
    t, j, u, _ = np.shape(weighted_overlaps_tj_uv)
    gen_tj = np.zeros([t, j], dtype=REAL_DTYPE)
    return np.array(np.real(_get_gen_tj_coef_tuu_jit(gen_tj, coef_tuu, weighted_overlaps_tj_uv, orbs_u, orbs_v)), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_gen_tj_coef_tuu_jit(gen_tj, coef_tuu, wk_cw_tj_uu, orbs_u, orbs_v):
    trange = list(range(np.shape(gen_tj)[0]))
    for u in orbs_u:
        for v in orbs_v:
            if not u == v:
                for t in trange:
                    val = coef_tuu[t, u, v] * wk_cw_tj_uu[t, :, u, v]
                    gen_tj[t,:] += val
    return gen_tj

def get_pdos_tj(
    proj_tju: np.ndarray[COMPLEX_DTYPE], orbs: list[int], wk_t: np.ndarray[REAL_DTYPE]
) -> np.ndarray[REAL_DTYPE]:
    r"""Return the projected density of states tensor PDOS_{s,a,b,c,j} = Sum_{u} |P_{u}^{j,s,a,b,c}|^2 w_{s,a,b,c}.

    Return the projected density of states tensor PDOS_{s,a,b,c,j} = Sum_{u} |P_{u}^{j,s,a,b,c}|^2 w_{s,a,b,c}.
    where u encompasses indices for orbitals of interest.

    Parameters
    ----------
    proj_tju : np.ndarray
        The projection vector T_{u}^{j,s,a,b,c} = <u | \phi_{j,s,a,b,c}>
    orbs : list[int]
        The list of orbitals to evaluate
    r"""
    t, j, u = np.shape(proj_tju)
    pdos_tj = np.zeros([t, j], dtype=REAL_DTYPE)
    for orb in orbs:
        pdos_tj += np.abs(proj_tju[:, :, orb]) ** 2
    pdos_tj *= wk_t[:, np.newaxis]
    return pdos_tj


def mod_weights_for_ebounds(
    weights_sabcj: np.ndarray[REAL_DTYPE], e_sabcj: np.ndarray[REAL_DTYPE], ebounds: list[REAL_DTYPE]
) -> np.ndarray:
    r"""Modify the weights array for the energy bounds of interest.

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
    r"""
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
    r"""Return a boolean array for the energy bounds of interest.

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
    r"""
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
    r"""Add k-point weights to the weights array.

    Add k-point weights to the weights array. The weights array is multiplied by the k-point weights.

    Parameters
    ----------
    weights_sabcj : np.ndarray
        The set of weights for the energies of interest
    wk_sabc : np.ndarray
        The set of k-point weights for the system of interest
    r"""
    wk_sabcj = np.moveaxis(np.array([wk_sabc] * np.shape(weights_sabcj)[-1]), 0, -1)
    return weights_sabcj * wk_sabcj


def los_projs_for_orbs(proj_tju: np.ndarray[COMPLEX_DTYPE]) -> np.ndarray[COMPLEX_DTYPE]:
    """Perform LOS on projections for projection orthogonality.

    Perform Lowdin symmetric orthogonalization on projections for orbital projection orthogonality.

    Parameters
    ----------
    proj_sabcju : np.ndarray[COMPLEX_DTYPE]
        Projections in shape (nstates, nbands, nproj).
    """
    low_proj_tju = np.zeros_like(proj_tju)
    nstates = np.shape(proj_tju)[0]
    for t in range(nstates):
        s_uu = np.tensordot(proj_tju[t].conj().T, proj_tju[t], axes=([1], [0]))
        eigs, low_u = np.linalg.eigh(s_uu)
        # TODO: The following two lines were stolen from pyscf. Figure out if pyscf should just be a dependency,
        # and dispatch orthogonalization to it.
        idx = eigs > 1e-15
        low_s_uu = np.dot(low_u[:,idx]/np.sqrt(eigs[idx]), low_u[:,idx].conj().T)
        low_proj_tju[t, :, :] += np.tensordot(proj_tju[t], low_s_uu, axes=([1], [0]))
    return low_proj_tju


def los_projs_for_bands(proj_tju: np.ndarray[COMPLEX_DTYPE]) -> np.ndarray[COMPLEX_DTYPE]:
    """Perform LOS on projections for band orthogonality.

    Perform Lowdin symmetric orthogonalization on projections for band orthogonality.

    Parameters
    ----------
    proj_sabcju : np.ndarray[COMPLEX_DTYPE]
        Projections in shape (nstates, nbands, nproj).
    """
    low_proj_tju = np.zeros_like(proj_tju)
    nstates = np.shape(proj_tju)[0]
    for t in range(nstates):
        s_uu = np.tensordot(proj_tju[t].conj().T, proj_tju[t], axes=([0], [1]))
        eigs, low_u = np.linalg.eigh(s_uu)
        nsqrt_ss_uu = np.eye(len(eigs)) * (eigs ** (-0.5))
        low_s_uu = np.dot(low_u, np.dot(nsqrt_ss_uu, low_u.T.conj()))
        low_proj_tju[t, :, :] += np.tensordot(proj_tju[t], low_s_uu, axes=([0], [1]))
    return low_proj_tju


# def get_t1_loss(proj_tju, nStates):
#     v = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=-1), axis=0) - nStates)
#     return v


# def get_t2_loss(proj_tju, nStates):
#     v = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=0), axis=1) - nStates)
#     return v


# def get_t3_loss(proj_tju, nStates):
#     v2 = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=0), axis=1) - nStates)
#     v1 = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=-1), axis=0) - nStates)
#     return abs(v1) + abs(v2)


def normalize_square_proj_tju(proj_tju: np.ndarray[COMPLEX_DTYPE], nloops=1000, conv=0.01):
    """Normalize projection matrices proj_tju.

    Normalize projection matrices proj_tju. Requires nproj == nbands. Performs the following:
    1. For each state t:
        1.a. For each band j:
            1.a.i. Sums proj_tju[t,j,u]^*proj_tju[t,j,u] for a given j and all u to "asum"
            1.a.ii. Divides proj_tju[t,j,:] by 1/(asum**0.5)
            1.b.iii. Adds |1-asum| to loss metric for state t
        1.b. For each projection u:
            1.b.i. Sums proj_tju[t,:,u]^*proj_tju[t,:,u] for a given u and all j to "asum"
            1.b.ii. Divides proj_tju[t,:,u] by 1/(asum**0.5)
            1.b.iii. Adds |1-asum| to loss metric for state t
        1.c. If loss metric for state t exceeds the "conv" threshold and 1.a/1.b have been
                performed for less than nloops, reset the loss metric and repeat 1.a/1.b for state t.
                Otherwise, move to the next state.
    2. Return the normalized proj_tju and the losses at each loop for each state.

    Parameters
    ----------
    proj_tju : np.ndarray[COMPLEX_DTYPE]
        Projection matrices in shape (nstates, nbands, nproj).
    nloops : int, optional
        Maximum number of loops to perform normalization, by default 1000.
    conv : float, optional
        Convergence threshold for loss metric, by default 0.01.
    """
    proj_tju_norm = np.array(proj_tju.copy(), dtype=COMPLEX_DTYPE)
    nproj: np.int64 = np.shape(proj_tju)[2]
    nstates: np.int64 = np.shape(proj_tju)[0]
    nbands: np.int64 = np.shape(proj_tju)[1]
    losses = np.zeros([nstates, nloops], dtype=REAL_DTYPE)
    return _normalize_square_proj_tju(proj_tju_norm, nloops, conv, losses, nstates, nproj, nbands)[0]


@jit(nopython=True)
def _normalize_square_proj_tju(
    proj_tju_norm: np.ndarray[COMPLEX_DTYPE],
    nloops: int,
    conv: REAL_DTYPE,
    losses: np.ndarray[REAL_DTYPE],
    nstates: int,
    nproj: int,
    nbands,
) -> tuple[np.ndarray[COMPLEX_DTYPE], np.ndarray[REAL_DTYPE]]:
    asum: REAL_DTYPE = 0
    for t in range(nstates):
        for i in range(nloops):
            for j in range(nbands):
                asum *= 0
                for u in range(nproj):
                    asum += np.real(np.conj(proj_tju_norm[t, j, u]) * proj_tju_norm[t, j, u])
                proj_tju_norm[t, j, :] *= 1 / (asum**0.5)
                losses[t, i] += np.abs(1 - asum)
            for u in range(nproj):
                asum *= 0
                for j in range(nbands):
                    asum += np.real(np.conj(proj_tju_norm[t, j, u]) * proj_tju_norm[t, j, u])
                proj_tju_norm[t, :, u] *= 1 / (asum**0.5)
                losses[t, i] += np.abs(1 - asum)
            if losses[t, i] < conv:
                break
    return proj_tju_norm, losses


@jit(nopython=True)
def _norm_projs_for_bands_jit_helper_1(nProj, nStates, nBands, proj_tju, j_sums):
    for u in range(nProj):
        for t in range(nStates):
            for j in range(nBands):
                j_sums[j] += abs(np.conj(proj_tju[t, j, u]) * proj_tju[t, j, u])
    for j in range(nBands):
        proj_tju[:, j, :] *= 1 / np.sqrt(j_sums[j])
    # proj_tju *= np.sqrt(nStates)
    return proj_tju


@jit(nopython=True)
def _mute_xs_bands(nProj, nBands, proj_tju):
    for j in range(nProj, nBands):
        proj_tju[:, j, :] *= 0
    return proj_tju


def _norm_projs_for_bands(proj_tju, nStates, nBands, nProj, restrict_band_norm_to_nproj=False):
    j_sums = np.zeros(nBands)
    proj_tju = _norm_projs_for_bands_jit_helper_1(nProj, nStates, nBands, proj_tju, j_sums)
    if restrict_band_norm_to_nproj:
        proj_tju = _mute_xs_bands(nProj, nBands, proj_tju)
    return proj_tju


@jit(nopython=True)
def _norm_projs_for_orbs_jit_helper(nProj, nStates, nBands, proj_tju, u_sums):
    for u in range(nProj):
        for t in range(nStates):
            for j in range(nBands):
                u_sums[u] += abs(np.conj(proj_tju[t, j, u]) * proj_tju[t, j, u])
    for u in range(nProj):
        proj_tju[:, :, u] *= 1 / np.sqrt(u_sums[u])
    # proj_tju *= np.sqrt(nStates)
    # proj_tju *= np.sqrt(2)
    # proj_tju *= np.sqrt(nStates*nBands/nProj)
    return proj_tju


def _norm_projs_for_orbs(proj_tju, nStates, nBands, nProj, mute_excess_bands=False):
    if mute_excess_bands:
        proj_tju = _mute_xs_bands(nProj, nBands, proj_tju)
    u_sums = np.zeros(nProj)
    # TODO: Identify the error types raised by division by zero within a jitted function
    # (if certain orbitals are only represented by bands above nProj, this error should
    # be clarified to the user)
    proj_tju = _norm_projs_for_orbs_jit_helper(nProj, nStates, nBands, proj_tju, u_sums)
    return proj_tju
