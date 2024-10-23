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


def get_real_s_tj_uu(proj_tju: np.ndarray[COMPLEX_DTYPE]):
    """ Get the real part of the overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*).
    
    Get the real part of the overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*).
    
    Parameters
    ----------
    proj_tju : np.ndarray
        The projection tensor T_{t,j,u} = <\phi_{t,j} | u>
    """
    t, j, u = np.shape(proj_tju)
    s_tj_uu = np.zeros([t, j, u, u], dtype=REAL_DTYPE)
    s_tj_uu += 0.5 * (
        np.conj(proj_tju[:, :, :, np.newaxis]) * proj_tju[:, :, np.newaxis, :] + 
        np.conj(np.conj(proj_tju[:, :, :, np.newaxis]) * proj_tju[:, :, np.newaxis, :])
    )
    return s_tj_uu

def get_p_tj_uu(s_tj_uu: np.ndarray[REAL_DTYPE], occ_tj: np.ndarray[REAL_DTYPE], wk_t: np.ndarray[REAL_DTYPE]):
    """ Get the projected density of states tensor P_{t,j,u,u} = Sum_{u} |T_{t,j,u}|^2 w_{t}.

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
    """
    t, j, u, _ = np.shape(s_tj_uu)
    p_tj_uu = np.zeros([t, j, u, u], dtype=REAL_DTYPE)
    p_tj_uu += (
        occ_tj[:, :, np.newaxis, np.newaxis] * wk_t[:, np.newaxis, np.newaxis, np.newaxis] / np.sum(np.abs(s_tj_uu),
                                                                                                    axis=(2, 3),
                                                                                                    keepdims=True)
        ) * np.abs(s_tj_uu)
    return p_tj_uu

def get_h_tj_uu(p_tj_uu: np.ndarray[REAL_DTYPE], e_tj: np.ndarray[REAL_DTYPE]):
    """ Get the Hamiltonian tensor H_{t,j,u,u} = P_{t,j,u,u} * E_{t,j}.

    Get the Hamiltonian tensor H_{t,j,u,u} = P_{t,j,u,u} * E_{t,j}.
    Assumes P_{t,j,u,u} is created with electron count preserved for each state/band (j/t).

    Parameters
    ----------
    p_tj_uu : np.ndarray
        The projected density of states tensor P_{t,j,u,u} = Sum_{u} |T_{t,j,u}|^2 w_{t}
    e_tj : np.ndarray
        The planewave eigenvalue tensor E_{t,j} = <\phi_{t,j} | \hat{H} | \phi_{t,j}>
    """
    t, j, u, _ = np.shape(p_tj_uu)
    h_tj_uu = np.zeros([t, j, u, u], dtype=REAL_DTYPE)
    h_tj_uu += p_tj_uu * e_tj[:, :, np.newaxis, np.newaxis]
    return h_tj_uu

def get_h_uu(proj_tju: np.ndarray[COMPLEX_DTYPE], e_tj: np.ndarray[REAL_DTYPE], occ_tj: np.ndarray[REAL_DTYPE], wk_t: np.ndarray[REAL_DTYPE]):
    """ Get the Hamiltonian tensor H_{u,u} = Sum_{t,j} P_{t,j,u,u} * E_{t,j}.

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
    """
    s_tj_uu = get_real_s_tj_uu(proj_tju)
    p_tj_uu = get_p_tj_uu(s_tj_uu, occ_tj, wk_t)
    h_tj_uu = get_h_tj_uu(p_tj_uu, e_tj)
    h_uu = np.sum(h_tj_uu, axis=(0, 1))
    return h_uu






    



# def get_s_uu(low_proj_sabcju):
#     s_uu = np.tensordot(low_proj_sabcju.conj().T, low_proj_sabcju, axes=([5, 4, 3, 2, 1], [0, 1, 2, 3, 4]))
#     return s_uu


# def get_h_uu(proj_sabcju, e_sabcj):
#     h_uu = np.tensordot(
#         proj_sabcju.conj().T,
#         np.tensordot(e_sabcj.T, proj_sabcju, axes=([3, 2, 1, 0], [4, 3, 2, 5])),
#         axes=([5, 4, 3, 2, 0], [0, 1, 2, 3, 5]),
#     )
#     return h_uu


# def los_projs_for_bands(proj_sabcju):
#     s_jj = np.tensordot(proj_sabcju.conj().T, proj_sabcju, axes=([5, 4, 3, 2, 0], [0, 1, 2, 3, 5]))
#     eigs, low_u = np.linalg.eigh(s_jj)
#     nsqrt_ss_jj = np.eye(len(eigs)) * (eigs ** (-0.5))
#     low_s_jj = np.dot(low_u, np.dot(nsqrt_ss_jj, low_u.T.conj()))
#     low_proj_sabcju = np.tensordot(proj_sabcju, low_s_jj, axes=([4], [0]))
#     low_proj_sabcju = np.swapaxes(low_proj_sabcju, 5, 4)
#     return low_proj_sabcju


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
