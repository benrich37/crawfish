r"""Module for common matrix operations.

Module for common methods yielding matrices/tensors
r"""

from __future__ import annotations
import numpy as np
from numba import jit
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE


# def get_p_uvjsabc(
#     proj_sabcju: np.ndarray[REAL_DTYPE], orbs_u: list[int] | None = None, orbs_v: list[int] | None = None
# ) -> np.ndarray[REAL_DTYPE]:
#     r"""Return the projection matrix P_{uv}^{j,s,a,b,c} = <\phi_{j,s,a,b,c}^j | u><v | \phi_{j,s,a,b,c}>.

#     Return the projection matrix P_{uv}^{j,s,a,b,c} = <\phi_{j,s,a,b,c}^j | u><v | \phi_{j,s,a,b,c}>.
#     Evaluated at P_uv as T_u^* T_v, where T is the bandprojections vector.
#     Parameters
#     ----------
#     proj_sabcju : np.ndarray
#         The projection matrix T_{a,b,c,j,u} = <\phi_{s,a,b,c,j}^j | u>
#     orbs_u : list[int] | None
#         The list of orbitals to evaluate for species 1
#     orbs_v : list[int] | None
#         The list of orbitals to evaluate for species 2
#     r"""
#     shape = np.shape(proj_sabcju)
#     nspin = shape[0]
#     nka = shape[1]
#     nkb = shape[2]
#     nkc = shape[3]
#     nbands = shape[4]
#     nproj = shape[5]
#     if orbs_u is None:
#         orbs_u = list(range(nproj))
#     if orbs_v is None:
#         orbs_v = list(range(nproj))
#     p_uvjsabc = np.zeros([nproj, nproj, nbands, nspin, nka, nkb, nkc], dtype=REAL_DTYPE)
#     _orbs_u = np.asarray(orbs_u)
#     _orbs_v = np.asarray(orbs_v)
#     p_uvjsabc = _get_p_uvjsabc_jit(proj_sabcju, p_uvjsabc, nproj, nbands, nka, nkb, nkc, nspin, _orbs_u, _orbs_v)
#     return np.real(p_uvjsabc)


# @jit(nopython=True)
# def _get_p_uvjsabc_jit(
#     proj_sabcju: np.ndarray[COMPLEX_DTYPE],
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     nproj: int,
#     nbands: int,
#     nka: int,
#     nkb: int,
#     nkc: int,
#     nspin: int,
#     orbs_u: np.ndarray[int],
#     orbs_v: np.ndarray[int],
# ):
#     for u in orbs_u:
#         for v in orbs_v:
#             for j in range(nbands):
#                 for a in range(nka):
#                     for b in range(nkb):
#                         for c in range(nkc):
#                             for s in range(nspin):
#                                 t1 = proj_sabcju[s, a, b, c, j, u]
#                                 t2 = proj_sabcju[s, a, b, c, j, v]
#                                 p_uvjsabc[u, v, j, s, a, b, c] += np.real(np.conj(t1) * t2)
#     return p_uvjsabc


# @jit(nopython=True)
# def _get_h_uvsabc_jit(
#     h_uvsabc: np.ndarray[REAL_DTYPE],
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     e_sabcj: np.ndarray[REAL_DTYPE],
#     nproj: int,
#     nbands: int,
#     nka: int,
#     nkb: int,
#     nkc: int,
#     nspin: int,
#     orbs_u: np.ndarray[int],
#     orbs_v: np.ndarray[int],
# ) -> np.ndarray:
#     for u in orbs_u:
#         for v in orbs_v:
#             for j in range(nbands):
#                 for s in range(nspin):
#                     for a in range(nka):
#                         for b in range(nkb):
#                             for c in range(nkc):
#                                 h_uvsabc[u, v, s, a, b, c] += p_uvjsabc[u, v, j, s, a, b, c] * e_sabcj[s, a, b, c, j]
#     return h_uvsabc


# def get_h_uvsabc(
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     e_sabcj: np.ndarray[REAL_DTYPE],
#     orbs_u: list[int] | None = None,
#     orbs_v: list[int] | None = None,
# ) -> np.ndarray[REAL_DTYPE]:
#     r"""Get the Hamiltonian matrix H_{u,v}^{s,a,b,c} = Sum_j <\phi_{s,a,b,c,j} | u><u | \hat{H} | v><v | \phi_{s,a,b,c,j}>.

#     Get the Hamiltonian matrix  H_{u,v}^{s,a,b,c} = Sum_j <\phi_{s,a,b,c,j} | u><u | \hat{H} | v><v | \phi_{s,a,b,c,j}>.
#     Evaluated at H_uv as T_u^* E T_v, where T is the bandprojections vector, and E is the band energies.

#     Parameters
#     ----------
#     p_uvjsabc : np.ndarray
#         The projection matrix P_{u,v}^{j,s,a,b,c} = <\phi_{j,s,a,b,c} | u><v | \phi_{j,s,a,b,c}>
#     e_sabcj : np.ndarray
#         The band energies E_{s,a,b,c,j} = <\phi_{s,a,b,c,j} | \hat{H} | \phi_{s,a,b,c,j}>
#     orbs_u : list[int] | None
#         The list of orbitals to evaluate for species 1
#     orbs_v : list[int] | None
#         The list of orbitals to evaluate for species 2
#     r"""
#     shape = np.shape(p_uvjsabc)
#     nproj = shape[0]
#     nbands = shape[2]
#     nspin = shape[3]
#     nka = shape[4]
#     nkb = shape[5]
#     nkc = shape[6]
#     h_uvsabc = np.zeros([nproj, nproj, nspin, nka, nkb, nkc], dtype=REAL_DTYPE)
#     if orbs_u is None:
#         orbs_u = list(range(nproj))
#     if orbs_v is None:
#         orbs_v = list(range(nproj))
#     _orbs_u = np.asarray(orbs_u)
#     _orbs_v = np.asarray(orbs_v)
#     return _get_h_uvsabc_jit(h_uvsabc, p_uvjsabc, e_sabcj, nproj, nbands, nka, nkb, nkc, nspin, _orbs_u, _orbs_v)


def get_s_tj_uu(proj_tju: np.ndarray[COMPLEX_DTYPE], pos=True, real=True) -> np.ndarray[REAL_DTYPE | COMPLEX_DTYPE]:
    r"""Get the real part of the overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*).

    Get the real part of the overlap tensor S_{t,j,u,u} = 0.5 * (T_{t,j,u}^* T_{t,j,u} + T_{t,j,u} T_{t,j,u}^*).

    Parameters
    ----------
    proj_tju : np.ndarray
        The projection tensor T_{t,j,u} = <\phi_{t,j} | u>
    pos : bool
        If True, removes negative component by adding to the tensor and scaling down
        to match previous tensor sum.
    r"""
    t, j, u = np.shape(proj_tju)
    s_tj_uu = np.zeros([t, j, u, u], dtype=COMPLEX_DTYPE)
    if real:
        s_tj_uu += np.real(np.conj(proj_tju[:, :, :, np.newaxis]) * proj_tju[:, :, np.newaxis, :])
        # s_tj_uu += 0.5 * (
        #     np.conj(proj_tju[:, :, :, np.newaxis]) * proj_tju[:, :, np.newaxis, :]
        #     + np.conj(np.conj(proj_tju[:, :, :, np.newaxis]) * proj_tju[:, :, np.newaxis, :])
        # )
        s_tj_uu = np.array(s_tj_uu, dtype=REAL_DTYPE)
    else:
        s_tj_uu += np.conj(proj_tju[:, :, :, np.newaxis]) * proj_tju[:, :, np.newaxis, :]
    if pos:
        sum1 = np.sum(s_tj_uu.flatten())
        s_tj_uu -= np.min(s_tj_uu.flatten())
        s_tj_uu *= sum1 / np.sum(s_tj_uu.flatten())
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
        p_tj_uu += (
            occ_tj[:, :, np.newaxis, np.newaxis]
            * wk_t[:, np.newaxis, np.newaxis, np.newaxis]
            / np.sum(np.abs(s_tj_uu), axis=(2, 3), keepdims=True)
        ) * np.abs(s_tj_uu)
    else:
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
    h_tj_uu += p_tj_uu * e_tj[:, :, np.newaxis, np.newaxis]
    return h_tj_uu


def get_h_uu_p_uu_s_uu(
    proj_tju: np.ndarray[COMPLEX_DTYPE],
    e_tj: np.ndarray[REAL_DTYPE],
    occ_tj: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    s_real: bool = True,
    s_pos: bool = True,
    p_sc: bool = True,
):
    r"""Get the Hamiltonian tensor H_{u,u} = Sum_{t,j} P_{t,j,u,u} * E_{t,j}.

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
    s_tj_uu = get_s_tj_uu(proj_tju, pos=s_pos, real=s_real)
    p_tj_uu = get_p_tj_uu(s_tj_uu, occ_tj, wk_t, sys_consistent=p_sc)
    h_tj_uu = get_h_tj_uu(p_tj_uu, e_tj)
    h_uu = np.sum(h_tj_uu, axis=(0, 1))
    p_uu = np.sum(p_tj_uu, axis=(0, 1))
    s_uu = np.sum(s_tj_uu, axis=(0, 1))
    return h_uu, p_uu, s_uu


def _get_gen_tj(
    proj_tju: np.ndarray[COMPLEX_DTYPE],
    gen_uu: np.ndarray[REAL_DTYPE],
    wk_t: np.ndarray[REAL_DTYPE],
    orbs_u: list[int],
    orbs_v: list[int],
) -> np.ndarray:
    wk_cw_tj_uu = np.real(
        np.conj(proj_tju)[:, :, np.newaxis, :]
        * proj_tju[:, :, :, np.newaxis]
        * wk_t[:, np.newaxis, np.newaxis, np.newaxis]
    )
    orbs = np.asarray(list(orbs_u) + list(orbs_v))
    t, j, u = np.shape(proj_tju)
    gen_tj = np.zeros([t, j], dtype=REAL_DTYPE)
    return np.array(_get_gen_tj_jit(gen_tj, gen_uu, wk_cw_tj_uu, orbs_u, orbs_v), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_gen_tj_jit(gen_tj, gen_uu, wk_cw_tj_uu, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            if not u == v:
                gen_tj += gen_uu[u, v] * wk_cw_tj_uu[:, :, u, v]
    return gen_tj


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


# @jit(nopython=True)
# def _get_pcohp_sabcj_jit(
#     nspin: int,
#     nka: int,
#     nkb: int,
#     nkc: int,
#     nbands: int,
#     orbs_u: np.ndarray[int],
#     orbs_v: np.ndarray[int],
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     h_uvsabc: np.ndarray[REAL_DTYPE],
#     pcohp_sabcj: np.ndarray[REAL_DTYPE],
# ) -> np.ndarray:
#     for s in range(nspin):
#         for a in range(nka):
#             for b in range(nkb):
#                 for c in range(nkc):
#                     for j in range(nbands):
#                         uv_sum = 0
#                         for u in orbs_u:
#                             for v in orbs_v:
#                                 p1 = p_uvjsabc[u, v, j, s, a, b, c]
#                                 p2 = h_uvsabc[u, v, s, a, b, c]
#                                 uv_sum += p1 * p2
#                         pcohp_sabcj[s, a, b, c, j] += uv_sum
#     return pcohp_sabcj


# def get_pcohp_sabcj(
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     h_uvsabc: np.ndarray[REAL_DTYPE],
#     orbs_u: list[int],
#     orbs_v: list[int],
# ) -> np.ndarray:
#     r"""Return the pCOHP tensor pCOHP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} H_{u,v}^{s,a,b,c} w_{s,a,b,c}.

#     Return the pCOHP tensor pCOHP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} H_{u,v}^{s,a,b,c} w_{s,a,b,c}.
#     where u encompasses indices for orbitals of interest for species 1, and v for species 2.

#     Parameters
#     ----------
#     p_uvjsabc : np.ndarray
#         The projection matrix P_{u,v}^{j,s,a,b,c} = <\phi_{j,s,a,b,c} | u><v | \phi_{j,s,a,b,c}>
#     h_uvsabc : np.ndarray
#         The Hamiltonian matrix H_{u,v}^{s,a,b,c} = Sum_j <\phi_{s,a,b,c,j} | u><u | \hat{H} | v><v | \phi_{s,a,b,c,j}>
#     orbs_u : list[int]
#         The list of orbitals to evaluate for species 1
#     orbs_v : list[int]
#         The list of orbitals to evaluate for species 2
#     r"""
#     shape = np.shape(p_uvjsabc)
#     nbands = shape[2]
#     nspin = shape[3]
#     nka = shape[4]
#     nkb = shape[5]
#     nkc = shape[6]
#     pcohp_sabcj = np.zeros([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE)
#     _orbs_u = np.asarray(orbs_u)
#     _orbs_v = np.asarray(orbs_v)
#     return _get_pcohp_sabcj_jit(nspin, nka, nkb, nkc, nbands, _orbs_u, _orbs_v, p_uvjsabc, h_uvsabc, pcohp_sabcj)


# @jit(nopython=True)
# def _get_pcoop_sabcj_jit(
#     nspin: int,
#     nka: int,
#     nkb: int,
#     nkc: int,
#     nbands: int,
#     orbs_u: np.ndarray[int],
#     orbs_v: np.ndarray[int],
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     pcoop_sabcj: np.ndarray[REAL_DTYPE],
# ) -> np.ndarray:
#     for s in range(nspin):
#         for a in range(nka):
#             for b in range(nkb):
#                 for c in range(nkc):
#                     for j in range(nbands):
#                         uv_sum = 0
#                         for u in orbs_u:
#                             for v in orbs_v:
#                                 uv_sum += p_uvjsabc[u, v, j, s, a, b, c]
#                         pcoop_sabcj[s, a, b, c, j] += uv_sum
#     return pcoop_sabcj


# def get_pcoop_sabcj(
#     p_uvjsabc: np.ndarray[REAL_DTYPE],
#     orbs_u: list[int],
#     orbs_v: list[int],
# ) -> np.ndarray[REAL_DTYPE]:
#     r"""Return the pCOOP tensor pCOOP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} w_{s,a,b,c}.

#     Return the pCOOP tensor pCOOP_{s,a,b,c,j} = Sum_{u,v} P_{u,v}^{j,s,a,b,c} w_{s,a,b,c}.
#     where u encompasses indices for orbitals of interest for species 1, and v for species 2.

#     Parameters
#     ----------
#     p_uvjsabc : np.ndarray
#         The projection matrix P_{u,v}^{j,s,a,b,c} = <\phi_{j,s,a,b,c} | u><v | \phi_{j,s,a,b,c}>
#     orbs_u : list[int]
#         The list of orbitals to evaluate for species 1
#     orbs_v : list[int]
#         The list of orbitals to evaluate for species 2
#     r"""
#     shape = np.shape(p_uvjsabc)
#     nbands = shape[2]
#     nspin = shape[3]
#     nka = shape[4]
#     nkb = shape[5]
#     nkc = shape[6]
#     pcoop_sabcj = np.zeros([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE)
#     _orbs_u = np.asarray(orbs_u)
#     _orbs_v = np.asarray(orbs_v)
#     return _get_pcoop_sabcj_jit(nspin, nka, nkb, nkc, nbands, _orbs_u, _orbs_v, p_uvjsabc, pcoop_sabcj)


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
