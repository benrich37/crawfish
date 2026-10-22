import pytest
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
import numpy as np

# def test_get_weighted_overlap_tjuv():
#     from crawfish.core.operations.matrix import get_weighted_overlap_tjuv

#     nproj = 3
#     nstates = 3
#     nbands = 3

#     proj_tju = (np.random.random([nstates, nbands, nproj]) - 0.5) + 1j*(np.random.random([nstates, nbands, nproj]) - 0.5)
#     wk_t = np.random.random([nstates])
#     weighted_overlap_tjuv = get_weighted_overlap_tjuv(proj_tju, wk_t)
#     for t in range(nstates):
#         for j in range(nbands):
#             for u in range(nproj):
#                 for v in range(u, nproj):
#                     assert np.isclose(
#                         weighted_overlap_tjuv[t,j,u,v],
#                         wk_t[t]*(np.conj(proj_tju[t,j,u])*proj_tju[t,j,v])
#                     )


# def test_get_overlap_tjuv():
#     from crawfish.core.operations.matrix import get_overlap_tjuv

#     nproj = 3
#     nstates = 3
#     nbands = 3

#     proj_tju = (np.random.random([nstates, nbands, nproj]) - 0.5) + 1j*(np.random.random([nstates, nbands, nproj]) - 0.5)
#     overlap_tjuv = get_overlap_tjuv(proj_tju)
#     for t in range(nstates):
#         for j in range(nbands):
#             for u in range(nproj):
#                 for v in range(u, nproj):
#                     assert np.isclose(
#                         overlap_tjuv[t,j,u,v],
#                         np.conj(proj_tju[t,j,u])*proj_tju[t,j,v]
#                     )

# def test_get_s_tj_uu():
#     from crawfish.core.operations.matrix import get_s_tj_uu

#     nproj = 3
#     nstates = 3
#     nbands = 3

#     proj_tju = (np.random.random([nstates, nbands, nproj]) - 0.5) + 1j*(np.random.random([nstates, nbands, nproj]) - 0.5)
#     s_tj_uu = get_s_tj_uu(proj_tju)
#     for t in range(nstates):
#         for j in range(nbands):
#             for u in range(nproj):
#                 for v in range(u, nproj):
#                     assert np.isclose(
#                         s_tj_uu[t,j,u,v],
#                         REAL_DTYPE(np.abs(np.conj(proj_tju[t,j,u])*proj_tju[t,j,v])**2)
#                     )

# def test_get_p_tj_uu():
#     from crawfish.core.operations.matrix import get_s_tj_uu, get_p_tj_uu

#     nproj = 3
#     nstates = 3
#     nbands = 3

#     proj_tju = (np.random.random([nstates, nbands, nproj]) - 0.5) + 1j*(np.random.random([nstates, nbands, nproj]) - 0.5)
#     s_tj_uu = get_s_tj_uu(proj_tju)
#     wk_t = np.random.random([nstates])
#     occ_tj = np.random.random([nstates, nbands])
#     p_tj_uu_direct = get_p_tj_uu(s_tj_uu, occ_tj, wk_t, sys_consistent=False)
#     p_tj_uu_consistent = get_p_tj_uu(s_tj_uu, occ_tj, wk_t, sys_consistent=True)
#     for t in range(nstates):
#         for j in range(nbands):
#             tj_sum = np.sum(s_tj_uu[t,j,:,:].flatten())
#             assert np.isclose(
#                 np.sum(p_tj_uu_consistent[t,j,:,:].flatten()),
#                 wk_t[t]*occ_tj[t,j]
#             )
#             for u in range(nproj):
#                 for v in range(u, nproj):
#                     assert np.isclose(
#                         p_tj_uu_direct[t,j,u,v],
#                         occ_tj[t,j]*wk_t[t]*s_tj_uu[t,j,u,v]
#                     )
#                     assert np.isclose(
#                         p_tj_uu_consistent[t,j,u,v],
#                         occ_tj[t,j]*wk_t[t]*s_tj_uu[t,j,u,v]/tj_sum
#                     )


# def test_get_h_tj_uu():
#     from crawfish.core.operations.matrix import get_s_tj_uu, get_p_tj_uu, get_h_tj_uu

#     nproj = 3
#     nstates = 3
#     nbands = 3

#     proj_tju = (np.random.random([nstates, nbands, nproj]) - 0.5) + 1j*(np.random.random([nstates, nbands, nproj]) - 0.5)
#     wk_t = np.random.random([nstates])
#     occ_tj = np.random.random([nstates, nbands])
#     e_tj = np.random.random([nstates, nbands]) - 0.5
#     s_tj_uu = get_s_tj_uu(proj_tju)
#     p_tj_uu = get_p_tj_uu(s_tj_uu, occ_tj, wk_t, sys_consistent=True)
#     h_tj_uu = get_h_tj_uu(p_tj_uu, e_tj)
#     for t in range(nstates):
#         for j in range(nbands):
#             for u in range(nproj):
#                 for v in range(u, nproj):
#                     assert np.isclose(
#                         h_tj_uu[t,j,u,v],
#                         p_tj_uu[t,j,u,v]*e_tj[t,j]
#                     )



def test_mod_weights_for_ebounds():
    from crawfish.core.operations.matrix import mod_weights_for_ebounds

    nspin = 1
    nka = 1
    nkb = 1
    nkc = 1
    nbands = 3
    e_sabcj = np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE) * (-1)
    e_sabcj[:, :, :, :, 0] -= 1
    e_sabcj[:, :, :, :, -1] += 1
    w_sabcj = np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE)
    ebounds = [-1.1, -0.1]
    w_sabcj = mod_weights_for_ebounds(w_sabcj, e_sabcj, ebounds)
    for j in [0, 2]:
        assert all(np.isclose(w_sabcj[:, :, :, :, j].flatten(), 0.0))
    for j in [1]:
        assert all(np.isclose(w_sabcj[:, :, :, :, j].flatten(), 1.0))


def test_get_ebound_arr():
    from crawfish.core.operations.matrix import get_ebound_arr

    nspin = 1
    nka = 1
    nkb = 1
    nkc = 1
    nbands = 3
    e_sabcj = np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE) * (-1)
    e_sabcj[:, :, :, :, 0] -= 1
    e_sabcj[:, :, :, :, -1] += 1
    ebounds = [-1.1, -0.1, 0.0]
    with pytest.raises(ValueError, match="The ebounds list must have an even number of elements."):
        get_ebound_arr(ebounds, e_sabcj)
