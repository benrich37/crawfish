import pytest
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
import numpy as np
from numba import jit
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR

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


######## Old vs new: matrix construction and pair weights ########
# The pre-`limited_memory` implementations below are copied verbatim from crawfish/core/operations/matrix.py
# (with an `_old` suffix), so the rewritten low-memory versions can be checked against them. The old h/p/s
# functions are still in matrix.py as `get_h_tj_uu_p_tj_uu_s_tj_uu`, so only their old wrappers are copied.


def get_h_t_uu_p_t_uu_s_t_uu_old(proj_tju, e_tj, occ_tj, wk_t, p_sc=True, norm_eigs_for_h=True, mu=None):
    from crawfish.core.operations.matrix import get_h_tj_uu_p_tj_uu_s_tj_uu

    h_tj_uu, p_tj_uu, s_tj_uu = get_h_tj_uu_p_tj_uu_s_tj_uu(
        proj_tju, e_tj, occ_tj, wk_t, p_sc=p_sc, mu=mu, norm_eigs_for_h=norm_eigs_for_h
    )
    h_t_uu = np.sum(h_tj_uu, axis=1)
    p_t_uu = np.sum(p_tj_uu, axis=1)
    s_t_uu = np.sum(s_tj_uu, axis=1)
    return h_t_uu, p_t_uu, s_t_uu


def get_h_uu_p_uu_s_uu_old(proj_tju, e_tj, occ_tj, wk_t, p_sc=True, norm_eigs_for_h=True, mu=None):
    from crawfish.core.operations.matrix import get_h_tj_uu_p_tj_uu_s_tj_uu

    h_tj_uu, p_tj_uu, s_tj_uu = get_h_tj_uu_p_tj_uu_s_tj_uu(
        proj_tju, e_tj, occ_tj, wk_t, p_sc=p_sc, norm_eigs_for_h=norm_eigs_for_h, mu=mu
    )
    h_uu = np.sum(h_tj_uu, axis=(0, 1))
    p_uu = np.sum(p_tj_uu, axis=(0, 1))
    s_uu = np.sum(s_tj_uu, axis=(0, 1))
    return h_uu, p_uu, s_uu


def _get_gen_tj_coef_uu_old(weighted_overlaps_tj_uv, coef_uu, orbs_u, orbs_v):
    t, j, u, _ = np.shape(weighted_overlaps_tj_uv)
    gen_tj = np.zeros([t, j], dtype=REAL_DTYPE)
    return np.array(np.real(_get_gen_tj_coef_uu_jit_old(gen_tj, coef_uu, weighted_overlaps_tj_uv, orbs_u, orbs_v)), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_gen_tj_coef_uu_jit_old(gen_tj, coef_uu, weighted_overlaps_tj_uv, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            if not u == v:
                gen_tj += coef_uu[u, v] * weighted_overlaps_tj_uv[:, :, u, v]
    return gen_tj


def _get_gen_tj_coef_tuu_old(weighted_overlaps_tj_uv, coef_tuu, orbs_u, orbs_v):
    t, j, u, _ = np.shape(weighted_overlaps_tj_uv)
    gen_tj = np.zeros([t, j], dtype=REAL_DTYPE)
    return np.array(np.real(_get_gen_tj_coef_tuu_jit_old(gen_tj, coef_tuu, weighted_overlaps_tj_uv, orbs_u, orbs_v)), dtype=REAL_DTYPE)


@jit(nopython=True)
def _get_gen_tj_coef_tuu_jit_old(gen_tj, coef_tuu, wk_cw_tj_uu, orbs_u, orbs_v):
    trange = list(range(np.shape(gen_tj)[0]))
    for u in orbs_u:
        for v in orbs_v:
            if not u == v:
                for t in trange:
                    val = coef_tuu[t, u, v] * wk_cw_tj_uu[t, :, u, v]
                    gen_tj[t,:] += val
    return gen_tj


def _random_calc(nstates, nbands, nproj, seed=0, zero_band=False):
    rng = np.random.default_rng(seed)
    proj_tju = (rng.random([nstates, nbands, nproj]) - 0.5) + 1j * (rng.random([nstates, nbands, nproj]) - 0.5)
    if zero_band:
        # A band with no projections makes the p_sc normalization 0/0
        proj_tju[:, -1, :] = 0
    e_tj = rng.random([nstates, nbands]) - 0.5
    occ_tj = rng.random([nstates, nbands])
    wk_t = rng.random([nstates])
    return (
        proj_tju.astype(COMPLEX_DTYPE), e_tj.astype(REAL_DTYPE), occ_tj.astype(REAL_DTYPE), wk_t.astype(REAL_DTYPE),
        REAL_DTYPE(0.1),
    )


def _example_calc(name):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(EXAMPLE_CALC_DIRS_DIR / name)
    return edata.proj_tju, edata.e_tj, edata.occ_tj, edata.wk_t, edata.mu


SAMPLE_CALCS = {
    "random_small": lambda: _random_calc(6, 5, 4),
    "random_more_bands": lambda: _random_calc(10, 12, 6, seed=1),
    "random_fewer_bands": lambda: _random_calc(4, 3, 7, seed=2),
    "random_large": lambda: _random_calc(40, 30, 16, seed=3),
    "random_zero_band": lambda: _random_calc(6, 5, 4, seed=4, zero_band=True),
    "N2_bare_min": lambda: _example_calc("N2_bare_min"),
}


def assert_matches_old(new, old):
    # Only summation order differs (float64 accumulation vs float32 numpy sums), so tolerance is relative to
    # the array's scale. NaNs must appear in the same places.
    assert new.shape == old.shape
    assert new.dtype == old.dtype
    scale = np.nanmax(np.abs(old)) if not np.all(np.isnan(old)) else 1.0
    np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-5 * scale, equal_nan=True)


@pytest.mark.parametrize("calc", SAMPLE_CALCS)
@pytest.mark.parametrize("p_sc", [True, False])
@pytest.mark.parametrize("norm_eigs_for_h", [True, False])
def test_h_p_s_mats_match_old(calc, p_sc, norm_eigs_for_h):
    from crawfish.core.operations.matrix import get_h_p_s_t_uu_and_uu, get_h_t_uu_p_t_uu_s_t_uu, get_h_uu_p_uu_s_uu

    proj_tju, e_tj, occ_tj, wk_t, mu = SAMPLE_CALCS[calc]()
    kwargs = {"p_sc": p_sc, "norm_eigs_for_h": norm_eigs_for_h, "mu": mu}
    old_t = get_h_t_uu_p_t_uu_s_t_uu_old(proj_tju, e_tj, occ_tj, wk_t, **kwargs)
    old_tot = get_h_uu_p_uu_s_uu_old(proj_tju, e_tj, occ_tj, wk_t, **kwargs)
    new_all = get_h_p_s_t_uu_and_uu(proj_tju, e_tj, occ_tj, wk_t, **kwargs)
    new_t = get_h_t_uu_p_t_uu_s_t_uu(proj_tju, e_tj, occ_tj, wk_t, **kwargs)
    new_tot = get_h_uu_p_uu_s_uu(proj_tju, e_tj, occ_tj, wk_t, **kwargs)
    for new, old in zip(new_all, old_t + old_tot):
        assert_matches_old(new, old)
    for new, old in zip(new_t + new_tot, old_t + old_tot):
        assert_matches_old(new, old)


@pytest.mark.parametrize("calc", [c for c in SAMPLE_CALCS if c != "random_zero_band"])
@pytest.mark.parametrize("orbs", [([0, 1], [1, 2, 3]), ([0], [1]), ([2, 0], [3, 1, 0])])
def test_gen_tj_coefs_match_old(calc, orbs):
    from crawfish.core.operations.matrix import (
        _get_gen_tj_coef_uu, _get_gen_tj_coef_tuu, get_weighted_overlap_tjuv, get_h_p_s_t_uu_and_uu,
    )

    proj_tju, e_tj, occ_tj, wk_t, mu = SAMPLE_CALCS[calc]()
    orbs_u, orbs_v = orbs
    w_tjuv = get_weighted_overlap_tjuv(proj_tju, wk_t)
    # Use the actual h/p/s matrices as coefficients, called the way pcohp.py calls them (old and new)
    mats = get_h_p_s_t_uu_and_uu(proj_tju, e_tj, occ_tj, wk_t, mu=mu)
    for coef_t_uu, coef_uu in zip(mats[:3], mats[3:]):
        old = _get_gen_tj_coef_uu_old(w_tjuv, coef_uu, orbs_u, orbs_v)
        new = _get_gen_tj_coef_uu(proj_tju, wk_t, coef_uu, orbs_u, orbs_v)
        assert_matches_old(new, old)
        old = _get_gen_tj_coef_tuu_old(w_tjuv, coef_t_uu * 1 / wk_t[:, np.newaxis, np.newaxis], orbs_u, orbs_v)
        new = _get_gen_tj_coef_tuu(proj_tju, coef_t_uu, orbs_u, orbs_v)
        assert_matches_old(new, old)
    # pcomo uses a matrix of ones
    ones_uu = np.ones(mats[3].shape, dtype=REAL_DTYPE)
    assert_matches_old(
        _get_gen_tj_coef_uu(proj_tju, wk_t, ones_uu, orbs_u, orbs_v),
        _get_gen_tj_coef_uu_old(w_tjuv, ones_uu, orbs_u, orbs_v),
    )


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
