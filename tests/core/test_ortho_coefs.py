"""F21: projections onto JDFTx's orthonormalized orbitals are the band coefficients (C = T) when los_orbs is off."""

import warnings
from pathlib import Path

import numpy as np

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR as calc_dirs


def _edata(name, **settings):
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edata = ElecData(calc_dirs / name)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata


def _occupied_density(edata):
    proj = np.asarray(edata.proj_tju, dtype=np.complex128)
    return np.einsum("tj,tju,tjv->tuv", edata.occ_tj, proj, proj.conj())


def test_ortho_projections_are_coefficients():
    edata = _edata("N2_bare_min", los_orbs=False)  # ortho-orbital-projections, Löwdin step off
    assert edata.bandprojfile_is_ortho
    assert np.array_equal(edata.coef_tju, edata.proj_tju)
    # D holds the occupied bands only, so dropping empty bands cannot change it
    assert np.allclose(edata.d_t_uv, _occupied_density(edata), atol=1e-6)
    assert "lgoc_False" in edata.cache_sub_dir.name


def test_legacy_ortho_coefs_restores_least_squares_fit():
    edata = _edata("N2_bare_min", los_orbs=False, legacy_ortho_coefs=True)
    assert "lgoc_True" in edata.cache_sub_dir.name
    assert not np.allclose(edata.coef_tju, edata.proj_tju, atol=1e-4)
    assert not np.allclose(edata.d_t_uv, _occupied_density(edata), atol=1e-4)


def test_los_orbs_path_unchanged():
    new, legacy = _edata("N2_bare_min"), _edata("N2_bare_min", legacy_ortho_coefs=True)
    assert new.los_orbs and legacy.los_orbs
    assert np.array_equal(new.coef_tju, legacy.coef_tju)


def _truncate(edata, nb, cache_dir):
    """Keep the lowest nb bands; the matrices are rebuilt in a fresh cache dir."""
    edata._proj_tju = np.ascontiguousarray(edata.proj_tju[:, :nb])
    edata._e_tj = np.ascontiguousarray(edata.e_tj[:, :nb])
    edata._occ_tj = np.ascontiguousarray(edata.occ_tj[:, :nb])
    edata._nbands = nb
    edata._coef_tju = edata._h_t_uv = edata._s_t_uv = edata._d_t_uv = None
    edata.cache_sub_dir = Path(cache_dir)
    edata.cache_sub_dir.mkdir(parents=True)
    return edata


def _nn_ipcobi(edata):
    from crawfish.funcs import get_ipcobi

    return float(np.ravel(get_ipcobi(edata, idcs1=0, idcs2=1, use_cache=False)[1])[-1])


def test_dropping_empty_bands_leaves_ipcobi_unchanged(tmp_path):
    """C = T: D holds occupied bands only, so dropping empty bands (to fewer bands than orbitals) changes nothing.
    The least-squares fit mixes the empty bands in, so it moves."""
    for legacy, moves in [(False, False), (True, True)]:
        edata = _edata("N2_bare_min", los_orbs=False, legacy_ortho_coefs=legacy)
        nb = int(np.max(np.nonzero(edata.occ_tj > 1e-8)[1])) + 2  # keep two empty bands
        assert nb < edata.nproj < edata.nbands
        full = _nn_ipcobi(edata)
        cut = _nn_ipcobi(_truncate(edata, nb, tmp_path / f"legacy_{legacy}"))
        assert (abs(cut - full) > 1e-3) == moves, (legacy, full, cut)


def test_nonortho_projections_keep_least_squares_fit():
    from crawfish.core.operations.matrix import get_coef_tju

    for legacy in [False, True]:
        edata = _edata("N2_nospin_nonortho15", los_orbs=False, legacy_ortho_coefs=legacy)
        assert not edata.bandprojfile_is_ortho
        assert np.allclose(edata.coef_tju, get_coef_tju(edata.proj_tju, orthonormal=False))
        assert not np.allclose(edata.coef_tju, edata.proj_tju, atol=1e-4)
