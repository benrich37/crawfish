"""F13: projections onto JDFTx's orthonormalized orbitals are the band coefficients (C = T) when los_orbs is off."""

import warnings

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
