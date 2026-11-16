"""Tests for JDFTx's exact atomic-orbital matrices (`dump End AO`, ElecData.use_ao_matrices).

The graphene_ao_* fixtures (README in graphene_ao_full) are SG15 graphene on a 6x6x1 mesh: the full mesh with ortho
projections, the same with non-orthogonal projections of the same wavefunctions, and a separate run reduced by its 24
symmetry operations, with the `sym` file. Norm-conserving, so ortho = S^-1/2 x non-ortho holds to the printed digits.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
full_dir, pg_dir, nonortho_dir = (calc_dirs / f"graphene_ao_{x}" for x in ("full", "pg", "full_nonortho"))


def _get_edata(calc_dir, **settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _band_totals_tj(edata, name):
    """sum_uv conj(C_u) M_uv C_v per state and band: basis-independent for the exact matrices."""
    mat = getattr(edata, f"{name}_t_uv").astype(np.complex128)
    return np.einsum("tju,tuv,tjv->tj", edata.coef_tju.conj(), mat, edata.coef_tju).real


def test_parser_reads_column_major(tmp_path):
    from crawfish.io.data_parsing import get_ao_mat_tuv_from_file

    rng = np.random.default_rng(0)
    m_tuv = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    path = tmp_path / "aoOverlap"
    np.concatenate([m.flatten(order="F") for m in m_tuv]).tofile(path)  # as JDFTx's matrix::write
    assert np.allclose(get_ao_mat_tuv_from_file(path, 3, 4), m_tuv, atol=1e-6)
    with pytest.raises(ValueError, match="same calculation"):
        get_ao_mat_tuv_from_file(path, 2, 4)


def test_exact_matrices_replace_band_sums():
    edata = _get_edata(full_dir)
    assert edata.ao_matrices_used
    s, h = edata.ao_s_t_uv, edata.ao_h_t_uv
    assert s.shape == h.shape == (36, 8, 8)
    assert np.allclose(s, s.conj().transpose(0, 2, 1)) and np.allclose(h, h.conj().transpose(0, 2, 1), atol=1e-6)
    # Ortho projections: Löwdin orbitals phi S^-1/2, so S = 1 exactly and the coefficients are the projections
    assert np.array_equal(edata.s_t_uv, np.broadcast_to(np.eye(8), s.shape))
    assert edata.coef_tju is edata.proj_tju
    # Without the matrices, the band sums come back (and the cache sub-directory differs)
    band = _get_edata(full_dir, use_ao_matrices=False)
    assert not band.ao_matrices_used and band.cache_sub_dir != edata.cache_sub_dir
    assert np.abs(band.s_t_uv - np.eye(8)).max() > 1e-3 or np.abs(band.h_t_uv - edata.h_t_uv).max() > 1e-2


def test_ritz_bound():
    """Eigenvalues of H in the orbital span are upper bounds on the lowest bands (Rayleigh-Ritz)."""
    edata = _get_edata(full_dir, norm_eigs_for_h=False)
    ritz_tu = np.linalg.eigvalsh(edata.h_t_uv.astype(np.complex128))
    assert np.all(ritz_tu - np.sort(edata.e_tj, axis=1)[:, :8] > -1e-5)


def test_nonortho_projections_give_the_same_lowdin_basis():
    ortho, nonortho = _get_edata(full_dir), _get_edata(nonortho_dir)
    assert not nonortho.bandprojfile_is_ortho
    # T_n S^-1/2 is JDFTx's ortho projection, to its 7 printed decimals; a transposed or unphased S is off by ~1
    assert np.abs(nonortho.proj_tju - ortho.proj_tju).max() < 1e-6
    assert np.allclose(nonortho.h_t_uv, ortho.h_t_uv)


def test_nonorthogonal_basis():
    ortho, bare = _get_edata(full_dir), _get_edata(nonortho_dir, los_orbs=False)
    assert np.allclose(bare.s_t_uv, bare.ao_s_t_uv)  # the bare orbitals' exact overlap
    # C = S^-1 T: each band's total energy and norm in the orbital span don't depend on the basis
    for name in ("h", "s"):
        assert np.abs(_band_totals_tj(bare, name) - _band_totals_tj(ortho, name)).max() < 1e-5
    from crawfish.funcs.pcohp import get_ipcoop

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # exact overlap: no band-completeness warning
        _, icoop = get_ipcoop(bare, idcs1=0, idcs2=1, image=(0, 0, 0), use_cache=False)
    assert icoop[-1] > 0.05  # sigma + pi overlap population of a C-C bond


@pytest.mark.parametrize("legacy_orbital_phase", [False, True])
def test_reduced_mesh_unfolds_like_the_full_mesh(legacy_orbital_phase):
    pg, full = (_get_edata(d, legacy_orbital_phase=legacy_orbital_phase) for d in (pg_dir, full_dir))
    assert pg.point_group_unfolded and pg.nstates == full.nstates
    assert np.allclose(pg.ks_t, full.ks_t, atol=1e-6)
    # Separate SCF runs, so they agree to its convergence
    assert np.abs(pg.ao_s_t_uv - full.ao_s_t_uv).max() < 1e-5
    assert np.abs(pg.ao_h_t_uv - full.ao_h_t_uv).max() < 1e-5
    assert np.abs(pg.h_t_uv - full.h_t_uv).max() < 1e-5
