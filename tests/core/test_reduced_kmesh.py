"""Tests for symmetry-reduced k-meshes (notes/theory_audit.md, finding F5).

graphene_zspin_full and graphene_zspin_symm are the same z-spin graphene calculation (12x12x1 mesh, 20 bands) run
with 'symmetries none' (144 k per spin) and 'symmetries automatic' (19 k per spin, non-uniform weights).
"""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

full_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "graphene_zspin_full"
symm_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "graphene_zspin_symm"


def _get_edata(calc_dir, **settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _icohp(edata, image, o1=None, o2=None):
    from crawfish.funcs.pcohp import _get_pcohp_tj

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = _get_pcohp_tj(edata, 0, None, o1, 1, None, o2, use_cache=False, image=image)
    return float(np.sum(edata.occ_tj * weights))


def _ipdos(edata, idcs, orbs):
    """Occupied pDOS (get_ipdos integrates over every band, where each orbital just sums to its completeness)."""
    from crawfish.funcs.dos import get_pdos_tj
    from crawfish.utils.indexing import get_orb_idcs

    orbs_u = get_orb_idcs(edata, idcs, None, orbs)
    return float(np.sum(edata.occ_tj * get_pdos_tj(edata.proj_tju, orbs_u, edata.wk_t)))


def test_mesh_detection():
    assert not _get_edata(full_dir).kmesh_is_reduced
    assert _get_edata(symm_dir).kmesh_is_reduced


def test_kpoint_weights_from_file():
    from crawfish.io.data_parsing import parse_kptsfile

    file_w = np.array(parse_kptsfile(symm_dir / "kPts")[0])
    assert len(np.unique(np.round(file_w, 6))) > 1  # symmetry weights are not uniform
    assert np.allclose(_get_edata(symm_dir).wk_t, file_w)
    # Legacy compared nspin*nk weights with nk and fell back to uniform weights
    legacy = _get_edata(symm_dir, legacy_reduced_kmesh=True)
    assert np.allclose(legacy.wk_t, legacy.wk_t[0])


def test_kpoint_weights_from_bandprojections_header():
    # Without kPts, the per-state weights in the bandProjections header are used
    edata = _get_edata(symm_dir)
    from_kpts = edata.wk_t.copy()
    edata.kptsfile_filepath = None
    edata._wk_t = None
    assert np.allclose(edata.wk_t, from_kpts, atol=1e-6)


def test_single_image_pair_spectra_raise_on_reduced_mesh():
    from crawfish.funcs import get_ipcohp, get_ipcomo

    edata = _get_edata(symm_dir)
    for image in ["nearest", (0, 0, 0)]:
        with pytest.raises(ValueError, match="single lattice image"):
            get_ipcohp(edata, idcs1=0, idcs2=1, image=image)
    with pytest.raises(ValueError, match="single lattice image"):
        get_ipcomo(edata, idcs1=0, idcs2=1)
    # Full mesh: fine
    get_ipcohp(_get_edata(full_dir), idcs1=0, idcs2=1)


def test_all_images_and_full_shells_are_exact_on_reduced_mesh():
    full, symm = _get_edata(full_dir), _get_edata(symm_dir)
    for o in [None, "s", "pz"]:  # complete shells, or orbitals the point group leaves invariant
        assert _icohp(symm, "all", o, o) == pytest.approx(_icohp(full, "all", o, o), abs=2e-4), o
    for idcs in [0, 1]:
        assert _ipdos(symm, idcs, None) == pytest.approx(_ipdos(full, idcs, None), abs=1e-3)
    assert _ipdos(symm, 0, ["px", "py"]) == pytest.approx(_ipdos(full, 0, ["px", "py"]), abs=1e-3)


def test_m_resolved_and_single_bond_are_wrong_on_reduced_mesh():
    # Documents why the guard exists: px alone is mixed with py by the point group, and a single bond is averaged
    full = _get_edata(full_dir)
    symm = _get_edata(symm_dir, legacy_reduced_kmesh=True)  # legacy: no guard, but uniform weights...
    symm._wk_t = _get_edata(symm_dir).wk_t                  # ...so use the correct weights to isolate the effect
    assert abs(_icohp(symm, "all", "px", "px") - _icohp(full, "all", "px", "px")) > 0.05
    assert abs(_ipdos(symm, 0, "px") - _ipdos(full, 0, "px")) > 0.02
    assert abs(_icohp(symm, [0, 0, 0]) - _icohp(full, [0, 0, 0])) > 0.3


def test_three_bonds_equivalent_on_full_mesh():
    full = _get_edata(full_dir)
    bonds = [_icohp(full, list(R)) for R in [(0, 0, 0), (-1, 0, 0), (0, 1, 0)]]
    assert np.allclose(bonds, bonds[0], rtol=1e-4)
    assert _icohp(full, "all") == pytest.approx(3 * bonds[0], rel=0.01)


def test_warnings_on_reduced_mesh():
    from crawfish.funcs import get_ipcohp, get_ipdos

    edata = _get_edata(symm_dir)
    with pytest.warns(UserWarning, match="incomplete orbital shells"):
        get_ipcohp(edata, idcs1=0, idcs2=1, orbs1="px", orbs2="px", image="all")
    with pytest.warns(UserWarning, match="averaged over symmetry-equivalent atoms"):
        get_ipdos(edata, idcs=0)
    # Loading graphene_zspin_full warns that the Löwdin step dropped a sigma combination at K (finding F10)
    full = _get_edata(full_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_ipdos(edata, idcs=[0, 1])  # every orbital: exact, no warning
        get_ipdos(full, idcs=0, orbs="px")  # full mesh: no warning


def test_orbital_shells():
    edata = _get_edata(symm_dir)
    assert list(edata.orb_shell_u) == [0, 1, 1, 1, 2, 3, 3, 3]


# ---- Time-reversal unfolding ----
# graphene_zspin_tr: same calculation as graphene_zspin_full, but with 'symmetries manual' and only the identity,
# so JDFTx reduces by k <-> -k alone (74 k per spin: 4 self-conjugate + 70 pairs).

tr_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "graphene_zspin_tr"


def test_time_reversal_mesh_is_unfolded_by_default():
    full, tr = _get_edata(full_dir), _get_edata(tr_dir)
    assert tr.unfold_time_reversal and tr.time_reversal_unfolded
    assert not tr.kmesh_is_reduced and tr.lti_allowed
    assert tr.nstates_file == 148 and tr.nstates == full.nstates == 288
    assert list(tr.kfolding) == [12, 12, 1]
    assert np.allclose(tr.ks_t, full.ks_t, atol=1e-5)
    assert np.allclose(tr.wk_t, full.wk_t, atol=1e-7)
    assert np.allclose(tr.e_tj, full.e_tj, atol=1e-5)  # separate SCF runs
    assert np.allclose(tr.occ_tj, full.occ_tj, atol=1e-4)


def test_unfolded_matches_full_mesh():
    full, tr = _get_edata(full_dir), _get_edata(tr_dir)
    for image, o in [([0, 0, 0], None), ([-1, 0, 0], None), ("nearest", None), ("all", None), ("all", "px"),
                     ([0, 0, 0], "px"), ([0, 1, 0], "py")]:
        assert _icohp(tr, image, o, o) == pytest.approx(_icohp(full, image, o, o), abs=1e-4), (image, o)
    for orbs in [None, "px", "py", "pz"]:
        assert _ipdos(tr, 0, orbs) == pytest.approx(_ipdos(full, 0, orbs), abs=3e-4), orbs  # separate SCF runs


def test_unfolding_is_controllable():
    from crawfish.core.elecdata import get_edata_settings_str

    tr = _get_edata(tr_dir)
    unfolded_dir = get_edata_settings_str(tr)
    tr.unfold_time_reversal = False
    assert not tr.time_reversal_unfolded and tr.kmesh_is_reduced and tr.nstates == 148
    assert get_edata_settings_str(tr) != unfolded_dir
    with pytest.raises(ValueError, match="single lattice image"):
        _icohp(tr, [0, 0, 0])
    tr.unfold_time_reversal = True
    assert tr.time_reversal_unfolded and tr.nstates == 288
    # The legacy F5 behaviour doesn't unfold either
    assert not _get_edata(tr_dir, legacy_reduced_kmesh=True).time_reversal_unfolded


def test_unfolding_with_raw_orbital_phase():
    # In JDFTx's raw convention, T(-k) = (-1)^l T(k)*: applying the i^l phase afterwards must give the same result
    from crawfish.core.operations.matrix import apply_jdftx_orbital_phase

    fixed, raw = _get_edata(tr_dir), _get_edata(tr_dir, legacy_orbital_phase=True)
    assert raw.time_reversal_unfolded
    assert np.allclose(apply_jdftx_orbital_phase(raw.proj_tju, raw.orb_l_u), fixed.proj_tju, atol=1e-5)


def test_point_group_reduction_is_not_unfolded():
    symm = _get_edata(symm_dir)
    assert not symm.time_reversal_unfolded and symm.kmesh_is_reduced


def test_unfolding_map_synthetic():
    from crawfish.io.data_parsing import get_time_reversal_unfolding

    # 4x4x1 Gamma-centred mesh, reduced by k <-> -k as JDFTx would (k in (-0.5, 0.5])
    grid = [((i - 4 * (i / 4 > 0.5)) / 4, (j - 4 * (j / 4 > 0.5)) / 4, 0.0) for i in range(4) for j in range(4)]
    kept, weights = [], []
    for k in grid:
        if any(np.allclose((np.array(k) + np.array(q) + 0.5) % 1 - 0.5, 0) for q in kept):
            continue
        self_conj = np.allclose((2 * np.array(k) + 0.5) % 1 - 0.5, 0)
        kept.append(k)
        weights.append((1 if self_conj else 2) * 2 / 16)  # no-spin weights: 2/N per full-mesh point
    out = get_time_reversal_unfolding(np.array(kept), np.array(weights), 1, [4, 4, 1], [0, 0, 0])
    assert out is not None and len(out["src_t"]) == 16
    assert np.isclose(np.sum(np.array(weights)[out["src_t"]] * out["wfactor_t"]), 2.0)
    assert out["conj_t"].sum() == 16 - len(kept)
    # Not a time-reversal reduction: drop a point
    assert get_time_reversal_unfolding(np.array(kept[:-1]), np.array(weights[:-1]), 1, [4, 4, 1], [0, 0, 0]) is None
    # Already the full mesh: nothing to do
    assert get_time_reversal_unfolding(np.array(grid), np.full(16, 2 / 16), 1, [4, 4, 1], [0, 0, 0]) is None
