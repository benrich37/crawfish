"""Tests for reading JDFTx fillings files (notes/theory_audit.md, finding F4)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import numpy as np
import pytest

n2_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
n2_nofill_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_no_fillings"


def _get_edata(calc_dir=n2_dir, **settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _n_electrons(edata):
    return float(np.sum(edata.wk_t[:, None].astype(np.float64) * edata.occ_tj))


def test_zspin_spin_weight_and_electron_count():
    edata = _get_edata()
    assert edata.spin_weight == 1
    assert _n_electrons(edata) == pytest.approx(edata.outfile.total_electrons, abs=1e-3)


def test_zspin_unchanged_by_fix():
    # z-spin fillings are already in [0, 1], so the fix must not change them
    fixed, legacy = _get_edata(), _get_edata(legacy_fillings=True)
    assert np.allclose(fixed.occ_tj, legacy.occ_tj)


def test_fillings_file_matches_computed_occupations():
    # Same calculation with and without the fillings file: both must give occupations in [0, 1] and N_e = 10
    with_file, without_file = _get_edata(), _get_edata(n2_nofill_dir)
    assert np.allclose(with_file.occ_tj, without_file.occ_tj, atol=1e-6)
    assert _n_electrons(without_file) == pytest.approx(10.0, abs=1e-3)


def test_electron_count_warning(monkeypatch):
    from crawfish.core.elecdata import ElecData

    # Pretend the z-spin data were no-spin: fillings get halved, so the electron count no longer matches
    monkeypatch.setattr(ElecData, "spin_weight", property(lambda self: 2))
    edata = _get_edata()
    edata._occ_tj = None
    with pytest.warns(UserWarning, match="electrons"):
        _ = edata.occ_tj


def test_integration_no_longer_rescales_by_max_occupation():
    # With fillings in [0, 1], the integrated pDOS over all atoms up to the top band is the electron count
    # (up to spilling), with no rescaling by max(occ)
    edata = _get_edata()
    from crawfish.funcs.general import get_generic_integrate
    from crawfish.funcs.dos import get_pdos_tj

    pdos_tj = get_pdos_tj(edata.proj_tju, list(range(edata.nproj)), edata.wk_t)
    _, cs = get_generic_integrate(edata, pdos_tj, use_fillings=True)
    expected = float(np.sum(edata.occ_tj * pdos_tj))
    assert float(np.ravel(cs)[-1]) == pytest.approx(expected, rel=1e-5)
    assert float(np.ravel(cs)[-1]) == pytest.approx(10.0, rel=0.05)  # 2% charge spilling for N2


# ---- Spin-unpolarized calculation: JDFTx writes fillings in [0, 2] ----

nospin_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_nospin"


def test_nospin_fillings_file_is_halved():
    raw = np.fromfile(nospin_dir / "fillings")
    assert raw.max() == pytest.approx(2.0)  # the convention F4 is about
    fixed, legacy = _get_edata(nospin_dir), _get_edata(nospin_dir, legacy_fillings=True)
    assert fixed.spin_weight == 2
    assert fixed.occ_tj.max() == pytest.approx(1.0)
    assert np.allclose(legacy.occ_tj, 2 * fixed.occ_tj)
    assert _n_electrons(fixed) == pytest.approx(fixed.outfile.total_electrons, abs=1e-3)
    assert _n_electrons(legacy) == pytest.approx(2 * fixed.outfile.total_electrons, abs=1e-3)


def test_nospin_fillings_file_matches_computed_occupations():
    # Before the fix, occupations computed from mu and the smearing (no fillings file) were half of those read
    # from the file for no-spin calculations
    edata = _get_edata(nospin_dir)
    from_file = edata.occ_tj.copy()
    edata.fillingsfile_filepath = None
    edata._occ_tj = None
    assert np.allclose(edata.occ_tj, from_file, atol=1e-6)


def test_nospin_pcobi_doubles_in_legacy_and_matches_zspin():
    from crawfish.funcs import get_ipcobi, get_ipcohp

    def last(func, edata):
        return float(np.ravel(func(edata, idcs1=0, idcs2=1, use_cache=False)[1])[-1])

    fixed, legacy = _get_edata(nospin_dir), _get_edata(nospin_dir, legacy_fillings=True)
    assert last(get_ipcobi, legacy) == pytest.approx(2 * last(get_ipcobi, fixed), rel=1e-5)
    # pCOHP's Hamiltonian has no occupations; the old max(occ) rescaling hid the x2 in integrated spectra
    assert last(get_ipcohp, legacy) == pytest.approx(last(get_ipcohp, fixed), rel=1e-5)
    # The same molecule computed spin-polarized (N2_bare_min: z-spin, 3x3x3 mesh, solvated, GBRV v1.5) agrees
    zspin = _get_edata()
    assert last(get_ipcobi, fixed) == pytest.approx(last(get_ipcobi, zspin), rel=0.01)
    assert last(get_ipcohp, fixed) == pytest.approx(last(get_ipcohp, zspin), rel=0.01)
