"""Tests for calculations without elec-smearing, which print no mu: the HOMO-LUMO midpoint (finding F20)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR, write_band_truncated_copy
from pathlib import Path
import re
import shutil
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
UNSMEARED = ["Si_nosmear_nospin", "Si_nosmear_zspin_M2"]


def _load(calc_dir, **settings):
    """Return the ElecData and the messages of the warnings about mu raised while loading it."""
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edata = ElecData(calc_dir)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata, [str(w.message) for w in caught if "chemical potential" in str(w.message)]


def _last(result):
    return float(np.ravel(result[1])[-1])


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _eigstats_gap(calc_dir):
    """(HOMO, LUMO) from JDFTx's eigStats dump (Dump.cpp: F >= 0.5 and F <= 0.5, as crawfish's estimate)."""
    text = (Path(calc_dir) / "eigStats").read_text()
    return tuple(float(re.search(rf"{key}: (\S+)", text)[1]) for key in ("HOMO", "LUMO"))


@pytest.fixture
def n2_without_mu(tmp_path):
    """N2_nospin (full 2x2x2 mesh) with every FillingsUpdate line removed from its outfile; its fillings file stays."""
    calc_dir = tmp_path / "N2_nomu"
    shutil.copytree(calc_dirs / "N2_nospin", calc_dir, ignore=shutil.ignore_patterns(".crawfish_cache"))
    lines = (calc_dir / "out").read_text().splitlines()
    (calc_dir / "out").write_text("\n".join(line for line in lines if "FillingsUpdate:" not in line) + "\n")
    return calc_dir


@pytest.mark.parametrize("name", UNSMEARED)
def test_mu_is_gap_midpoint(name):
    edata, msgs = _load(calc_dirs / name)
    assert msgs == []  # loading alone needs no mu
    homo, lumo = _eigstats_gap(calc_dirs / name)
    with pytest.warns(UserWarning, match="midpoint of the HOMO") as record:
        mu = edata.mu
    assert mu == pytest.approx(0.5 * (homo + lumo), abs=1e-6) and edata.mu_is_estimated
    assert edata.norm_eigs_for_h and "h-en_True" in edata.cache_sub_dir.name
    # z-spin M2: constant fillings per channel put the HOMO (spin up) above the LUMO (spin down)
    assert ("no single Fermi level" in str(record[0].message)) == (homo > lumo)


def test_si_midpoint_values():
    edata, _ = _load(calc_dirs / "Si_nosmear_nospin")
    assert _quiet(lambda: edata.mu) == pytest.approx(0.24226, abs=1e-5)
    edata, _ = _load(calc_dirs / "Si_nosmear_zspin_M2")
    assert _quiet(lambda: edata.mu) == pytest.approx(0.5 * (0.36062 + 0.10321), abs=1e-5)


@pytest.mark.parametrize("name", ["N2_nospin", "ZnO_wurtzite_full", "Al_fcc_Fermi"])
def test_printed_mu_unchanged(name):
    from crawfish.io.data_parsing import get_mu_from_outfile_filepath

    edata, _ = _load(calc_dirs / name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning about an estimate
        assert edata.mu == get_mu_from_outfile_filepath(edata.outfile_filepath)
    assert not edata.mu_is_estimated


def test_mu_by_hand_overrides():
    edata, _ = _load(calc_dirs / "Si_nosmear_nospin")
    assert edata.mu_is_estimated
    edata.mu = 0.2419  # e.g. the eigStats mu line
    assert edata.mu == pytest.approx(0.2419) and not edata.mu_is_estimated


@pytest.mark.parametrize("name", UNSMEARED)
@pytest.mark.parametrize("los_orbs", [True, False])
def test_pair_spectra_use_the_midpoint(name, los_orbs):
    # The reduced meshes of the Si runs guard single images; image="all" is exact for the whole Si1-Si2 sum
    from crawfish.funcs import get_ipcohp, get_pcohp

    def ipcohp(edata):
        return _last(_quiet(get_ipcohp, edata, idcs1=0, idcs2=1, image="all", use_cache=False))

    default, _ = _load(calc_dirs / name, los_orbs=los_orbs)
    spec = _quiet(get_pcohp, default, idcs1=0, idcs2=1, image="all", use_cache=False, use_cached_spectrum=False,
                  save_spectrum=False)[1]
    assert np.all(np.isfinite(spec)) and np.any(spec)
    by_hand, _ = _load(calc_dirs / name, los_orbs=los_orbs)
    by_hand.mu = _quiet(lambda: default.mu)
    assert ipcohp(default) == pytest.approx(ipcohp(by_hand), abs=1e-6)
    unshifted, _ = _load(calc_dirs / name, los_orbs=los_orbs, norm_eigs_for_h=False)
    if los_orbs:
        # Löwdin basis, nothing dropped (test_band_span): the energy zero only changes the diagonal
        assert ipcohp(default) == pytest.approx(ipcohp(unshifted), abs=1e-6)
    else:
        assert abs(ipcohp(default) - ipcohp(unshifted)) > 1e-3
    # Nothing cached for norm_eigs_for_h False (the F20 interim default) is read as shifted data
    assert unshifted.cache_sub_dir != default.cache_sub_dir


def test_warns_once():
    from crawfish.funcs import get_ipcohp

    edata, _ = _load(calc_dirs / "Si_nosmear_nospin", los_orbs=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = edata.mu
        get_ipcohp(edata, idcs1=0, idcs2=1, image="all", use_cache=False)
        edata.legacy_lattice_image = True
        edata.los_orbs = True
        get_ipcohp(edata, idcs1=0, idcs2=1, image="all", use_cache=False)
        _ = edata.mu
    assert len([w for w in caught if "midpoint of the HOMO" in str(w.message)]) == 1


def test_stripped_mu_uses_midpoint(n2_without_mu):
    # A smeared run whose outfile lost its mu, with a fillings file: the occupations don't need mu, so it is estimated
    from crawfish.funcs import get_ipcohp

    edata, _ = _load(n2_without_mu)
    ref, _ = _load(calc_dirs / "N2_nospin")
    e, f = np.asarray(ref.e_tj, np.float64), np.asarray(ref.occ_tj, np.float64)
    assert _quiet(lambda: edata.mu) == pytest.approx(0.5 * (e[f >= 0.5].max() + e[f <= 0.5].min()), abs=1e-6)
    assert edata.mu_is_estimated and ref.mu == pytest.approx(-0.1)
    assert _last(_quiet(get_ipcohp, edata, idcs1=0, idcs2=1, use_cache=False)) == pytest.approx(
        _last(get_ipcohp(ref, idcs1=0, idcs2=1, use_cache=False)), abs=1e-6)  # Löwdin, nothing dropped


def test_smeared_without_mu_or_fillings_raises(n2_without_mu):
    # The occupations would need mu from the smearing, so there is nothing to estimate from: raises as before
    from crawfish.core.elecdata import ElecData

    (n2_without_mu / "fillings").unlink()
    with pytest.raises(ValueError, match="No chemical potential"):
        ElecData(n2_without_mu)


def test_no_lumo_falls_back_to_unshifted(tmp_path):
    # Only the occupied bands: no LUMO, no estimate. norm_eigs_for_h is switched off, as before the midpoint (F20)
    from crawfish.funcs import get_ipcohp

    calc_dir = write_band_truncated_copy(calc_dirs / "Si_nosmear_nospin", 4, tmp_path / "Si_4bands")
    edata, msgs = _load(calc_dir)  # 4 bands, 8 orbitals: also no Löwdin step, so the energy zero matters
    assert not edata.norm_eigs_for_h and "h-en_False" in edata.cache_sub_dir.name and not edata.mu_is_estimated
    assert len(msgs) == 1 and "no LUMO" in msgs[0]
    with pytest.raises(ValueError, match="No chemical potential"):
        _ = edata.mu
    with pytest.raises(ValueError, match="No chemical potential"):
        edata.norm_eigs_for_h = True
    assert np.isfinite(_last(_quiet(get_ipcohp, edata, idcs1=0, idcs2=1, image="all", use_cache=False)))
    edata.mu = 0.24
    edata.norm_eigs_for_h = True
    assert "h-en_True" in edata.cache_sub_dir.name


def test_mu_by_hand_after_reset_cache(tmp_path):
    # A hand-set mu isn't part of the cache key: results cached with the estimate are reused until reset_cache()
    from crawfish.funcs import get_ipcohp

    def ipcohp(edata):
        return _last(_quiet(get_ipcohp, edata, idcs1=0, idcs2=1, image="all"))

    calc_dir = tmp_path / "Si"
    shutil.copytree(calc_dirs / "Si_nosmear_nospin", calc_dir, ignore=shutil.ignore_patterns(".crawfish_cache"))
    ipcohp(_load(calc_dir, los_orbs=False)[0])  # caches the estimate-based matrices and weights
    fresh_dir = tmp_path / "Si_fresh"
    shutil.copytree(calc_dirs / "Si_nosmear_nospin", fresh_dir, ignore=shutil.ignore_patterns(".crawfish_cache"))
    fresh, _ = _load(fresh_dir, los_orbs=False)
    fresh.mu = 0.20
    edata, _ = _load(calc_dir, los_orbs=False)
    edata.mu = 0.20
    edata.reset_cache()
    assert edata.mu == pytest.approx(0.20) and not edata.mu_is_estimated
    assert ipcohp(edata) == pytest.approx(ipcohp(fresh), abs=1e-6)
