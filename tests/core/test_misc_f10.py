"""Tests for the smaller fixes of finding F10 in notes/theory_audit.md: mu read in Hartree from the outfile, the
relative Löwdin cutoff (legacy_lowdin_cutoff), trim_excess_bands re-trimming, and fallback k-point weights."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
graphene_bonds = [(-1, -1, 0), (0, -1, 0), (0, 0, 0)]  # C1's three nearest C2 images in graphene_1x1 (F3 notes)


def _edata(name, **settings):
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edata = ElecData(calc_dirs / name)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata


# ---- mu ----


@pytest.mark.parametrize("name", ["Al_fcc_Fermi", "Al_fcc_Cold", "N2_bare_min", "N2_nospin", "graphene_zspin_symm"])
def test_mu_matches_pymatgen_main_path(name):
    from crawfish.io.data_parsing import get_mu_from_outfile_filepath
    from pymatgen.io.jdftx.outputs import JDFTXOutfile
    from pymatgen.core.units import Ha_to_eV

    mu = get_mu_from_outfile_filepath(calc_dirs / name / "out")
    assert mu == pytest.approx(JDFTXOutfile.from_file(calc_dirs / name / "out").mu / Ha_to_eV, abs=1e-15)
    assert _edata(name).mu == mu


def test_mu_skips_lcao_lines(tmp_path):
    # Without elec-smearing, JDFTx prints FillingsUpdate only during LCAO (with LCAO's own smearing). pymatgen's
    # fallback picks that up in Hartree (_mu_backup = 0.267 here). Without an eigStats dump it becomes
    # outfile.mu, which crawfish used to divide by Ha_to_eV.
    from crawfish.io.data_parsing import get_mu_from_outfile_filepath

    out = calc_dirs / "Si_nosmear_nospin" / "out"
    assert "FillingsUpdate:  mu: +0.267051037" in out.read_text()
    assert get_mu_from_outfile_filepath(out) is None
    with pytest.raises(ValueError, match="No chemical potential"):
        _ = _edata("Si_nosmear_nospin").mu
    lines = out.read_text().split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("Dumping 'eigStats'"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith("\t"))
    (tmp_path / "out").write_text("\n".join(lines[:start] + lines[end:]))
    assert get_mu_from_outfile_filepath(tmp_path / "out") is None


# ---- Löwdin cutoff ----


def _synthetic_proj(eig_small, nbands=6, nproj=4, seed=0):
    # P = U diag(sqrt(eigs)) V^dag, so P^dag P has eigenvalues eigs, one of them tiny
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(nbands, nproj)) + 1j * rng.normal(size=(nbands, nproj)))
    v, _ = np.linalg.qr(rng.normal(size=(nproj, nproj)) + 1j * rng.normal(size=(nproj, nproj)))
    eigs = np.array([1.0, 0.8, 0.5, eig_small])
    return (u * np.sqrt(eigs)) @ v.conj().T, v[:, -1]


def test_lowdin_drops_unspanned_direction():
    from crawfish.core.operations.matrix import los_projs_for_orbs

    p_ju, v_weak = _synthetic_proj(5e-7)
    proj_tju = np.array([p_ju], dtype=np.complex64)
    low, ndropped = los_projs_for_orbs(proj_tju, return_ndropped=True)
    assert list(ndropped) == [1]
    # P'^dag P' is the projector onto the kept directions: the weak one gets no weight
    s_uu = low[0].conj().T @ low[0]
    assert np.allclose(s_uu @ s_uu, s_uu, atol=1e-5)
    assert abs(v_weak.conj() @ s_uu @ v_weak) < 1e-5
    # The legacy (absolute 1e-15) cutoff keeps it and scales it up to unit weight
    low_legacy, ndropped_legacy = los_projs_for_orbs(proj_tju, legacy=True, return_ndropped=True)
    assert list(ndropped_legacy) == [0]
    assert abs(v_weak.conj() @ (low_legacy[0].conj().T @ low_legacy[0]) @ v_weak) > 0.5


def test_legacy_lowdin_cutoff_depends_on_noise_sign():
    # At eigenvalue 1e-8 (float32 noise is ~1e-7), the legacy step keeps or drops the direction depending on the
    # sign of the rounding error; the relative cutoff always drops it
    from crawfish.core.operations.matrix import los_projs_for_orbs

    proj_tju = np.array([_synthetic_proj(1e-8, seed=seed)[0] for seed in range(10)], dtype=np.complex64)
    assert list(los_projs_for_orbs(proj_tju, return_ndropped=True)[1]) == [1] * 10
    assert 0 < np.sum(los_projs_for_orbs(proj_tju, legacy=True, return_ndropped=True)[1]) < 10


def test_lowdin_unchanged_when_bands_span_orbitals():
    from crawfish.core.operations.matrix import los_projs_for_orbs

    p_ju, _ = _synthetic_proj(1e-2)
    proj_tju = np.array([p_ju], dtype=np.complex64)
    low = los_projs_for_orbs(proj_tju)
    assert low.dtype == np.complex64
    assert np.allclose(low[0].conj().T @ low[0], np.eye(4), atol=1e-5)
    assert np.allclose(low, los_projs_for_orbs(proj_tju, legacy=True), atol=1e-5)
    # N2: nothing dropped, and only float32 rounding separates the two
    n2, n2_legacy = _edata("N2_bare_min"), _edata("N2_bare_min", legacy_lowdin_cutoff=True)
    assert np.allclose(n2.proj_tju, n2_legacy.proj_tju, atol=1e-5)


def test_lowdin_warning_and_legacy_switch():
    from crawfish.core.elecdata import ElecData, get_edata_settings_str

    with pytest.warns(UserWarning, match="dropped 4 orbital combination"):
        edata = ElecData(calc_dirs / "graphene_zspin_full")
    assert "lglw_False" in get_edata_settings_str(edata)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        edata.legacy_lowdin_cutoff = True  # no warning on the legacy path
    assert edata.cache_sub_dir.name == get_edata_settings_str(edata)
    assert "lglw_True" in edata.cache_sub_dir.name


def test_graphene_equivalent_bonds_agree_better():
    # graphene_1x1 has 8 bands for 8 orbitals; 446 overlap eigenvalues sit below the cutoff (F10, F11)
    from crawfish.funcs import get_ipcohp

    def bonds(edata):
        return np.array([get_ipcohp(edata, idcs1=0, idcs2=1, image=list(R))[1][-1] for R in graphene_bonds])

    fixed = bonds(_edata("graphene_1x1"))
    legacy = bonds(_edata("graphene_1x1", legacy_lowdin_cutoff=True))
    assert legacy == pytest.approx([-0.14569, -0.14604, -0.14539], abs=2e-5)  # values before F10
    assert fixed == pytest.approx([-0.14352, -0.14346, -0.14348], abs=2e-5)
    assert np.ptp(fixed) < 1e-4 < 5e-4 < np.ptp(legacy)


# ---- trim_excess_bands ----


def test_trim_excess_bands_deprecated_and_retrims():
    edata = _edata("N2_bare_min")
    e_tj = np.array(edata.e_tj)
    with pytest.warns(FutureWarning, match="deprecated"):
        edata.trim_excess_bands = True
    assert edata.nbands == edata.nproj == edata.e_tj.shape[1] == edata.proj_tju.shape[1]
    trimmed = np.array(edata.e_tj)
    # Used to raise IndexError: the reloaded projections were trimmed again, but e_tj/occ_tj already had been
    edata.los_orbs = False
    assert np.array_equal(edata.e_tj, trimmed) and edata.proj_tju.shape[1] == edata.nproj
    edata.los_orbs = True
    edata.trim_excess_bands = False
    assert np.array_equal(edata.e_tj, e_tj) and edata.nbands == edata.proj_tju.shape[1] == e_tj.shape[1]
    assert edata.occ_tj.shape == e_tj.shape


# ---- fallback k-point weights ----


@pytest.mark.parametrize("nspin", [1, 2])
def test_arbitrary_wk_sums_to_two(nspin):
    from crawfish.io.data_parsing import _get_arbitrary_wk, get_wk_t

    assert np.sum(_get_arbitrary_wk(6, nspin)) == pytest.approx(2.0)
    assert np.sum(_get_arbitrary_wk(6, nspin, legacy=True)) == pytest.approx(nspin)  # pre-F10
    assert np.sum(get_wk_t(None, nspin, [3, 2, 1], True)) == pytest.approx(2.0)
    assert np.sum(get_wk_t(None, nspin, [3, 2, 1], True, legacy=True)) == pytest.approx(nspin)
