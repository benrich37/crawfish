"""Tests for occupations computed without a fillings file (notes/theory_audit.md, finding F8).

Every fixture below dumped `fillings`, so the computed occupations can be compared against JDFTx's. See
tests/files/io/example_calc_dirs/Al_fcc_Fermi/README.md.
"""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
SMEARINGS = ["Fermi", "Gauss", "MP1", "Cold"]


def _from_file_and_computed(calc_dir):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    from_file = edata.occ_tj.copy()
    edata.fillingsfile_filepath = None
    edata._occ_tj = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # includes the electron-count check
        computed = edata.occ_tj
    return edata, from_file, computed


@pytest.mark.parametrize("smearing", SMEARINGS)
def test_smeared_occupations_match_jdftx(smearing):
    edata, from_file, computed = _from_file_and_computed(calc_dirs / f"Al_fcc_{smearing}")
    assert edata.outfile.broadening_type == smearing
    assert np.sum((from_file > 1e-3) & (from_file < 1 - 1e-3)) >= 4  # the metal has fractional fillings
    assert np.allclose(computed, from_file, atol=5e-5)
    assert float(np.sum(edata.wk_t[:, None] * computed)) == pytest.approx(3.0, abs=1e-3)


@pytest.mark.parametrize("name", ["Si_nosmear_nospin", "Si_nosmear_zspin_M2"])
def test_constant_occupations_match_jdftx(name):
    edata, from_file, computed = _from_file_and_computed(calc_dirs / name)
    assert edata.outfile.broadening_type is None and edata.outfile.mu is None  # no smearing, no mu printed
    assert np.array_equal(computed, from_file)


def test_constant_occupations_are_per_spin_channel():
    # M = 2: 5 electrons per spin-up state, 3 per spin-down state, although the spin-up HOMO lies above the
    # spin-down LUMO
    edata, _, computed = _from_file_and_computed(calc_dirs / "Si_nosmear_zspin_M2")
    nk = edata.nstates // 2
    assert np.all(computed[:nk].sum(axis=1) == 5) and np.all(computed[nk:].sum(axis=1) == 3)
    homo_up = edata.e_tj[:nk][computed[:nk] > 0].max()
    lumo_down = edata.e_tj[nk:][computed[nk:] == 0].min()
    assert homo_up > lumo_down


def test_helpers_take_arrays_and_scalars():
    from crawfish.core.elecdata import SMEARING_FUNCS, calculate_filling_nobroad

    eig = np.linspace(-0.3, 0.3, 61)
    for name, func in SMEARING_FUNCS.items():
        arr = func(0.01, eig, 0.0)
        assert np.allclose(arr, [func(0.01, float(e), 0.0) for e in eig])
        assert arr[0] == pytest.approx(1.0, abs=1e-6) and arr[-1] == pytest.approx(0.0, abs=1e-6)
        if name != "cold":  # cold smearing is asymmetric about mu
            assert func(0.01, 0.0, 0.0) == pytest.approx(0.5)
    # The old np.heaviside(eig, efermi) gave [0, 0, 1, 1] here
    assert np.array_equal(calculate_filling_nobroad(0.0, np.array([-0.5, -0.1, 0.1, 0.5]), 0.0), [1, 1, 0, 0])
    assert np.array_equal(calculate_filling_nobroad(0.0, np.array([0.1, 0.2, 0.3]), 0.2), [1, 0.5, 0])


def test_broadening_type_accepts_jdftx_spellings():
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dirs / "Al_fcc_Cold")
    for value in ["Fermi", "Gauss", "MP1", "Cold", "cold", "none", None]:
        edata.broadening_type = value
    with pytest.raises(ValueError):
        edata.broadening_type = "Lorentz"


def test_mu_without_smearing_raises_clear_error():
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dirs / "Si_nosmear_nospin")
    with pytest.raises(ValueError, match="No chemical potential"):
        _ = edata.mu
    edata.mu = 0.24  # e.g. from eigStats
    assert edata.mu == pytest.approx(0.24)
