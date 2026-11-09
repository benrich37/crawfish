"""Tests for the JDFTx orbital phase correction (notes/theory_audit.md, finding F1)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import numpy as np
import pytest

excddir = Path(EXAMPLE_CALC_DIRS_DIR)
n2_dir = excddir / "N2_bare_min"
# N2 along x. Orbitals per atom in JDFTx order: s, py, pz, px
S_A, PX_B = 0, 7


def _gamma_state(edata):
    return [t for t in range(edata.nstates) if np.allclose(edata.ks_t[t], 0)][0]


def _get_edata(legacy: bool, legacy_matrices: bool = False):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(n2_dir)
    edata.legacy_orbital_phase = legacy
    edata.legacy_matrices = legacy_matrices
    return edata


def test_orb_l_u_from_header():
    from crawfish.io.data_parsing import get_orb_l_u_from_bandfile_filepath

    l_u = get_orb_l_u_from_bandfile_filepath(n2_dir / "bandProjections")
    assert list(l_u) == [0, 1, 1, 1, 0, 1, 1, 1]


def test_apply_jdftx_orbital_phase():
    from crawfish.core.operations.matrix import apply_jdftx_orbital_phase
    from crawfish.utils.typing import COMPLEX_DTYPE

    proj_tju = np.ones([1, 1, 4], dtype=COMPLEX_DTYPE)
    out = apply_jdftx_orbital_phase(proj_tju, np.array([0, 1, 2, 3]))
    assert out.dtype == COMPLEX_DTYPE
    assert np.array_equal(out[0, 0], np.array([1, 1j, -1, -1j], dtype=COMPLEX_DTYPE))


@pytest.mark.parametrize("legacy", [False, True])
def test_gamma_sp_products(legacy):
    # At Gamma the KS states are real up to a per-band phase, so for real orbitals T*_s T_p must be real.
    # The raw JDFTx projections make it purely imaginary instead.
    edata = _get_edata(legacy)
    t = _gamma_state(edata)
    prod = np.conj(edata.proj_tju[t, :, S_A]) * edata.proj_tju[t, :, PX_B]
    assert np.max(np.abs(prod)) > 0.1
    if legacy:
        assert np.max(np.abs(prod.real)) < 1e-4
    else:
        assert np.max(np.abs(prod.imag)) < 1e-4


def test_pdos_unaffected():
    edata_new = _get_edata(False)
    edata_old = _get_edata(True)
    assert np.allclose(np.abs(edata_new.proj_tju), np.abs(edata_old.proj_tju))


def test_proj_tju_cache_holds_raw_projections():
    from crawfish.io.data_parsing import get_proj_tju_from_file

    edata = _get_edata(False)
    cached = np.load(edata.proj_tju_filepath)
    assert np.array_equal(cached, get_proj_tju_from_file(edata.bandfile_filepath))


# Values recorded before the fix (notes/theory_audit.md section 6, crawfish 3efe98f, default settings).
# The legacy path must keep reproducing them. They predate F2 too, so the legacy matrices are used as well.
LEGACY_N2_VALUES = {
    ("s", "s"): (-0.22098, 0.40719),
    ("s", "px"): (0.0, 0.0),
    ("px", "px"): (0.15313, -0.58770),
    ("py", "py"): (-0.11463, 0.99778),
}


@pytest.mark.parametrize("orbs", list(LEGACY_N2_VALUES))
def test_legacy_reproduces_recorded_values(orbs):
    from crawfish.funcs.pcohp import get_ipcohp, get_ipcomo

    edata = _get_edata(True, legacy_matrices=True)
    o1, o2 = orbs
    kwargs = {"idcs1": 0, "idcs2": 1, "orbs1": o1, "orbs2": o2, "use_cache": False}
    ipcohp = float(np.ravel(get_ipcohp(edata, **kwargs)[1])[-1])
    ipcomo = float(np.ravel(get_ipcomo(edata, **kwargs)[1])[-1])
    assert ipcohp == pytest.approx(LEGACY_N2_VALUES[orbs][0], abs=2e-5)
    assert ipcomo == pytest.approx(LEGACY_N2_VALUES[orbs][1], abs=2e-5)


def test_sp_interaction_present_after_fix():
    from crawfish.funcs.pcohp import get_ipcomo

    edata = _get_edata(False)
    kwargs = {"idcs1": 0, "idcs2": 1, "orbs1": "s", "orbs2": "px", "use_cache": False}
    assert abs(float(np.ravel(get_ipcomo(edata, **kwargs)[1])[-1])) > 0.1
