"""Tests for the too-few-bands diagnostic and warning (notes/theory_audit.md, finding F11)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)


def _load(name, **settings):
    """Return the ElecData and the too-few-bands warnings raised while loading it."""
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edata = ElecData(calc_dirs / name)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata, [str(w.message) for w in caught if "Too few bands" in str(w.message)]


def _graphene_bond_ipcohp(edata):
    # Mean all-orbital IpCOHP of C1's three nearest C2 images
    from crawfish.funcs import get_ipcohp

    st = edata.structure
    d = {R: np.linalg.norm(st.lattice.get_cartesian_coords(st.frac_coords[1] + np.array(R) - st.frac_coords[0]))
         for R in [(a, b, 0) for a in (-1, 0, 1) for b in (-1, 0, 1)]}
    nearest = sorted(d, key=d.get)[:3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.mean([get_ipcohp(edata, idcs1=0, idcs2=1, image=list(R))[1][-1] for R in nearest])


@pytest.mark.parametrize(
    "name", ["N2_bare_min", "N2_nospin", "N2_nospin_nonortho15", "N2_nospin_nonortho40", "Al_fcc_Fermi",
             "Si_nosmear_nospin", "graphene_zspin_tr_30bands"]
)
def test_no_warning_when_bands_span_orbitals(name):
    edata, msgs = _load(name)
    assert msgs == []
    assert edata.orbital_span_t.min() > 1e-3


@pytest.mark.parametrize(
    ("name", "nbad", "orbitals"),
    [("N2_nospin_10bands", 5, "N px, N s"), ("graphene_zspin_tr", 4, "C px, C py"),
     ("graphene_zspin_full", 4, "C py, C px"), ("graphene_1x1", 830, "C px, C py")],
)
def test_warning_names_states_and_orbitals(name, nbad, orbitals):
    edata, msgs = _load(name)
    assert len(msgs) == 1
    assert f"at {nbad} of {edata.nstates} states" in msgs[0] and f"mostly {orbitals}." in msgs[0]
    assert np.sum(edata.orbital_span_t < 1e-4) == nbad


def test_warning_tracks_band_convergence():
    # The warned runs are measurably unconverged in the number of bands; the quiet ones are within ~1-2%
    from crawfish.funcs import get_ipcohp

    def ipcohp(name):
        edata, msgs = _load(name)
        return get_ipcohp(edata, idcs1=0, idcs2=1)[1][-1], bool(msgs)

    (n2_10, warned_10), (n2_15, warned_15) = ipcohp("N2_nospin_10bands"), ipcohp("N2_nospin")
    assert warned_10 and not warned_15
    assert n2_10 == pytest.approx(-0.8497, abs=2e-4) and n2_15 == pytest.approx(-0.8668, abs=2e-4)
    gr20, gr30 = _load("graphene_zspin_tr"), _load("graphene_zspin_tr_30bands")
    assert gr20[1] and not gr30[1]
    bond20, bond30 = _graphene_bond_ipcohp(gr20[0]), _graphene_bond_ipcohp(gr30[0])
    assert bond20 == pytest.approx(-0.5013, abs=2e-4) and bond30 == pytest.approx(-0.5193, abs=2e-4)
    assert abs(bond20 / bond30 - 1) > 0.03


def test_orbital_span_is_independent_of_settings():
    edata, _ = _load("graphene_zspin_tr")
    span = np.array(edata.orbital_span_t)
    for key, val in [("los_orbs", False), ("legacy_lowdin_cutoff", True), ("legacy_orbital_phase", True)]:
        other, _ = _load("graphene_zspin_tr", **{key: val})
        assert np.array_equal(other.orbital_span_t, span)
    # Unfolded onto the full mesh: the same as the calculation without any k-point reduction
    full, _ = _load("graphene_zspin_full")
    assert edata.nstates == full.nstates
    assert np.allclose(np.maximum(span, 1e-4), np.maximum(full.orbital_span_t, 1e-4), atol=1e-3)
    # Ortho-orbital projections: the true overlap is the identity, so with enough bands the span approaches 1
    gr30, _ = _load("graphene_zspin_tr_30bands")
    assert gr30.bandprojfile_is_ortho and gr30.orbital_span_t.min() == pytest.approx(0.93, abs=0.01)


def test_legacy_lowdin_path_reports_no_dropped_directions():
    _, msgs = _load("graphene_zspin_full")
    assert "dropped 4 of these combination" in msgs[0]
    _, msgs = _load("graphene_zspin_full", legacy_lowdin_cutoff=True)
    assert msgs and "dropped" not in msgs[-1]
