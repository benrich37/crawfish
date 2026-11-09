"""Tests for pCOOP and the orbital basis (notes/theory_audit.md, finding F7).

N2_nospin has projections onto orthonormalized orbitals ('band-projection-params yes no'). N2_nospin_nonortho15/40
are the same calculation with 'band-projection-params no no' and 15 / 40 bands.
"""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
ortho_dir = calc_dirs / "N2_nospin"
nonortho15_dir = calc_dirs / "N2_nospin_nonortho15"
nonortho40_dir = calc_dirs / "N2_nospin_nonortho40"
ORBS = ["s", "py", "pz", "px"]  # JDFTx order per atom; N2 lies along x


def _get_edata(calc_dir, **settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _integrated(edata, mode, o):
    from crawfish.funcs import pcohp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = getattr(pcohp, f"_get_{mode}_tj")(edata, 0, None, o, 1, None, o, use_cache=False)
    return float(np.sum(edata.occ_tj * weights))


def test_nonortho_coop_has_no_warning():
    from crawfish.funcs import get_ipcoop

    edata = _get_edata(nonortho15_dir, los_orbs=False)
    assert not edata.bandprojfile_is_ortho
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_ipcoop(edata, idcs1=0, idcs2=1, use_cache=False)


def test_nonortho_overlap_signs():
    # Textbook signs for the N1-N2 overlaps: s-s and pi (py-py) positive, sigma between parallel px negative
    edata = _get_edata(nonortho15_dir, los_orbs=False)
    S = edata.get_mat_r0_suv("s")[0].real
    i = {o: k for k, o in enumerate(ORBS)}
    assert S[i["s"], 4 + i["s"]] > 0.3
    assert S[i["py"], 4 + i["py"]] > 0.2
    assert S[i["px"], 4 + i["px"]] < -0.1
    # On-site: JDFTx normalizes <phi|O|phi> = 1, reached up to the band-space completeness
    assert np.allclose(np.diag(S), np.tile([0.986, 0.976, 0.976, 0.895], 2), atol=0.01)


def test_nonortho_coop_values_and_band_convergence():
    e15, e40 = _get_edata(nonortho15_dir, los_orbs=False), _get_edata(nonortho40_dir, los_orbs=False)
    for o in [None, "s", "px", "py"]:
        assert _integrated(e15, "pcoop", o) == pytest.approx(_integrated(e40, "pcoop", o), abs=0.03), o
    assert _integrated(e40, "pcoop", None) > 0.4  # net bonding
    assert _integrated(e40, "pcoop", "px") > 0 and _integrated(e40, "pcoop", "py") > 0
    # s-s is net antibonding in N2 (2sigma_g and 2sigma_u* both filled), in COOP and COHP alike
    assert _integrated(e40, "pcoop", "s") < 0 and _integrated(e40, "pcohp", "s") > 0


def test_pcomo_sign_depends_on_orientation():
    # px-px sigma bonding: COOP > 0 and COHP < 0, but the bare coefficient product (pCOMO) is negative
    edata = _get_edata(nonortho40_dir, los_orbs=False)
    assert _integrated(edata, "pcoop", "px") > 0
    assert _integrated(edata, "pcohp", "px") < 0
    assert _integrated(edata, "pcomo", "px") < 0


def test_coop_warnings():
    from crawfish.funcs import get_ipcoop
    from crawfish.funcs import pcohp

    with pytest.warns(UserWarning, match="identically zero"):
        get_ipcoop(_get_edata(nonortho15_dir), idcs1=0, idcs2=1, use_cache=False)  # Löwdin basis
    with pytest.warns(UserWarning, match="orthonormalized orbitals"):
        get_ipcoop(_get_edata(ortho_dir, los_orbs=False), idcs1=0, idcs2=1, use_cache=False)
    # Too few bands: px is only 0.90 represented, so a stricter threshold triggers the completeness warning
    old = pcohp.COOP_COMPLETENESS_WARN
    try:
        pcohp.COOP_COMPLETENESS_WARN = 0.95
        with pytest.warns(UserWarning, match="Increase elec-n-bands"):
            get_ipcoop(_get_edata(nonortho15_dir, los_orbs=False), idcs1=0, idcs2=1, orbs1="px", orbs2="px",
                       use_cache=False)
    finally:
        pcohp.COOP_COMPLETENESS_WARN = old


def test_orbital_completeness_independent_of_settings():
    base = _get_edata(nonortho15_dir).orbital_completeness_u
    for settings in [{"los_orbs": False}, {"legacy_orbital_phase": True}, {"legacy_matrices": True}]:
        assert np.allclose(_get_edata(nonortho15_dir, **settings).orbital_completeness_u, base)
    assert np.allclose(base, np.tile([0.986, 0.976, 0.976, 0.895], 2), atol=0.005)
    # Projections onto orthonormalized orbitals are less complete with the same 15 bands
    assert _get_edata(ortho_dir).orbital_completeness_u[3] < 0.75
