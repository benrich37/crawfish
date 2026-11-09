"""Tests for the normalization of smeared spectra (notes/theory_audit.md, finding F6)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest
from scipy.integrate import trapezoid

n2_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
graphene_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "graphene_zspin_full"


def _get_edata(calc_dir, **settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _erange(edata, step=0.0005):
    return np.arange(edata.e_tj.min() - 0.05, edata.e_tj.max() + 0.05, step, dtype=np.float32)


def _dos_integral(edata, lti, sig=None):
    from crawfish.funcs import get_dos

    np.random.seed(0)  # rattle_eigenvals is random
    er = _erange(edata)
    kwargs = {"lti": lti, "rattle_eigenvals": lti}
    if sig is not None:
        kwargs["sig"] = sig
    return float(trapezoid(np.asarray(get_dos(edata, erange=er, **kwargs)[1]), er))


@pytest.mark.parametrize("calc_dir", [n2_dir, graphene_dir])
def test_gaussian_dos_integrates_to_state_count(calc_dir):
    edata = _get_edata(calc_dir)
    expected = float(np.sum(edata.wk_t) * edata.nbands)
    assert _dos_integral(edata, lti=False) == pytest.approx(expected, rel=1e-4)
    assert _dos_integral(edata, lti=False, sig=1e-4) == pytest.approx(expected, rel=1e-4)  # any width


def test_lti_dos_integrates_to_state_count():
    edata = _get_edata(graphene_dir)  # dispersive bands; flat molecular bands are poorly resolved by LTI
    expected = float(np.sum(edata.wk_t) * edata.nbands)
    assert _dos_integral(edata, lti=True) == pytest.approx(expected, rel=1e-3)


def test_gaussian_and_lti_agree():
    from crawfish.funcs import get_pdos

    edata = _get_edata(graphene_dir)
    er = _erange(edata)
    np.random.seed(0)
    gauss = np.asarray(get_pdos(edata, idcs=[0, 1], erange=er, use_cached_spectrum=False, save_spectrum=False)[1])
    lti = np.asarray(get_pdos(edata, idcs=[0, 1], erange=er, lti=True, rattle_eigenvals=True,
                              use_cached_spectrum=False, save_spectrum=False)[1])
    assert trapezoid(gauss, er) == pytest.approx(trapezoid(lti, er), rel=2e-3)


def test_pcohp_spectrum_integrates_to_its_weights():
    # Integrating the smeared pCOHP over all energies gives the sum of its (t, j) weights (no occupations)
    from crawfish.funcs import get_pcohp
    from crawfish.funcs.pcohp import _get_pcohp_tj

    edata = _get_edata(graphene_dir)
    er = _erange(edata)
    _, spec = get_pcohp(edata, idcs1=0, idcs2=1, image="all", erange=er, use_cached_spectrum=False,
                        save_spectrum=False)
    weights = _get_pcohp_tj(edata, 0, None, None, 1, None, None, use_cache=False, image="all")
    assert trapezoid(np.asarray(spec), er) == pytest.approx(float(np.sum(weights)), rel=1e-3)


def test_legacy_scales():
    from crawfish.funcs.general import SIGMA_DEFAULT

    edata = _get_edata(graphene_dir, legacy_normalization=True)
    expected = float(np.sum(edata.wk_t) * edata.nbands)
    # Gaussian integrated to sqrt(pi * sig) per state, LTI to 1/N_k per state
    assert _dos_integral(edata, lti=False) == pytest.approx(expected * np.sqrt(np.pi * SIGMA_DEFAULT), rel=1e-4)
    assert _dos_integral(edata, lti=True) == pytest.approx(expected / np.prod(edata.kfolding), rel=1e-3)


def test_norm_max_unchanged():
    from crawfish.funcs import get_dos

    fixed, legacy = _get_edata(n2_dir), _get_edata(n2_dir, legacy_normalization=True)
    er = _erange(fixed)
    a = np.asarray(get_dos(fixed, erange=er, norm_max=True)[1])
    b = np.asarray(get_dos(legacy, erange=er, norm_max=True)[1])
    assert np.allclose(a, b, atol=1e-5)
