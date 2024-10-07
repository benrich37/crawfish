from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR, approx_idx, get_pocket_idx
from crawfish.funcs.general import get_generic_gsmear_spectrum
import numpy as np
import pytest


def test_get_generic_gsmear_spectrum():
    edata = ElecData(EXAMPLE_CALC_DIRS_DIR / "N2_bare_min")
    sabcj = [edata.nspin] + list(edata.kfolding) + [edata.nbands]
    weights_sabcj = np.ones(sabcj, dtype=REAL_DTYPE)
    sig = 0.0001
    res = 0.01
    erange1, spectrum1 = get_generic_gsmear_spectrum(edata, weights_sabcj, None, False, sig, res=res)
    erange2, spectrum2 = get_generic_gsmear_spectrum(edata, weights_sabcj, None, True, sig, res=res)
    erange3, spectrum3 = get_generic_gsmear_spectrum(edata, weights_sabcj, None, False, sig * 2, res=res)
    assert len(erange1) == len(erange2)
    assert len(spectrum1.shape) == 1
    assert len(spectrum2.shape) == 2
    for eig in edata.e_sabcj.flatten():
        idx = approx_idx(erange1, eig)
        assert spectrum1[idx] >= 1
        assert spectrum3[idx] >= 1
        assert spectrum3[idx + 1] >= spectrum1[idx + 1]
    assert spectrum1[get_pocket_idx(edata.e_sabcj, erange1)] == pytest.approx(0)
