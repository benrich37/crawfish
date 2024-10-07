from crawfish.core.elecdata import ElecData
from crawfish.core.operations.matrix import _add_kweights
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR, approx_idx
from crawfish.funcs.pcohp import get_pcohp
import numpy as np
import pytest


def test_get_pcohp():
    edata = ElecData(EXAMPLE_CALC_DIRS_DIR / "N2_bare_min")
    sig = 0.0001
    res = 0.001
    erange, spectrum = get_pcohp(edata, idcs1=0, orbs1="s", idcs2=1, orbs2="s", sig=sig, res=res)
    eigs = edata.e_sabcj.flatten()
    orb1 = edata.orbs_idx_dict["N #1"][0]
    orb2 = edata.orbs_idx_dict["N #2"][0]
    weights_jsabc = edata.p_uvjsabc[orb1, orb2]
    weights_sabcj = _add_kweights(np.moveaxis(weights_jsabc, 0, -1), edata.wk_sabc)
    weights = weights_sabcj.flatten()
    for i, eig in enumerate(eigs):
        idx = approx_idx(erange, eig)
        enval = spectrum[idx]
        wval = weights[i]
        if pytest.approx(enval) == 0:
            assert pytest.approx(wval) == 0
        elif enval < 0:
            assert enval <= wval
        else:
            assert enval >= wval
    erange2, spectrum2 = get_pcohp(edata, idcs1=0, orbs1="s", idcs2=1, orbs2="s", sig=sig, res=res, lite=True)
    assert np.allclose(spectrum, spectrum2)
