from crawfish.core.elecdata import ElecData
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR, approx_idx
from crawfish.funcs.pdos import get_pdos
import numpy as np
import pytest


def test_get_pdos():
    edata = ElecData(EXAMPLE_CALC_DIRS_DIR / "N2_bare_min")
    sig = 0.0001
    res = 0.001
    erange, spectrum = get_pdos(edata, idcs=0, orbs="s", sig=sig, res=res)
    eigs = edata.e_sabcj.flatten()
    orb = edata.orbs_idx_dict["N #1"][0]
    proj_sabcj = edata.proj_sabcju[:, :, :, :, :, orb]
    w_sabcj = np.conj(proj_sabcj) * proj_sabcj
    wk_sabcj = np.moveaxis(np.array([edata.wk_sabc] * edata.nbands), 0, -1)
    weights_sabcj = w_sabcj * wk_sabcj
    weights = weights_sabcj.flatten()
    for i, eig in enumerate(eigs):
        idx = approx_idx(erange, eig)
        enval = spectrum[idx]
        wval = weights[i]
        if pytest.approx(enval) == 0:
            assert pytest.approx(wval) == 0
        else:
            assert enval >= wval
