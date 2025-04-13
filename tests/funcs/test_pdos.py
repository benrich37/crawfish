from crawfish.core.elecdata import ElecData
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR, approx_idx
from crawfish.utils.typing import REAL_DTYPE
from crawfish.funcs.pdos import get_pdos
import numpy as np
import pytest


n2_calcdir = EXAMPLE_CALC_DIRS_DIR / "N2_bare_min"
edata = ElecData(n2_calcdir)
erange, pdos = get_pdos(edata, idcs=0)


def test_get_pdos_tj():
    from crawfish.core.operations.matrix import get_pdos_tj

    nproj = 3
    orbs_1 = np.array(list(range(nproj)))
    orbs_2 = np.array([0,int(nproj-1)])
    nstates = 3
    nbands = 3

    proj_tju = (np.random.random([nstates, nbands, nproj]) - 0.5) + 1j*(np.random.random([nstates, nbands, nproj]) - 0.5)
    wk_t = np.random.random([nstates])
    pdos_tj_1 = get_pdos_tj(proj_tju, orbs_1, wk_t)
    pdos_tj_2 = get_pdos_tj(proj_tju, orbs_2, wk_t)
    for t in range(nstates):
        for j in range(nbands):
            assert np.isclose(
                pdos_tj_1[t,j], 
                np.sum((np.abs(proj_tju[t,j,orbs_1])**2).flatten())*wk_t[t]
            )
            assert np.isclose(
                pdos_tj_2[t,j], 
                np.sum((np.abs(proj_tju[t,j,orbs_2])**2).flatten())*wk_t[t]
            )


def test_get_pdos():
    edata = ElecData(EXAMPLE_CALC_DIRS_DIR / "N2_bare_min")
    edata._wk_sabc = np.ones(np.shape(edata.wk_sabc), dtype=REAL_DTYPE)
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
