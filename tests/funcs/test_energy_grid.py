"""Tests for the default energy grid and its padding (finding F17)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest
from scipy.integrate import trapezoid

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
OLD_RES = np.float32(0.01)  # the default grid step before F17
NOCACHE = dict(use_cache=False, use_cached_spectrum=False, save_spectrum=False)


def _get_edata(name, **settings):
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edata = ElecData(calc_dirs / name)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata


def _grid_warnings(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = func(*args, **kwargs)
    return out, [str(w.message) for w in caught if "grid step" in str(w.message)]


def _mid_gap(edata):
    e, f = np.asarray(edata.e_tj, np.float64), np.asarray(edata.occ_tj, np.float64)
    return 0.5 * (e[f > 0.5].max() + e[f < 0.5].min())


@pytest.mark.parametrize("name", ["N2_nospin", "graphene_zspin_full"])
def test_default_dos_integrates_to_state_count(name):
    from crawfish.funcs import get_dos

    edata = _get_edata(name)
    (erange, dos), msgs = _grid_warnings(get_dos, edata)
    assert msgs == []  # the defaults are fine enough
    assert trapezoid(dos, erange) == pytest.approx(float(np.sum(edata.wk_t)) * edata.nbands, rel=1e-3)


def test_default_pcohp_integrates_to_exact_sum():
    from crawfish.funcs import get_pcohp, get_ipcohp

    edata = _get_edata("N2_nospin")
    e_cut = _mid_gap(edata)
    es, cs = get_ipcohp(edata, idcs1=0, idcs2=1, use_fillings=False, use_cache=False)
    exact = float(np.asarray(cs)[np.asarray(es) < e_cut][-1])
    erange, spec = get_pcohp(edata, idcs1=0, idcs2=1, **NOCACHE)
    below = erange < e_cut
    assert trapezoid(spec[below], erange[below]) == pytest.approx(exact, abs=1e-4)
    # On the pre-F17 default grid, the same integral was -1.0215 Ha instead of -0.8668 Ha (check_V3)
    erange, spec = get_pcohp(edata, idcs1=0, idcs2=1, res=OLD_RES, **NOCACHE)
    below = erange < e_cut
    assert trapezoid(spec[below], erange[below]) == pytest.approx(-1.021535, abs=1e-5)


@pytest.mark.parametrize("name", ["N2_nospin", "graphene_zspin_full"])
def test_old_res_reproduces_old_grid(name):
    from crawfish.funcs import get_dos, get_pcohp

    edata = _get_edata(name)
    # The pre-F17 automatic grid: 10 res beyond the extreme eigenvalues
    old_grid = np.arange(np.min(edata.e_sabcj) - 10 * OLD_RES, np.max(edata.e_sabcj) + 10 * OLD_RES, OLD_RES,
                         dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        erange, dos = get_dos(edata, res=OLD_RES)
        assert np.array_equal(erange, old_grid)
        assert np.array_equal(dos, get_dos(edata, erange=old_grid)[1])
        erange, spec = get_pcohp(edata, idcs1=0, idcs2=1, res=OLD_RES, **NOCACHE)
        assert np.array_equal(erange, old_grid)
        assert np.array_equal(spec, get_pcohp(edata, idcs1=0, idcs2=1, erange=old_grid, **NOCACHE)[1])


def test_coarse_grid_warns():
    from crawfish.funcs import get_dos, get_pdos

    edata = _get_edata("N2_nospin")
    _, msgs = _grid_warnings(get_dos, edata, res=OLD_RES)
    assert len(msgs) == 1 and "sig" in msgs[0] and "res" in msgs[0] and "erange" in msgs[0]
    # Also when the spectrum comes from the cache
    get_pdos(edata, idcs=0, res=OLD_RES, save_spectrum=True)
    _, msgs = _grid_warnings(get_pdos, edata, idcs=0, res=OLD_RES)
    assert len(msgs) == 1
    # A wider Gaussian on the same grid is fine, and LTI has no Gaussian
    _, msgs = _grid_warnings(get_dos, edata, res=OLD_RES, sig=np.float32(2e-4))
    assert msgs == []


def test_padding_keeps_outermost_peaks():
    from crawfish.funcs import get_dos
    from crawfish.funcs.general import SIGMA_DEFAULT
    from crawfish.utils.arg_correction import get_erange

    edata = _get_edata("N2_nospin")
    std = np.sqrt(SIGMA_DEFAULT / 2)
    for res in (1e-3, 1e-4):
        erange = get_erange(edata, None, res=np.float32(res), sig=SIGMA_DEFAULT)
        assert np.min(edata.e_tj) - erange[0] == pytest.approx(5 * std, rel=1e-3)
    # LTI (no sig): 10 res
    erange = get_erange(edata, None, res=np.float32(1e-4))
    assert np.min(edata.e_tj) - erange[0] == pytest.approx(1e-3, rel=1e-3)
    # Before F17, res=1e-4 padded by 0.45 standard deviations and the DOS integral was 0.976 of the state count
    erange, dos = get_dos(edata, res=np.float32(1e-4))
    assert trapezoid(dos, erange) == pytest.approx(float(np.sum(edata.wk_t)) * edata.nbands, rel=1e-4)


def test_returned_grid_matches_cached_spectrum():
    # The cache path, the spectrum and the returned erange come from one grid, also when the spectrum is cached
    from crawfish.funcs import get_pcohp

    edata = _get_edata("N2_nospin")
    for res in (None, OLD_RES):
        kw = {} if res is None else {"res": res}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e1, s1 = get_pcohp(edata, idcs1=0, idcs2=1, use_cache=False, **kw)
            e2, s2 = get_pcohp(edata, idcs1=0, idcs2=1, use_cache=False, **kw)  # from the spectrum cache
        assert np.array_equal(e1, e2) and np.array_equal(s1, s2) and len(e2) == len(s2)
    # The two grids are different cache entries
    assert len(get_pcohp(edata, idcs1=0, idcs2=1, use_cache=False)[0]) > 5 * len(
        get_pcohp(edata, idcs1=0, idcs2=1, use_cache=False, res=OLD_RES)[0])
