"""Tests for the pCOBI scale: the Wiberg/Mayer bond index (finding F16, legacy_cobi_scale)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
# N2_nospin: no-spin, 2x2x2; N2_bare_min: the same molecule, z-spin 3x3x3 (solvated); graphene: z-spin, full mesh
FIXTURES = ["N2_nospin", "N2_bare_min", "graphene_zspin_full"]


def _get_edata(name, **settings):
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # graphene's too-few-bands and tied-image warnings
        edata = ElecData(calc_dirs / name)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata


def _ipcobi(edata, **kwargs):
    from crawfish.funcs import get_ipcobi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.ravel(get_ipcobi(edata, use_cache=False, **kwargs)[1])[-1])


def _wiberg_mayer(edata, a, b):
    """Float64 bond index of atoms a and b (nearest image), from the raw projections and occupations.

    Löwdin per state with crawfish's cutoff, D_s(R) = sum_{t in s} w_t e^{-2 pi i k.R} D(t) / sum_{t in s} w_t with
    D(t) = sum_j f C C^dag; no-spin: Wiberg sum |2 D_uv|^2, z-spin: Mayer 2 sum_s |D^s_uv|^2.
    """
    from crawfish.core.operations.matrix import LOWDIN_RTOL
    from crawfish.funcs import get_pair_images
    from crawfish.io.data_parsing import get_proj_tju_from_file
    from crawfish.utils.indexing import get_orb_idcs

    P = get_proj_tju_from_file(edata.bandfile_filepath).astype(np.complex128) * (1j ** edata.orb_l_u)
    C = np.empty_like(P)
    for t in range(P.shape[0]):
        eigs, vecs = np.linalg.eigh(P[t].conj().T @ P[t])
        keep = eigs > LOWDIN_RTOL * eigs.max()
        C[t] = P[t] @ (vecs[:, keep] / np.sqrt(eigs[keep])) @ vecs[:, keep].conj().T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = get_pair_images(edata, idcs1=a, idcs2=b)[0]["R"]
    w, k = np.asarray(edata.wk_t, np.float64), np.asarray(edata.ks_t, np.float64)
    D_t = np.einsum("tj,tju,tjv->tuv", np.asarray(edata.occ_tj, np.float64), C, C.conj())
    wph = w * np.exp(-2j * np.pi * k @ np.asarray(R, float))
    nk = len(w) // edata.nspin
    sl = [slice(s * nk, (s + 1) * nk) for s in range(edata.nspin)]
    D_s = [np.tensordot(wph[s], D_t[s], axes=1) / w[s].sum() for s in sl]
    blk = np.ix_(get_orb_idcs(edata, a, None, None), get_orb_idcs(edata, b, None, None))
    if edata.nspin == 1:
        return float(np.sum(np.abs(2 * D_s[0][blk]) ** 2))
    return float(2 * sum(np.sum(np.abs(d[blk]) ** 2) for d in D_s))


@pytest.mark.parametrize("name", FIXTURES)
def test_ipcobi_is_wiberg_mayer_index(name):
    edata = _get_edata(name)
    assert _ipcobi(edata, idcs1=0, idcs2=1) == pytest.approx(_wiberg_mayer(edata, 0, 1), rel=1e-5)


@pytest.mark.parametrize("name", FIXTURES)
def test_legacy_scale_is_half(name):
    from crawfish.funcs.pcohp import _get_pcobi_tj

    edata = _get_edata(name)
    fixed = _get_pcobi_tj(edata, 0, None, None, 1, None, None, use_cache=False)
    edata.legacy_cobi_scale = True
    legacy = _get_pcobi_tj(edata, 0, None, None, 1, None, None, use_cache=False)
    assert np.array_equal(fixed, 2 * legacy)
    assert _ipcobi(edata, idcs1=0, idcs2=1) == pytest.approx(_wiberg_mayer(edata, 0, 1) / 2, rel=1e-5)


def _h2(nspin):
    """Closed-shell H2, minimal orthonormal basis: occupied band (1, 1)/sqrt2, empty band (1, -1)/sqrt2."""
    from crawfish.core.operations.matrix import get_h_s_d_t_uv
    from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE

    coef = np.repeat((np.array([[1, 1], [1, -1]], dtype=COMPLEX_DTYPE) / np.sqrt(2))[None], nspin, axis=0)
    occ = np.repeat(np.array([[1.0, 0.0]], dtype=REAL_DTYPE), nspin, axis=0)
    e = np.repeat(np.array([[-0.5, 0.3]], dtype=REAL_DTYPE), nspin, axis=0)
    _, _, d = get_h_s_d_t_uv(coef, coef, e, occ)
    # One no-spin state of weight 2, or two spin channels of weight 1
    return coef, d, np.full(nspin, 2.0 / nspin, dtype=REAL_DTYPE), occ


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("legacy", [False, True])
def test_h2_single_bond_index_is_one(nspin, legacy):
    from crawfish.funcs.pcohp import _compute_pcobi_tj

    coef, d, wk, occ = _h2(nspin)
    edata = SimpleNamespace(coef_tju=coef, d_t_uv=d, wk_t=wk, legacy_matrices=False, legacy_cobi_scale=legacy)
    bond_index = float(np.sum(occ * _compute_pcobi_tj(edata, [0], [1], "all")))
    assert bond_index == pytest.approx(0.5 if legacy else 1.0, abs=1e-6)


def test_nospin_and_zspin_n2_agree():
    # The same molecule computed spin-unpolarized and spin-polarized gives the same bond index: about 3 (triple bond)
    nospin, zspin = _ipcobi(_get_edata("N2_nospin"), idcs1=0, idcs2=1), _ipcobi(_get_edata("N2_bare_min"), idcs1=0, idcs2=1)
    assert nospin == pytest.approx(zspin, rel=0.01)
    assert nospin == pytest.approx(2.98, abs=0.01)


@pytest.mark.parametrize("name", ["N2_nospin", "graphene_zspin_full"])
def test_every_orbital_pair_is_non_negative(name):
    edata = _get_edata(name)
    labels = edata.atom_orb_labels_dict[edata.ion_names[0]]
    vals = [_ipcobi(edata, idcs1=0, orbs1=o1, idcs2=1, orbs2=o2) for o1 in labels for o2 in labels]
    assert min(vals) >= -1e-6
    assert sum(vals) == pytest.approx(_ipcobi(edata, idcs1=0, idcs2=1), rel=1e-5)


def test_setting_is_part_of_cache_key():
    from crawfish.core.elecdata import get_edata_settings_str
    from crawfish.funcs import get_ipcobi

    edata = _get_edata("N2_nospin")
    assert "lgcb_False" in get_edata_settings_str(edata)
    # Weight caches are per setting, so the cached default weights aren't returned for the legacy scale
    fixed = float(np.ravel(get_ipcobi(edata, idcs1=0, idcs2=1, use_cache=True)[1])[-1])
    edata.legacy_cobi_scale = True
    assert edata.cache_sub_dir.name == get_edata_settings_str(edata) and "lgcb_True" in edata.cache_sub_dir.name
    assert float(np.ravel(get_ipcobi(edata, idcs1=0, idcs2=1, use_cache=True)[1])[-1]) == pytest.approx(fixed / 2)


def test_legacy_matrices_path_unchanged():
    # The legacy (pre-F2) matrices aren't rescaled: legacy_cobi_scale has no effect there
    edata = _get_edata("N2_bare_min", legacy_matrices=True)
    before = _ipcobi(edata, idcs1=0, idcs2=1)
    edata.legacy_cobi_scale = True
    assert _ipcobi(edata, idcs1=0, idcs2=1) == pytest.approx(before, rel=1e-6)
