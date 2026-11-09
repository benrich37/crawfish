"""Tests for the signed projected matrices (notes/theory_audit.md, finding F2)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

n2_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
PAIRS = [("s", "s"), ("s", "px"), ("px", "px"), ("py", "py")]


def _get_edata(**settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(n2_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _ipcohp(edata, o1, o2, state_sep=False):
    from crawfish.funcs.pcohp import _get_pcohp_tj

    weights_tj = _get_pcohp_tj(edata, 0, None, o1, 1, None, o2, use_cache=False, state_sep=state_sep)
    return float(np.sum(weights_tj * edata.occ_tj))


def _independent_reference(edata, orthonormal: bool, shift: float):
    """Reference matrices from the raw projections, in float64 (independent of the crawfish kernels)."""
    from crawfish.io.data_parsing import get_proj_tju_from_file

    P = get_proj_tju_from_file(edata.bandfile_filepath).astype(np.complex128) * (1j ** edata.orb_l_u)
    if orthonormal:
        for t in range(P.shape[0]):
            eigs, vecs = np.linalg.eigh(P[t].conj().T @ P[t])
            P[t] = P[t] @ (vecs / np.sqrt(eigs)) @ vecs.conj().T
        C = P
    else:
        C = np.einsum("tjw,twu->tju", P, np.linalg.inv(np.einsum("tju,tjv->tuv", P.conj(), P)))
    H = np.einsum("tju,tj,tjv->tuv", P, edata.e_tj - shift, P.conj())
    return C, H


def test_matrices_hermitian_and_orthonormal():
    edata = _get_edata()
    for name in ["h_t_uv", "s_t_uv", "d_t_uv"]:
        mat = getattr(edata, name)
        assert np.iscomplexobj(mat)
        assert np.allclose(mat, mat.conj().transpose(0, 2, 1))
    # Löwdin basis (default): band-space overlap is the identity
    assert np.allclose(edata.s_t_uv, np.eye(edata.nproj)[None], atol=1e-5)


def test_band_energy_sum_rule():
    # sum over all orbital pairs (including on-site) of the integrated COHP ~ band energy, up to spilling (~0.3% here)
    edata = _get_edata(norm_eigs_for_h=False)
    C, H, w, f = edata.coef_tju, edata.h_t_uv, edata.wk_t, edata.occ_tj
    total = np.einsum("t,tj,tju,tuv,tjv->", w, f, C.conj(), H, C).real
    band_energy = np.sum(w[:, None] * f * edata.e_tj)
    assert total / band_energy == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("orthonormal", [True, False])
@pytest.mark.parametrize("norm_eigs_for_h", [True, False])
def test_matches_independent_reference(orthonormal, norm_eigs_for_h):
    edata = _get_edata(los_orbs=orthonormal, norm_eigs_for_h=norm_eigs_for_h)
    C, H = _independent_reference(edata, orthonormal, edata.mu if norm_eigs_for_h else 0.0)
    for o1, o2 in PAIRS:
        u = edata.orbs_idx_dict["N #1"][["s", "py", "pz", "px"].index(o1)]
        v = edata.orbs_idx_dict["N #2"][["s", "py", "pz", "px"].index(o2)]
        expected = np.sum(edata.wk_t[:, None] * edata.occ_tj * (C[:, :, u].conj() * H[:, u, v][:, None] * C[:, :, v]).real)
        assert _ipcohp(edata, o1, o2, state_sep=True) == pytest.approx(expected, abs=2e-5), (o1, o2)


def test_n2_pairs_are_bonding():
    # Every N-N orbital pair in N2 is net bonding (negative ICOHP). Legacy matrices give px-px antibonding.
    edata = _get_edata()
    for o1, o2 in PAIRS:
        assert _ipcohp(edata, o1, o2) < -0.05, (o1, o2)
    assert _ipcohp(_get_edata(legacy_matrices=True), "px", "px") > 0


def test_r0_vs_all_images_isolated_molecule():
    # N2 is isolated in a large box, so the R=0 element carries essentially all of the interaction
    edata = _get_edata()
    for o1, o2 in PAIRS:
        assert _ipcohp(edata, o1, o2, state_sep=False) == pytest.approx(_ipcohp(edata, o1, o2, state_sep=True), abs=2e-4)


def test_coop_warnings_and_zero_in_lowdin_basis():
    from crawfish.funcs import get_ipcoop

    edata = _get_edata()
    with pytest.warns(UserWarning, match="identically zero"):
        _, icoop = get_ipcoop(edata, idcs1=0, idcs2=1)
    assert np.max(np.abs(icoop)) < 1e-4
    with pytest.warns(UserWarning, match="orthonormalized orbitals"):
        get_ipcoop(_get_edata(los_orbs=False), idcs1=0, idcs2=1)


# Values with F1 fixed and the legacy matrices (crawfish a22c558, default settings); the legacy path must keep them
LEGACY_MATRICES_N2 = {("s", "s"): -0.12203, ("s", "px"): -0.08856, ("px", "px"): 0.08452, ("py", "py"): -0.11462}


@pytest.mark.parametrize("orbs", list(LEGACY_MATRICES_N2))
def test_legacy_matrices_reproduce_recorded_values(orbs):
    from crawfish.funcs import get_ipcohp

    edata = _get_edata(legacy_matrices=True)
    _, cs = get_ipcohp(edata, idcs1=0, idcs2=1, orbs1=orbs[0], orbs2=orbs[1], use_cache=False)
    assert float(np.ravel(cs)[-1]) == pytest.approx(LEGACY_MATRICES_N2[orbs], abs=2e-5)
