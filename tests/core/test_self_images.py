"""Tests for an atom's pair spectra with its own periodic images (finding F19)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
# Graphene primitive cell on a full 12x12x1 mesh: C #1's six second neighbours (2.46 A) are its own images
graphene_dir = calc_dirs / "graphene_zspin_full"
# Wurtzite ZnO on a full mesh: Zn #1 and its in-plane images at a = 3.25 A
zno_dir = calc_dirs / "ZnO_wurtzite_full"
NOCACHE = dict(use_cache=False)


def _get_edata(calc_dir=graphene_dir, **settings):
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # graphene's too-few-bands warning
        edata = ElecData(calc_dir)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata


def _last(result):
    return float(np.ravel(result[1])[-1])


def _reference(edata, atom, R):
    """Float64 IpCOHP and IpCOBI of `atom` with its image at R, every orbital pair including u == v.

    Raw projections with the i^l phase, Löwdin per state with crawfish's cutoff (C), H_t = C (e - mu) C^dag,
    D_t = sum_j f C C^dag, M_s(R) = sum_{t in s} w_t e^{-2 pi i k.R} M_t / sum_{t in s} w_t, and
    sum_t w_t f Re[conj(C_u) M_s(R)_uv e^{2 pi i k.R} C_v]; pCOBI carries the factor 2 of F16.
    """
    from crawfish.core.operations.matrix import LOWDIN_RTOL
    from crawfish.io.data_parsing import get_proj_tju_from_file
    from crawfish.utils.indexing import get_orb_idcs

    P = get_proj_tju_from_file(edata.bandfile_filepath).astype(np.complex128) * (1j ** edata.orb_l_u)
    C = np.empty_like(P)
    for t in range(P.shape[0]):
        eigs, vecs = np.linalg.eigh(P[t].conj().T @ P[t])
        keep = eigs > LOWDIN_RTOL * eigs.max()
        C[t] = P[t] @ (vecs[:, keep] / np.sqrt(eigs[keep])) @ vecs[:, keep].conj().T
    e, f = np.asarray(edata.e_tj, np.float64) - edata.mu, np.asarray(edata.occ_tj, np.float64)
    w, k = np.asarray(edata.wk_t, np.float64), np.asarray(edata.ks_t, np.float64)
    ph = np.exp(2j * np.pi * k @ np.asarray(R, float))
    orbs = get_orb_idcs(edata, atom, None, None)
    nk = len(w) // edata.nspin
    out = []
    for M_t in (np.einsum("tju,tj,tjv->tuv", C, e, C.conj()), np.einsum("tj,tju,tjv->tuv", f, C, C.conj())):
        total = 0.0
        for s in range(edata.nspin):
            sl = slice(s * nk, (s + 1) * nk)
            M_R = np.tensordot(w[sl] * ph[sl].conj(), M_t[sl], axes=1)[np.ix_(orbs, orbs)] / w[sl].sum()
            c = C[sl][:, :, orbs]
            total += np.sum(w[sl, None] * f[sl] * (np.einsum("tju,uv,tjv->tj", c.conj(), M_R, c) * ph[sl, None]).real)
        out.append(total)
    return out[0], 2 * out[1]


def test_graphene_self_image_matches_reference():
    from crawfish.funcs import get_ipcohp, get_ipcobi

    edata = _get_edata()
    icohp_ref, icobi_ref = _reference(edata, 0, (1, 0, 0))
    assert _last(get_ipcohp(edata, idcs1=0, idcs2=0, image=(1, 0, 0), **NOCACHE)) == pytest.approx(icohp_ref, abs=1e-5)
    assert _last(get_ipcobi(edata, idcs1=0, idcs2=0, image=(1, 0, 0), **NOCACHE)) == pytest.approx(icobi_ref, abs=1e-5)
    # About half of this interaction is u with its own image (check_V5: the kernel used to keep 49.8%)
    diag = sum(_last(get_ipcohp(edata, idcs1=0, orbs1=o, idcs2=0, orbs2=o, image=(1, 0, 0), **NOCACHE))
               for o in edata.atom_orb_labels_dict["C"])
    assert diag / icohp_ref == pytest.approx(0.502, abs=0.01)


@pytest.mark.parametrize("calc_dir", [graphene_dir, zno_dir])
@pytest.mark.parametrize("func_name", ["get_ipcohp", "get_ipcobi", "get_ipcomo"])
def test_image_and_its_inverse_agree(calc_dir, func_name):
    # The bond from a to a + R is the bond from a - R to a
    from crawfish import funcs

    edata = _get_edata(calc_dir)
    func = getattr(funcs, func_name)
    forward = _last(func(edata, idcs1=0, idcs2=0, image=(1, 0, 0), **NOCACHE))
    assert forward == pytest.approx(_last(func(edata, idcs1=0, idcs2=0, image=(-1, 0, 0), **NOCACHE)), abs=1e-6)
    assert abs(forward) > 1e-3


def test_zno_includes_orbital_with_its_image():
    from crawfish.funcs import get_ipcohp
    from crawfish.funcs.pcohp import _compute_signed_tj
    from crawfish.utils.indexing import get_orb_idcs

    edata = _get_edata(zno_dir)
    icohp_ref, _ = _reference(edata, 0, (1, 0, 0))
    full = _last(get_ipcohp(edata, idcs1=0, idcs2=0, image=(1, 0, 0), **NOCACHE))
    assert full == pytest.approx(icohp_ref, abs=1e-5)
    assert full == pytest.approx(-0.025691, abs=1e-5)
    # Without the u == v terms (the kernel before F19) it was 69% of that (check_V5)
    from crawfish.core.operations.matrix import _get_gen_tj_block

    orbs = np.asarray(get_orb_idcs(edata, 0, None, None))
    w_full = _compute_signed_tj(edata, orbs, orbs, "h", [1, 0, 0])
    assert float(np.sum(edata.occ_tj * w_full)) == pytest.approx(full, abs=1e-6)
    mat_t = edata.get_mat_r_suv("h", (1, 0, 0))[np.zeros(edata.nstates, int) if edata.nspin == 1 else
                                                  np.arange(edata.nstates) // (edata.nstates // edata.nspin)]
    mat_t = mat_t[:, orbs][:, :, orbs] * np.exp(2j * np.pi * edata.ks_t.astype(float) @ [1.0, 0, 0])[:, None, None]
    off_diagonal = float(np.sum(edata.occ_tj * _get_gen_tj_block(edata.coef_tju, mat_t, edata.wk_t, orbs, orbs)))
    assert off_diagonal / full == pytest.approx(0.694, abs=0.005)


@pytest.mark.parametrize("image", ["nearest", (0, 0, 0), "all"])
def test_overlapping_sets_still_raise(image):
    from crawfish.funcs import get_ipcohp, get_ipcobi, get_ipcoop

    edata = _get_edata()
    for func in (get_ipcohp, get_ipcobi, get_ipcoop):
        with pytest.raises(ValueError, match="image=<lattice vector>"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                func(edata, idcs1=0, idcs2=0, image=image)
    with pytest.raises(ValueError, match="image=<lattice vector>"):
        get_ipcohp(edata, idcs1=0, idcs2=0, state_sep=True, image=(1, 0, 0))


def test_disjoint_sets_unchanged():
    # Same-atom disjoint sets keep their on-site meaning with "nearest" (R = 0), and an explicit include mask that
    # adds nothing reproduces the default (u != v) kernel bit for bit
    from crawfish.core.operations.matrix import _get_gen_tj_block
    from crawfish.funcs import get_pair_images

    edata = _get_edata()
    assert get_pair_images(edata, idcs1=0, orbs1="s", idcs2=0, orbs2="px")[0]["R"] == (0, 0, 0)
    orbs_u, orbs_v = np.array([0, 1]), np.array([4, 5, 6])
    mat = np.asarray(edata.h_t_uv[:, orbs_u][:, :, orbs_v])
    default = _get_gen_tj_block(edata.coef_tju, mat, edata.wk_t, orbs_u, orbs_v)
    explicit = _get_gen_tj_block(edata.coef_tju, mat, edata.wk_t, orbs_u, orbs_v, np.ones((2, 3), dtype=bool))
    assert np.array_equal(default, explicit)


def test_get_nearest_images():
    from crawfish.funcs import get_nearest_images

    edata = _get_edata()
    second = get_nearest_images(edata, 0, 0)
    assert len(second) == 6
    assert {img["R"] for img in second} == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0)}
    assert all(img["distance"] == pytest.approx(2.46, abs=1e-3) for img in second)
    assert all(img["atom1"] == img["atom2"] == "C #1" for img in second)
    first = get_nearest_images(edata, 0, 1)
    assert len(first) == 3 and all(img["distance"] == pytest.approx(1.42, abs=5e-3) for img in first)
    # The nearest pair image ("nearest") is one of them
    from crawfish.funcs import get_pair_images

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert get_pair_images(edata, idcs1=0, idcs2=1)[0]["R"] in {img["R"] for img in first}


def test_spectrum_cache_keeps_permuted_images_apart():
    # Lattice vectors that are permutations of each other used to share one cached spectrum
    from crawfish.funcs import get_pcohp

    edata = _get_edata()
    erange = np.arange(-1.0, 0.5, 1e-3, dtype=np.float32)
    a = get_pcohp(edata, idcs1=0, idcs2=0, image=(1, 0, 0), erange=erange, **NOCACHE)[1]
    b = get_pcohp(edata, idcs1=0, idcs2=0, image=(0, 0, 1), erange=erange, **NOCACHE)[1]  # cached call
    fresh = get_pcohp(edata, idcs1=0, idcs2=0, image=(0, 0, 1), erange=erange, use_cached_spectrum=False,
                      save_spectrum=False, **NOCACHE)[1]
    assert np.array_equal(b, fresh) and not np.allclose(a, b)


def test_reduced_mesh_one_atom_cell_stays_guarded():
    # fcc Al's mesh was reduced by symmetry and has no `sym` file, so single images can't be recovered
    from crawfish.funcs import get_ipcohp

    edata = _get_edata(calc_dirs / "Al_fcc_Fermi")
    with pytest.raises(ValueError, match="single lattice image"):
        get_ipcohp(edata, idcs1=0, idcs2=0, image=(1, 0, 0))
