"""Tests for unfolding k-meshes reduced by JDFTx's space group (notes/theory_audit.md, finding F12).

graphene_zspin_pg is graphene_zspin_symm (24 operations, 19 of 144 k per spin) rerun with `dump End Symmetries`;
graphene_zspin_full is its full-mesh SCF reference. ZnO_wurtzite_pg (12 operations with c/2 screws, no inversion,
Zn d orbitals, 8 of 48 k) has ZnO_wurtzite_full as its reference, computed on the full mesh at the same density.
"""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
gr_pg_dir, gr_full_dir, gr_symm_dir = (calc_dirs / f"graphene_zspin_{x}" for x in ("pg", "full", "symm"))
zno_pg_dir, zno_full_dir = calc_dirs / "ZnO_wurtzite_pg", calc_dirs / "ZnO_wurtzite_full"


def _get_edata(calc_dir, **settings):
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the 20-band graphene runs warn about too few bands (F11)
        edata = ElecData(calc_dir)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata


def _icohp(edata, a, b, image, orbs=None):
    from crawfish.funcs.pcohp import _get_pcohp_tj

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = _get_pcohp_tj(edata, a, None, orbs, b, None, orbs, use_cache=False, image=image)
    return float(np.sum(edata.occ_tj * weights))


def _ipdos(edata, idcs, orbs):
    from crawfish.funcs.dos import get_pdos_tj
    from crawfish.utils.indexing import get_orb_idcs

    return float(np.sum(edata.occ_tj * get_pdos_tj(edata.proj_tju, get_orb_idcs(edata, idcs, None, orbs), edata.wk_t)))


# ---- Real spherical harmonics and their rotations ----


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))[None, :]
        out.append(q if rng.random() < 0.5 else -q)  # proper and improper
    return out


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_ylm_are_orthonormal_and_rotate_as_a_representation(l):
    from crawfish.utils.symmetry import real_ylm, get_ylm_rotation

    # Orthonormal on the sphere (Lebedev-free check: Monte Carlo is too noisy, so use a product Gauss grid)
    x, wx = np.polynomial.legendre.leggauss(24)
    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ct, ph = np.meshgrid(x, phi, indexing="ij")
    st = np.sqrt(1 - ct**2)
    pts = np.stack([st * np.cos(ph), st * np.sin(ph), ct], -1).reshape(-1, 3)
    w = (wx[:, None] * np.full(len(phi), 2 * np.pi / len(phi))[None, :]).ravel()
    y = real_ylm(l, pts)
    assert np.allclose(y.T @ (w[:, None] * y), np.eye(2 * l + 1), atol=1e-12)
    # Y_m'(S x) = sum_m M(S)[m, m'] Y_m(x), and M(S1 S2) = M(S2) M(S1)
    s1, s2 = _random_rotations(2, seed=l)
    m1, m2, m12 = get_ylm_rotation(l, s1), get_ylm_rotation(l, s2), get_ylm_rotation(l, s1 @ s2)
    assert np.allclose(real_ylm(l, pts @ s1.T), y @ m1, atol=1e-10)
    assert np.allclose(m12, m2 @ m1, atol=1e-10)
    assert np.allclose(get_ylm_rotation(l, -np.eye(3)), (-1) ** l * np.eye(2 * l + 1), atol=1e-12)


def test_p_orbital_rotation_follows_jdftx_order():
    # JDFTx's p order is (y, z, x): M[m, m'] = S[c(m'), c(m)] with c = (1, 2, 0)
    from crawfish.utils.symmetry import get_ylm_rotation

    s = _random_rotations(1, seed=7)[0]
    c = [1, 2, 0]
    assert np.allclose(get_ylm_rotation(1, s), s[np.ix_(c, c)].T, atol=1e-12)


# ---- Parsing ----


def test_parse_symfile_and_manual_symmetries():
    from crawfish.io.data_parsing import parse_symfile, get_symmetries_from_outfile_filepath

    ops = parse_symfile(zno_pg_dir / "sym")
    assert len(ops) == 12 and np.array_equal(ops[0][0], np.eye(3)) and np.allclose(ops[0][1], 0)
    assert sum(np.isclose(abs(a[2]), 0.5) for _, a in ops) == 6  # the 6_3 screws and c-glides
    assert get_symmetries_from_outfile_filepath(zno_pg_dir / "out") == ("automatic", None)
    assert get_symmetries_from_outfile_filepath(gr_full_dir / "out") == ("none", None)
    mode, manual = get_symmetries_from_outfile_filepath(calc_dirs / "graphene_zspin_tr" / "out")
    assert mode == "manual" and len(manual) == 1 and np.array_equal(manual[0][0], np.eye(3))


@pytest.mark.parametrize("calc_dir", [gr_pg_dir, zno_pg_dir])
def test_kpoint_map_matches_jdftx(calc_dir):
    # JDFTx's kMap: for each full-mesh k-point, the computed one and whether inversion (time reversal) was used
    kmap = np.loadtxt(calc_dir / "kMap", dtype=int, comments="#")
    edata = _get_edata(calc_dir)
    u = edata.kmesh_unfolding
    nk = len(kmap)
    assert np.array_equal(u["src_t"][:nk], kmap[:, 0])
    assert np.array_equal(u["conj_t"][:nk], kmap[:, 2] < 0)


# ---- Unfolding ----


@pytest.mark.parametrize(
    ("pg_dir", "full_dir", "nfile"), [(gr_pg_dir, gr_full_dir, 38), (zno_pg_dir, zno_full_dir, 8)]
)
def test_point_group_mesh_is_unfolded_by_default(pg_dir, full_dir, nfile):
    pg, full = _get_edata(pg_dir), _get_edata(full_dir)
    assert pg.unfold_point_group and pg.point_group_unfolded and pg.kmesh_unfolded
    assert not pg.time_reversal_unfolded and not pg.kmesh_is_reduced and pg.lti_allowed
    assert pg.nstates_file == nfile and pg.nstates == full.nstates
    assert list(pg.kfolding) == list(full.kfolding)
    assert np.allclose(pg.ks_t, full.ks_t, atol=1e-6)
    assert np.allclose(pg.wk_t, full.wk_t, rtol=1e-5)
    assert np.allclose(pg.e_tj, full.e_tj, atol=2e-4)  # graphene: separate SCF runs, empty vacuum bands


@pytest.mark.parametrize("calc_dir", [gr_pg_dir, zno_pg_dir])
def test_little_group_leaves_states_invariant(calc_dir):
    # An operation that maps a computed k-point onto itself must map its band space onto itself: S and H over the
    # bands below a gap (no degenerate multiplet cut) are unchanged. Independent of any second calculation.
    from crawfish.core.operations.matrix import apply_jdftx_orbital_phase
    from crawfish.io.data_parsing import get_proj_tju_from_file, parse_kptsfrom_bandfile_header, get_e_tj_helper
    from crawfish.utils.lattice import get_orb_atom_idx_u
    from crawfish.utils.symmetry import get_orbital_transform, apply_orbital_transform

    ed = _get_edata(calc_dir)
    proj = apply_jdftx_orbital_phase(get_proj_tju_from_file(ed.bandfile_filepath), ed.orb_l_u).astype(np.complex128)
    e = get_e_tj_helper(ed.eigfile_filepath, ed.nstates_file, ed.nbands).astype(np.float64)
    ks = np.array(parse_kptsfrom_bandfile_header(ed.bandfile_header_filepath)[1])
    st, npairs, worst = ed.structure, 0, 0.0
    for rot, a in ed.kmesh_unfolding["ops"]:
        tr = get_orbital_transform(rot, a, st.lattice.matrix, st.frac_coords, [str(s) for s in st.species],
                                   get_orb_atom_idx_u(ed), ed.orb_l_u, ed.orb_shell_u)
        rot_inv = np.round(np.linalg.inv(rot)).astype(int)
        for t in range(len(ks)):
            for conj, sign in ((False, 1), (True, -1)):
                diff = sign * (ks[t] @ rot_inv) - ks[t]
                if np.abs(diff - np.round(diff)).max() > 1e-6:
                    continue
                nb = max(n for n in range(1, ed.nbands) if e[t, n] - e[t, n - 1] > 1e-3)
                p1 = proj[t, :nb]
                p2 = apply_orbital_transform(np.conj(proj[t]) if conj else proj[t], tr, ks[t])[:nb]
                worst = max(worst, np.abs(p1.conj().T @ p1 - p2.conj().T @ p2).max(),
                            np.abs(p1.conj().T @ (e[t, :nb, None] * p1) - p2.conj().T @ (e[t, :nb, None] * p2)).max())
                npairs += 1
    assert npairs > 50
    assert worst < 3e-5


def test_rotated_states_match_directly_computed_ones():
    # ZnO_wurtzite_full has the same Hamiltonian, so every state that needed an operation or time reversal must
    # match it over the bands below a gap, to the 7-decimal precision of the files. (At Γ, which needs neither,
    # JDFTx's two runs differ by themselves by up to 2e-3.)
    pg, full = _get_edata(zno_pg_dir), _get_edata(zno_full_dir)
    u = pg.kmesh_unfolding
    p1, p2 = np.asarray(pg.proj_tju, np.complex128), np.asarray(full.proj_tju, np.complex128)
    e1, e2 = np.asarray(pg.e_tj, np.float64), np.asarray(full.e_tj, np.float64)
    transformed = [t for t in range(pg.nstates) if u["op_t"][t] != 0 or u["conj_t"][t]]
    assert len(transformed) == 40
    for t in transformed:
        nb = max(n for n in range(1, pg.nbands) if min(e1[t, n] - e1[t, n - 1], e2[t, n] - e2[t, n - 1]) > 1e-3)
        a, b = p1[t, :nb], p2[t, :nb]
        assert np.abs(a.conj().T @ a - b.conj().T @ b).max() < 2e-5, t
        assert np.abs(a.conj().T @ (e1[t, :nb, None] * a) - b.conj().T @ (e2[t, :nb, None] * b)).max() < 2e-5, t


def test_graphene_bonds_and_orbitals_match_full_mesh():
    pg, full = _get_edata(gr_pg_dir), _get_edata(gr_full_dir)
    for image, orbs in [([0, 0, 0], None), ([-1, 0, 0], None), ([0, 1, 0], None), ("nearest", "px"),
                        ([0, 0, 0], "py"), ("all", "px"), ([1, 1, 0], None)]:
        assert _icohp(pg, 0, 1, image, orbs) == pytest.approx(_icohp(full, 0, 1, image, orbs), abs=2e-5), (image, orbs)
    for orbs in ["s", "px", "py", "pz"]:
        assert _ipdos(pg, 0, orbs) == pytest.approx(_ipdos(full, 0, orbs), abs=2e-5), orbs
    # The three nearest bonds are equivalent: equal on the unfolded mesh, as on the full one
    bonds = [_icohp(pg, 0, 1, R) for R in ([0, 0, 0], [-1, 0, 0], [0, 1, 0])]
    assert bonds[0] == pytest.approx(-0.50126, abs=2e-5) and np.ptp(bonds) < 2e-5


def test_zno_bonds_and_d_orbitals_match_full_mesh():
    pg, full = _get_edata(zno_pg_dir), _get_edata(zno_full_dir)
    # Zn1-O1 axial (1.989 A) and two of the basal Zn1-O2 bonds (1.974 A)
    for b, image in [(2, [0, 0, 0]), (3, [0, 0, -1]), (3, [0, 1, -1])]:
        for orbs in [None, "pz", "px"]:
            assert _icohp(pg, 0, b, image, orbs) == pytest.approx(_icohp(full, 0, b, image, orbs), abs=3e-5)
    assert _icohp(pg, 0, 2, [0, 0, 0]) == pytest.approx(-0.14438, abs=1e-5)
    assert _icohp(pg, 0, 3, [0, 0, -1]) == pytest.approx(-0.14943, abs=1e-5)
    for idx in (0, 1, 2):
        for orbs in ["dxy", "dyz", "dz2", "dxz", "dx2-y2", "0px", "1py", "pz", "px"]:
            if idx == 2 and orbs.startswith("d"):
                continue
            assert _ipdos(pg, idx, orbs) == pytest.approx(_ipdos(full, idx, orbs), abs=1e-5), (idx, orbs)


def test_band_diagnostics_use_the_unfolded_mesh():
    # graphene_zspin_full is a separate SCF run (span differs by up to 3e-4); ZnO_wurtzite_full shares the density
    for pg_dir, full_dir, tol in [(gr_pg_dir, gr_full_dir, 5e-4), (zno_pg_dir, zno_full_dir, 1e-6)]:
        pg, full = _get_edata(pg_dir), _get_edata(full_dir)
        assert np.allclose(pg.orbital_completeness_u, full.orbital_completeness_u, atol=3e-5)
        assert np.allclose(np.maximum(pg.orbital_span_t, 1e-4), np.maximum(full.orbital_span_t, 1e-4), atol=tol)
    # graphene's in-plane σ* combination at K is missing from 20 bands: reported for the K points of the full mesh
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from crawfish.core.elecdata import ElecData

        ElecData(gr_pg_dir)
    msgs = [str(w.message) for w in caught if "Too few bands" in str(w.message)]
    assert msgs and "at 4 of 288 states" in msgs[0] and "mostly C p" in msgs[0]


def test_unfolding_is_controllable():
    from crawfish.core.elecdata import get_edata_settings_str

    pg = _get_edata(zno_pg_dir)
    unfolded_dir = get_edata_settings_str(pg)
    pg.unfold_point_group = False  # the F5 behaviour: time reversal alone can't unfold this mesh
    assert not pg.kmesh_unfolded and pg.kmesh_is_reduced and pg.nstates == 8
    assert get_edata_settings_str(pg) != unfolded_dir
    with pytest.raises(ValueError, match="dump End"):
        _icohp(pg, 0, 2, [0, 0, 0])
    pg.unfold_point_group = True
    assert pg.point_group_unfolded and pg.nstates == 48
    # Without time reversal the operations alone don't reach every k-point of this non-centrosymmetric crystal
    assert not _get_edata(zno_pg_dir, unfold_time_reversal=False).kmesh_unfolded
    # Graphene's point group contains inversion, so the operations suffice there
    assert _get_edata(gr_pg_dir, unfold_time_reversal=False).point_group_unfolded
    assert not _get_edata(zno_pg_dir, legacy_reduced_kmesh=True).kmesh_unfolded
    # Without the sym file (graphene_zspin_symm) the mesh stays reduced and guarded
    symm = _get_edata(gr_symm_dir)
    assert symm.symfile_filepath is None and not symm.kmesh_unfolded and symm.kmesh_is_reduced


def test_unfolding_with_raw_orbital_phase():
    from crawfish.core.operations.matrix import apply_jdftx_orbital_phase

    fixed, raw = _get_edata(zno_pg_dir), _get_edata(zno_pg_dir, legacy_orbital_phase=True)
    assert raw.point_group_unfolded
    assert np.allclose(apply_jdftx_orbital_phase(raw.proj_tju, raw.orb_l_u), fixed.proj_tju, atol=1e-5)


def test_operations_that_dont_fit_the_structure_raise(tmp_path):
    import shutil

    for name in ["bandProjections", "eigenvals", "fillings", "kPts", "out"]:
        shutil.copy(zno_pg_dir / name, tmp_path / name)
    lines = (zno_pg_dir / "sym").read_text().splitlines()
    lines[8] = "0 0 0.25"  # the 2_1 screw's c/2 translation, changed to c/4: no longer a symmetry of ZnO
    (tmp_path / "sym").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="don't match this structure"):
        _get_edata(tmp_path)
