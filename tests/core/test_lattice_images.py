"""Tests for lattice-image selection in pair spectra (notes/theory_audit.md, finding F3)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import itertools
import numpy as np
import pytest

n2_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
# Graphene primitive cell (2 C, 23x23x1 k-mesh, GBRV USPP, z-spin, symmetries none, ortho s/p projections).
# Each C has three nearest neighbours (~1.456 A) of the other sublattice, two of them in neighbouring cells.
graphene_dir = Path(EXAMPLE_CALC_DIRS_DIR) / "graphene_1x1"
BOHR_TO_ANG = 0.529177210903


def _get_edata(calc_dir=n2_dir, **settings):
    from crawfish.core.elecdata import ElecData

    edata = ElecData(calc_dir)
    for key, val in settings.items():
        setattr(edata, key, val)
    return edata


def _weights(edata, func_name, image, o1=None, o2=None):
    from crawfish.funcs import pcohp

    get_tj = getattr(pcohp, f"_get_{func_name}_tj")
    return get_tj(edata, 0, None, o1, 1, None, o2, use_cache=False, image=image)


def _shift_atom2(edata, monkeypatch, L):
    """Describe the same crystal with atom 2 moved by lattice vector L.

    JDFTx's Bloch sums carry cis(-2 pi pos.(k+G)) (SpeciesInfo_internal.h::Vnl_calc), so moving an atom by L
    multiplies its Bloch orbitals by exp(-2 pi i k.L) and its projections <phi|O|psi> by exp(+2 pi i k.L).
    """
    from crawfish.core.elecdata import ElecData
    from crawfish.core.operations.matrix import get_h_s_d_t_uv

    shifted = edata.structure.copy()
    shifted.replace(1, shifted[1].species, shifted.frac_coords[1] + np.asarray(L), coords_are_cartesian=False)
    monkeypatch.setattr(ElecData, "structure", property(lambda self: shifted))
    phase_t = np.exp(2j * np.pi * (edata.ks_t.astype(np.float64) @ np.asarray(L, dtype=np.float64)))
    proj = edata.proj_tju.copy()
    orbs_b = edata.orbs_idx_dict[edata.kmap[1]]
    proj[:, :, orbs_b] *= phase_t[:, None, None].astype(proj.dtype)
    edata._proj_tju = proj
    edata._coef_tju = None
    e_tj = edata.e_tj - edata.mu if edata.norm_eigs_for_h else edata.e_tj
    edata._h_t_uv, edata._s_t_uv, edata._d_t_uv = get_h_s_d_t_uv(proj, edata.coef_tju, e_tj, edata.occ_tj)
    return edata


def test_nearest_image_n2():
    from crawfish.funcs import get_pair_images

    (pair,) = get_pair_images(_get_edata(), idcs1=0, idcs2=1)
    assert pair["R"] == (0, 0, 0)
    # The N-N separation in the outfile is 13.895656 - 11.832075 bohr
    assert pair["distance"] == pytest.approx((13.895656 - 11.832075) * BOHR_TO_ANG, abs=1e-4)


@pytest.mark.parametrize("func_name", ["pcohp", "pcoop", "pcobi"])
def test_images_sum_to_all(func_name):
    # Discrete Fourier identity: on a 3x3x3 mesh, summing the R-resolved weights over one R per residue class
    # (here R in {-1,0,1}^3) reproduces the state-resolved ("all images") weights exactly.
    edata = _get_edata()
    total = sum(_weights(edata, func_name, list(R)).astype(np.float64)
                for R in itertools.product((-1, 0, 1), repeat=3))
    assert np.allclose(total, _weights(edata, func_name, "all"), atol=1e-6)


def test_far_image_is_negligible():
    # The next copy of N2 is ~13 A away along x: essentially no interaction
    edata = _get_edata()
    f = edata.occ_tj
    near = np.sum(f * _weights(edata, "pcohp", "nearest"))
    far = np.sum(f * _weights(edata, "pcohp", [1, 0, 0]))
    assert near < -0.5
    assert abs(far) < 1e-3 * abs(near)


@pytest.mark.parametrize("func_name", ["pcohp", "pcobi", "pcomo"])
@pytest.mark.parametrize("L", [(1, 0, 0), (-1, 2, 0)])
def test_nearest_image_independent_of_wrapping(monkeypatch, func_name, L):
    from crawfish.funcs import get_pair_images

    reference = _weights(_get_edata(), func_name, "nearest")
    edata = _shift_atom2(_get_edata(), monkeypatch, L)
    (pair,) = get_pair_images(edata, idcs1=0, idcs2=1)
    assert pair["R"] == tuple(-x for x in L)
    assert np.allclose(_weights(edata, func_name, "nearest"), reference, atol=1e-5)


def test_legacy_same_cell_pairing_depends_on_wrapping(monkeypatch):
    # Before the fix, the same-cell copy was used whatever its distance: after wrapping atom 2 one cell over,
    # that copy is ~13 A away and the N-N pCOHP all but vanishes.
    reference = np.sum(_get_edata().occ_tj * _weights(_get_edata(), "pcohp", "nearest"))
    edata = _shift_atom2(_get_edata(legacy_lattice_image=True), monkeypatch, (1, 0, 0))
    legacy = np.sum(edata.occ_tj * _weights(edata, "pcohp", "nearest"))
    assert abs(legacy) < 1e-3 * abs(reference)


def test_state_sep_is_all_images():
    edata = _get_edata()
    from crawfish.funcs.pcohp import _get_pcohp_tj

    w_sep = _get_pcohp_tj(edata, 0, None, None, 1, None, None, use_cache=False, state_sep=True, image=[1, 0, 0])
    assert np.allclose(w_sep, _weights(edata, "pcohp", "all"))


def test_image_argument_validation():
    from crawfish.funcs import get_ipcohp, get_ipcomo

    edata = _get_edata()
    with pytest.raises(ValueError):
        get_ipcohp(edata, idcs1=0, idcs2=1, image="farthest")
    with pytest.raises(ValueError):
        get_ipcohp(edata, idcs1=0, idcs2=1, image=(1, 0))
    with pytest.raises(ValueError):
        get_ipcomo(edata, idcs1=0, idcs2=1, image="all")
    with pytest.raises(ValueError):
        get_ipcohp(_get_edata(legacy_matrices=True), idcs1=0, idcs2=1, image=(1, 0, 0))


# ---- Graphene: inter-cell pairs carry most of the A-B bonding ----

GRAPHENE_NEAREST = [(-1, -1, 0), (0, -1, 0), (0, 0, 0)]


def _icohp(edata, image, o1=None, o2=None):
    return float(np.sum(edata.occ_tj * _weights(edata, "pcohp", image, o1, o2)))


def test_graphene_three_nearest_bonds():
    from crawfish.funcs import get_pair_images

    edata = _get_edata(graphene_dir)
    with pytest.warns(UserWarning, match="3 images within"):
        (pair,) = get_pair_images(edata, idcs1=0, idcs2=1)
    assert pair["R"] in GRAPHENE_NEAREST
    assert pair["distance"] == pytest.approx(1.456, abs=2e-3)
    bonds = [_icohp(edata, list(R)) for R in GRAPHENE_NEAREST]
    # The three bonds are equivalent, and two of them are inter-cell
    assert np.allclose(bonds, bonds[0], rtol=0.01)
    assert bonds[0] < -0.1
    # Summed over all images: the 3 nearest bonds plus further shells (2.91 A shell ~ -0.026 Ha per image)
    beyond_nearest = _icohp(edata, "all") - sum(bonds)
    assert beyond_nearest < -0.05


def test_graphene_orbital_resolved_bonds_differ():
    # Each bond points a different way, so m-resolved pCOHP differs between the three equivalent bonds even
    # though the totals agree: px-px is large for one bond and small for the others.
    edata = _get_edata(graphene_dir)
    pxpx = [_icohp(edata, list(R), "px", "px") for R in GRAPHENE_NEAREST]
    pzpz = [_icohp(edata, list(R), "pz", "pz") for R in GRAPHENE_NEAREST]
    assert max(pxpx) - min(pxpx) > 0.02
    assert np.allclose(pzpz, pzpz[0], rtol=0.02)  # pi bonding is the same for every bond


def test_graphene_images_sum_to_all():
    # Discrete Fourier identity on the 23x23x1 mesh: one R per residue class
    edata = _get_edata(graphene_dir)
    total = sum(_weights(edata, "pcohp", [n1, n2, 0]).astype(np.float64)
                for n1 in range(-11, 12) for n2 in range(-11, 12))
    assert np.allclose(total, _weights(edata, "pcohp", "all"), atol=1e-6)


@pytest.mark.parametrize("L", [(1, 0, 0), (2, -1, 0)])
def test_graphene_wrapping(monkeypatch, L):
    reference = _icohp(_get_edata(graphene_dir), "nearest")
    edata = _shift_atom2(_get_edata(graphene_dir), monkeypatch, L)
    assert _icohp(edata, "nearest") == pytest.approx(reference, abs=1e-5)
    # Legacy pairs with the same-cell copy, which after wrapping is no longer a nearest neighbour
    legacy = _shift_atom2(_get_edata(graphene_dir, legacy_lattice_image=True), monkeypatch, L)
    assert abs(_icohp(legacy, "nearest")) < 0.25 * abs(reference)
