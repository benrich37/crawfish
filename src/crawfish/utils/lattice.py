"""Module for lattice-image bookkeeping.

Pair spectra between orbitals u (on atom a) and v (on atom b) depend on which periodic image of b is meant.
JDFTx's Bloch sums are phi_uk(r) = sum_R e^{ik.R} phi_u(r - tau_u - R), with tau_u the atom positions exactly as
JDFTx held them (SpeciesInfo_internal.h::Vnl_calc; JDFTx does not wrap them, and neither does pymatgen's
JDFTXOutfile.structure), so the image of b at tau_b + R is selected by the phase e^{2 pi i k.R}.
See notes/theory_audit.md, finding F3.
"""

from __future__ import annotations

import warnings

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawfish.core.elecdata import ElecData

# Images whose bond lengths differ by less than this (Angstrom) count as equally near. Relaxed structures carry
# ~1e-3 A noise between symmetry-equivalent bonds (e.g. the three C-C bonds of the graphene test calculation differ
# by up to 0.003 A), so a numerically tight tolerance would hide that the choice is arbitrary.
IMAGE_TIE_TOL = 1e-2


def get_orb_atom_idx_u(edata: ElecData) -> np.ndarray[int]:
    """Return the structure site index of the atom each orbital belongs to, in shape (nproj,)."""
    atom_u = np.zeros(edata.nproj, dtype=np.int64)
    for site, label in enumerate(edata.kmap):
        atom_u[edata.orbs_idx_dict[label]] = site
    return atom_u


def get_image_distance(edata: ElecData, atom_a: int, atom_b: int, R) -> float:
    """Return the distance (Angstrom) from atom a to the image of atom b at tau_b + R (R in lattice vectors)."""
    frac = edata.structure.frac_coords
    d_frac = frac[atom_b] + np.asarray(R, dtype=float) - frac[atom_a]
    return float(np.linalg.norm(d_frac @ edata.structure.lattice.matrix))


def get_nearest_image(edata: ElecData, atom_a: int, atom_b: int, search: int = 2) -> tuple[tuple[int, int, int], float, int]:
    """Return the lattice vector R of the image of atom b nearest to atom a, its distance, and the number of ties.

    Ties are images within IMAGE_TIE_TOL of the nearest distance (the nearest itself included). The returned R is
    the strictly nearest, with exact ties going to the smallest |R|.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest
    atom_a, atom_b : int
        Structure site indices
    search : int
        Search R within +/- search of the minimum-image guess along each lattice vector
    """
    frac = edata.structure.frac_coords
    lattice = edata.structure.lattice.matrix
    d_frac = frac[atom_b] - frac[atom_a]
    guess = -np.round(d_frac).astype(int)
    rng = np.arange(-search, search + 1)
    offsets = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    candidates = guess[None, :] + offsets
    dists = np.linalg.norm((d_frac[None, :] + candidates) @ lattice, axis=1)
    order = np.lexsort((np.abs(candidates).sum(axis=1), dists))  # nearest first; ties prefer small |R|
    best = order[0]
    nties = int(np.sum(dists < dists[best] + IMAGE_TIE_TOL))
    return tuple(int(x) for x in candidates[best]), float(dists[best]), nties


def normalize_image(image) -> str | list[int]:
    """Return image as "nearest", "all", or a list of 3 ints (the form stored in spectrum-cache metadata)."""
    if isinstance(image, str):
        if image not in ("nearest", "all"):
            raise ValueError(f"image must be 'nearest', 'all' or a lattice vector of 3 ints, not {image!r}")
        return image
    vec = [int(x) for x in image]
    if len(vec) != 3 or not np.allclose(vec, np.asarray(image, dtype=float)):
        raise ValueError(f"image must be 'nearest', 'all' or a lattice vector of 3 ints, not {image!r}")
    return vec


def get_pair_image_map(
    edata: ElecData, orbs_u: list[int], orbs_v: list[int], image: str | list[int]
) -> dict[tuple[int, int], tuple[tuple[int, int, int], float]]:
    """Return {(atom_a, atom_b): (R, distance)} for every atom pair spanned by the two orbital sets.

    image="nearest" picks each pair's nearest image (R=(0,0,0) for all pairs when edata.legacy_lattice_image is
    True), and an explicit lattice vector is used for every pair. Warns when a nearest image is not unique.
    """
    atom_u = get_orb_atom_idx_u(edata)
    pairs = sorted({(int(atom_u[u]), int(atom_u[v])) for u in orbs_u for v in orbs_v})
    out = {}
    for a, b in pairs:
        if image == "nearest" and not edata.legacy_lattice_image:
            R, dist, nties = get_nearest_image(edata, a, b)
            if nties > 1:
                warnings.warn(
                    f"Atoms {edata.kmap[a]} and {edata.kmap[b]} have {nties} images within {IMAGE_TIE_TOL} A of the "
                    f"nearest ({dist:.4f} A); using R={R}. Orbital-resolved results differ between such bonds. Pass "
                    "image=<lattice vector> to choose another, or image='all' to sum over every image.",
                    stacklevel=4,
                )
        else:
            R = (0, 0, 0) if image == "nearest" else tuple(image)
            dist = get_image_distance(edata, a, b, R)
        out[(a, b)] = (R, dist)
    return out


REDUCED_KMESH_HINT = (
    "For atom-, bond- or orbital-resolved results, add 'Symmetries' to the JDFTx 'dump End' line (crawfish then "
    "unfolds the mesh with JDFTx's operations), or rerun with 'symmetries none' (notes/theory_audit.md, F5 and F12)."
)


def check_reduced_kmesh(edata: ElecData, orbs: list[int], what: str, image=None) -> None:
    """Guard atom/bond/orbital-resolved quantities on a symmetry-reduced k-mesh (notes/theory_audit.md, F5).

    On a mesh reduced by the point group, a quantity for one atom, bond or m-orbital is averaged over its
    symmetry-equivalent copies. It is exact only for selections the group maps onto themselves: the sum over all
    lattice images (pair spectra), complete orbital shells, and complete sets of equivalent atoms. Meshes that
    ElecData unfolds (time reversal, F5; JDFTx's space group from the `sym` file, F12) are full meshes and pass.

    Parameters
    ----------
    orbs : list[int]
        All orbitals the quantity involves (both sets for pair spectra)
    what : str
        Name of the quantity, for the messages
    image : str | list[int] | None
        The lattice-image selection of a pair spectrum (None for pDOS)
    """
    if edata.legacy_reduced_kmesh or not edata.kmesh_is_reduced:
        return
    if image is not None and image != "all":
        raise ValueError(
            f"{what} for a single lattice image (image={image!r}) can't be recovered from a k-mesh reduced by "
            "symmetry: every k-point stands in for its symmetry-equivalent ones, which belong to other, equivalent "
            "bonds. Use image='all' (exact when summed over symmetry-equivalent atoms and complete orbital shells). "
            + REDUCED_KMESH_HINT
        )
    shell_u = edata.orb_shell_u
    selected = np.zeros(len(shell_u), dtype=bool)
    selected[np.asarray(orbs, dtype=int)] = True
    partial = sorted({int(shell_u[u]) for u in np.flatnonzero(selected)
                      if not selected[shell_u == shell_u[u]].all()})
    if image is None and selected.all():
        return  # total DOS-like quantity: exact
    msg = (
        f"This calculation's k-points were reduced by symmetry. {what} for a subset of atoms is averaged over "
        "symmetry-equivalent atoms, and is exact only when summed over complete sets of equivalent atoms."
    )
    if partial:
        msg += (
            " The selection also contains incomplete orbital shells (e.g. px without py), which the point group "
            "mixes, so those m-resolved values are averaged as well."
        )
    warnings.warn(msg + " " + REDUCED_KMESH_HINT, stacklevel=4)
