"""Module for applying JDFTx's space-group operations to band projections (notes/theory_audit.md, finding F12).

A JDFTx operation g = (rot, a) acts on lattice coordinates as x -> rot x + a (`Symmetries.cpp::findSpaceGroup`), so
its Cartesian rotation is S = R rot R^-1, with the lattice vectors as the columns of R. If psi is a Bloch state at
k, g psi = psi(g^-1 r) is one at S k, whose fractional coordinates are rot^-T k. JDFTx's Bloch-sum orbitals are
phi_{a,m,k}(r) = sum_R e^{ik.R} chi_m(r - tau_a - R) with chi_m = f(r) Y_m(r_hat) (SpeciesInfo_internal.h::Vnl_calc;
after the F1 phase fix the orbitals are real). If g takes atom a to atom b, S tau_a + tau_g = tau_b + L_a with L_a a
lattice vector, then

    <phi_{b,m',Sk}| O |g psi_k> = e^{-2 pi i k'.L_a} sum_m M_{m m'} <phi_{a,m,k}| O |psi_k>,

where k' is any k-point equivalent to S k (e.g. the full-mesh point), and Y_m'(S x) = sum_m M_{m m'} Y_m(x) defines
M (a real orthogonal matrix per l). O commutes with g because the augmentation spheres map onto equivalent atoms.
Time reversal (psi(-k) = psi(k)*, collinear spin) conjugates the projections. The derivation is in the notes, F12.
"""

from __future__ import annotations

import numpy as np

# Sample directions for fitting the rotation matrices of the real spherical harmonics (any set on which the Ylm are
# linearly independent works; these are fixed so the result is reproducible)
_YLM_SAMPLES = np.random.default_rng(12).normal(size=(64, 3))
_YLM_SAMPLES /= np.linalg.norm(_YLM_SAMPLES, axis=1)[:, None]


def real_ylm(l: int, xyz: np.ndarray) -> np.ndarray:
    """Return JDFTx's real spherical harmonics of angular momentum l at unit vectors xyz, in shape (..., 2l+1).

    The order (m = -l..l), signs and normalization are those of `jdftx/core/SphericalHarmonics.h` (YlmInternal),
    which JDFTx uses for its atomic orbitals: p = (y, z, x), d = (xy, yz, z^2, xz, x^2-y^2), f = (y(3x^2-y^2), xyz,
    yz^2, z^3, xz^2, z(x^2-y^2), x(x^2-3y^2)). Signs matter: they fix how the orbitals transform.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if l == 0:
        out = [0.28209479177387814 * np.ones_like(x)]
    elif l == 1:
        out = [0.4886025119029199 * y, 0.4886025119029199 * z, 0.4886025119029199 * x]
    elif l == 2:
        out = [
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            -0.31539156525252005 * (x * x + y * y - 2.0 * z * z),
            1.0925484305920792 * x * z,
            0.5462742152960396 * (x - y) * (x + y),
        ]
    elif l == 3:
        out = [
            -0.5900435899266435 * y * (y * y - 3.0 * x * x),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (x * x + y * y - 4.0 * z * z),
            0.3731763325901154 * z * (2.0 * z * z - 3.0 * (x * x + y * y)),
            -0.4570457994644658 * x * (x * x + y * y - 4.0 * z * z),
            1.445305721320277 * (x - y) * (x + y) * z,
            0.5900435899266435 * x * (x * x - 3.0 * y * y),
        ]
    else:
        raise ValueError(f"Real spherical harmonics are only implemented for l <= 3 (got l={l})")
    return np.stack(out, axis=-1)


def get_ylm_rotation(l: int, rot_cart: np.ndarray) -> np.ndarray:
    """Return M, shape (2l+1, 2l+1), with Y_m'(S x) = sum_m M[m, m'] Y_m(x) for the Cartesian rotation S = rot_cart.

    S may be improper (det -1). M is real and orthogonal. JDFTx's `Symmetries::getSphericalMatrices` builds the
    transpose of this matrix the same way, by fitting at sample directions.
    """
    rot_cart = np.asarray(rot_cart, dtype=np.float64)
    a = real_ylm(l, _YLM_SAMPLES)
    b = real_ylm(l, _YLM_SAMPLES @ rot_cart.T)
    m, *_ = np.linalg.lstsq(a, b, rcond=None)
    if not np.allclose(m @ m.T, np.eye(2 * l + 1), atol=1e-8) or not np.allclose(a @ m, b, atol=1e-8):
        raise ValueError(f"Not a rotation of the l={l} harmonics:\n{rot_cart}")
    return m


def get_cartesian_rotation(rot: np.ndarray, lattice_rows: np.ndarray) -> np.ndarray:
    """Return S = R rot R^-1 for an operation in lattice coordinates, with R = lattice_rows.T (columns = vectors).

    Lattice vectors are usually given to ~7 digits (e.g. a sqrt(3)/2 component), so S is only orthogonal to that
    precision. It is replaced by the nearest orthogonal matrix (polar decomposition) once it passes the check.
    """
    lat = np.asarray(lattice_rows, dtype=np.float64).T
    s = lat @ np.asarray(rot, dtype=np.float64) @ np.linalg.inv(lat)
    if not np.allclose(s @ s.T, np.eye(3), atol=1e-5):
        raise ValueError(f"Symmetry operation is not orthogonal in Cartesian coordinates:\n{rot}")
    u, _, vt = np.linalg.svd(s)
    return u @ vt


def get_atom_map(
    frac_coords: np.ndarray, species: list, rot: np.ndarray, a: np.ndarray, tol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, L): operation (rot, a) takes atom i to atom perm[i] in cell L[i], rot x_i + a = x_perm[i] + L[i].

    Raises ValueError if the operation doesn't map the structure onto itself within tol (fractional coordinates).
    """
    frac = np.asarray(frac_coords, dtype=np.float64)
    mapped = frac @ np.asarray(rot, dtype=np.float64).T + np.asarray(a, dtype=np.float64)[None, :]
    perm = np.zeros(len(frac), dtype=np.int64)
    lat_vec = np.zeros((len(frac), 3), dtype=np.int64)
    for i in range(len(frac)):
        diff = mapped[i][None, :] - frac
        resid = np.abs(diff - np.round(diff)).max(axis=1)
        match = [j for j in range(len(frac)) if resid[j] < tol and species[j] == species[i]]
        if len(match) != 1:
            raise ValueError(
                f"Symmetry operation (rot={np.asarray(rot).tolist()}, a={np.asarray(a).tolist()}) maps atom {i} onto "
                f"{len(match)} atoms of the structure (expected 1). The operations don't match this structure."
            )
        perm[i] = match[0]
        lat_vec[i] = np.round(diff[match[0]]).astype(np.int64)
    if len(set(perm.tolist())) != len(perm):
        raise ValueError("Symmetry operation does not permute the atoms")
    return perm, lat_vec


def get_orbital_transform(
    rot: np.ndarray,
    a: np.ndarray,
    lattice_rows: np.ndarray,
    frac_coords: np.ndarray,
    species: list,
    atom_u: np.ndarray,
    l_u: np.ndarray,
    shell_u: np.ndarray,
) -> dict:
    """Return how operation (rot, a) maps projections T(k) onto projections at the image k-point (module docstring).

    The projections at the image are T'[j, v] = e^{-2 pi i k'.L_v} sum_k T[j, idx[v, k]] coef[v, k], with k' the
    image k-point in fractional coordinates.

    Parameters
    ----------
    atom_u, l_u, shell_u : np.ndarray
        Structure site, angular momentum and shell index (same atom, l and shell n) of each orbital, in JDFTx's
        order (l, then shell, then m = -l..l)

    Returns
    -------
    dict
        "idx": int array (nproj, kmax), the source orbitals of each destination orbital (padded with 0);
        "coef": float array (nproj, kmax), their coefficients M[m, m'] (padded with 0);
        "L_v": int array (nproj, 3), the lattice offset L_a of each destination orbital's source atom a.
    """
    atom_u, l_u, shell_u = (np.asarray(x, dtype=np.int64) for x in (atom_u, l_u, shell_u))
    nproj = len(atom_u)
    s_cart = get_cartesian_rotation(rot, lattice_rows)
    perm, lat_vec = get_atom_map(frac_coords, species, rot, a)
    mats = {l: get_ylm_rotation(l, s_cart) for l in sorted(set(l_u.tolist()))}
    kmax = max(2 * l + 1 for l in mats)
    idx = np.zeros((nproj, kmax), dtype=np.int64)
    coef = np.zeros((nproj, kmax), dtype=np.float64)
    l_v = np.zeros((nproj, 3), dtype=np.int64)
    orbs_by_atom = {int(s): np.flatnonzero(atom_u == s) for s in np.unique(atom_u)}
    for src, src_orbs in orbs_by_atom.items():
        dst_orbs = orbs_by_atom[int(perm[src])]
        if len(dst_orbs) != len(src_orbs) or np.any(l_u[dst_orbs] != l_u[src_orbs]):
            raise ValueError(f"Atoms {src} and {perm[src]} have different orbitals but are related by symmetry")
        p = 0
        while p < len(src_orbs):
            l = int(l_u[src_orbs[p]])
            block = src_orbs[p : p + 2 * l + 1]
            if len(block) != 2 * l + 1 or np.any(shell_u[block] != shell_u[block[0]]):
                raise ValueError("Orbital shells are not complete (2l+1 consecutive m); spinor runs aren't supported")
            for mp in range(2 * l + 1):
                v = dst_orbs[p + mp]
                idx[v, : 2 * l + 1] = block
                coef[v, : 2 * l + 1] = mats[l][:, mp]
                l_v[v] = lat_vec[src]
            p += 2 * l + 1
    return {"idx": idx, "coef": coef, "L_v": l_v}


def apply_orbital_transform(proj_ju: np.ndarray, transform: dict, k_frac: np.ndarray) -> np.ndarray:
    """Return T'[j, v] = e^{-2 pi i k.L_v} sum_k T[j, idx[v, k]] coef[v, k] for one state (see get_orbital_transform)."""
    phase_v = np.exp(-2j * np.pi * (transform["L_v"] @ np.asarray(k_frac, dtype=np.float64)))
    mixed = np.einsum("jvk,vk->jv", proj_ju[:, transform["idx"]], transform["coef"])
    return mixed * phase_v[None, :]
