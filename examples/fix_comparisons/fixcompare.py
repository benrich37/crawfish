"""Shared helpers for the fix-comparison notebooks in this directory.

Each notebook compares crawfish results before and after one fix from notes/theory_audit.md:

- Fixes that are implemented are compared through their `legacy_*` ElecData setting (old behaviour) against the
  default (corrected behaviour), as required by section 4.0 of the notes.
- Fixes that are not implemented yet are compared against the reference implementations below, which follow the
  target formalism in section 2.3 of the notes. Once a fix lands with the legacy flag listed in `LEGACY_FLAGS`,
  `fix_available` turns True and the notebooks switch to comparing legacy vs. corrected crawfish, checking the
  corrected result against the reference.

Reference implementations work in float64/complex128 from the raw JDFTx output files, so they don't depend on the
crawfish code paths they are meant to check (other than file parsing and orbital indexing).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*Numba.*")

from crawfish.core.elecdata import ElecData
from crawfish.io.data_parsing import get_proj_tju_from_file, get_orb_l_u_from_bandfile_filepath
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from crawfish.utils.typing import REAL_DTYPE
from pymatgen.core.units import Ha_to_eV

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
N2_CALC_DIR = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"

# Finding id -> ElecData setting that restores the pre-fix behaviour. Names for unimplemented fixes are the ones
# proposed in notes/theory_audit.md section 4.0; update them here if the implementation picks different names.
LEGACY_FLAGS = {
    "F1": "legacy_orbital_phase",
    "F2": "legacy_matrices",
    "F3": "legacy_lattice_image",
    "F4": "legacy_fillings",
    "F5": "legacy_kpoint_weights",
    "F6": "legacy_normalization",
}
# F8 (crashing/inverted occupation helpers) is a bug fix without a legacy switch (notes section 4.0), so its
# notebook detects the fix from behaviour instead of a flag.


def fix_available(finding: str) -> bool:
    """Return True if the legacy switch for `finding` exists, i.e. the fix has been implemented."""
    return hasattr(ElecData, LEGACY_FLAGS[finding])


def load_edata(calc_dir: Path = N2_CALC_DIR, **settings) -> ElecData:
    """Return an ElecData with the given settings (e.g. legacy_orbital_phase=True) applied through their setters."""
    edata = ElecData(calc_dir)
    for key, val in settings.items():
        if not hasattr(ElecData, key):
            raise AttributeError(f"ElecData has no setting {key!r} (fix not implemented yet?)")
        setattr(edata, key, val)
    return edata


def savefig(fig, name: str) -> Path:
    """Save a figure as figures/<name>.png and return the path."""
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def to_ev_rel_mu(edata: ElecData, erange: np.ndarray) -> np.ndarray:
    """Energies in Hartree (absolute) -> eV relative to the Fermi level, for plotting."""
    return (np.asarray(erange) - edata.mu) * Ha_to_eV


def last_value(result) -> float:
    """Final value of an integrated (cumulative) crawfish spectrum, i.e. the integral up to the highest band."""
    return float(np.ravel(np.asarray(result[1]))[-1])


def orb_idcs(edata: ElecData, idx: int, orbs: str | list[str] | None) -> list[int]:
    return get_orb_idcs(edata, idx, None, orbs)


# --------------------------------------------------------------------------------------------------------------
# Reference implementation (notes/theory_audit.md, section 2.3)
# --------------------------------------------------------------------------------------------------------------


def jdftx_occupations(calc_dir: Path, nspin: int, shape: tuple[int, int]) -> np.ndarray:
    """Fillings in [0, 1] from the JDFTx `fillings` file (the dump is x2 for spin-unpolarized runs, finding F4)."""
    f = np.fromfile(Path(calc_dir) / "fillings").reshape(shape)
    return f / 2.0 if nspin == 1 else f


def _lowdin(proj_tju: np.ndarray) -> np.ndarray:
    out = np.empty_like(proj_tju)
    for t in range(proj_tju.shape[0]):
        s_uu = proj_tju[t].conj().T @ proj_tju[t]
        eigs, vecs = np.linalg.eigh(s_uu)
        keep = eigs > 1e-10 * eigs.max()
        out[t] = proj_tju[t] @ (vecs[:, keep] / np.sqrt(eigs[keep])) @ vecs[:, keep].conj().T
    return out


class Reference:
    """Reference projected-COHP quantities for one JDFTx calculation.

    basis="lowdin": band-space Löwdin-orthonormalized orbitals (S = I, so COOP vanishes off-diagonal).
    basis="nonorthogonal": band-projected orbitals, C = S^-1 T.
    All matrices follow crawfish's layout, P[t, j, u] = <phi_u|O|psi_tj>, with H_t[u,v] = <chi_u|H|chi_v>.
    """

    def __init__(self, edata: ElecData, basis: str = "lowdin", energy_shift: float = 0.0):
        calc_dir = Path(edata.calc_dir)
        raw = get_proj_tju_from_file(calc_dir / "bandProjections").astype(np.complex128)
        l_u = get_orb_l_u_from_bandfile_filepath(calc_dir / "bandProjections")
        P = raw * (1j ** l_u)[None, None, :]  # F1 phase fix
        self.e_tj = np.fromfile(calc_dir / "eigenvals").reshape(raw.shape[:2])
        self.f_tj = jdftx_occupations(calc_dir, edata.nspin, raw.shape[:2])
        self.w_t = np.asarray(edata.wk_t, dtype=np.float64)
        e_h = self.e_tj - energy_shift
        if basis == "lowdin":
            P = _lowdin(P)
            self.C = P
        elif basis == "nonorthogonal":
            s_conj = np.einsum("tju,tjv->tuv", P.conj(), P)
            self.C = np.einsum("tjw,twu->tju", P, np.linalg.inv(s_conj))
        else:
            raise ValueError(basis)
        self.S = np.einsum("tju,tjv->tuv", P, P.conj())
        self.H = np.einsum("tju,tj,tjv->tuv", P, e_h, P.conj())
        self.D = np.einsum("tj,tju,tjv->tuv", self.f_tj, self.C, self.C.conj())
        self.nproj = P.shape[2]

    def weights_tj(self, orbs_u: list[int], orbs_v: list[int], mode: str = "cohp") -> np.ndarray:
        """w_t * sum_{u in A, v in B, u != v} Re[conj(C_u) M_uv C_v] for M = H (cohp), S (coop), D (cobi), 1 (como)."""
        mats = {"cohp": self.H, "coop": self.S, "cobi": self.D, "como": np.ones((1, self.nproj, self.nproj))}
        M = mats[mode]
        mask = np.zeros((self.nproj, self.nproj))
        for u in orbs_u:
            for v in orbs_v:
                if u != v:
                    mask[u, v] = 1.0
        M = M * mask[None]
        val = np.einsum("tju,tuv,tjv->tj", self.C.conj(), np.broadcast_to(M, (self.C.shape[0],) + M.shape[1:]), self.C)
        return np.asarray(self.w_t[:, None] * val.real, dtype=REAL_DTYPE)

    def integrated(self, orbs_u: list[int], orbs_v: list[int], mode: str = "cohp") -> float:
        """Integral up to the Fermi level (occupation-weighted sum over all states and bands)."""
        return float(np.sum(self.f_tj * self.weights_tj(orbs_u, orbs_v, mode)))

    def band_energy(self) -> float:
        return float(np.sum(self.w_t[:, None] * self.f_tj * self.e_tj))

    def sum_rule(self) -> float:
        """sum over all orbital pairs (including on-site) of the integrated COHP; ~ band energy for small spilling."""
        val = np.einsum("t,tj,tju,tuv,tjv->", self.w_t, self.f_tj, self.C.conj(), self.H, self.C)
        return float(val.real)


def reference_spectrum(edata: ElecData, weights_tj: np.ndarray, erange: np.ndarray, sig: float) -> np.ndarray:
    """Smear reference weights with crawfish's own Gaussian, so reference and crawfish share the broadening."""
    from crawfish.funcs.general import get_generic_spectrum

    _, spectrum = get_generic_spectrum(edata, np.asarray(weights_tj, dtype=REAL_DTYPE), erange=erange, sig=sig)
    return np.asarray(spectrum)
